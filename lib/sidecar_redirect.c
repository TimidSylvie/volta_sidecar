/*
 * This is the little switchboard that points one process at the sidecar driver instead of the primary one.
 *
 * NVIDIA userspace hardcodes the normal /dev/nvidia* and /proc/driver/nvidia* paths. When this library is active,
 * it rewrites those paths to /dev/nvidia-<suffix> and /proc/driver/nvidia_<suffix>. The application keeps speaking
 * ordinary CUDA/NVML while every open and stat quietly lands on the legacy driver.
 *
 * Activation:
 *   NVIDIA_SIDECAR=1                       always redirect (what `nvidia-sidecar run` sets)
 *   CUDA_VISIBLE_DEVICES=<unified indices>  redirect when every listed GPU is a legacy one (indices at or above the
 *                                          number of GPUs on the open driver, or legacy UUIDs); the variable is then
 *                                          rewritten to the legacy side's own numbering (CUDA_DEVICE_ORDER=PCI_BUS_ID)
 *   otherwise                              inert: every call passes straight through
 * NVIDIA_SIDECAR_SUFFIX (default "sidecar") selects the driver instance.
 */
#define _GNU_SOURCE
#include <dirent.h>
#include <dlfcn.h>
#include <fcntl.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <sys/un.h>
#include <sys/types.h>
#include <unistd.h>

static int active = 0;
static char dev_dir[64] = "/dev/nvidia-sidecar";
static char proc_dir[64] = "/proc/driver/nvidia_sidecar";
static char proc_uvm[80] = "/proc/driver/nvidia_sidecar-uvm";
static int open_major = 195, open_uvm_major = -1;

static int major_of(const char *name) {   /* Read /proc/devices before path redirection is active. */
    FILE *f = fopen("/proc/devices", "r");
    char line[128], n[96];
    int m, found = -1;
    if (!f) return -1;
    while (fgets(line, sizeof line, f))
        if (sscanf(line, "%d %95s", &m, n) == 2 && strcmp(n, name) == 0) { found = m; break; }
    fclose(f);
    return found;
}

static int count_open_gpus(void) {        /* The open driver creates one directory per GPU here. */
    DIR *d = opendir("/proc/driver/nvidia/gpus");
    struct dirent *e;
    int n = 0;
    if (!d) return 0;
    while ((e = readdir(d))) if (e->d_name[0] != '.') n++;
    closedir(d);
    return n;
}

static int is_legacy_uuid(const char *tok) {   /* Look for this GPU UUID in the sidecar's own inventory. */
    char path[256], line[256];
    int hit = 0;
    snprintf(path, sizeof path, "%s/gpus", proc_dir);
    DIR *d = opendir(path);
    if (!d) return 0;
    struct dirent *e;
    while (!hit && (e = readdir(d))) {
        if (e->d_name[0] == '.') continue;
        char info[768];
        snprintf(info, sizeof info, "%s/%s/information", path, e->d_name);
        FILE *f = fopen(info, "r");
        if (!f) continue;
        while (fgets(line, sizeof line, f))
            if (strstr(line, "GPU UUID") && strstr(line, tok)) { hit = 1; break; }
        fclose(f);
    }
    closedir(d);
    return hit;
}

__attribute__((constructor)) static void sidecar_init(void) {
    const char *suffix = getenv("NVIDIA_SIDECAR_SUFFIX");
    if (!suffix || !*suffix) suffix = "sidecar";
    snprintf(dev_dir, sizeof dev_dir, "/dev/nvidia-%s", suffix);
    snprintf(proc_dir, sizeof proc_dir, "/proc/driver/nvidia_%s", suffix);
    snprintf(proc_uvm, sizeof proc_uvm, "/proc/driver/nvidia_%s-uvm", suffix);
    char ctl[96];
    snprintf(ctl, sizeof ctl, "%s/nvidiactl", dev_dir);
    if (access(ctl, F_OK) != 0) return;                        /* No sidecar means this library does nothing. */
    const char *force = getenv("NVIDIA_SIDECAR");
    const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
    if (force && strcmp(force, "0") == 0) return;
    if (force && *force && strcmp(force, "0") != 0) {
        active = 1;
    } else if (cvd && *cvd) {
        int host = count_open_gpus(), all_legacy = 1, any = 0;
        char buf[512], out[512] = "";
        snprintf(buf, sizeof buf, "%s", cvd);
        for (char *save = NULL, *t = strtok_r(buf, ",", &save); t; t = strtok_r(NULL, ",", &save)) {
            while (*t == ' ') t++;
            any = 1;
            char piece[96];
            if (strncmp(t, "GPU-", 4) == 0 && is_legacy_uuid(t)) {
                snprintf(piece, sizeof piece, "%s", t);
            } else {
                char *end;
                long i = strtol(t, &end, 10);
                if (*end || i < host) { all_legacy = 0; break; }
                snprintf(piece, sizeof piece, "%ld", i - host);
            }
            if (*out) strncat(out, ",", sizeof out - strlen(out) - 1);
            strncat(out, piece, sizeof out - strlen(out) - 1);
        }
        if (any && all_legacy) {
            active = 1;
            setenv("CUDA_VISIBLE_DEVICES", out, 1);
            setenv("CUDA_DEVICE_ORDER", "PCI_BUS_ID", 1);
        }
    }
    if (!active) return;
    open_uvm_major = major_of("nvidia-uvm");
    setenv("NVIDIA_SIDECAR", "1", 1);   /* Keep child workers on the same driver as their parent. */
}

/* The CUDA/NVML proxies load this into a private namespace and call here to force only that namespace onto the
 * sidecar, regardless of the process environment. A true result means the sidecar is actually up. */
int nvidia_sidecar_redirect_force(const char *suffix) {
    if (!suffix || !*suffix) suffix = "sidecar";
    snprintf(dev_dir, sizeof dev_dir, "/dev/nvidia-%s", suffix);
    snprintf(proc_dir, sizeof proc_dir, "/proc/driver/nvidia_%s", suffix);
    snprintf(proc_uvm, sizeof proc_uvm, "/proc/driver/nvidia_%s-uvm", suffix);
    char ctl[96];
    snprintf(ctl, sizeof ctl, "%s/nvidiactl", dev_dir);
    active = 0;
    if (access(ctl, F_OK) != 0) return 0;
    open_uvm_major = major_of("nvidia-uvm");
    active = 1;
    return 1;
}

/* Rewrite NVIDIA-owned paths and leave everything else completely alone. */
static const char *map(const char *p, char *buf, size_t n) {
    if (!active || !p) return p;
    if (strncmp(p, "/dev/nvidia", 11) == 0) {
        const char *rest = p + 11;     /* ctl, a GPU number, -uvm, -uvm-tools, -modeset or -caps/... */
        if (strcmp(rest, "ctl") == 0 || strcmp(rest, "-uvm") == 0 || strcmp(rest, "-uvm-tools") == 0 ||
            (rest[0] >= '0' && rest[0] <= '9')) {
            snprintf(buf, n, "%s/nvidia%s", dev_dir, rest);
            return buf;
        }
        return p;
    }
    if (strncmp(p, "/dev/char/", 10) == 0) {
        int ma, mi;
        if (sscanf(p + 10, "%d:%d", &ma, &mi) == 2) {
            if (ma == open_major && mi == 255) { snprintf(buf, n, "%s/nvidiactl", dev_dir); return buf; }
            if (ma == open_major && mi < 248) { snprintf(buf, n, "%s/nvidia%d", dev_dir, mi); return buf; }
            if (ma == open_uvm_major && mi == 0) { snprintf(buf, n, "%s/nvidia-uvm", dev_dir); return buf; }
            if (ma == open_uvm_major && mi == 1) { snprintf(buf, n, "%s/nvidia-uvm-tools", dev_dir); return buf; }
        }
        return p;
    }
    if (strncmp(p, "/proc/driver/nvidia-uvm", 23) == 0 && (p[23] == '/' || p[23] == 0)) {
        snprintf(buf, n, "%s%s", proc_uvm, p + 23);
        return buf;
    }
    if (strncmp(p, "/proc/driver/nvidia", 19) == 0 && (p[19] == '/' || p[19] == 0)) {
        snprintf(buf, n, "%s%s", proc_dir, p + 19);
        return buf;
    }
    return p;
}

#define REAL(ret, name, ...) static ret (*real_##name)(__VA_ARGS__); if (!real_##name) real_##name = dlsym(RTLD_NEXT, #name)
#define VREAL(ret, name, ver, ...) static ret (*real_##name)(__VA_ARGS__); if (!real_##name) real_##name = dlvsym(RTLD_NEXT, #name, ver)
#define MAPPED char _b[512]; path = map(path, _b, sizeof _b)

static int open_has_mode(int flags) {
    if (flags & O_CREAT) return 1;
#ifdef O_TMPFILE
    if ((flags & O_TMPFILE) == O_TMPFILE) return 1;
#endif
    return 0;
}

int open(const char *path, int flags, ...) {
    REAL(int, open, const char *, int, ...); MAPPED;
    if (open_has_mode(flags)) {
        va_list a; va_start(a, flags); mode_t m = va_arg(a, mode_t); va_end(a);
        return real_open(path, flags, m);
    }
    return real_open(path, flags);
}
int open64(const char *path, int flags, ...) {
    REAL(int, open64, const char *, int, ...); MAPPED;
    if (open_has_mode(flags)) {
        va_list a; va_start(a, flags); mode_t m = va_arg(a, mode_t); va_end(a);
        return real_open64(path, flags, m);
    }
    return real_open64(path, flags);
}
int openat(int dir, const char *path, int flags, ...) {
    REAL(int, openat, int, const char *, int, ...); MAPPED;
    if (open_has_mode(flags)) {
        va_list a; va_start(a, flags); mode_t m = va_arg(a, mode_t); va_end(a);
        return real_openat(dir, path, flags, m);
    }
    return real_openat(dir, path, flags);
}
int openat64(int dir, const char *path, int flags, ...) {
    REAL(int, openat64, int, const char *, int, ...); MAPPED;
    if (open_has_mode(flags)) {
        va_list a; va_start(a, flags); mode_t m = va_arg(a, mode_t); va_end(a);
        return real_openat64(dir, path, flags, m);
    }
    return real_openat64(dir, path, flags);
}
FILE *fopen(const char *path, const char *mode) { REAL(FILE *, fopen, const char *, const char *); MAPPED; return real_fopen(path, mode); }
FILE *fopen64(const char *path, const char *mode) { REAL(FILE *, fopen64, const char *, const char *); MAPPED; return real_fopen64(path, mode); }
DIR *opendir(const char *path) { REAL(DIR *, opendir, const char *); MAPPED; return real_opendir(path); }
int access(const char *path, int mode) { REAL(int, access, const char *, int); MAPPED; return real_access(path, mode); }
int faccessat(int d, const char *path, int mode, int fl) { REAL(int, faccessat, int, const char *, int, int); MAPPED; return real_faccessat(d, path, mode, fl); }
ssize_t readlink(const char *path, char *b, size_t n) { REAL(ssize_t, readlink, const char *, char *, size_t); MAPPED; return real_readlink(path, b, n); }
int stat(const char *path, struct stat *s) { REAL(int, stat, const char *, struct stat *); MAPPED; return real_stat(path, s); }
int lstat(const char *path, struct stat *s) { REAL(int, lstat, const char *, struct stat *); MAPPED; return real_lstat(path, s); }
int stat64(const char *path, struct stat64 *s) { REAL(int, stat64, const char *, struct stat64 *); MAPPED; return real_stat64(path, s); }
int lstat64(const char *path, struct stat64 *s) { REAL(int, lstat64, const char *, struct stat64 *); MAPPED; return real_lstat64(path, s); }
int fstatat(int d, const char *path, struct stat *s, int f) { REAL(int, fstatat, int, const char *, struct stat *, int); MAPPED; return real_fstatat(d, path, s, f); }
int fstatat64(int d, const char *path, struct stat64 *s, int f) { REAL(int, fstatat64, int, const char *, struct stat64 *, int); MAPPED; return real_fstatat64(d, path, s, f); }
int statx(int d, const char *path, int f, unsigned m, struct statx *s) { REAL(int, statx, int, const char *, int, unsigned, struct statx *); MAPPED; return real_statx(d, path, f, m, s); }
/* libcuda and libnvidia-ml still import these older glibc stat entry points. */
int __xstat(int v, const char *path, struct stat *s) { VREAL(int, __xstat, "GLIBC_2.2.5", int, const char *, struct stat *); MAPPED; return real___xstat(v, path, s); }
int __xstat64(int v, const char *path, struct stat64 *s) { VREAL(int, __xstat64, "GLIBC_2.2.5", int, const char *, struct stat64 *); MAPPED; return real___xstat64(v, path, s); }
int __lxstat(int v, const char *path, struct stat *s) { VREAL(int, __lxstat, "GLIBC_2.2.5", int, const char *, struct stat *); MAPPED; return real___lxstat(v, path, s); }
int __lxstat64(int v, const char *path, struct stat64 *s) { VREAL(int, __lxstat64, "GLIBC_2.2.5", int, const char *, struct stat64 *); MAPPED; return real___lxstat64(v, path, s); }
int __fxstatat(int v, int d, const char *path, struct stat *s, int f) { VREAL(int, __fxstatat, "GLIBC_2.4", int, int, const char *, struct stat *, int); MAPPED; return real___fxstatat(v, d, path, s, f); }

/* Each libcuda copy tries to own the same abstract cuda-uvmfd socket. Give the sidecar copy a -sc suffix for both
 * bind and connect, so the two drivers stop colliding while sidecar peers can still find one another. */
static const struct sockaddr *uvmfd_name(const struct sockaddr *a, socklen_t *len, struct sockaddr_un *buf) {
    static const char tag[] = "cuda-uvmfd-";
    if (!active || !a || a->sa_family != AF_UNIX || *len <= (socklen_t)(sizeof(sa_family_t) + sizeof tag)) return a;
    const struct sockaddr_un *u = (const struct sockaddr_un *)a;
    if (u->sun_path[0] != 0 || strncmp(u->sun_path + 1, tag, sizeof tag - 1) != 0) return a;
    size_t n = *len - sizeof(sa_family_t);                   /* Abstract names include their leading NUL. */
    while (n > 1 && u->sun_path[n - 1] == 0) n--;            /* Do not carry trailing NULs into the suffix. */
    if (n + 3 > sizeof buf->sun_path) return a;
    memset(buf, 0, sizeof *buf);
    buf->sun_family = AF_UNIX;
    memcpy(buf->sun_path, u->sun_path, n);
    memcpy(buf->sun_path + n, "-sc", 3);
    *len = (socklen_t)(sizeof(sa_family_t) + n + 3);
    return (const struct sockaddr *)buf;
}
int bind(int fd, const struct sockaddr *a, socklen_t len) {
    REAL(int, bind, int, const struct sockaddr *, socklen_t);
    struct sockaddr_un b; a = uvmfd_name(a, &len, &b); return real_bind(fd, a, len);
}
int connect(int fd, const struct sockaddr *a, socklen_t len) {
    REAL(int, connect, int, const struct sockaddr *, socklen_t);
    struct sockaddr_un b; a = uvmfd_name(a, &len, &b); return real_connect(fd, a, len);
}
