/*
 * This is the trick that makes two real NVIDIA drivers look like one libcuda.
 *
 * One process holds two isolated copies of NVIDIA's real libcuda: the host copy opens the primary driver, while the
 * legacy copy goes through the path redirect and opens nvidia_<suffix>. A call chooses its copy from the strongest
 * clue it carries: device ordinal, owned object, device pointer, or the calling thread's current context. Merged
 * ordinals are translated back to each driver's local numbering. Cross-driver copies go through host memory.
 * gen_cuda_proxy.py supplies the boring per-function wrappers around this routing core.
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <unistd.h>
#include <dirent.h>

#ifndef REAL_CUDA
#define REAL_CUDA "/usr/local/lib/nvidia-sidecar/real/libcuda.so.1"
#endif
#ifndef HOST_LOADER
#define HOST_LOADER "/usr/local/lib/nvidia-sidecar/libnvidia-sidecar-cuda-host.so"
#endif
#ifndef LEGACY_LOADER
#define LEGACY_LOADER "/usr/local/lib/nvidia-sidecar/libnvidia-sidecar-cuda-legacy.so"
#endif

typedef int CUresult;
typedef unsigned long long CUdeviceptr;
#define CUDA_SUCCESS 0
#define CUDA_ERROR_INVALID_VALUE 1
#define CUDA_ERROR_OUT_OF_MEMORY 2
#define CUDA_ERROR_NOT_INITIALIZED 3
#define CUDA_ERROR_NO_DEVICE 100
#define CUDA_ERROR_INVALID_DEVICE 101
#define CUDA_ERROR_INVALID_HANDLE 400
#define CUDA_ERROR_PEER_ACCESS_UNSUPPORTED 217
#define CUDA_ERROR_NOT_FOUND 500
#define CUDA_ERROR_NOT_SUPPORTED 801
#define CU_POINTER_ATTRIBUTE_CONTEXT 1

enum { HOST = 0, LEG = 1 };

extern const char *const cp_names[];   /* Generated export names in function-table order. */
extern const int cp_count;
struct cp_gpa_entry { const char *name; int version; unsigned pt; const char *export_name; };
extern const struct cp_gpa_entry cp_gpa_map[];
extern const int cp_gpa_count;

void **cp_fn[2];                       /* One real function table for each driver copy. */
int cp_have[2];
int cp_ndev[2];                        /* Each copy's device count after cuInit. */
static pthread_once_t cp_once = PTHREAD_ONCE_INIT;
static int cp_debug;
static __thread int cur_side = -1;
static __thread int side_stack[128];
static __thread int side_sp;

int cp_trace_all;                      /* NVIDIA_SIDECAR_DEBUG=2 logs every call, not only failures. */
void cp_trace(const char *name, int side, CUresult r);
#define DBG(...) do { if (cp_debug) { fprintf(stderr, "[cuda-proxy] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

int cp_index(const char *name) {
    for (int i = 0; i < cp_count; i++) if (strcmp(cp_names[i], name) == 0) return i;
    return -1;
}
static void **resolve_all(void *lib) {
    void **t = calloc((size_t)cp_count, sizeof(void *));
    for (int i = 0; t && i < cp_count; i++) t[i] = dlsym(lib, cp_names[i]);
    return t;
}
/* CUDA_VISIBLE_DEVICES describes the merged list. Split it into one private list per driver, while remembering the
 * user's requested order so the public ordinals can be rebuilt the same way after cuInit. */
#define MAX_DEV 64
static int ord_side[MAX_DEV], ord_local[MAX_DEV], nord = -1;      /* merged ordinal -> driver copy and local ordinal */
static int plan_side[MAX_DEV], nplan = -1;                         /* requested order, captured before cuInit */
static char side_cvd[2][512];
static int side_listed[2];
static int proc_gpus(const char *dir, char uuids[][64], int max) {  /* Read one driver's GPUs in PCI order. */
    char path[640], line[256], names[MAX_DEV][64];
    int n = 0;
    DIR *d = opendir(dir);
    if (!d) return 0;
    for (struct dirent *e; (e = readdir(d)) && n < MAX_DEV;)
        if (e->d_name[0] != '.') snprintf(names[n++], 64, "%.63s", e->d_name);
    closedir(d);
    qsort(names, (size_t)n, 64, (int (*)(const void *, const void *))strcmp);
    for (int i = 0; i < n && i < max; i++) {
        uuids[i][0] = 0;
        snprintf(path, sizeof path, "%.511s/%.63s/information", dir, names[i]);
        FILE *f = fopen(path, "r");
        if (!f) continue;
        while (fgets(line, sizeof line, f))
            if (sscanf(line, "GPU UUID: %63s", uuids[i]) == 1) break;
        fclose(f);
    }
    return n < max ? n : max;
}
static int unified_inventory_present(void) {
    char host_uuid[1][64], legacy_uuid[1][64], dir[128];
    const char *sfx = getenv("NVIDIA_SIDECAR_SUFFIX");
    snprintf(dir, sizeof dir, "/proc/driver/nvidia_%s/gpus", sfx && *sfx ? sfx : "sidecar");
    return proc_gpus("/proc/driver/nvidia/gpus", host_uuid, 1) > 0 && proc_gpus(dir, legacy_uuid, 1) > 0;
}
static void plan_visible(int want_host, int want_leg) {
    const char *cvd = getenv("CUDA_VISIBLE_DEVICES");
    if (!cvd || !want_host || !want_leg) return;
    char hu[MAX_DEV][64], lu[MAX_DEV][64], dir[128], buf[512];
    const char *sfx = getenv("NVIDIA_SIDECAR_SUFFIX");
    snprintf(dir, sizeof dir, "/proc/driver/nvidia_%s/gpus", sfx && *sfx ? sfx : "sidecar");
    int nh = proc_gpus("/proc/driver/nvidia/gpus", hu, MAX_DEV), nl = proc_gpus(dir, lu, MAX_DEV);
    nplan = 0;
    snprintf(buf, sizeof buf, "%s", cvd);
    for (char *save = NULL, *t = strtok_r(buf, ",", &save); t && nplan < MAX_DEV; t = strtok_r(NULL, ",", &save)) {
        while (*t == ' ') t++;
        int side = -1; char piece[96];
        if (strncmp(t, "GPU-", 4) == 0) {                  /* A full UUID or a unique UUID prefix. */
            for (int i = 0; i < nh && side < 0; i++) if (strncmp(hu[i], t, strlen(t)) == 0) side = HOST;
            for (int i = 0; i < nl && side < 0; i++) if (strncmp(lu[i], t, strlen(t)) == 0) side = LEG;
            snprintf(piece, sizeof piece, "%s", t);
        } else if (strncmp(t, "MIG-", 4) == 0) {
            side = HOST; snprintf(piece, sizeof piece, "%s", t);
        } else {
            char *end; long i = strtol(t, &end, 10);
            if (end == t || *end || i < 0) break;           /* CUDA itself stops at the first bad entry. */
            if (i < nh) { side = HOST; snprintf(piece, sizeof piece, "%ld", i); }
            else if (i < nh + nl) { side = LEG; snprintf(piece, sizeof piece, "%ld", i - nh); }
        }
        if (side < 0) break;
        char *dst = side_cvd[side];
        if (*dst) strncat(dst, ",", sizeof side_cvd[side] - strlen(dst) - 1);
        strncat(dst, piece, sizeof side_cvd[side] - strlen(dst) - 1);
        side_listed[side]++;
        plan_side[nplan++] = side;
    }
    DBG("CUDA_VISIBLE_DEVICES=%s -> main driver \"%s\", sidecar \"%s\"", cvd, side_cvd[HOST], side_cvd[LEG]);
}
static void private_env(void *ns, const char *cvd) {   /* Give one namespace its own environment. */
    extern char **environ;
    int (*ce)(void) = (int (*)(void))dlsym(ns, "clearenv");
    int (*pe)(char *) = (int (*)(char *))dlsym(ns, "putenv");
    int (*se)(const char *, const char *, int) = (int (*)(const char *, const char *, int))dlsym(ns, "setenv");
    if (!ce || !pe || !se) { DBG("no libc in namespace: environment shared"); return; }
    size_t n = 0;
    while (environ[n]) n++;
    char **snap = malloc((n + 1) * sizeof *snap);       /* clearenv drops libc's pointer, not the strings themselves. */
    if (!snap) return;
    memcpy(snap, environ, (n + 1) * sizeof *snap);
    ce();
    for (size_t i = 0; i < n; i++)
        if (strncmp(snap[i], "CUDA_VISIBLE_DEVICES=", 21) != 0 && strncmp(snap[i], "NVIDIA_SIDECAR=", 15) != 0) pe(snap[i]);
    if (cvd) se("CUDA_VISIBLE_DEVICES", cvd, 1);
    free(snap);
}

static void *load_side(const char *loader, int legacy) {
    void *l = dlmopen(LM_ID_NEWLM, loader, RTLD_NOW | RTLD_LOCAL);
    if (!l) { DBG("loader %s: %s", loader, dlerror()); return NULL; }
    if (nplan >= 0) private_env(l, side_cvd[legacy ? LEG : HOST]);
    if (legacy) {
        const char *sfx = getenv("NVIDIA_SIDECAR_SUFFIX");
        int (*force)(const char *) = (int (*)(const char *))dlsym(l, "nvidia_sidecar_redirect_force");
        if (!force || !force(sfx && *sfx ? sfx : "sidecar")) { DBG("sidecar not up"); return NULL; }
    }
    Lmid_t lm;
    if (dlinfo(l, RTLD_DI_LMID, &lm) != 0) return NULL;
    return dlmopen(lm, REAL_CUDA, RTLD_NOW | RTLD_LOCAL);   /* Absolute path so lookup cannot loop back here. */
}
static void setup(void) {
    cp_debug = getenv("NVIDIA_SIDECAR_DEBUG") != NULL;
    cp_trace_all = cp_debug && atoi(getenv("NVIDIA_SIDECAR_DEBUG")) >= 2;
    const char *mode = getenv("NVIDIA_SIDECAR_CUDA"), *routed = getenv("NVIDIA_SIDECAR");
    int want_host = 1, want_leg = !(mode && strcmp(mode, "0") == 0);
    if (routed && *routed && strcmp(routed, "0") != 0) { want_host = 0; want_leg = 1; }
    /* NCCL sees one inventory and assumes its local SHM path can exchange CUDA IPC handles. It cannot: an IPC
     * handle belongs to exactly one kernel driver. Default mixed-driver runs to sockets, but never override an
     * explicit choice made by the user. */
    if (want_host && want_leg && unified_inventory_present() && !getenv("NCCL_SHM_DISABLE")) {
        setenv("NCCL_SHM_DISABLE", "1", 0);
        DBG("defaulted NCCL_SHM_DISABLE=1 for cross-driver correctness");
    }
    plan_visible(want_host, want_leg);
    if (nplan >= 0) { want_host = side_listed[HOST] > 0; want_leg = side_listed[LEG] > 0; }
    /* Keep redirection asleep while its library loads; load_side turns it on only inside the legacy namespace. */
    int set_marker = want_leg && !routed;
    if (set_marker) setenv("NVIDIA_SIDECAR", "0", 1);
    if (want_host) { void *h = load_side(HOST_LOADER, 0); if (h) { cp_fn[HOST] = resolve_all(h); cp_have[HOST] = cp_fn[HOST] != NULL; } }
    if (want_leg) { void *l = load_side(LEGACY_LOADER, 1); if (l) { cp_fn[LEG] = resolve_all(l); cp_have[LEG] = cp_fn[LEG] != NULL; } }
    if (set_marker) unsetenv("NVIDIA_SIDECAR");
    DBG("host copy %d, legacy copy %d", cp_have[HOST], cp_have[LEG]);
}
void cp_init(void) { pthread_once(&cp_once, setup); }

int cp_cur(void) {
    if (cur_side >= 0 && cp_have[cur_side]) return cur_side;
    return (cp_have[HOST] && (cp_ndev[HOST] > 0 || !cp_have[LEG])) ? HOST : LEG;
}
void cp_set_cur(int side) { cur_side = side; }

/* Remember which driver created each opaque CUDA object. */
#define OWN_BITS 16
#define OWN_DEAD ((uintptr_t)1)
static struct { uintptr_t key; int side; } own_tab[1 << OWN_BITS];
static pthread_mutex_t own_mu = PTHREAD_MUTEX_INITIALIZER;
static inline unsigned own_hash(uintptr_t k) { return (unsigned)((k >> 4) * 2654435761u) & ((1u << OWN_BITS) - 1); }
void cp_own(const void *h, int side) {
    uintptr_t k = (uintptr_t)h;
    if (k < 16) return;                            /* NULL and CUDA's special legacy/per-thread stream handles. */
    pthread_mutex_lock(&own_mu);
    unsigned dead = 1u << OWN_BITS;
    for (unsigned i = own_hash(k), n = 0; n < (1u << OWN_BITS); i = (i + 1) & ((1u << OWN_BITS) - 1), n++) {
        if (own_tab[i].key == k) { own_tab[i].side = side; dead = 1u << OWN_BITS; break; }
        if (own_tab[i].key == OWN_DEAD && dead == (1u << OWN_BITS)) dead = i;
        if (own_tab[i].key == 0) {
            if (dead != (1u << OWN_BITS)) i = dead;
            own_tab[i].key = k; own_tab[i].side = side; dead = 1u << OWN_BITS; break;
        }
    }
    if (dead != (1u << OWN_BITS)) { own_tab[dead].key = k; own_tab[dead].side = side; }
    pthread_mutex_unlock(&own_mu);
}
void cp_disown(const void *h) {
    uintptr_t k = (uintptr_t)h;
    if (k < 16) return;
    pthread_mutex_lock(&own_mu);
    for (unsigned i = own_hash(k), n = 0; n < (1u << OWN_BITS); i = (i + 1) & ((1u << OWN_BITS) - 1), n++) {
        if (own_tab[i].key == k) { own_tab[i].key = OWN_DEAD; break; }
        if (own_tab[i].key == 0) break;
    }
    pthread_mutex_unlock(&own_mu);
}
int cp_owner(const void *h, int dflt) {
    uintptr_t k = (uintptr_t)h;
    if (k < 16) return dflt;
    int r = dflt;
    pthread_mutex_lock(&own_mu);
    for (unsigned i = own_hash(k), n = 0; n < (1u << OWN_BITS); i = (i + 1) & ((1u << OWN_BITS) - 1), n++) {
        if (own_tab[i].key == k) { r = own_tab[i].side; break; }
        if (own_tab[i].key == 0) break;
    }
    pthread_mutex_unlock(&own_mu);
    return r;
}

/* Ask which driver owns a device pointer, then cache the answer per 2 MiB page. */
#define PTR_BITS 14
static struct { uintptr_t g; int side; } ptr_tab[1 << PTR_BITS];
static pthread_mutex_t ptr_mu = PTHREAD_MUTEX_INITIALIZER;
void cp_ptr_own(CUdeviceptr p, int side) {
    uintptr_t g = (uintptr_t)(p >> 21) + 1;
    pthread_mutex_lock(&ptr_mu);
    unsigned i = (unsigned)(g * 2654435761u) & ((1u << PTR_BITS) - 1);
    ptr_tab[i].g = g; ptr_tab[i].side = side;
    pthread_mutex_unlock(&ptr_mu);
}
void cp_ptr_forget(CUdeviceptr p, int side) {
    if (!p) return;
    uintptr_t g = (uintptr_t)(p >> 21) + 1;
    unsigned i = (unsigned)(g * 2654435761u) & ((1u << PTR_BITS) - 1);
    pthread_mutex_lock(&ptr_mu);
    if (ptr_tab[i].g == g && ptr_tab[i].side == side) ptr_tab[i].g = 0;
    pthread_mutex_unlock(&ptr_mu);
}
int cp_ptr_owner(CUdeviceptr p) {                   /* -1 means neither driver recognizes it as device memory. */
    if (!p) return -1;
    uintptr_t g = (uintptr_t)(p >> 21) + 1;
    unsigned i = (unsigned)(g * 2654435761u) & ((1u << PTR_BITS) - 1);
    pthread_mutex_lock(&ptr_mu);
    int s = ptr_tab[i].g == g ? ptr_tab[i].side : -2;
    pthread_mutex_unlock(&ptr_mu);
    static int ix = -2;
    if (ix == -2) ix = cp_index("cuPointerGetAttribute");
    if (s >= 0 && ix >= 0 && cp_have[s] && cp_fn[s][ix]) {
        void *ctx = NULL;                           /* Recheck hits because a freed allocation can leave cached pages. */
        CUresult r = ((CUresult (*)(void *, int, CUdeviceptr))cp_fn[s][ix])(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, p);
        if (r == CUDA_SUCCESS) return s;
        cp_ptr_forget(p, s);
    }
    for (int side = 0; side < 2 && ix >= 0; side++) {
        if (!cp_have[side] || !cp_fn[side][ix]) continue;
        void *ctx = NULL;
        CUresult r = ((CUresult (*)(void *, int, CUdeviceptr))cp_fn[side][ix])(&ctx, CU_POINTER_ATTRIBUTE_CONTEXT, p);
        if (r == CUDA_SUCCESS) { cp_ptr_own(p, side); return side; }
    }
    return -1;
}

/* Translate between public merged ordinals and each driver's private ordinals. */
static void build_order(void) {                       /* Build merged numbering after cuInit. */
    int seen[2] = {0, 0};
    nord = 0;
    if (nplan >= 0) {
        for (int k = 0; k < nplan && nord < MAX_DEV; k++) {
            int s = plan_side[k];
            if (!cp_have[s] || seen[s] >= cp_ndev[s]) continue;
            ord_side[nord] = s; ord_local[nord++] = seen[s]++;
        }
        return;
    }
    for (int s = 0; s < 2; s++)
        for (int i = 0; cp_have[s] && i < cp_ndev[s] && nord < MAX_DEV; i++) { ord_side[nord] = s; ord_local[nord++] = i; }
}
int cp_dev_side(int g, int *local) {
    if (nord < 0) build_order();
    if (g >= 0 && g < nord) { *local = ord_local[g]; return ord_side[g]; }
    *local = 1 << 20;                                  /* Feed an impossible ordinal to the real driver's validator. */
    return cp_have[HOST] ? HOST : LEG;
}
int cp_dev_global(int side, int local) {
    if (nord < 0) build_order();
    for (int g = 0; g < nord; g++) if (ord_side[g] == side && ord_local[g] == local) return g;
    return local;
}

/* Calls whose routing needs more context than the generated wrappers have. */
#define REALFN(side, name, type) ((type)cp_fn[side][cp_index(name)])

CUresult cuInit(unsigned flags) {
    cp_init();
    CUresult rh = CUDA_ERROR_NO_DEVICE, rl = CUDA_ERROR_NO_DEVICE;
    for (int s = 0; s < 2; s++) {
        if (!cp_have[s]) continue;
        CUresult r = REALFN(s, "cuInit", CUresult (*)(unsigned))(flags);
        int n = 0;
        if (r == CUDA_SUCCESS) REALFN(s, "cuDeviceGetCount", CUresult (*)(int *))(&n);
        else cp_have[s] = 0;
        cp_ndev[s] = n;
        if (s == HOST) rh = r; else rl = r;
        DBG("cuInit side %d: rc %d, %d device(s)", s, r, n);
    }
    build_order();
    return rh == CUDA_SUCCESS ? rh : rl;
}
CUresult cuDeviceGetCount(int *count) {
    cp_init();
    if (!count) return CUDA_ERROR_INVALID_VALUE;
    if (nord < 0) build_order();
    *count = nord;
    return CUDA_SUCCESS;
}
CUresult cuDeviceGet(int *dev, int ordinal) {
    cp_init();
    int local, side = cp_dev_side(ordinal, &local);
    if (!cp_have[side]) return CUDA_ERROR_INVALID_DEVICE;
    int d;
    CUresult r = REALFN(side, "cuDeviceGet", CUresult (*)(int *, int))(&d, local);
    if (r == CUDA_SUCCESS && dev) *dev = cp_dev_global(side, d);
    return r;
}
CUresult cuDeviceGetByPCIBusId(int *dev, const char *id) {
    cp_init();
    for (int s = 0; s < 2; s++) {
        if (!cp_have[s]) continue;
        int d;
        if (REALFN(s, "cuDeviceGetByPCIBusId", CUresult (*)(int *, const char *))(&d, id) == CUDA_SUCCESS) {
            if (dev) *dev = cp_dev_global(s, d);
            return CUDA_SUCCESS;
        }
    }
    return CUDA_ERROR_INVALID_VALUE;
}

/* VMM and memory-pool structs hide the device ordinal inside CUmemLocation. Translate that field too, or merged
 * device 1 reaches the sidecar as local device 1 even though its V100 is local device 0. CUmemPoolProps stays 88
 * bytes; newer headers merely gave names to bytes that used to live in its reserved tail. */
struct cp_mem_location { int type, id; };
struct cp_mem_pool_props {
    int alloc_type;
    int handle_types;
    struct cp_mem_location location;
    void *win32_security_attributes;
    size_t max_size;
    unsigned short usage;
    unsigned char reserved[54];
};
struct cp_mem_allocation_prop {
    int type;
    int requested_handle_types;
    struct cp_mem_location location;
    void *win32_handle_metadata;
    unsigned char allocation_flags[8];
};
struct cp_mem_access_desc {
    struct cp_mem_location location;
    unsigned int flags;                                  /* CUmemAccess_flags is a four-byte enum. */
};
_Static_assert(sizeof(struct cp_mem_location) == 8, "CUmemLocation ABI changed");
_Static_assert(sizeof(struct cp_mem_allocation_prop) == 32, "CUmemAllocationProp ABI changed");
_Static_assert(sizeof(struct cp_mem_pool_props) == 88, "CUmemPoolProps ABI changed");
_Static_assert(sizeof(struct cp_mem_access_desc) == 12, "CUmemAccessDesc ABI changed");
static int translate_mem_location(struct cp_mem_location *location) {
    int side = cp_cur();
    if (location->type == 1) {                             /* CU_MEM_LOCATION_TYPE_DEVICE */
        int local;
        side = cp_dev_side(location->id, &local);
        location->id = local;
    }
    return side;
}
CUresult cuMemPoolCreate(void **pool, const struct cp_mem_pool_props *props) {
    cp_init();
    if (!pool || !props) return CUDA_ERROR_INVALID_VALUE;
    struct cp_mem_pool_props local_props = *props;
    int side = translate_mem_location(&local_props.location);
    CUresult r = REALFN(side, "cuMemPoolCreate",
                        CUresult (*)(void **, const struct cp_mem_pool_props *))(pool, &local_props);
    if (r || cp_trace_all) cp_trace("cuMemPoolCreate", side, r);
    if (r == CUDA_SUCCESS) cp_own(*pool, side);
    return r;
}
CUresult cuMemGetAllocationGranularity(size_t *granularity, const struct cp_mem_allocation_prop *props, int option) {
    cp_init();
    if (!granularity || !props) return CUDA_ERROR_INVALID_VALUE;
    struct cp_mem_allocation_prop local_props = *props;
    int side = translate_mem_location(&local_props.location);
    CUresult r = REALFN(side, "cuMemGetAllocationGranularity",
                        CUresult (*)(size_t *, const struct cp_mem_allocation_prop *, int))
                       (granularity, &local_props, option);
    if (r || cp_trace_all) cp_trace("cuMemGetAllocationGranularity", side, r);
    return r;
}
CUresult cuMemCreate(unsigned long long *handle, size_t size, const struct cp_mem_allocation_prop *props,
                     unsigned long long flags) {
    cp_init();
    if (!handle || !props) return CUDA_ERROR_INVALID_VALUE;
    struct cp_mem_allocation_prop local_props = *props;
    int side = translate_mem_location(&local_props.location);
    CUresult r = REALFN(side, "cuMemCreate",
                        CUresult (*)(unsigned long long *, size_t, const struct cp_mem_allocation_prop *, unsigned long long))
                       (handle, size, &local_props, flags);
    if (r || cp_trace_all) cp_trace("cuMemCreate", side, r);
    if (r == CUDA_SUCCESS) cp_own((void *)(uintptr_t)*handle, side);
    return r;
}
CUresult cuMemRelease(unsigned long long handle) {
    cp_init();
    int side = cp_owner((void *)(uintptr_t)handle, cp_cur());
    CUresult r = REALFN(side, "cuMemRelease", CUresult (*)(unsigned long long))(handle);
    if (r || cp_trace_all) cp_trace("cuMemRelease", side, r);
    if (r == CUDA_SUCCESS) cp_disown((void *)(uintptr_t)handle);
    return r;
}
static CUresult local_access_descs(int side, const struct cp_mem_access_desc *desc, size_t count,
                                   struct cp_mem_access_desc **local_out, size_t *count_out) {
    *local_out = NULL;
    *count_out = 0;
    if (!count) return CUDA_SUCCESS;
    if (!desc) return CUDA_ERROR_INVALID_VALUE;
    struct cp_mem_access_desc *local = malloc(count * sizeof *local);
    if (!local) return CUDA_ERROR_OUT_OF_MEMORY;
    for (size_t i = 0; i < count; i++) {
        local[*count_out] = desc[i];
        int location_side = translate_mem_location(&local[*count_out].location);
        if (location_side == side) {
            (*count_out)++;
        } else if (local[*count_out].flags != 0) {
            free(local);
            return CUDA_ERROR_NOT_SUPPORTED;               /* Never pretend foreign read/write access was granted. */
        }
    }
    *local_out = local;
    return CUDA_SUCCESS;
}
CUresult cuMemSetAccess(CUdeviceptr ptr, size_t size, const struct cp_mem_access_desc *desc, size_t count) {
    cp_init();
    int side = cp_ptr_owner(ptr);
    if (side < 0) side = cp_cur();
    struct cp_mem_access_desc *local;
    size_t local_count;
    CUresult r = local_access_descs(side, desc, count, &local, &local_count);
    if (r == CUDA_SUCCESS && local_count)
        r = REALFN(side, "cuMemSetAccess",
                   CUresult (*)(CUdeviceptr, size_t, const struct cp_mem_access_desc *, size_t))
                  (ptr, size, local, local_count);
    free(local);
    if (r || cp_trace_all) cp_trace("cuMemSetAccess", side, r);
    return r;
}
CUresult cuMemGetAccess(unsigned long long *flags, const struct cp_mem_location *location, CUdeviceptr ptr) {
    cp_init();
    if (!flags || !location) return CUDA_ERROR_INVALID_VALUE;
    int side = cp_ptr_owner(ptr);
    if (side < 0) side = cp_cur();
    struct cp_mem_location local = *location;
    if (translate_mem_location(&local) != side) {
        *flags = 0;                                       /* CU_MEM_ACCESS_FLAGS_PROT_NONE */
        return CUDA_SUCCESS;
    }
    CUresult r = REALFN(side, "cuMemGetAccess",
                        CUresult (*)(unsigned long long *, const struct cp_mem_location *, CUdeviceptr))
                       (flags, &local, ptr);
    if (r || cp_trace_all) cp_trace("cuMemGetAccess", side, r);
    return r;
}
CUresult cuMemPoolSetAccess(void *pool, const struct cp_mem_access_desc *desc, size_t count) {
    cp_init();
    int side = cp_owner(pool, cp_cur());
    struct cp_mem_access_desc *local;
    size_t local_count;
    CUresult r = local_access_descs(side, desc, count, &local, &local_count);
    if (r == CUDA_SUCCESS && local_count)
        r = REALFN(side, "cuMemPoolSetAccess",
                   CUresult (*)(void *, const struct cp_mem_access_desc *, size_t))(pool, local, local_count);
    free(local);
    if (r || cp_trace_all) cp_trace("cuMemPoolSetAccess", side, r);
    return r;
}
CUresult cuMemPoolGetAccess(unsigned int *flags, void *pool, const struct cp_mem_location *location) {
    cp_init();
    if (!flags || !location) return CUDA_ERROR_INVALID_VALUE;
    int side = cp_owner(pool, cp_cur());
    struct cp_mem_location local = *location;
    if (translate_mem_location(&local) != side) {
        *flags = 0;
        return CUDA_SUCCESS;
    }
    CUresult r = REALFN(side, "cuMemPoolGetAccess",
                        CUresult (*)(unsigned int *, void *, struct cp_mem_location *))
                       (flags, pool, &local);
    if (r || cp_trace_all) cp_trace("cuMemPoolGetAccess", side, r);
    return r;
}
CUresult cuMemPrefetchAsync_v2(CUdeviceptr ptr, size_t count, struct cp_mem_location location,
                               unsigned flags, void *stream) {
    cp_init();
    int side = cp_ptr_owner(ptr);
    if (side < 0) side = cp_owner(stream, cp_cur());
    int location_side = translate_mem_location(&location);
    if (location.type == 1 && location_side != side) return CUDA_ERROR_NOT_SUPPORTED;
    if (stream && cp_owner(stream, side) != side) return CUDA_ERROR_NOT_SUPPORTED;
    CUresult r = REALFN(side, "cuMemPrefetchAsync_v2",
                        CUresult (*)(CUdeviceptr, size_t, struct cp_mem_location, unsigned, void *))
                       (ptr, count, location, flags, stream);
    if (r || cp_trace_all) cp_trace("cuMemPrefetchAsync_v2", side, r);
    return r;
}
CUresult cuMemAdvise_v2(CUdeviceptr ptr, size_t count, int advice, struct cp_mem_location location) {
    cp_init();
    int side = cp_ptr_owner(ptr);
    if (side < 0) side = cp_cur();
    int location_side = translate_mem_location(&location);
    if (location.type == 1 && location_side != side) return CUDA_ERROR_NOT_SUPPORTED;
    CUresult r = REALFN(side, "cuMemAdvise_v2",
                        CUresult (*)(CUdeviceptr, size_t, int, struct cp_mem_location))
                       (ptr, count, advice, location);
    if (r || cp_trace_all) cp_trace("cuMemAdvise_v2", side, r);
    return r;
}
CUresult cuCtxGetCurrent(void **ctx) { cp_init(); return REALFN(cp_cur(), "cuCtxGetCurrent", CUresult (*)(void **))(ctx); }
CUresult cuCtxSetCurrent(void *ctx) {
    cp_init();
    int side = ctx ? cp_owner(ctx, cp_cur()) : cp_cur();
    CUresult r = REALFN(side, "cuCtxSetCurrent", CUresult (*)(void *))(ctx);
    if (r || cp_trace_all) cp_trace("cuCtxSetCurrent", side, r);
    if (r == CUDA_SUCCESS && ctx) cur_side = side;
    return r;
}
static CUresult push(const char *name, void *ctx) {
    cp_init();
    int side = cp_owner(ctx, cp_cur());
    CUresult r = REALFN(side, name, CUresult (*)(void *))(ctx);
    if (r || cp_trace_all) cp_trace(name, side, r);
    if (r == CUDA_SUCCESS) { if (side_sp < 128) side_stack[side_sp++] = cp_cur(); cur_side = side; }
    return r;
}
static CUresult pop(const char *name, void **ctx) {
    cp_init();
    CUresult r = REALFN(cp_cur(), name, CUresult (*)(void **))(ctx);
    if (r == CUDA_SUCCESS && side_sp > 0) cur_side = side_stack[--side_sp];
    return r;
}
CUresult cuCtxPushCurrent(void *ctx) { return push("cuCtxPushCurrent", ctx); }
CUresult cuCtxPushCurrent_v2(void *ctx) { return push("cuCtxPushCurrent_v2", ctx); }
CUresult cuCtxPopCurrent(void **ctx) { return pop("cuCtxPopCurrent", ctx); }
CUresult cuCtxPopCurrent_v2(void **ctx) { return pop("cuCtxPopCurrent_v2", ctx); }

CUresult cuDeviceCanAccessPeer(int *can, int dev, int peer) {
    cp_init();
    int ld, lp, sd = cp_dev_side(dev, &ld), sp = cp_dev_side(peer, &lp);
    if (sd != sp) { if (can) *can = 0; return CUDA_SUCCESS; }
    return REALFN(sd, "cuDeviceCanAccessPeer", CUresult (*)(int *, int, int))(can, ld, lp);
}
CUresult cuCtxEnablePeerAccess(void *peer, unsigned flags) {
    cp_init();
    int side = cp_owner(peer, cp_cur());
    if (side != cp_cur()) return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
    return REALFN(side, "cuCtxEnablePeerAccess", CUresult (*)(void *, unsigned))(peer, flags);
}
CUresult cuCtxDisablePeerAccess(void *peer) {
    cp_init();
    int side = cp_owner(peer, cp_cur());
    if (side != cp_cur()) return CUDA_ERROR_PEER_ACCESS_UNSUPPORTED;
    return REALFN(side, "cuCtxDisablePeerAccess", CUresult (*)(void *))(peer);
}

CUresult cp_cross_wait_event(void *event, int side) {
    cp_init();
    if (side < HOST || side > LEG || !cp_have[side]) return CUDA_ERROR_INVALID_HANDLE;
    int ix = cp_index("cuEventSynchronize");
    if (ix < 0 || !cp_fn[side][ix]) return CUDA_ERROR_NOT_SUPPORTED;
    CUresult r = ((CUresult (*)(void *))cp_fn[side][ix])(event);
    if (r || cp_trace_all) cp_trace("cuStreamWaitEvent(cross-driver fallback)", side, r);
    return r;
}

/* Cross-driver copies have no peer DMA path, so move them through a bounded host buffer. Async entry points use the
 * same synchronized fallback: drain both contexts, copy, then return. Ordering stays correct even though overlap is
 * lost. */
static CUresult with_ctx_of(int side, CUdeviceptr p, void **ctx_out) {
    *ctx_out = NULL;
    return ((CUresult (*)(void *, int, CUdeviceptr))cp_fn[side][cp_index("cuPointerGetAttribute")])(ctx_out, CU_POINTER_ATTRIBUTE_CONTEXT, p);
}
CUresult cp_stage_copy(CUdeviceptr dst, int dside, CUdeviceptr src, int sside, size_t n) {
    void *sctx, *dctx, *tmp;
    CUresult r;
    if ((r = with_ctx_of(sside, src, &sctx)) || (r = with_ctx_of(dside, dst, &dctx))) return r;
    CUresult (*spush)(void *) = REALFN(sside, "cuCtxPushCurrent_v2", CUresult (*)(void *));
    CUresult (*spop)(void **) = REALFN(sside, "cuCtxPopCurrent_v2", CUresult (*)(void **));
    CUresult (*dpush)(void *) = REALFN(dside, "cuCtxPushCurrent_v2", CUresult (*)(void *));
    CUresult (*dpop)(void **) = REALFN(dside, "cuCtxPopCurrent_v2", CUresult (*)(void **));
    CUresult (*sync_s)(void) = REALFN(sside, "cuCtxSynchronize", CUresult (*)(void));
    CUresult (*sync_d)(void) = REALFN(dside, "cuCtxSynchronize", CUresult (*)(void));
    CUresult (*dtoh)(void *, CUdeviceptr, size_t) = REALFN(sside, "cuMemcpyDtoH_v2", CUresult (*)(void *, CUdeviceptr, size_t));
    CUresult (*htod)(CUdeviceptr, const void *, size_t) = REALFN(dside, "cuMemcpyHtoD_v2", CUresult (*)(CUdeviceptr, const void *, size_t));
    const size_t chunk = 64u << 20;
    char *buf = malloc(n < chunk ? (n ? n : 1) : chunk);
    if (!buf) return 2; /* CUDA_ERROR_OUT_OF_MEMORY */
    r = spush(sctx);                                                /* Finish pending source work first. */
    if (!r) {
        r = sync_s();
        CUresult pr = spop(&tmp);
        if (!r) r = pr;
    }
    if (!r) r = dpush(dctx);                                       /* Then establish destination ordering. */
    if (!r) {
        r = sync_d();
        CUresult pr = dpop(&tmp);
        if (!r) r = pr;
    }
    for (size_t off = 0; !r && off < n; off += chunk) {
        size_t k = n - off < chunk ? n - off : chunk;
        r = spush(sctx);
        if (!r) {
            r = dtoh(buf, src + off, k);
            CUresult pr = spop(&tmp);
            if (!r) r = pr;
        }
        if (!r) r = dpush(dctx);
        if (!r) {
            r = htod(dst + off, buf, k);
            CUresult pr = dpop(&tmp);
            if (!r) r = pr;
        }
    }
    free(buf);
    DBG("staged copy %zu bytes side %d -> %d: rc %d", n, sside, dside, r);
    return r;
}

/* cuGetProcAddress must hand out routed proxy functions, never a raw function from just one driver. */
static void *self_handle(void) {
    static void *self;
    if (!self) { Dl_info di; if (dladdr((void *)cp_init, &di)) self = dlopen(di.dli_fname, RTLD_NOW | RTLD_NOLOAD); }
    return self;
}
static void *proxy_for(void *real_fn) {
    Dl_info di;
    if (!real_fn || !dladdr(real_fn, &di) || !di.dli_sname || di.dli_saddr != real_fn) return NULL;
    return dlsym(self_handle(), di.dli_sname);
}
static void *proxy_for_request(const char *symbol, int ver, unsigned long long flags) {
    const char *export_name = NULL;
    int best = -1, pt = (flags & 2u) != 0;          /* CU_GET_PROC_ADDRESS_PER_THREAD_DEFAULT_STREAM */
    for (int i = 0; i < cp_gpa_count; i++) {
        const struct cp_gpa_entry *e = &cp_gpa_map[i];
        if (e->pt != (unsigned)pt || strcmp(e->name, symbol) != 0 || e->version > ver || e->version < best) continue;
        best = e->version;
        export_name = e->export_name;
    }
    return export_name ? dlsym(self_handle(), export_name) : NULL;
}
CUresult cuGetProcAddress_v2(const char *symbol, void **pfn, int ver, unsigned long long flags, int *status) {
    cp_init();
    int side = cp_have[HOST] ? HOST : LEG;
    void *real = NULL;
    CUresult r = REALFN(side, "cuGetProcAddress_v2", CUresult (*)(const char *, void **, int, unsigned long long, int *))(symbol, &real, ver, flags, status);
    if (r != CUDA_SUCCESS || !real) { if (pfn) *pfn = real; return r; }
    static int calls; int k = calls++;
    void *mine = proxy_for_request(symbol, ver, flags);
    if (!mine) mine = proxy_for(real);
    if (getenv("NVIDIA_SIDECAR_GPA_LOG")) fprintf(stderr, "[gpa] %d %s %d\n", k, symbol, ver);
    DBG("cuGetProcAddress(%s, %d) -> %s", symbol, ver, mine ? "proxy" : "UNMAPPED (blocked)");
    if (mine) { if (pfn) *pfn = mine; return r; }
    if (pfn) *pfn = NULL;
    if (status) *status = 1;                         /* CU_GET_PROC_ADDRESS_SYMBOL_NOT_FOUND */
    return CUDA_SUCCESS;
}
CUresult cuGetProcAddress(const char *symbol, void **pfn, int ver, unsigned long long flags) {
    int status;
    return cuGetProcAddress_v2(symbol, pfn, ver, flags, &status);
}

/* NVIDIA's private export tables need the same per-call routing as the public API. */
#define MAX_TABLES 64
#define MAX_ENTRIES 512
static struct { unsigned char id[16]; void **merged; void **real[2]; size_t n; } tables[MAX_TABLES];
static int ntables;
static pthread_mutex_t tab_mu = PTHREAD_MUTEX_INITIALIZER;
void *cp_table_route(unsigned slot, void *arg1) {     /* Dynamic trampolines land here with slot and argument 1. */
    unsigned t = slot / MAX_ENTRIES, e = slot % MAX_ENTRIES;
    int side = cp_owner(arg1, cp_cur());
    if (cp_trace_all) DBG("table %u slot %u arg1 %p -> side %d", t, e, arg1, side);
    void *f = tables[t].real[side] ? tables[t].real[side][e] : NULL;
    return f ? f : tables[t].real[HOST][e];
}
extern void cp_table_dispatch(void);                 /* Assembly preserves args, asks for a route, then jumps. */
static void *make_tramp(unsigned slot) {
    static unsigned char *page; static size_t used, cap;
    if (!page || used + 16 > cap) {
        cap = 1 << 16;
        page = mmap(NULL, cap, PROT_READ | PROT_WRITE | PROT_EXEC, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        if (page == MAP_FAILED) { page = NULL; return NULL; }
        used = 0;
    }
    unsigned char *p = page + used;
    /* mov $slot, %r11d ; movabs $cp_table_dispatch, %rax ; jmp *%rax */
    p[0] = 0x41; p[1] = 0xbb; memcpy(p + 2, &slot, 4);
    p[6] = 0x48; p[7] = 0xb8; uintptr_t d = (uintptr_t)cp_table_dispatch; memcpy(p + 8, &d, 8);
    p[16] = 0xff; p[17] = 0xe0;
    used += 24;
    return p;
}
static int in_text(void *lib_fn_sample, void *p) {   /* Is p code from the same object as a known function? */
    Dl_info a, b;
    return dladdr(lib_fn_sample, &a) && dladdr(p, &b) && a.dli_fbase == b.dli_fbase;
}
/* cudart hashes the tables it received, so the proxy must answer that integrity check itself. */
static const unsigned char UUID_INTEGRITY[16] = {0xD4,0x08,0x20,0x55,0xBD,0xE6,0x70,0x4B,0x8D,0x34,0xBA,0x12,0x3C,0x66,0xE1,0xF2};
static const unsigned char UUID_CUDART[16]    = {0x6B,0xD5,0xFB,0x6C,0x5B,0xF4,0xE7,0x4A,0x89,0x87,0xD9,0x39,0x12,0xFD,0x9D,0xF9};
struct sc_ic_input { uint32_t driver_version, version, process, thread; const void *cudart_table, *integrity_table, *fn_address; uint64_t unix_seconds; };
struct sc_ic_device { unsigned char uuid[16]; int32_t pci_domain, pci_bus, pci_device; };
void sc_integrity_hash(const struct sc_ic_input *, const struct sc_ic_device *, unsigned, uint64_t[2]);
CUresult cuDriverGetVersion(int *);                   /* The proxy's own merged-view exports. */
CUresult cuDeviceGetCount(int *);
CUresult cuDeviceGet(int *, int);
CUresult cuDeviceGetUuid_v2(void *, int);
CUresult cuDeviceGetAttribute(int *, int, int);
CUresult cuGetExportTable(const void **pp, const void *uuid);
static CUresult proxy_integrity_check(uint32_t version, uint64_t unix_seconds, uint64_t *result) {
    const void *ic = NULL, *rt = NULL;
    int drv = 0, n = 0;
    if (cuGetExportTable(&ic, UUID_INTEGRITY) || cuGetExportTable(&rt, UUID_CUDART)) return CUDA_ERROR_NOT_INITIALIZED;
    CUresult r = cuDriverGetVersion(&drv);
    if (r == CUDA_SUCCESS) r = cuDeviceGetCount(&n);
    if (r != CUDA_SUCCESS) return r;
    struct sc_ic_device dev[64];
    memset(dev, 0, sizeof dev);
    if (n > 64) n = 64;
    for (int i = 0; i < n && r == CUDA_SUCCESS; i++) {          /* Hash the same merged list cudart can see. */
        int d;
        if ((r = cuDeviceGet(&d, i)) || (r = cuDeviceGetUuid_v2(dev[i].uuid, d)) ||
            (r = cuDeviceGetAttribute(&dev[i].pci_domain, 50, d)) ||
            (r = cuDeviceGetAttribute(&dev[i].pci_bus, 33, d)) ||
            (r = cuDeviceGetAttribute(&dev[i].pci_device, 34, d))) break;
    }
    if (r != CUDA_SUCCESS) return r;
    struct sc_ic_input in = {(uint32_t)drv, version, (uint32_t)getpid(), (uint32_t)pthread_self(), rt, ic,
                             ((void *const *)ic)[1], unix_seconds};
    sc_integrity_hash(&in, dev, (unsigned)n, result);
    DBG("integrity_check(%u) answered for %d device(s)", version, n);
    return CUDA_SUCCESS;
}

/* A few private-table calls hide their device or context somewhere other than argument 1, so they get typed
 * wrappers. The signatures come from ZLUDA's dark_api/src/lib.rs. Everything else uses the generic trampoline. */
static const unsigned char UUID_CLS[16]    = {0xC6,0x93,0x33,0x6E,0x11,0x21,0xDF,0x11,0xA8,0xC3,0x68,0xF3,0x55,0xD8,0x95,0x93};
static const unsigned char UUID_BYPASS[16] = {0x0C,0xA5,0x0B,0x8C,0x10,0x04,0x92,0x9A,0x89,0xA7,0xD0,0xDF,0x10,0xE7,0x72,0x86};
static const unsigned char UUID_DEVEXT[16] = {0xB1,0x05,0x41,0xE1,0xF7,0xC7,0xC7,0x4A,0x9F,0x64,0xF2,0x23,0xBE,0x99,0xF1,0xE2};
static void *slot_real(const unsigned char *uuid, unsigned e, int side);
#define SLOT(uuid, e, side, type) ((type)slot_real(uuid, e, side))
typedef void *CTX;
static CUresult rt_module_from_cubin(void **mod, const void *fatbin) {             /* CUDART_INTERFACE[1] */
    int s = cp_cur();
    CUresult r = SLOT(UUID_CUDART, 1, s, CUresult (*)(void **, const void *))(mod, fatbin);
    if (r || cp_trace_all) cp_trace("rt_module_from_cubin", s, r);
    if (r == CUDA_SUCCESS && mod) cp_own(*mod, s);
    return r;
}
static CUresult rt_fn2(CTX *pctx, int dev) {                                         /* CUDART_INTERFACE[2] */
    int local, s = cp_dev_side(dev, &local);
    CUresult r = SLOT(UUID_CUDART, 2, s, CUresult (*)(CTX *, int))(pctx, local);
    if (r || cp_trace_all) cp_trace("rt_fn2", s, r);
    if (r == CUDA_SUCCESS && pctx) cp_own(*pctx, s);
    return r;
}
static CUresult rt_primary_retain_flags(int dev, unsigned flags) {                  /* CUDART_INTERFACE[4] */
    int local, s = cp_dev_side(dev, &local);
    return SLOT(UUID_CUDART, 4, s, CUresult (*)(int, unsigned))(local, flags);
}
static CUresult rt_module_ext1(void **mod, const void *fatbin, void *a3, void *a4, unsigned a5) {   /* [6] */
    int s = cp_cur();
    CUresult r = SLOT(UUID_CUDART, 6, s, CUresult (*)(void **, const void *, void *, void *, unsigned))(mod, fatbin, a3, a4, a5);
    if (r || cp_trace_all) cp_trace("rt_module_ext1", s, r);
    if (r == CUDA_SUCCESS && mod) cp_own(*mod, s);
    return r;
}
static void rt_fn7(size_t a) {                                                       /* [7]: both copies */
    for (int s = 0; s < 2; s++) if (cp_have[s] && slot_real(UUID_CUDART, 7, s)) SLOT(UUID_CUDART, 7, s, void (*)(size_t))(a);
}
static CUresult rt_module_ext2(const void *hdr, void **mod, void *a3, void *a4, unsigned a5) {      /* [8] */
    int s = cp_cur();
    CUresult r = SLOT(UUID_CUDART, 8, s, CUresult (*)(const void *, void **, void *, void *, unsigned))(hdr, mod, a3, a4, a5);
    if (r || cp_trace_all) cp_trace("rt_module_ext2", s, r);
    if (r == CUDA_SUCCESS && mod) cp_own(*mod, s);
    return r;
}
static CUresult rt_load_compilers(void) {                                            /* [12]: both copies */
    CUresult r = CUDA_SUCCESS;
    for (int s = 0; s < 2; s++)
        if (cp_have[s] && slot_real(UUID_CUDART, 12, s)) { CUresult q = SLOT(UUID_CUDART, 12, s, CUresult (*)(void))(); if (!r) r = q; }
    return r;
}
static CUresult cls_get(void **value, CTX ctx, void *key) {                         /* CLS[2]: context is arg 2 */
    int s = cp_owner(ctx, cp_cur());
    CUresult r = SLOT(UUID_CLS, 2, s, CUresult (*)(void **, CTX, void *))(value, ctx, key);
    if (r || cp_trace_all) cp_trace("cls_get", s, r);
    return r;
}
static CUresult bypass_ctx_create(CTX *pctx, unsigned flags, int dev) {             /* CTX_CREATE_BYPASS[1] */
    int local, s = cp_dev_side(dev, &local);
    CUresult r = SLOT(UUID_BYPASS, 1, s, CUresult (*)(CTX *, unsigned, int))(pctx, flags, local);
    if (r || cp_trace_all) cp_trace("bypass_ctx_create", s, r);
    if (r == CUDA_SUCCESS && pctx) cp_own(*pctx, s);
    return r;
}
static CUresult devext_attr(int dev, unsigned attr, int unk, size_t *res) {         /* DEVICE_EXTENDED_RT[5] */
    int local, s = cp_dev_side(dev, &local);
    return SLOT(UUID_DEVEXT, 5, s, CUresult (*)(int, unsigned, int, size_t *))(local, attr, unk, res);
}
static CUresult devext_13(unsigned char *res, int dev) {                            /* DEVICE_EXTENDED_RT[13] */
    int local, s = cp_dev_side(dev, &local);
    return SLOT(UUID_DEVEXT, 13, s, CUresult (*)(unsigned char *, int))(res, local);
}
static const struct { const unsigned char *uuid; unsigned slot; void *fn; } typed_slots[] = {
    {UUID_CUDART, 1, (void *)rt_module_from_cubin}, {UUID_CUDART, 2, (void *)rt_fn2},
    {UUID_CUDART, 4, (void *)rt_primary_retain_flags}, {UUID_CUDART, 6, (void *)rt_module_ext1},
    {UUID_CUDART, 7, (void *)rt_fn7}, {UUID_CUDART, 8, (void *)rt_module_ext2}, {UUID_CUDART, 12, (void *)rt_load_compilers},
    {UUID_CLS, 2, (void *)cls_get}, {UUID_BYPASS, 1, (void *)bypass_ctx_create},
    {UUID_DEVEXT, 5, (void *)devext_attr}, {UUID_DEVEXT, 13, (void *)devext_13},
};

CUresult cuGetExportTable(const void **pp, const void *uuid) {
    cp_init();
    pthread_mutex_lock(&tab_mu);
    for (int i = 0; i < ntables; i++)
        if (memcmp(tables[i].id, uuid, 16) == 0) { *pp = tables[i].merged; pthread_mutex_unlock(&tab_mu); return CUDA_SUCCESS; }
    void **real[2] = {NULL, NULL};
    CUresult r = CUDA_ERROR_NOT_FOUND;
    for (int s = 0; s < 2; s++) {
        if (!cp_have[s]) continue;
        const void *t = NULL;
        CUresult rs = REALFN(s, "cuGetExportTable", CUresult (*)(const void **, const void *))(&t, uuid);
        if (rs == CUDA_SUCCESS) { real[s] = (void **)t; r = rs; }
    }
    int base = cp_have[HOST] && real[HOST] ? HOST : LEG;
    if (r != CUDA_SUCCESS || ntables >= MAX_TABLES || !real[base]) {
        pthread_mutex_unlock(&tab_mu);
        if (r == CUDA_SUCCESS) *pp = real[base];
        return r;
    }
    /* Most tables start with a byte size; a few, such as context-local storage, start directly with code. */
    size_t bytes = (size_t)(uintptr_t)real[base][0], n = bytes >= 16 && bytes < MAX_ENTRIES * 8 ? bytes / 8 : 0;
    int sized = n > 0;
    void *sample = cp_fn[base][cp_index("cuInit")];
    if (!n) { for (n = 1; n < MAX_ENTRIES && in_text(sample, real[base][n]); n++) ; }
    void **m = calloc(n + 1, sizeof(void *));
    int t = ntables++;
    memcpy(tables[t].id, uuid, 16);
    tables[t].real[HOST] = real[HOST]; tables[t].real[LEG] = real[LEG]; tables[t].n = n; tables[t].merged = m;
    int wrapped = 0;
    for (size_t e = 0; e < n; e++) {
        void *v = real[base][e];
        if ((e > 0 || !sized) && in_text(sample, v) && real[HOST] && real[LEG]) { m[e] = make_tramp((unsigned)(t * MAX_ENTRIES + e)); wrapped++; }
        else m[e] = v;                                   /* Keep size/data slots and one-sided tables untouched. */
    }
    if (memcmp(uuid, UUID_INTEGRITY, 16) == 0 && n > 1) m[1] = (void *)proxy_integrity_check;
    if (real[HOST] && real[LEG])
        for (size_t k = 0; k < sizeof typed_slots / sizeof typed_slots[0]; k++)
            if (memcmp(uuid, typed_slots[k].uuid, 16) == 0 && typed_slots[k].slot < n && m[typed_slots[k].slot] != real[base][typed_slots[k].slot])
                m[typed_slots[k].slot] = typed_slots[k].fn;     /* Replace only slots already proven to hold code. */
    *pp = m;
    pthread_mutex_unlock(&tab_mu);
    { const unsigned char *u = uuid;
      DBG("export table %d {%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x}: %zu entries, %d dispatched", t,
          u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15], n, wrapped); }
    return CUDA_SUCCESS;
}

/* Unknown signatures still route safely through register-preserving trampolines. */
void *cp_generic_route(void *arg1, int idx) {
    cp_init();
    int side = cp_owner(arg1, cp_cur());
    if (!cp_have[side]) side = cp_have[HOST] ? HOST : LEG;
    void *f = cp_fn[side] ? cp_fn[side][idx] : NULL;
    return f;
}

static void *slot_real(const unsigned char *uuid, unsigned e, int side) {
    for (int i = 0; i < ntables; i++)
        if (memcmp(tables[i].id, uuid, 16) == 0) {
            void **t = tables[i].real[side] ? tables[i].real[side] : tables[i].real[HOST];
            return t && e < tables[i].n ? t[e] : NULL;
        }
    return NULL;
}

void cp_trace(const char *name, int side, CUresult r) {
    DBG("%s on %s -> %d", name, side ? "legacy" : "host", r);
}

/* cudart treats CUlibrary and CUkernel as context-independent, but each real driver only knows its own objects.
 * Load the fatbinary into both copies, expose one public handle, and translate it back to the right real handle on
 * every call. */
#define PAIR_BITS 14
#define PAIR_DEAD ((void *)1)
static struct { void *pub; void *h[2]; void *lib; } pair_tab[1 << PAIR_BITS];
static pthread_mutex_t pair_mu = PTHREAD_MUTEX_INITIALIZER;
static inline unsigned pair_hash(const void *k) { return (unsigned)(((uintptr_t)k >> 4) * 2654435761u) & ((1u << PAIR_BITS) - 1); }
static int pair_slot(const void *pub) {                    /* Reads are lock-free after release publication. */
    for (unsigned i = pair_hash(pub), n = 0; n < (1u << PAIR_BITS); i = (i + 1) & ((1u << PAIR_BITS) - 1), n++) {
        void *k = __atomic_load_n(&pair_tab[i].pub, __ATOMIC_ACQUIRE);
        if (k == pub) return (int)i;
        if (!k) return -1;
    }
    return -1;
}
static void pair_put(void *pub, void *h0, void *h1, void *lib) {
    pthread_mutex_lock(&pair_mu);
    int i = pair_slot(pub);
    if (i < 0)
        for (unsigned j = pair_hash(pub), n = 0; n < (1u << PAIR_BITS); j = (j + 1) & ((1u << PAIR_BITS) - 1), n++)
            if (!pair_tab[j].pub || pair_tab[j].pub == PAIR_DEAD) { i = (int)j; break; }
    if (i >= 0) {
        pair_tab[i].h[0] = h0; pair_tab[i].h[1] = h1; pair_tab[i].lib = lib;
        __atomic_store_n(&pair_tab[i].pub, pub, __ATOMIC_RELEASE);
    }
    pthread_mutex_unlock(&pair_mu);
}
void *cp_pair_get(const void *h, int side) {               /* Paired handle -> this copy; ordinary handles pass through. */
    int i = pair_slot(h);
    return i < 0 ? (void *)h : pair_tab[i].h[side];
}
void *cp_pair_pub(const void *h, int side) {               /* Turn a real library/kernel handle back into its public one. */
    for (unsigned i = 0; i < (1u << PAIR_BITS); i++) {
        void *k = __atomic_load_n(&pair_tab[i].pub, __ATOMIC_ACQUIRE);
        if (k && k != PAIR_DEAD && pair_tab[i].h[side] == h) return k;
    }
    return (void *)h;
}

typedef CUresult (*lib_load_t)(void **, const void *, void *, void **, unsigned, void *, void **, unsigned);
static CUresult lib_load(const char *name, void **lib, const void *src, void *jo, void **jv, unsigned nj,
                         void *lo, void **lv, unsigned nl) {
    cp_init();
    if (!lib) return CUDA_ERROR_INVALID_VALUE;
    void *h[2] = {NULL, NULL};
    CUresult first = CUDA_ERROR_NO_DEVICE;
    int ok = 0;
    for (int s = 0; s < 2; s++) {
        if (!cp_have[s] || !cp_ndev[s]) continue;
        CUresult r = REALFN(s, name, lib_load_t)(&h[s], src, jo, jv, nj, lo, lv, nl);
        if (r || cp_trace_all) cp_trace(name, s, r);
        if (r == CUDA_SUCCESS) ok = 1; else { h[s] = NULL; if (first == CUDA_ERROR_NO_DEVICE) first = r; }
    }
    if (!ok) return first;
    void *pub = h[HOST] ? h[HOST] : h[LEG];
    pair_put(pub, h[HOST], h[LEG], NULL);
    *lib = pub;
    return CUDA_SUCCESS;
}
CUresult cuLibraryLoadData(void **lib, const void *code, void *jo, void **jv, unsigned nj, void *lo, void **lv, unsigned nl) {
    return lib_load("cuLibraryLoadData", lib, code, jo, jv, nj, lo, lv, nl);
}
CUresult cuLibraryLoadFromFile(void **lib, const char *file, void *jo, void **jv, unsigned nj, void *lo, void **lv, unsigned nl) {
    return lib_load("cuLibraryLoadFromFile", lib, file, jo, jv, nj, lo, lv, nl);
}
CUresult cuLibraryUnload(void *lib) {
    cp_init();
    int i = pair_slot(lib);
    if (i < 0) return REALFN(cp_owner(lib, cp_cur()), "cuLibraryUnload", CUresult (*)(void *))(lib);
    CUresult r = CUDA_SUCCESS;
    for (int s = 0; s < 2; s++)
        if (pair_tab[i].h[s]) { CUresult q = REALFN(s, "cuLibraryUnload", CUresult (*)(void *))(pair_tab[i].h[s]); if (!r) r = q; }
    pthread_mutex_lock(&pair_mu);                          /* Retire the library and all of its kernels with tombstones. */
    for (unsigned j = 0; j < (1u << PAIR_BITS); j++)
        if (pair_tab[j].pub == lib || (pair_tab[j].lib == lib && pair_tab[j].pub)) {
            pair_tab[j].h[0] = pair_tab[j].h[1] = NULL; pair_tab[j].lib = NULL;
            __atomic_store_n(&pair_tab[j].pub, PAIR_DEAD, __ATOMIC_RELEASE);
        }
    pthread_mutex_unlock(&pair_mu);
    return r;
}
CUresult cuLibraryGetKernel(void **kernel, void *lib, const char *name) {
    cp_init();
    if (!kernel) return CUDA_ERROR_INVALID_VALUE;
    int i = pair_slot(lib);
    if (i < 0) return REALFN(cp_owner(lib, cp_cur()), "cuLibraryGetKernel", CUresult (*)(void **, void *, const char *))(kernel, lib, name);
    void *k[2] = {NULL, NULL};
    CUresult first = CUDA_ERROR_INVALID_HANDLE;
    int ok = 0;
    for (int s = 0; s < 2; s++) {
        if (!pair_tab[i].h[s]) continue;
        CUresult r = REALFN(s, "cuLibraryGetKernel", CUresult (*)(void **, void *, const char *))(&k[s], pair_tab[i].h[s], name);
        if (r || cp_trace_all) cp_trace("cuLibraryGetKernel", s, r);
        if (r == CUDA_SUCCESS) ok = 1; else { k[s] = NULL; if (first == CUDA_ERROR_INVALID_HANDLE) first = r; }
    }
    if (!ok) return first;
    void *pub = k[HOST] ? k[HOST] : k[LEG];
    pair_put(pub, k[HOST], k[LEG], lib);
    *kernel = pub;
    return CUDA_SUCCESS;
}
