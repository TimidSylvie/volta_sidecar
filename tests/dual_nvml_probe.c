/* Prove two isolated NVML copies can coexist: one on the open driver and one redirected to the sidecar.
 * Build: gcc -O2 -o dual_nvml_probe dual_nvml_probe.c -ldl
 * Run: ./dual_nvml_probe /path/to/libnvidia-sidecar-redirect.so */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <stdio.h>
typedef int (*init_t)(void);
typedef int (*count_t)(unsigned *);
typedef int (*handle_t)(unsigned, void **);
typedef int (*name_t)(void *, char *, unsigned);
typedef int (*temp_t)(void *, int, unsigned *);
static int side(const char *label, Lmid_t lm, const char *nvml) {
    void *h = dlmopen(lm, nvml, RTLD_NOW | RTLD_LOCAL);
    if (!h) { printf("%s: dlmopen NVML failed: %s\n", label, dlerror()); return 1; }
    init_t init = (init_t)dlsym(h, "nvmlInit_v2"); count_t cnt = (count_t)dlsym(h, "nvmlDeviceGetCount_v2");
    handle_t get = (handle_t)dlsym(h, "nvmlDeviceGetHandleByIndex_v2"); name_t nm = (name_t)dlsym(h, "nvmlDeviceGetName");
    temp_t tp = (temp_t)dlsym(h, "nvmlDeviceGetTemperature");
    int rc = init(); unsigned n = 0; cnt(&n);
    printf("%s: nvmlInit rc=%d, %u device(s)\n", label, rc, n);
    for (unsigned i = 0; i < n; i++) {
        void *d; char name[96] = ""; unsigned t = 0;
        get(i, &d); nm(d, name, sizeof name); tp(d, 0, &t);
        printf("  %s[%u] handle=%p  %s  %u C\n", label, i, d, name, t);
    }
    return 0;
}
int main(int argc, char **argv) {
    const char *nvml = "/lib/x86_64-linux-gnu/libnvidia-ml.so.1";
    void *anchor = dlmopen(LM_ID_NEWLM, "libc.so.6", RTLD_NOW);
    if (!anchor) { printf("host namespace: %s\n", dlerror()); return 1; }
    Lmid_t host; dlinfo(anchor, RTLD_DI_LMID, &host);
    void *r = dlmopen(LM_ID_NEWLM, argv[1], RTLD_NOW | RTLD_LOCAL);
    if (!r) { printf("dlmopen redirect failed: %s\n", dlerror()); return 1; }
    Lmid_t legacy; dlinfo(r, RTLD_DI_LMID, &legacy);
    int (*force)(const char *) = (int (*)(const char *))dlsym(r, "nvidia_sidecar_redirect_force");
    printf("redirect force: %d\n", force("sidecar"));
    side("host", host, nvml);
    side("legacy", legacy, nvml);
    return 0;
}
