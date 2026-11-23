/*
 * This does the same little trick as the CUDA proxy, but for NVML: two real drivers, one ordinary GPU inventory.
 *
 * Each real libnvidia-ml lives in its own link-map namespace. The host copy opens the primary driver normally; the
 * legacy copy loads behind the path redirect. Device handles point into the library that created them, which gives
 * the trampoline enough information to route almost every call. Enumeration is simply host GPUs followed by
 * sidecar GPUs. Mixed pair queries get honest conservative answers: different boards, system-level topology and no
 * P2P. Legacy event registration remains unsupported because event sets belong to one NVML copy.
 *
 * Environment: NVIDIA_SIDECAR_NVML=0  only the open driver's GPUs (a plain pass-through)
 *              NVIDIA_SIDECAR=1       only the legacy GPUs (a process routed to the legacy side)
 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <link.h>
#include <pthread.h>
#include <stdint.h>
#include <stdlib.h>
#include <stdio.h>
#include <string.h>

#ifndef REAL_NVML
#define REAL_NVML "/usr/local/lib/nvidia-sidecar/real/libnvidia-ml.so.1"
#endif
#ifndef HOST_LOADER
#define HOST_LOADER "/usr/local/lib/nvidia-sidecar/libnvidia-sidecar-host-nvml.so"
#endif
#ifndef LEGACY_LOADER
#define LEGACY_LOADER "/usr/local/lib/nvidia-sidecar/libnvidia-sidecar-legacy-nvml.so"
#endif

typedef int nvmlReturn_t;
typedef void *nvmlDevice_t;
#define NVML_SUCCESS 0
#define NVML_ERROR_UNINITIALIZED 1
#define NVML_ERROR_INVALID_ARGUMENT 2
#define NVML_ERROR_NOT_SUPPORTED 3
#define NVML_ERROR_NOT_FOUND 6
#define NVML_ERROR_LIBRARY_NOT_FOUND 12
#define NVML_ERROR_FUNCTION_NOT_FOUND 13
#define NVML_TOPOLOGY_SYSTEM 50
#define NVML_P2P_STATUS_NOT_SUPPORTED 5

/* Generated export names in trampoline order. */
extern const char *const sc_names[];
extern const int sc_count;

static void *host_lib, *legacy_lib;
static void **host_fn, **legacy_fn;
static uintptr_t legacy_lo, legacy_hi;    /* Address range occupied by the legacy NVML copy. */
static int use_host = 1, use_legacy = 0;
static pthread_once_t once = PTHREAD_ONCE_INIT;
static int debug;
#define DBG(...) do { if (debug) { fprintf(stderr, "[nvml-proxy] " __VA_ARGS__); fputc('\n', stderr); } } while (0)

/* Find a loaded object's mapped range from its ELF headers; dl_iterate_phdr cannot see this other namespace. */
static void image_range(struct link_map *m) {
    const ElfW(Ehdr) *eh = (const ElfW(Ehdr) *)m->l_addr;
    if (memcmp(eh->e_ident, ELFMAG, SELFMAG) != 0) return;
    const ElfW(Phdr) *ph = (const ElfW(Phdr) *)(m->l_addr + eh->e_phoff);
    for (int i = 0; i < eh->e_phnum; i++) {
        if (ph[i].p_type != PT_LOAD) continue;
        uintptr_t lo = m->l_addr + ph[i].p_vaddr, hi = lo + ph[i].p_memsz;
        if (!legacy_lo || lo < legacy_lo) legacy_lo = lo;
        if (hi > legacy_hi) legacy_hi = hi;
    }
}

static void *fn(void **t, const char *name);
static void **resolve_all(void *lib) {
    void **t = calloc((size_t)sc_count, sizeof(void *));
    for (int i = 0; t && i < sc_count; i++) t[i] = dlsym(lib, sc_names[i]);
    return t;
}

static void setup(void) {
    const char *nvml = getenv("NVIDIA_SIDECAR_NVML"), *routed = getenv("NVIDIA_SIDECAR"), *sfx = getenv("NVIDIA_SIDECAR_SUFFIX");
    int want_legacy = !(nvml && strcmp(nvml, "0") == 0);
    debug = getenv("NVIDIA_SIDECAR_DEBUG") != NULL;
    if (routed && *routed && strcmp(routed, "0") != 0) { use_host = 0; want_legacy = 1; }
    /* Put each side's interposers into a new namespace first, then load the real NVML by absolute path so library
     * search can never loop back into this proxy. */
    void *hl = dlmopen(LM_ID_NEWLM, HOST_LOADER, RTLD_NOW | RTLD_LOCAL);
    Lmid_t hlm;
    if (hl && dlinfo(hl, RTLD_DI_LMID, &hlm) == 0) host_lib = dlmopen(hlm, REAL_NVML, RTLD_NOW | RTLD_LOCAL);
    if (host_lib) host_fn = resolve_all(host_lib);
    DBG("host copy %p (%s); count fn %p", host_lib, host_lib ? HOST_LOADER : dlerror(), host_fn ? fn(host_fn, "nvmlDeviceGetCount_v2") : NULL);
    if (!want_legacy) return;
    void *loader = dlmopen(LM_ID_NEWLM, LEGACY_LOADER, RTLD_NOW | RTLD_LOCAL);
    if (!loader) return;
    int (*force)(const char *) = (int (*)(const char *))dlsym(loader, "nvidia_sidecar_redirect_force");
    if (!force || !force(sfx && *sfx ? sfx : "sidecar")) return;      /* A missing sidecar leaves host NVML alone. */
    Lmid_t lm;
    if (dlinfo(loader, RTLD_DI_LMID, &lm) != 0) return;
    legacy_lib = dlmopen(lm, REAL_NVML, RTLD_NOW | RTLD_LOCAL);
    if (!legacy_lib) return;
    struct link_map *m;
    if (dlinfo(legacy_lib, RTLD_DI_LINKMAP, &m) != 0) return;
    image_range(m);
    legacy_fn = resolve_all(legacy_lib);
    use_legacy = legacy_fn && legacy_hi > legacy_lo;
    DBG("legacy copy %p image %#lx-%#lx use_legacy=%d", legacy_lib, (unsigned long)legacy_lo, (unsigned long)legacy_hi, use_legacy);
}

static void init_once(void) { pthread_once(&once, setup); }
static int is_legacy(const void *p) { uintptr_t a = (uintptr_t)p; return use_legacy && a >= legacy_lo && a < legacy_hi; }

static nvmlReturn_t not_found(void) { return NVML_ERROR_FUNCTION_NOT_FOUND; }
static nvmlReturn_t no_library(void) { return NVML_ERROR_LIBRARY_NOT_FOUND; }

/* Every generated trampoline asks here which real function should receive the call. */
void *sc_route(void *arg1, int idx) {
    init_once();
    void **t = is_legacy(arg1) ? legacy_fn : (use_host ? host_fn : legacy_fn);
    DBG("route %s(%p) -> %s", sc_names[idx], arg1, t == legacy_fn ? "legacy" : "host");
    if (!t) return (void *)no_library;
    return t[idx] ? t[idx] : (void *)not_found;
}

static void *fn(void **t, const char *name) {
    if (!t) return NULL;
    for (int i = 0; i < sc_count; i++) if (strcmp(sc_names[i], name) == 0) return t[i];
    return NULL;
}
#define CALL(side, name, type, ...) ({ __typeof__((type)0) _f = (type)fn(side, name); _f ? _f(__VA_ARGS__) : NVML_ERROR_FUNCTION_NOT_FOUND; })

/* Init and shutdown belong to both copies. */
static nvmlReturn_t init_both(const char *name, unsigned flags, int with_flags) {
    init_once();
    nvmlReturn_t rh = NVML_SUCCESS, rl = NVML_SUCCESS;
    if (use_host) rh = with_flags ? CALL(host_fn, name, nvmlReturn_t (*)(unsigned), flags) : CALL(host_fn, name, nvmlReturn_t (*)(void));
    if (use_legacy) {
        rl = with_flags ? CALL(legacy_fn, name, nvmlReturn_t (*)(unsigned), flags) : CALL(legacy_fn, name, nvmlReturn_t (*)(void));
        if (rl != NVML_SUCCESS && use_host) use_legacy = 0;   /* A broken sidecar must not take host NVML down with it. */
    }
    return use_host ? rh : rl;
}
nvmlReturn_t nvmlInit(void) { return init_both("nvmlInit", 0, 0); }
nvmlReturn_t nvmlInit_v2(void) { return init_both("nvmlInit_v2", 0, 0); }
nvmlReturn_t nvmlInitWithFlags(unsigned flags) { return init_both("nvmlInitWithFlags", flags, 1); }
nvmlReturn_t nvmlShutdown(void) {
    init_once();
    nvmlReturn_t r = NVML_SUCCESS;
    if (use_legacy) r = CALL(legacy_fn, "nvmlShutdown", nvmlReturn_t (*)(void));
    if (use_host) r = CALL(host_fn, "nvmlShutdown", nvmlReturn_t (*)(void));
    return r;
}

/* Public numbering is host GPUs first, then sidecar GPUs. */
static nvmlReturn_t count_of(void **t, const char *name, unsigned *count) {
    if (!t) return NVML_ERROR_LIBRARY_NOT_FOUND;
    *count = 0;
    return CALL(t, name, nvmlReturn_t (*)(unsigned *), count);
}
static nvmlReturn_t get_count(const char *name, unsigned *count) {
    init_once();
    if (!count) return NVML_ERROR_INVALID_ARGUMENT;
    if (use_host && !use_legacy) return CALL(host_fn, name, nvmlReturn_t (*)(unsigned *), count);
    unsigned h = 0, l = 0;
    nvmlReturn_t r;
    if (use_host && (r = count_of(host_fn, name, &h)) != NVML_SUCCESS) return r;
    if (use_legacy && (r = count_of(legacy_fn, name, &l)) != NVML_SUCCESS) return r;
    *count = h + l;
    return NVML_SUCCESS;
}
nvmlReturn_t nvmlDeviceGetCount(unsigned *count) { return get_count("nvmlDeviceGetCount", count); }
nvmlReturn_t nvmlDeviceGetCount_v2(unsigned *count) { return get_count("nvmlDeviceGetCount_v2", count); }

static nvmlReturn_t by_index(const char *name, const char *count_name, unsigned index, nvmlDevice_t *dev) {
    init_once();
    unsigned h = 0;
    nvmlReturn_t r;
    if (use_host && (r = count_of(host_fn, count_name, &h)) != NVML_SUCCESS) return r;
    if (index < h) return CALL(host_fn, name, nvmlReturn_t (*)(unsigned, nvmlDevice_t *), index, dev);
    if (use_legacy) return CALL(legacy_fn, name, nvmlReturn_t (*)(unsigned, nvmlDevice_t *), index - h, dev);
    return NVML_ERROR_INVALID_ARGUMENT;
}
nvmlReturn_t nvmlDeviceGetHandleByIndex(unsigned i, nvmlDevice_t *d) { return by_index("nvmlDeviceGetHandleByIndex", "nvmlDeviceGetCount", i, d); }
nvmlReturn_t nvmlDeviceGetHandleByIndex_v2(unsigned i, nvmlDevice_t *d) { return by_index("nvmlDeviceGetHandleByIndex_v2", "nvmlDeviceGetCount_v2", i, d); }

/* Identifier lookups try the host first, then the sidecar. */
#define LOOKUP(fname, ktype)                                                                                  \
    nvmlReturn_t fname(ktype key, nvmlDevice_t *dev) {                                                       \
        init_once();                                                                                          \
        nvmlReturn_t r = NVML_ERROR_NOT_FOUND;                                                                \
        if (use_host) { r = CALL(host_fn, #fname, nvmlReturn_t (*)(ktype, nvmlDevice_t *), key, dev); if (r == NVML_SUCCESS) return r; } \
        if (use_legacy) r = CALL(legacy_fn, #fname, nvmlReturn_t (*)(ktype, nvmlDevice_t *), key, dev);      \
        return r;                                                                                             \
    }
LOOKUP(nvmlDeviceGetHandleByUUID, const char *)
LOOKUP(nvmlDeviceGetHandleByPciBusId, const char *)
LOOKUP(nvmlDeviceGetHandleByPciBusId_v2, const char *)
LOOKUP(nvmlDeviceGetHandleBySerial, const char *)
LOOKUP(nvmlDeviceGetHandleByUUIDV, const void *)

nvmlReturn_t nvmlDeviceGetIndex(nvmlDevice_t dev, unsigned *index) {
    init_once();
    if (!is_legacy(dev)) return CALL(use_host ? host_fn : legacy_fn, "nvmlDeviceGetIndex", nvmlReturn_t (*)(nvmlDevice_t, unsigned *), dev, index);
    nvmlReturn_t r = CALL(legacy_fn, "nvmlDeviceGetIndex", nvmlReturn_t (*)(nvmlDevice_t, unsigned *), dev, index);
    if (r == NVML_SUCCESS && index && use_host) {
        unsigned h = 0;
        r = count_of(host_fn, "nvmlDeviceGetCount_v2", &h);
        if (r == NVML_SUCCESS) *index += h;
    }
    return r;
}

/* Neither real NVML copy can accept a handle from the other. For mixed pairs, return the same conservative facts
 * NVML gives for two GPUs without a direct link instead of leaking our split as an API failure. NCCL uses these
 * answers to choose its host-memory/socket path. */
#define PAIR(fname, ...)                                                                                      \
    nvmlReturn_t fname(nvmlDevice_t a, nvmlDevice_t b, __VA_ARGS__)
PAIR(nvmlDeviceOnSameBoard, int *same) {
    init_once();
    if (is_legacy(a) != is_legacy(b)) {
        if (!same) return NVML_ERROR_INVALID_ARGUMENT;
        *same = 0;
        return NVML_SUCCESS;
    }
    return CALL(is_legacy(a) ? legacy_fn : host_fn, "nvmlDeviceOnSameBoard",
                nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t, int *), a, b, same);
}
PAIR(nvmlDeviceGetTopologyCommonAncestor, int *info) {
    init_once();
    if (is_legacy(a) != is_legacy(b)) {
        if (!info) return NVML_ERROR_INVALID_ARGUMENT;
        *info = NVML_TOPOLOGY_SYSTEM;
        return NVML_SUCCESS;
    }
    return CALL(is_legacy(a) ? legacy_fn : host_fn, "nvmlDeviceGetTopologyCommonAncestor",
                nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t, int *), a, b, info);
}
PAIR(nvmlDeviceGetP2PStatus, int index, int *status) {
    init_once();
    if (is_legacy(a) != is_legacy(b)) {
        if (!status) return NVML_ERROR_INVALID_ARGUMENT;
        *status = NVML_P2P_STATUS_NOT_SUPPORTED;
        return NVML_SUCCESS;
    }
    return CALL(is_legacy(a) ? legacy_fn : host_fn, "nvmlDeviceGetP2PStatus",
                nvmlReturn_t (*)(nvmlDevice_t, nvmlDevice_t, int, int *), a, b, index, status);
}

/* An event set belongs to one NVML copy, so a legacy handle cannot be registered into the host's set. */
nvmlReturn_t nvmlDeviceRegisterEvents(nvmlDevice_t dev, unsigned long long types, void *set) {
    init_once();
    if (is_legacy(dev)) return NVML_ERROR_NOT_SUPPORTED;
    return CALL(use_host ? host_fn : legacy_fn, "nvmlDeviceRegisterEvents", nvmlReturn_t (*)(nvmlDevice_t, unsigned long long, void *), dev, types, set);
}
