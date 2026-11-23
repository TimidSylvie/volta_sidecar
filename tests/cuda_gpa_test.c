/* Never let cuGetProcAddress leak a raw function tied to only one underlying driver. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>

typedef int CUresult;
typedef CUresult (*gpa_v2_t)(const char *, void **, int, unsigned long long, int *);

struct test_case { const char *name; int version; unsigned long long flags; };

int main(void) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen libcuda: %s\n", dlerror()); return 1; }
    gpa_v2_t gpa = (gpa_v2_t)dlsym(lib, "cuGetProcAddress_v2");
    if (!gpa) { fprintf(stderr, "cuGetProcAddress_v2 is missing\n"); return 1; }
    Dl_info proxy;
    if (!dladdr((void *)gpa, &proxy) || !proxy.dli_fbase) { fprintf(stderr, "cannot identify proxy image\n"); return 1; }

    const struct test_case cases[] = {
        {"cuInit", 2000, 0},
        {"cuDeviceGetCount", 2000, 0},
        {"cuMemAlloc", 12000, 0},
        {"cuMemcpyAsync", 12000, 0},
        {"cuMemcpyAsync", 7000, 2},
        {"cuLaunchKernel", 12000, 0},
        {"cuLaunchKernel", 7000, 2},
        {"cuGraphCreate", 12000, 0},
    };
    for (size_t i = 0; i < sizeof cases / sizeof cases[0]; i++) {
        void *fn = NULL;
        int status = -1;
        CUresult r = gpa(cases[i].name, &fn, cases[i].version, cases[i].flags, &status);
        Dl_info got;
        if (r || status || !fn || !dladdr(fn, &got) || got.dli_fbase != proxy.dli_fbase) {
            fprintf(stderr, "%s v%d flags=%llu escaped proxy: rc=%d status=%d fn=%p image=%s\n",
                    cases[i].name, cases[i].version, cases[i].flags, r, status, fn,
                    fn && dladdr(fn, &got) && got.dli_fname ? got.dli_fname : "unknown");
            return 1;
        }
        printf("%s v%d flags=%llu -> proxy\n", cases[i].name, cases[i].version, cases[i].flags);
    }

    void *fn = (void *)1;
    int status = -1;
    CUresult r = gpa("cuDefinitelyNotARealFunction", &fn, 12000, 0, &status);
    if (r || fn || status != 1) {
        fprintf(stderr, "unknown GPA contract wrong: rc=%d status=%d fn=%p\n", r, status, fn);
        return 1;
    }
    puts("unknown symbol -> NULL / SYMBOL_NOT_FOUND");
    return 0;
}
