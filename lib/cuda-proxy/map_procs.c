/* Ask the real libcuda what exported symbol backs each name/version/PTDS request. The generator uses this map so
 * cuGetProcAddress can return our matching wrapper instead of leaking a raw one-driver function. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
typedef int (*gpa_t)(const char *, void **, int, unsigned long long, int *);
int main(int argc, char **argv) {
    void *h = dlopen(argc > 1 ? argv[1] : "libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!h) { fprintf(stderr, "%s\n", dlerror()); return 1; }
    gpa_t gpa = (gpa_t)dlsym(h, "cuGetProcAddress_v2");
    char name[256]; int ver, pt;
    while (scanf("%255s %d %d", name, &ver, &pt) == 3) {
        void *fn = NULL; int status = 0;
        int rc = gpa(name, &fn, ver, pt ? 2ULL : 0ULL, &status);   /* 2 selects per-thread default streams. */
        Dl_info di; const char *sym = "-";
        if (rc == 0 && fn && dladdr(fn, &di) && di.dli_sname && di.dli_saddr == fn) sym = di.dli_sname;
        printf("%s %d %d %s\n", name, ver, pt, sym);
    }
    return 0;
}
