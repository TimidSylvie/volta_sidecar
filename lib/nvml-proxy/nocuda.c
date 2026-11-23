/* NVML tries to dlopen libcuda during init for extra CUDA facts. A second isolated libcuda copy is unsafe in some
 * host processes, while NVML already treats it as optional. Refuse only that load inside the NVML namespaces. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <string.h>
void *dlopen(const char *file, int mode) {
    static void *(*real)(const char *, int);
    if (!real) real = (void *(*)(const char *, int))dlsym(RTLD_NEXT, "dlopen");
    if (file && strstr(file, "libcuda.so")) return NULL;
    return real(file, mode);
}
