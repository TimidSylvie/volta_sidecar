/* Hammer ownership-map retirement and make sure tombstones do not break a collision chain. */
#include <dlfcn.h>
#include <stdint.h>
#include <stdio.h>

typedef void (*own_t)(const void *, int);
typedef int (*owner_t)(const void *, int);
typedef void (*disown_t)(const void *);

int main(void) {
    void *lib = dlopen("libcuda.so.1", RTLD_NOW | RTLD_LOCAL);
    if (!lib) { fprintf(stderr, "dlopen libcuda: %s\n", dlerror()); return 1; }
    own_t own = (own_t)dlsym(lib, "cp_own");
    owner_t owner = (owner_t)dlsym(lib, "cp_owner");
    disown_t disown = (disown_t)dlsym(lib, "cp_disown");
    if (!own || !owner || !disown) { fprintf(stderr, "proxy ownership test hooks are missing\n"); return 1; }

    for (uintptr_t i = 1; i <= 200000; i++) {
        const void *h = (const void *)((i + 16) << 4);
        int side = (int)(i & 1);
        own(h, side);
        if (owner(h, -1) != side) { fprintf(stderr, "ownership lost at iteration %lu\n", (unsigned long)i); return 1; }
        disown(h);
        if (owner(h, -1) != -1) { fprintf(stderr, "retired handle remained owned at iteration %lu\n", (unsigned long)i); return 1; }
    }

    const void *a = (const void *)0x1000;
    const void *b = (const void *)(0x1000 + ((uintptr_t)1 << (16 + 4))); /* same 16-bit hash bucket */
    own(a, 0); own(b, 1); disown(a);
    if (owner(b, -1) != 1) { fprintf(stderr, "deletion broke an ownership collision chain\n"); return 1; }
    disown(b);
    puts("ownership lifecycle: 200000 retirements and collision-chain lookup OK");
    return 0;
}
