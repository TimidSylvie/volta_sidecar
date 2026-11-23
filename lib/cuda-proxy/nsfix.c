/* A private libc keeps a private pthread-key allocator even though the actual thread slots are shared. That lets
 * libcuda reuse a key Python already owns and read Python's data as its own. Forward these four calls to the main
 * libc so the whole process has one honest key table. */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
static void *base(const char *name) {
    static void *libc;
    if (!libc) libc = dlmopen(LM_ID_BASE, "libc.so.6", RTLD_NOW | RTLD_NOLOAD);
    return libc ? dlsym(libc, name) : 0;
}
#define FWD(ret, name, params, args) \
    ret name params { static ret (*f) params; if (!f) f = (ret (*) params)base(#name); return f args; }
FWD(int, pthread_key_create, (pthread_key_t *k, void (*d)(void *)), (k, d))
FWD(int, pthread_key_delete, (pthread_key_t k), (k))
FWD(void *, pthread_getspecific, (pthread_key_t k), (k))
FWD(int, pthread_setspecific, (pthread_key_t k, const void *v), (k, v))
