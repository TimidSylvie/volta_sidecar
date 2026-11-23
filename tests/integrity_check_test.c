/* Ask a real libcuda for its private integrity answer, recompute it from the same tables and devices, and compare.
 * Usage: integrity_check_test /path/libcuda.so.1 */
#define _GNU_SOURCE
#include <dlfcn.h>
#include <pthread.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <unistd.h>
#include <sys/syscall.h>
struct sc_ic_input { uint32_t driver_version, version, process, thread; const void *cudart_table, *integrity_table, *fn_address; uint64_t unix_seconds; };
struct sc_ic_device { unsigned char uuid[16]; int32_t pci_domain, pci_bus, pci_device; };
void sc_integrity_hash(const struct sc_ic_input *, const struct sc_ic_device *, unsigned, uint64_t[2]);
static const unsigned char IC[16] = {0xD4,0x08,0x20,0x55,0xBD,0xE6,0x70,0x4B,0x8D,0x34,0xBA,0x12,0x3C,0x66,0xE1,0xF2};
static const unsigned char RT[16] = {0x6B,0xD5,0xFB,0x6C,0x5B,0xF4,0xE7,0x4A,0x89,0x87,0xD9,0x39,0x12,0xFD,0x9D,0xF9};
int main(int argc, char **argv) {
    void *h = dlopen(argv[1], RTLD_NOW); if (!h) { puts(dlerror()); return 1; }
    int (*init)(unsigned) = dlsym(h, "cuInit");
    int (*exp)(const void **, const void *) = dlsym(h, "cuGetExportTable");
    int (*cnt)(int *) = dlsym(h, "cuDeviceGetCount");
    int (*get)(int *, int) = dlsym(h, "cuDeviceGet");
    int (*uuid)(unsigned char *, int) = dlsym(h, "cuDeviceGetUuid_v2");
    int (*attr)(int *, int, int) = dlsym(h, "cuDeviceGetAttribute");
    int (*dv)(int *) = dlsym(h, "cuDriverGetVersion");
    printf("cuInit %d\n", init(0));
    const void *ic = 0, *rt = 0;
    printf("tables: ic rc=%d %p  rt rc=%d %p\n", exp(&ic, IC), ic, exp(&rt, RT), rt);
    if (!ic || !rt) return 1;
    int (*check)(uint32_t, uint64_t, uint64_t *) = ((void *const *)ic)[1];
    int n = 0; cnt(&n);
    struct sc_ic_device devs[16]; memset(devs, 0, sizeof devs);
    for (int i = 0; i < n; i++) {
        int d; get(&d, i); uuid(devs[i].uuid, d);
        attr(&devs[i].pci_domain, 50, d); attr(&devs[i].pci_bus, 33, d); attr(&devs[i].pci_device, 34, d);
        printf("dev %d pci %04x:%02x:%02x\n", i, devs[i].pci_domain, devs[i].pci_bus, devs[i].pci_device);
    }
    int drv = 0; dv(&drv);
    uint32_t vers[] = {12093, 12085, 13002};
    int matched = 0;
    for (int v = 0; v < 3; v++) {
        uint64_t real[2] = {0}, mine[2]; uint64_t now = (uint64_t)time(0);
        int rc = check(vers[v], now, real);
        uint32_t dvs[] = {(uint32_t)drv, 12090, 12080, 13000, 13010, 12000};
        uint32_t tids[] = {(uint32_t)pthread_self(), (uint32_t)syscall(SYS_gettid)};
        int hit = 0;
        for (int a = 0; a < 6 && !hit; a++) for (int b = 0; b < 2 && !hit; b++) {
            struct sc_ic_input in = {dvs[a], vers[v], (uint32_t)getpid(), tids[b], rt, ic, ((void *const *)ic)[1], now};
            sc_integrity_hash(&in, devs, n, mine);
            if (mine[0] == real[0] && mine[1] == real[1]) { printf("version %u: MATCH (driver_version=%u, thread=%s)\n", vers[v], dvs[a], b ? "gettid" : "pthread_self"); hit = 1; }
        }
        if (!hit) printf("version %u: rc=%d real=%016llx%016llx  no match\n", vers[v], rc, (unsigned long long)real[0], (unsigned long long)real[1]);
        matched += hit;
    }
    return matched == 3 ? 0 : 2;
}
