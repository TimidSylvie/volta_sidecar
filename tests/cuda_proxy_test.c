/* Use whichever libcuda the loader gives us, run one kernel per device, then copy across the two ends of the merged
 * inventory with both UVA and peer-copy entry points. Build: gcc -o t t.c -ldl */
#include <dlfcn.h>
#include <stdio.h>
#include <string.h>
typedef int CUresult; typedef int CUdevice; typedef void *CUcontext, *CUmodule, *CUfunction;
typedef unsigned long long CUdeviceptr;
static const char *ptx =
 ".version 6.0\n.target sm_70\n.address_size 64\n.visible .entry add1(.param .u64 p, .param .u32 n) {\n"
 " .reg .u32 %r<5>; .reg .u64 %d<4>; .reg .f32 %f<3>; .reg .pred %q;\n"
 " ld.param.u64 %d1, [p]; ld.param.u32 %r1, [n];\n"
 " mov.u32 %r2, %ctaid.x; mov.u32 %r3, %ntid.x; mov.u32 %r4, %tid.x; mad.lo.u32 %r2, %r2, %r3, %r4;\n"
 " setp.ge.u32 %q, %r2, %r1; @%q ret;\n"
 " cvta.to.global.u64 %d1, %d1; mul.wide.u32 %d2, %r2, 4; add.u64 %d3, %d1, %d2;\n"
 " ld.global.f32 %f1, [%d3]; add.f32 %f2, %f1, 0f3F800000; st.global.f32 [%d3], %f2; ret; }\n";
#define F(name, ...) CUresult (*name)(__VA_ARGS__) = dlsym(h, #name)
int main(void) {
    setvbuf(stdout, NULL, _IONBF, 0);
    void *h = dlopen("libcuda.so.1", RTLD_NOW);
    F(cuInit, unsigned); F(cuDeviceGetCount, int *); F(cuDeviceGet, CUdevice *, int); F(cuDeviceGetName, char *, int, CUdevice);
    F(cuDevicePrimaryCtxRetain, CUcontext *, CUdevice); F(cuCtxSetCurrent, CUcontext); F(cuModuleLoadData, CUmodule *, const void *);
    F(cuModuleGetFunction, CUfunction *, CUmodule, const char *); F(cuMemAlloc_v2, CUdeviceptr *, size_t);
    F(cuMemcpyHtoD_v2, CUdeviceptr, const void *, size_t); F(cuMemcpyDtoH_v2, void *, CUdeviceptr, size_t);
    F(cuLaunchKernel, CUfunction, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, unsigned, void *, void **, void **);
    F(cuCtxSynchronize, void); F(cuMemcpy, CUdeviceptr, CUdeviceptr, size_t);
    F(cuMemcpyPeer, CUdeviceptr, CUcontext, CUdeviceptr, CUcontext, size_t); F(cuDeviceGetAttribute, int *, int, CUdevice);
    int rc = cuInit(0), n = 0; cuDeviceGetCount(&n);
    printf("cuInit=%d, %d device(s)\n", rc, n);
    CUcontext ctx[8]; CUdeviceptr buf[8]; float host[1024];
    for (int i = 0; i < n && i < 8; i++) {
        CUdevice d; char name[96]; int major = 0, minor = 0;
        cuDeviceGet(&d, i); cuDeviceGetName(name, sizeof name, d);
        cuDeviceGetAttribute(&major, 75, d); cuDeviceGetAttribute(&minor, 76, d);
        CUmodule m; CUfunction f;
        int r1 = cuDevicePrimaryCtxRetain(&ctx[i], d), r2 = cuCtxSetCurrent(ctx[i]);
        int r3 = cuModuleLoadData(&m, ptx), r4 = cuModuleGetFunction(&f, m, "add1");
        for (int k = 0; k < 1024; k++) host[k] = (float)(i + 1);
        cuMemAlloc_v2(&buf[i], sizeof host); cuMemcpyHtoD_v2(buf[i], host, sizeof host);
        unsigned nn = 1024; void *args[2] = {&buf[i], &nn};
        int r5 = cuLaunchKernel(f, 4, 1, 1, 256, 1, 1, 0, 0, args, 0), r6 = cuCtxSynchronize();
        cuMemcpyDtoH_v2(host, buf[i], sizeof host);
        printf("  device %d (handle %d) %s sm_%d%d: ctx=%d set=%d module=%d func=%d launch=%d sync=%d -> %s\n", i, d, name,
               major, minor, r1, r2, r3, r4, r5, r6, host[0] == i + 2.0f && host[1023] == i + 2.0f ? "kernel OK" : "WRONG");
    }
    if (n >= 2) {
        cuCtxSetCurrent(ctx[0]);
        int r = cuMemcpy(buf[0], buf[n - 1], sizeof host);           /* Last device -> first device. */
        cuMemcpyDtoH_v2(host, buf[0], sizeof host);
        printf("cuMemcpy device %d -> 0: rc %d, values %s\n", n - 1, r, host[0] == n + 1.0f && host[1023] == n + 1.0f ? "OK" : "WRONG");
        for (int k = 0; k < 1024; k++) host[k] = 42.0f;
        cuMemcpyHtoD_v2(buf[0], host, sizeof host);
        r = cuMemcpyPeer(buf[n - 1], ctx[n - 1], buf[0], ctx[0], sizeof host);   /* First device -> last device. */
        cuCtxSetCurrent(ctx[n - 1]); cuMemcpyDtoH_v2(host, buf[n - 1], sizeof host);
        printf("cuMemcpyPeer device 0 -> %d: rc %d, values %s\n", n - 1, r, host[0] == 42.0f && host[1023] == 42.0f ? "OK" : "WRONG");
    }
    return 0;
}
