/* Catch VMM struct-layout mistakes and make sure merged device ordinals reach the right driver. */
#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#define CK(expr) do { CUresult r_ = (expr); if (r_ != CUDA_SUCCESS) { \
    const char *name_ = NULL; cuGetErrorName(r_, &name_); \
    fprintf(stderr, "%s:%d %s: %d (%s)\n", __FILE__, __LINE__, #expr, (int)r_, name_ ? name_ : "?"); return 1; \
} } while (0)

int main(void) {
    CK(cuInit(0));
    int count = 0;
    CK(cuDeviceGetCount(&count));
    if (count < 2) { fprintf(stderr, "need two merged devices\n"); return 1; }

    CUdevice modern, legacy;
    CK(cuDeviceGet(&modern, 0));
    CK(cuDeviceGet(&legacy, count - 1));
    CUcontext ctx = NULL;
    CK(cuCtxCreate(&ctx, 0, legacy));

    CUmemAllocationProp prop;
    memset(&prop, 0, sizeof prop);
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = legacy;
    size_t granularity = 0;
    CK(cuMemGetAllocationGranularity(&granularity, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM));

    CUdeviceptr address = 0;
    CUmemGenericAllocationHandle allocation = 0;
    CK(cuMemAddressReserve(&address, granularity, 0, 0, 0));
    CK(cuMemCreate(&allocation, granularity, &prop, 0));
    CK(cuMemMap(address, granularity, 0, allocation, 0));

    CUmemAccessDesc access[2];
    memset(access, 0, sizeof access);
    access[0].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access[0].location.id = legacy;
    access[0].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    access[1].location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    access[1].location.id = modern;
    access[1].flags = CU_MEM_ACCESS_FLAGS_PROT_NONE;
    CK(cuMemSetAccess(address, granularity, access, 2));

    unsigned long long flags = 0;
    CK(cuMemGetAccess(&flags, &access[0].location, address));
    if (flags != CU_MEM_ACCESS_FLAGS_PROT_READWRITE) {
        fprintf(stderr, "legacy mapping access flags were %#llx\n", flags); return 1;
    }
    flags = ~0ULL;
    CK(cuMemGetAccess(&flags, &access[1].location, address));
    if (flags != CU_MEM_ACCESS_FLAGS_PROT_NONE) {
        fprintf(stderr, "foreign mapping unexpectedly reports access %#llx\n", flags); return 1;
    }
    access[1].flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    CUresult cross = cuMemSetAccess(address, granularity, &access[1], 1);
    if (cross != CUDA_ERROR_NOT_SUPPORTED) {
        fprintf(stderr, "foreign access request returned %d, expected CUDA_ERROR_NOT_SUPPORTED\n", (int)cross); return 1;
    }

    CUmemoryPool default_pool = NULL, current_pool = NULL, custom_pool = NULL;
    CK(cuDeviceGetDefaultMemPool(&default_pool, legacy));
    CK(cuDeviceGetMemPool(&current_pool, legacy));
    CUmemPoolProps pool_props;
    memset(&pool_props, 0, sizeof pool_props);
    pool_props.allocType = CU_MEM_ALLOCATION_TYPE_PINNED;
    pool_props.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    pool_props.location.id = legacy;
    CK(cuMemPoolCreate(&custom_pool, &pool_props));
    CK(cuDeviceSetMemPool(legacy, custom_pool));
    CK(cuDeviceGetMemPool(&current_pool, legacy));
    if (current_pool != custom_pool) { fprintf(stderr, "legacy current pool did not round-trip\n"); return 1; }

    struct {
        uint32_t before;
        CUmemAccess_flags value;
        uint32_t after;
    } canary = {0x11223344u, CU_MEM_ACCESS_FLAGS_PROT_NONE, 0x55667788u};
    CUmemLocation legacy_location = {CU_MEM_LOCATION_TYPE_DEVICE, legacy};
    CK(cuMemPoolGetAccess(&canary.value, custom_pool, &legacy_location));
    if (canary.before != 0x11223344u || canary.after != 0x55667788u) {
        fprintf(stderr, "cuMemPoolGetAccess overwrote a four-byte output canary\n"); return 1;
    }
    CK(cuDeviceSetMemPool(legacy, default_pool));
    CK(cuMemPoolDestroy(custom_pool));

    CUdeviceptr managed = 0;
    CUstream stream = NULL;
    CK(cuMemAllocManaged(&managed, 4096, CU_MEM_ATTACH_GLOBAL));
    CK(cuStreamCreate(&stream, CU_STREAM_DEFAULT));
    CK(cuMemAdvise_v2(managed, 4096, CU_MEM_ADVISE_SET_PREFERRED_LOCATION, legacy_location));
    CK(cuMemPrefetchAsync_v2(managed, 4096, legacy_location, 0, stream));
    CK(cuStreamSynchronize(stream));
    CK(cuStreamDestroy(stream));
    CK(cuMemFree(managed));

    CK(cuMemUnmap(address, granularity));
    CK(cuMemRelease(allocation));
    CK(cuMemAddressFree(address, granularity));
    CK(cuCtxDestroy(ctx));
    puts("CUDA VMM descriptors, access semantics, and legacy ordinal translation: OK");
    return 0;
}
