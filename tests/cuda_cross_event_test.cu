/* Make the cross-driver event fallback prove ordering, not merely return success. */
#include <cstdio>
#include <cuda_runtime.h>

__global__ void delayed_write(int *value, int expected) {
    unsigned long long start = clock64();
    while (clock64() - start < 50000000ULL) {}
    if (!threadIdx.x) *value = expected;
}
__global__ void write_value(int *value, int expected) { if (!threadIdx.x) *value = expected; }
#define CK(e) do { cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #e, cudaGetErrorString(r_)); return 1; } } while (0)

int main() {
    int count = 0; CK(cudaGetDeviceCount(&count));
    if (count < 2) return 1;
    int source = count - 1, *source_data = nullptr, *destination_data = nullptr, got = 0;
    cudaStream_t source_stream{}, destination_stream{}; cudaEvent_t ready{};
    CK(cudaSetDevice(source)); CK(cudaStreamCreate(&source_stream)); CK(cudaEventCreate(&ready));
    CK(cudaMalloc(&source_data, sizeof *source_data));
    delayed_write<<<1, 1, 0, source_stream>>>(source_data, 8675309); CK(cudaGetLastError());
    CK(cudaEventRecord(ready, source_stream));
    CK(cudaSetDevice(0)); CK(cudaStreamCreate(&destination_stream)); CK(cudaMalloc(&destination_data, sizeof *destination_data));
    CK(cudaStreamWaitEvent(destination_stream, ready, 0));
    write_value<<<1, 1, 0, destination_stream>>>(destination_data, 424242); CK(cudaGetLastError());
    CK(cudaMemcpyAsync(&got, destination_data, sizeof got, cudaMemcpyDeviceToHost, destination_stream));
    CK(cudaStreamSynchronize(destination_stream));
    if (got != 424242) { std::fprintf(stderr, "destination work after the cross-device wait did not complete\n"); return 1; }
    CK(cudaSetDevice(source)); CK(cudaEventQuery(ready)); CK(cudaMemcpy(&got, source_data, sizeof got, cudaMemcpyDeviceToHost));
    if (got != 8675309) { std::fprintf(stderr, "cross-device event wait returned before source work\n"); return 1; }
    CK(cudaFree(source_data)); CK(cudaEventDestroy(ready)); CK(cudaStreamDestroy(source_stream));
    CK(cudaSetDevice(0)); CK(cudaFree(destination_data)); CK(cudaStreamDestroy(destination_stream));
    puts("cross-driver cudaStreamWaitEvent fallback: OK");
    return 0;
}
