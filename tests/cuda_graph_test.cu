/* Build an ordinary per-device graph on every GPU the merged inventory exposes. */
#include <cstdio>
#include <cuda_runtime.h>

__global__ void add(float *x, float v) { x[threadIdx.x] += v; }
#define CK(e) do { cudaError_t r_ = (e); if (r_ != cudaSuccess) { \
    std::fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #e, cudaGetErrorString(r_)); return 1; } } while (0)

int main() {
    int count = 0; CK(cudaGetDeviceCount(&count));
    for (int dev = 0; dev < count; dev++) {
        CK(cudaSetDevice(dev));
        cudaDeviceProp prop{}; CK(cudaGetDeviceProperties(&prop, dev));
        cudaStream_t stream{}; cudaGraph_t graph{}; cudaGraphExec_t exec{};
        float *data = nullptr, host[32]{};
        CK(cudaStreamCreate(&stream));
        CK(cudaMalloc(&data, sizeof host));
        CK(cudaMemsetAsync(data, 0, sizeof host, stream));
        CK(cudaStreamBeginCapture(stream, cudaStreamCaptureModeGlobal));
        add<<<1, 32, 0, stream>>>(data, float(dev + 3));
        CK(cudaGetLastError());
        CK(cudaStreamEndCapture(stream, &graph));
        CK(cudaGraphInstantiate(&exec, graph));
        CK(cudaGraphLaunch(exec, stream));
        CK(cudaMemcpyAsync(host, data, sizeof host, cudaMemcpyDeviceToHost, stream));
        CK(cudaStreamSynchronize(stream));
        for (float v : host) if (v != float(dev + 3)) { std::fprintf(stderr, "device %d graph result wrong\n", dev); return 1; }
        std::printf("device %d %s: graph OK\n", dev, prop.name);
        CK(cudaGraphExecDestroy(exec)); CK(cudaGraphDestroy(graph)); CK(cudaFree(data)); CK(cudaStreamDestroy(stream));
    }
    return 0;
}
