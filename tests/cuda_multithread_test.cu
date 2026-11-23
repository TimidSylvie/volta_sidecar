/* Drive both underlying drivers from separate host threads at the same time. */
#include <cstdio>
#include <thread>
#include <vector>
#include <cuda_runtime.h>

__global__ void stamp(int *value, int expected) { if (!threadIdx.x) *value = expected; }

static void worker(int dev, int *failure) {
    if (cudaSetDevice(dev) != cudaSuccess) { *failure = 1; return; }
    for (int i = 0; i < 128; i++) {
        int *data = nullptr, got = -1, expected = dev * 1000 + i;
        if (cudaMalloc(&data, sizeof *data) != cudaSuccess) { *failure = 2; return; }
        stamp<<<1, 1>>>(data, expected);
        if (cudaGetLastError() != cudaSuccess || cudaMemcpy(&got, data, sizeof got, cudaMemcpyDeviceToHost) != cudaSuccess ||
            cudaFree(data) != cudaSuccess || got != expected) { *failure = 3; return; }
    }
}

int main() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess || count < 2) return 1;
    int failures[2] = {0, 0};
    std::thread a(worker, 0, &failures[0]), b(worker, count - 1, &failures[1]);
    a.join(); b.join();
    if (failures[0] || failures[1]) { std::fprintf(stderr, "thread failures: %d %d\n", failures[0], failures[1]); return 1; }
    std::printf("concurrent host threads: device 0 and device %d OK\n", count - 1);
    return 0;
}
