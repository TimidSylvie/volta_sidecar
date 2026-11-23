// Ordinary CUDA runtime code: one kernel per GPU, then cross-driver copies in both directions.
#include <cstdio>
#include <cuda_runtime.h>
__global__ void scale(float *x, float a, int n) { int i = blockIdx.x * blockDim.x + threadIdx.x; if (i < n) x[i] *= a; }
#define CK(e) do { cudaError_t r_ = (e); if (r_ != cudaSuccess) { printf("  %s -> %s\n", #e, cudaGetErrorString(r_)); return 1; } } while (0)
int main() {
    int n = 0; CK(cudaGetDeviceCount(&n));
    printf("cudaGetDeviceCount: %d\n", n);
    const int N = 1 << 20; float *d[8], h[16];
    for (int i = 0; i < n && i < 8; i++) {
        cudaDeviceProp p; CK(cudaGetDeviceProperties(&p, i)); CK(cudaSetDevice(i));
        CK(cudaMalloc(&d[i], N * sizeof(float)));
        float *tmp = new float[N]; for (int k = 0; k < N; k++) tmp[k] = 1.0f;
        CK(cudaMemcpy(d[i], tmp, N * sizeof(float), cudaMemcpyHostToDevice)); delete[] tmp;
        scale<<<(N + 255) / 256, 256>>>(d[i], (float)(i + 2), N); CK(cudaGetLastError()); CK(cudaDeviceSynchronize());
        CK(cudaMemcpy(h, d[i], sizeof h, cudaMemcpyDeviceToHost));
        size_t fr, tot; CK(cudaMemGetInfo(&fr, &tot));
        printf("  device %d %s sm_%d%d, %.1f GiB: kernel %s, free %.1f GiB\n", i, p.name, p.major, p.minor,
               tot / 1073741824.0, h[0] == i + 2.0f ? "OK" : "WRONG", fr / 1073741824.0);
    }
    if (n >= 2) {
        int can = -1; CK(cudaDeviceCanAccessPeer(&can, 0, n - 1));
        CK(cudaMemcpyPeer(d[0], 0, d[n - 1], n - 1, N * sizeof(float)));          // Last GPU -> first GPU.
        CK(cudaSetDevice(0)); CK(cudaMemcpy(h, d[0], sizeof h, cudaMemcpyDeviceToHost));
        printf("cudaMemcpyPeer %d -> 0: %s (canAccessPeer=%d)\n", n - 1, h[0] == n + 1.0f ? "OK" : "WRONG", can);
        CK(cudaMemcpy(d[n - 1], d[0], N * sizeof(float), cudaMemcpyDefault));        // UVA copy back the other way.
        CK(cudaSetDevice(n - 1)); CK(cudaMemcpy(h, d[n - 1], sizeof h, cudaMemcpyDeviceToHost));
        printf("cudaMemcpy (default, UVA) 0 -> %d: %s\n", n - 1, h[0] == n + 1.0f ? "OK" : "WRONG");
        CK(cudaSetDevice(0));
        cudaStream_t s; CK(cudaStreamCreate(&s));
        CK(cudaMemcpyPeerAsync(d[n - 1], n - 1, d[0], 0, N * sizeof(float), s));   // Correct, synchronized fallback.
        CK(cudaStreamSynchronize(s)); CK(cudaStreamDestroy(s));
        CK(cudaSetDevice(n - 1)); CK(cudaMemcpy(h, d[n - 1], sizeof h, cudaMemcpyDeviceToHost));
        printf("cudaMemcpyPeerAsync (synchronized fallback) 0 -> %d: %s\n", n - 1,
               h[0] == n + 1.0f ? "OK" : "WRONG");
    }
    for (int i = 0; i < n && i < 8; i++) { cudaSetDevice(i); cudaFree(d[i]); }
    printf("done\n");
    return 0;
}
