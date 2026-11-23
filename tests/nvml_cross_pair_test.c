/* A mixed pair should look like two ordinary GPUs without a direct link, not a broken NVML call. */
#include <nvml.h>
#include <stdio.h>

#define NK(expr) do { nvmlReturn_t r_ = (expr); if (r_ != NVML_SUCCESS) { \
    fprintf(stderr, "%s:%d %s: %s\n", __FILE__, __LINE__, #expr, nvmlErrorString(r_)); return 1; \
} } while (0)

int main(void) {
    NK(nvmlInit_v2());
    unsigned count = 0;
    NK(nvmlDeviceGetCount_v2(&count));
    if (count < 2) { fprintf(stderr, "need two merged NVML devices\n"); return 1; }
    nvmlDevice_t modern, legacy;
    NK(nvmlDeviceGetHandleByIndex_v2(0, &modern));
    NK(nvmlDeviceGetHandleByIndex_v2(count - 1, &legacy));

    int same = 1;
    NK(nvmlDeviceOnSameBoard(modern, legacy, &same));
    if (same != 0) { fprintf(stderr, "mixed-driver devices reported the same board\n"); return 1; }
    nvmlGpuTopologyLevel_t topology = NVML_TOPOLOGY_INTERNAL;
    NK(nvmlDeviceGetTopologyCommonAncestor(modern, legacy, &topology));
    if (topology != NVML_TOPOLOGY_SYSTEM) {
        fprintf(stderr, "mixed-driver topology was %d, expected NVML_TOPOLOGY_SYSTEM\n", (int)topology); return 1;
    }
    for (int capability = NVML_P2P_CAPS_INDEX_READ; capability <= NVML_P2P_CAPS_INDEX_ATOMICS; capability++) {
        nvmlGpuP2PStatus_t status = NVML_P2P_STATUS_UNKNOWN;
        NK(nvmlDeviceGetP2PStatus(modern, legacy, (nvmlGpuP2PCapsIndex_t)capability, &status));
        if (status != NVML_P2P_STATUS_NOT_SUPPORTED) {
            fprintf(stderr, "P2P capability %d returned status %d\n", capability, (int)status); return 1;
        }
    }
    NK(nvmlShutdown());
    puts("mixed-driver NVML pair semantics: OK");
    return 0;
}
