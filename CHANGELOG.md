# Changelog

## 0.2.0 - 2026-11-23

- Adds `install.sh` and the guarded one-command `setup` workflow with `core`, per-shell and system-wide activation
  modes, a dry-run plan, explicit confirmation and fresh-process verification.
- Adds exact kernel/userspace version matching, a pinned SHA-256 for the physically proven NVIDIA installer,
  automatic support for all Tesla V100/V100S PCI IDs on any exact-match R580 point release, and validation of every
  selected ID against NVIDIA's own downloaded GPU catalog. Physical-test metadata is kept separate from support.
- Makes the main install transactional, refuses unowned system paths, stages device nodes atomically, preserves
  recoverable userspace upgrades, restores the prior activation after post-install failures, and makes uninstall
  validate and report every critical teardown step.
- Verifies the exact selected PCI IDs, bound device count and CUDA-visible PCI BDFs; absent requested devices fail
  before download or mutation.
- Adds installer transaction tests with injected activation failure and rollback hash checks.
- Documents the CUDA 13 boundary: it can run on R580, but Maxwell/Pascal/Volta workload kernels and libraries must
  come from CUDA 12.x-compatible builds.
- Hardens the merged userspace ABI and lifecycle routing, including `cuGetProcAddress`, VMM/memory-pool locations,
  destroyed handles, cross-driver events, per-device graphs and mixed-driver NVML topology/P2P queries.
- Adds an automatic NCCL socket fallback for cross-driver ranks and verifies a two-rank all-reduce across an RTX
  5060 Ti and V100 with a CUDA 12.9 NCCL build containing both architectures.
- Adds the optional userspace layer: one CUDA process can enumerate and drive GPUs owned by the open and sidecar
  drivers, public NVML clients see one merged device list, and a merged `nvidia-smi` routes queries and management
  commands to the owning driver.
- Adds per-shell `env` activation and reversible system-wide `seamless on|off`, preserving a pre-existing
  `/usr/local/bin/nvidia-smi` and rolling back a failed userspace replacement.
- Cross-driver copies use bounded host staging. Async copy entry points currently fall back to a synchronized copy;
  peer DMA, CUDA IPC/graphics interop and cross-driver graph objects are not supported.
- Frameworks must contain kernels for every participating architecture; no claim is made that an official PyTorch
  wheel contains both Volta `sm_70` and Blackwell `sm_120`.

## 0.1.0 - 2026-11-22
- First version: `doctor`, `fetch`, `patch`, `install` (DKMS), `up`/`down`, `run`, `bindpaths`, `status`, `uninstall`.
- Renames NVIDIA's proprietary 580-series kernel modules to `nvidia_<suffix>` so they load beside the open driver and
  drive only the listed legacy GPUs.
