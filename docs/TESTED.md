# Tested

## The method (renamed proprietary driver beside the open one)

Machine: Ubuntu 24.04, kernel 6.8.0-149, x86_64, Secure Boot off.
GPUs: RTX 5060 Ti (Blackwell, 10de:2d04) on the open driver 580.178.04; Tesla V100 PCIe 32GB (Volta, 10de:1db6) on the
proprietary 580.178.04 module, renamed with exactly the edits `lib/patch_driver.py` makes (a hand-built instance with
the suffix `volta` before this tool existed).

Physical validation and automatic support are deliberately separate. `10de:1db6` is the board in the machine above;
it is not the only V100 the installer supports. The first-class V100/V100S device set is:

| PCI device ID | board family |
|---|---|
| `1db1` | Tesla V100 SXM2 16GB (including LS subsystem variants) |
| `1db3` | Tesla V100 FHHL 16GB |
| `1db4` | Tesla V100 PCIe 16GB (including LS subsystem variants) |
| `1db5` | Tesla V100 SXM2 32GB (including LS subsystem variants) |
| `1db6` | Tesla V100 PCIe 32GB |
| `1db7` | Tesla V100 DGXS 32GB |
| `1db8` | Tesla V100 SXM3 32GB (including H subsystem variants) |
| `1df6` | Tesla V100S PCIe 32GB |

All eight IDs take the automatic setup path on any exact-match R580 point release. A different point release is
reported as not yet physically tested, not rejected. The package's exact GPU catalog, patch application, DKMS build
and post-install device verification remain mandatory gates.

| check | result |
|---|---|
| both modules loaded together, no kernel warnings | yes |
| CUDA driver API on the V100 (init, context, kernel launch, managed memory) | yes |
| `nvidia-smi` on the V100 (temperatures, power, power limit) | yes |
| `nvidia-persistenced` for the V100 only; a 170 W power limit holding | yes |
| PyTorch 2.8 (cu126) training on the V100 (4-step LoRA job; 140-step pipeline job with the 5060 Ti) | yes |
| llama.cpp CUDA 12 server on the V100 | yes |
| the 5060 Ti unaffected (its own CUDA workloads during all of the above) | yes |
| unload and reload of the renamed modules | yes |

## The tool

| step | status |
|---|---|
| `doctor` | tested on the machine above |
| `fetch 580.178.04` | tested: download, integrity check, unpack; the unpacked `kernel/` matches the source used above |
| `patch --ids 1db6` | tested: all edits apply |
| build (`make modules`) | tested: `nvidia_sidecar.ko`, `nvidia_sidecar_uvm.ko`; only `10de:1db6` claimed; all exports prefixed |
| `install` (DKMS build + install, modprobe options, boot service) | tested |
| `up` (modules, device files, persistence daemon, `POWER_LIMIT_W=170`) | tested |
| `run` (as root: `nvidia-smi`; as a user: a CUDA kernel), `bindpaths`, `status` | tested |
| `down` then `up` again | tested: modules unloaded and the GPU released, then everything back |
| long-running services through `bindpaths` (llama.cpp server, a PyTorch KV store, telemetry) | tested |
| `uninstall` | not yet tested |

## Optional seamless userspace layer

Same machine, driver and GPUs as above. Built with CUDA 12.9 headers; the CUDA runtime acceptance binary was compiled
for both `sm_70` and `sm_120`. Tests ran with the live tower workloads left in place and with host-RAM caps on each
test process.

| check | result |
|---|---|
| fresh userspace build: 682 CUDA exports and 404 NVML exports covered | yes |
| CUDA Driver API: enumerate both GPUs, create contexts, load a kernel and execute it on each | yes |
| CUDA Runtime: execute native `sm_120` and `sm_70` kernels in one process | yes |
| cross-driver `cuMemcpy`, `cudaMemcpyPeer`, `cudaMemcpyDefault` in both directions | yes |
| cross-driver `cudaMemcpyPeerAsync` synchronized fallback | yes; correct result, deliberately no overlap |
| `CUDA_VISIBLE_DEVICES=1,0`: V100 becomes `cuda:0`, 5060 Ti becomes `cuda:1` | yes; kernels and copies still pass |
| `cuGetProcAddress`: version/PTDS mappings stay inside the proxy; unknown symbols fail safely | yes |
| CUDA handle lifecycle: 200,000 create/retire operations plus collision-chain lookup | yes |
| per-device CUDA graphs on both GPUs; concurrent host threads driving both drivers | yes |
| cross-driver `cudaStreamWaitEvent`: source completion and downstream destination work | yes; synchronized fallback |
| CUDA VMM: two-entry access descriptors, output canaries, mixed-access rejection, legacy ordinal/pool translation | yes |
| public NVML: names, UUIDs, indices, temperature, memory, utilization, power, processes and UUID lookup | yes |
| mixed-driver NVML pair semantics: board, topology and P2P status | yes; conservative successful answers |
| merged `nvidia-smi`: default table/processes, `-L`, CSV query, `-q`, XML, `-i`, `dmon`, `pmon` | yes |
| NCCL 2.27.3+cuda12.9: two local ranks, 5060 Ti + V100, all-reduce through automatic socket fallback | yes; both ranks received `3.0` |
| system-wide `seamless off` → `userspace` rebuild → `seamless on` | yes; plain mode restored between steps |
| existing body/router, sidecar persistence, V100 KV, shadow and telemetry services after enable | all active; no Xid, OOM or kernel errors |

PyTorch 2.7.1+cu128 was also checked. Through the proxy it enumerates both GPUs, reports the correct properties, moves
tensors between host memory and each GPU, and runs cuBLAS matmul on each. Its official wheel omits `sm_70`, however,
so its built-in V100 elementwise kernels correctly fail before the sidecar acceptance can be called full PyTorch
support. Full PyTorch use requires one source build containing both architectures (CUDA 12.8,
`TORCH_CUDA_ARCH_LIST="7.0;12.0"`) or another
distribution that explicitly ships both.

The PyTorch wheel's bundled NCCL 2.26.2+cuda12.2 crashes on the RTX 5060 Ti even when the proxy is bypassed. The
machine's NCCL 2.27.7+cuda13.0 supports the 5060 Ti but contains no V100 kernel image. NCCL 2.27.3+cuda12.9 contains
both architectures and is the build used for the successful mixed-driver collective. This is the ordinary
heterogeneous-binary requirement, not a sidecar routing failure.

Other 580.x versions, the seven V100/V100S IDs not present in this machine, other legacy GPUs (Pascal, Maxwell), more
than one legacy GPU, other distributions and kernels have not yet been physically exercised. V100/V100S variants and
R580 point releases are nevertheless automatically supported targets; non-V100 GPU families and other distributions
need the explicit `--allow-untested` path. Reports welcome.

## Guarded installer 0.2.0

The release preflight was run on the live machine above from a clean candidate tree. It identified Ubuntu 24.04,
kernel 6.8.0-149, open kernel module and userspace 580.178.04, the 5060 Ti on `nvidia`, the V100 on
`nvidia_sidecar`, CUDA 12.9 headers, disabled Secure Boot, and every required tool, then reported `ready`.

`setup --mode seamless --plan` produced the exact expected mutation plan without changing the host. GPU catalog
validation accepted `1db6` (Tesla V100-PCIE-32GB) and rejected `2d04` (RTX 5060 Ti) because NVIDIA marks it
`kernelopen`.

The same downloaded R580.178.04 catalog was also checked against the complete automatic family set. It accepts
`1db1`, `1db3`, `1db4`, `1db5`, `1db6`, `1db7`, `1db8` and `1df6`; none is marked `kernelopen` or assigned to an older
legacy branch. This is package-level validation, distinct from the live-board test above.

The exact verifier matched the selected `1DB6` to the sidecar's bound `0000:04:00.0` device, then confirmed that a
fresh merged CUDA process enumerated both `0000:03:00.0` (RTX 5060 Ti) and `0000:04:00.0` (V100), with NVML also
reporting two devices. Setup rejected an otherwise valid GTX 1080 ID because that device is not present on the host,
and normalized duplicate/case-varied V100 IDs to one `1DB6` entry in the plan.

Repository tests cover all eight V100/V100S IDs on both physically tested and new R580 point versions, version/ID
parsing, modern and pre-R580 GPU rejection, fresh transactional install, identical reinstall, refusal of an unowned
systemd unit, injected late activation failure with tree/config/manifest rollback, post-install
userspace/activation snapshot restoration, uninstall, and repeated uninstall. A complete uninstall/reinstall cycle
has not been run on the live workload host because doing so would interrupt its running services; that remains an
explicit release limitation.
