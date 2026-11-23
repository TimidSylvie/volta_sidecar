# nvidia-sidecar

Run **legacy NVIDIA GPUs (Maxwell, Pascal, Volta) with CUDA beside newer ones (Turing … Blackwell) on the same Linux
machine — no VM.**

A Tesla V100 needs NVIDIA's *proprietary* kernel module. An RTX 50-series card needs the *open* kernel module. Both are
called `nvidia.ko`, both claim the same names in the kernel, and NVIDIA's documentation says the two flavors cannot be
used at the same time. So people either give up one card or pass the old one into a VM.

nvidia-sidecar takes your own copy of NVIDIA's proprietary driver and **renames everything it claims in the kernel**
(module names, exported symbols, the character-device number, `/proc` folders, memory caches, the PCI driver name), so it
loads *beside* the open driver as `nvidia_sidecar` and drives only your legacy GPUs.

On top of that, an optional userspace layer makes the machine look as if one driver supported every card: `nvidia-smi`
lists all GPUs, NVML tools (nvtop, pynvml, exporters) see all of them, and **one CUDA program can use the legacy and
the newer GPUs at the same time** (`cuda:0` = RTX 5060 Ti, `cuda:1` = V100 in one PyTorch process). Or keep it
simple: run a program in a private view where only the legacy GPUs exist.

```
 ┌──────────────── one Linux kernel ─────────────────┐
 │  nvidia.ko (open)            nvidia_sidecar.ko     │
 │  RTX 5090 / 5060 Ti / ...    V100 / P100 / 1080 Ti │
 │  /dev/nvidia0, major 195     /dev/nvidia-sidecar/* │
 └────────────────────────────────────────────────────┘
   your programs                nvidia-sidecar run -- python train.py
   ── with "seamless on": every program sees every GPU ──
```

## Status

Version 0.2.0. The method and merged CUDA/NVML layer are proven on one machine: Ubuntu 24.04, kernel 6.8,
open driver **580.178.04** driving an RTX 5060 Ti, and a Tesla V100 PCIe 32GB on the renamed proprietary module.
That is the physically validated configuration, not the support boundary. The installer automatically supports every
Tesla V100/V100S board ID and every exact-match R580 point release. It identifies a new point release as not yet
physically tested, then still applies the same fail-closed source patch, build, device-catalog and post-install checks.
Other operating systems and non-V100 legacy GPU families require `--allow-untested`.

This does not flash a GPU or modify the running primary driver. It downloads NVIDIA's matching `.run` package, checks
it, patches a copy of the proprietary kernel interface source, builds renamed modules with DKMS, and installs a
reversible userspace compatibility layer. See [docs/TESTED.md](docs/TESTED.md) and
[docs/INSTALLER.md](docs/INSTALLER.md).

## Requirements

- Linux x86_64 with systemd. The guarded installer currently accepts Ubuntu 24.04; other distributions require
  `--allow-untested`.
- Kernel headers for the running kernel, `dkms`, `make`, `gcc`, `python3`, `curl`, PCI/kmod utilities and standard GNU
  system tools. `doctor` names anything missing before setup changes the host.
- **NVIDIA's open R580 driver**, already installed and driving the newer GPU(s). NVIDIA's
  [driver/architecture matrix](https://docs.nvidia.com/datacenter/tesla/drivers/latest/cuda-toolkit-driver-and-architecture-matrix.html)
  lists R580 as the last driver branch for Maxwell, Pascal and Volta. The sidecar proprietary modules and installed
  NVIDIA userspace must match the loaded open module's complete version exactly.
- Your legacy GPU must not be claimed by `vfio-pci` or `nouveau` (shut down any VM that owns it; blacklist nouveau).
- Secure Boot off. The installer fails closed when Secure Boot is enabled or unknown because automated MOK signing is
  not release-tested; `--allow-secure-boot` is an expert override.
- `shell` and `seamless` modes need CUDA headers (`cuda.h` and `cudaTypedefs.h`) to build the proxy.

R580 can run CUDA 13 applications, but NVIDIA's
[CUDA 13 release notes](https://docs.nvidia.com/cuda/archive/13.0.0/cuda-toolkit-release-notes/)
remove Maxwell/Pascal/Volta compilation and library support from the CUDA 13 toolkit. Code that executes on those
GPUs must therefore carry compatible CUDA 12.x-built device kernels and libraries. The sidecar makes the device
reachable; it cannot add a missing `sm_50`, `sm_60`, or `sm_70` binary.

## One-command guarded install

```bash
git clone https://github.com/TimidSylvie/volta_sidecar && cd volta_sidecar
./install.sh --mode shell
```

`install.sh` elevates with `sudo`, runs strict preflight, displays the exact GPU IDs and persistent paths it will
change, asks for confirmation, transactionally switches the managed core files, starts the service, and verifies
CUDA/NVML from a fresh process. A later proxy, activation or verification failure restores the prior activation mode
and userspace proxy; the updated core remains installed for diagnosis.

- `--mode core`: only the renamed kernel modules; use `nvidia-sidecar run -- ...` for a legacy-only private view.
- `--mode shell` (default): also build the merged CUDA/NVML layer, but activate it only with
  `eval "$(nvidia-sidecar env)"`.
- `--mode seamless`: put the proxies first system-wide and install the merged `nvidia-smi` wrapper.
- `--plan`: print preflight and the mutation plan without changing anything.
- `--yes`: non-interactive confirmation.
- `--ids 1db6,...`: explicitly select legacy PCI device IDs. Every ID is checked against NVIDIA's own catalog; a
  modern/open-module GPU or a pre-R580 legacy GPU is rejected.

Automatic V100-family support covers `10de:1db1`, `1db3`, `1db4`, `1db5`, `1db6`, `1db7`, `1db8` and `1df6`: the
Tesla V100 SXM2/SXM3, FHHL, PCIe, DGXS and V100S PCIe variants. Subsystem-specific LS/H variants use those same
device IDs. The default is auto-detection, so most people do not need to pass `--ids` at all.

For the fully merged host experience:

```bash
./install.sh --mode seamless
nvidia-smi
python3 -c 'import torch; print([torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())])'
```

Use `--allow-untested` only after reading the plan when using an operating system outside the tested matrix or a
legacy GPU family other than V100/V100S. An exact-match R580 point release and every V100/V100S ID need no override.
The NVIDIA installer still performs its internal integrity check, and every selected ID must pass the downloaded
package's own GPU catalog before kernel source is copied.

## Seamless mode: every program sees every GPU

- **CUDA**: a proxy `libcuda.so.1` holds both drivers' CUDA libraries in one process and routes every call to the
  driver that owns the device, context, stream or memory it names. Copies between the two drivers' GPUs (peer copies,
  `cudaMemcpyDefault`, `tensor.to("cuda:1")`) are staged through host memory. Cross-driver async copies currently use
  a correctness-first synchronized fallback: they preserve ordering and results but not copy/compute overlap.
  `CUDA_VISIBLE_DEVICES` works across both drivers: indices in the merged numbering, UUIDs, and any order (`1,0`
  puts the V100 first).
- **NCCL**: mixed-driver ranks use NCCL's ordinary socket transport because CUDA IPC handles cannot be imported by
  the other kernel driver. In unified mode the proxy defaults `NCCL_SHM_DISABLE=1` only when the variable is unset;
  an explicit user setting wins. NCCL 2.27.3+cuda12.9 was tested with one rank on each card and an ordinary
  all-reduce. As on any heterogeneous system, the NCCL build itself must contain kernels for every participating
  compute capability.
- **NVML**: a proxy `libnvidia-ml.so.1` merges both drivers' GPUs, so monitoring tools and `torch.cuda.device_count()`
  agree with CUDA.
- **nvidia-smi**: a wrapper runs NVIDIA's `nvidia-smi` against each driver and merges the output (tables, `-L`,
  `--query-*`, `-q`, `-q -x`, `dmon`, `pmon`, `-l`); `-i N` goes to the driver that owns GPU N.

Your framework still needs kernels for every architecture you use. The tested PyTorch 2.7.1+cu128 wheel sees both
cards and runs cuBLAS on both, but omits PyTorch's `sm_70` elementwise kernels. For full PyTorch operation, build it
with a CUDA 12.8 toolkit and
`TORCH_CUDA_ARCH_LIST="7.0;12.0"`, or use a distribution that explicitly lists both architectures. See
[docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md#one-process-both-drivers).

## Manual/developer flow

The original phases remain available for debugging:

```bash
sudo ./nvidia-sidecar fetch
sudo ./nvidia-sidecar patch
sudo ./nvidia-sidecar install
sudo nvidia-sidecar up
sudo nvidia-sidecar userspace
sudo nvidia-sidecar seamless on
sudo nvidia-sidecar verify
```

## Use (private view: legacy GPUs only)

```bash
sudo nvidia-sidecar run -- python train.py               # foreground, as you (sudo's user)
sudo nvidia-sidecar run --bg job1 -- python train.py     # background unit nvidia-sidecar-run-job1
nvidia-sidecar bindpaths                                  # lines to add to your own systemd service
sudo nvidia-sidecar status
```

For a long-running service, put the lines from `bindpaths` into its `[Service]` section; it then sees the legacy
GPUs as `/dev/nvidia0…`. Optional settings live in `/etc/nvidia-sidecar/sidecar.conf` (for example
`POWER_LIMIT_W=170`, applied at every `up`).

## How it works

See [docs/HOW-IT-WORKS.md](docs/HOW-IT-WORKS.md): what collides between the two drivers, why NVIDIA's binary core needs no
changes, why CUDA accepts a different device number (`NVreg_ModifyDeviceFiles=0`), and how the private view is built.

## Limits

- Without the userspace layer, one process talks to one driver. With it, one process can use both, but there is no
  peer-to-peer DMA or NVLink between the two drivers' GPUs (copies go through host memory, and
  `cudaDeviceCanAccessPeer` says 0), and cross-driver async copies currently serialize both contexts. Per-device CUDA
  graphs work; one graph cannot contain objects from both drivers. CUDA IPC handles and graphics interop are not
  translated across drivers. NCCL works through its socket transport rather than its faster local CUDA-IPC/SHM path,
  so collectives pay an additional host/network transport cost.
- Same-version userspace: the open driver must stay on the 580 branch while nvidia-sidecar is in use.
- A V100 remains an `sm_70` V100. It does not gain FP8/FP4, Blackwell instructions, CUDA 13 toolkit libraries, or
  any other newer hardware capability.
- This is independent nvidia-sidecar support, not a claim that NVIDIA ships the mixed open/proprietary configuration.
  The modules are NVIDIA's own code with renamed identifiers; a bad build can still crash a kernel, so first install
  on a machine you can reboot.
- nvidia-sidecar never redistributes NVIDIA software: it downloads NVIDIA's installer from download.nvidia.com and
  patches it on your machine. NVIDIA's license applies to NVIDIA's files.

## Uninstall

```bash
sudo nvidia-sidecar uninstall
```

Uninstall refuses unknown paths, stops before removing files if modules are busy, removes the DKMS package for every
installed kernel, restores the prior `nvidia-smi`, and keeps downloads/configuration under `/var/lib/nvidia-sidecar`
and `/etc/nvidia-sidecar` for recovery. See [docs/INSTALLER.md](docs/INSTALLER.md) for owned paths.

## Credit

Created by Benjamin (Sylvie) Edgar — see [NOTICE](NOTICE). If you build on it, please keep the NOTICE and cite it
([CITATION.cff](CITATION.cff)). Apache License 2.0.
