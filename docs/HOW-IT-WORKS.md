# How nvidia-sidecar works

## The problem

Since the 560 driver series NVIDIA ships two flavors of its Linux kernel module:

| flavor | source folder in the installer | GPUs |
|---|---|---|
| proprietary | `kernel/` (open interface layer + closed `nv-kernel.o_binary`) | Maxwell … Hopper (no Blackwell) |
| open | `kernel-open/` (fully open, needs the GPU's GSP processor) | Turing … Blackwell |

Maxwell, Pascal and Volta have no GSP, so only the proprietary flavor drives them; Blackwell requires the open flavor.
Both build a module called `nvidia.ko`, so a kernel can hold only one of them.

R580 is the last NVIDIA driver branch for Maxwell, Pascal and Volta. It is new enough to host CUDA 13 applications,
but the CUDA 13 toolkit no longer compiles or ships libraries for those architectures. The sidecar solves the
kernel-driver coexistence problem; applications still need CUDA 12.x-built device code and compatible libraries for
the legacy GPU.

## Installer safety boundary

`nvidia-sidecar setup` never replaces the primary open module. It requires the installed open kernel module and
NVIDIA userspace to have the same complete R580 version, downloads that exact NVIDIA package, validates its pinned
hash when the version has a physically tested record, and runs NVIDIA's internal package check. Every R580 point
release is eligible; an untested point release is labeled as such rather than blocked. Tesla V100/V100S device IDs are
first-class automatic targets. After extraction, every selected PCI ID is still validated against that exact
package's bundled `supported-gpus.json` before patching.

The DKMS source, generated configuration, installed repository and systemd units are staged before activation.
Existing paths must be recognized as an earlier sidecar install. The prior managed files and install manifest are
restored if a late activation step fails. See [INSTALLER.md](INSTALLER.md) for the complete ownership and recovery
contract.

## What collides, and what the patch renames

Everything the module claims in the kernel lives in the *open interface layer* of the proprietary flavor. The closed
`nv-kernel.o_binary` defines none of the exported symbols, contains no `/dev` or `/proc` paths, and only calls
functions of the interface layer (plus the kernel's retpoline thunks). So renaming the interface layer is enough; the
binary core is linked in unchanged.

| claimed in the kernel | why it collides | renamed to |
|---|---|---|
| module files `nvidia`, `nvidia-uvm` | one module name per kernel | `nvidia_sidecar`, `nvidia_sidecar_uvm` |
| ~94 `EXPORT_SYMBOL`s (`nvUvmInterface*`, `nvidia_p2p_*`, …) | the kernel refuses a duplicate export | `sidecar_*` via one `-include` header |
| char device major 195, minors 0–247 and 255 | the open driver registered them | a major assigned by the kernel at load |
| PCI driver name, `/proc/driver/nvidia`, IRQ and thread names | duplicate driver/proc names fail | `nvidia_sidecar` (NVIDIA's own `MODULE_INSTANCE_STRING` macro) |
| `/proc/driver/nvidia-{uvm,caps,nvlink,nvswitch}`, their device labels | duplicate proc folders fail | `nvidia_sidecar-*` |
| slab caches (`nvidia_stack_cache`, `uvm_*`) | NVIDIA gives them a constructor, so they cannot merge and register their exact name in sysfs | `sidecar_` prefix |
| PCI ID table (every NVIDIA display device) | both drivers would probe every GPU | only your legacy GPUs' IDs |

The open driver already refuses pre-Turing GPUs at probe time, so it never fights over the legacy card.

## Why CUDA accepts the different device number

`libcuda` and `nvidia-smi` embed NVIDIA's open-source device-file helper (the code of `nvidia-modprobe`). It compares a
device file's major number with 195 only while *maintaining* device files (creating or repairing them). With the module
option `NVreg_ModifyDeviceFiles=0` (a documented option, used by container setups) userspace never maintains them and
only opens them by path; it then needs `/dev/char/195:<minor>` to resolve to the device file, which it does because
those links point at `../nvidia<minor>` relative to the view's `/dev`. This was verified by making every NVIDIA device
file report a different major (an `LD_PRELOAD` shim over the old glibc `__xstat` entry points, which is what `libcuda`
imports) and running CUDA and `nvidia-smi` unchanged.

## The private view

`libcuda` opens fixed paths: `/dev/nvidiactl`, `/dev/nvidia<N>`, `/dev/nvidia-uvm`, and reads `/proc/driver/nvidia`.
A program that should use the legacy GPUs runs in its own mount namespace where those paths are bind-mounted onto the
sidecar's device files (`/dev/nvidia-sidecar/*`, on devtmpfs — a `nodev` filesystem such as `/run` would block them) and
onto `/proc/driver/nvidia_sidecar`. systemd builds the namespace (`BindPaths=` in a unit, or `systemd-run -p BindPaths=`
in `nvidia-sidecar run`), so nothing runs privileged inside it.

Because both drivers are the same version, the one installed CUDA userspace serves both: no second `libcuda`, no
containers.

## Persistence and power

The legacy GPUs get their own `nvidia-persistenced` instance (inside the view, with its own runtime folder), so they stay
initialized and settings such as a power limit (`POWER_LIMIT_W`) hold when no program is using them.

## One process, both drivers

The optional userspace layer (`nvidia-sidecar userspace`, `seamless on`) runs both drivers inside one process.

**Two copies of NVIDIA's libraries.** Each copy is loaded with `dlmopen` into its own link-map namespace, so each has
its own globals and its own libc. A small loader library goes first into each namespace. For the legacy copy, its
first dependency is the path-redirect library, which points that copy's `open()`/`stat()` of `/dev/nvidia*` and
`/proc/driver/nvidia` at the sidecar's files. The real library is then added to that namespace by absolute path. Two
conflicts between the copies had to be removed:
- Both copies name an abstract Unix socket after the process (`cuda-uvmfd-<mntns>-<pid>`). The legacy copy's socket
  is renamed.
- Each libc copy hands out pthread keys from its own table. A helper (`nsfix.c`) forwards key creation and lookup to
  the process's main libc, so the copies stop overwriting each other's thread-local data.

**Routing.** The proxy `libcuda.so.1` exports every driver function. Typed wrappers are generated from
`cudaTypedefs.h`, plus the mapping that the real library's `cuGetProcAddress` uses. Each call goes to the copy that
owns its first known argument: a context, stream, event, module or other handle (tracked when it is created); a
device pointer (asked of each copy and cached per 2 MiB); or a device ordinal (the merged numbering: the main driver's
GPUs first, unless `CUDA_VISIBLE_DEVICES` orders them differently). Without such an argument, the call goes to the
copy whose context is current on that thread. Copies between the two copies' memory are staged through host memory.
Libraries and kernels are not tied to a context (`cuLibraryLoadData`, used by the CUDA runtime for every
fatbinary), so they are loaded into both copies behind one handle, and translated per call.

There is no peer DMA between drivers. A cross-driver copy is staged through a bounded host buffer. The current
implementation deliberately turns cross-driver `*Async` copies into a synchronized fallback: it drains the source
and destination contexts, stages the bytes, and returns only when the copy is complete. This is slower and removes
copy/compute overlap, but it keeps stream-visible data ordering honest until a two-stream staging engine exists.

**NCCL.** CUDA IPC and virtual-memory shareable handles belong to the kernel driver that exported them. NCCL's local
SHM transport normally imports those handles in the peer rank, which cannot work when the ranks are backed by
different kernel drivers. In unified mode the proxy therefore defaults `NCCL_SHM_DISABLE=1` when the user has not
set it, making NCCL select its standard socket transport. This default is intentionally applied before
`CUDA_VISIBLE_DEVICES` narrows a rank: two ranks can each expose one GPU yet still belong to opposite drivers. Set
`NCCL_SHM_DISABLE=0` explicitly to retain SHM when every participating rank is known to use one underlying driver.
The socket path is correct but slower than CUDA-IPC SHM. A two-rank all-reduce across the RTX 5060 Ti and V100 was
verified with NCCL 2.27.3+cuda12.9. The NCCL binary must itself contain device kernels for both `sm_120` and `sm_70`;
the proxy cannot manufacture a missing architecture image.

Framework binaries remain responsible for device code. A CUDA 12.8 program compiled for `sm_70` and `sm_120` runs
on both sides through the proxy. An official PyTorch 2.7.1+cu128 wheel, however, contains `sm_120` but not `sm_70`:
enumeration and cuBLAS work on both cards, while a built-in V100 elementwise kernel correctly fails with "no kernel
image is available." Full PyTorch use therefore needs a source build with both architectures, for example
`TORCH_CUDA_ARCH_LIST="7.0;12.0"` against CUDA 12.8.

**The CUDA runtime's private interface.** `libcudart` also uses undocumented export tables (`cuGetExportTable`). The
proxy merges them: each entry dispatches to the right copy. Entries whose device or context is not argument 1 have
typed wrappers. One entry is an integrity check. The runtime asks the driver for a hash over the driver's own table
addresses and every GPU's UUID and PCI location, and compares it with its own computation. The proxy hands out its
own tables, so it answers that check itself, with the same algorithm over its addresses and the merged device list.
The algorithm was documented by the ZLUDA project, and the port in `integrity.c` was checked against the real
driver's answers. Without this, the runtime stops with "integrity checks failed".

**NVML and nvidia-smi.** The NVML proxy works the same way: two copies, calls routed by which copy issued the device
handle, and a merged enumeration. Mixed-driver pair queries report conservative, successful answers equivalent to
an ordinary pair without direct P2P: different boards, system-level topology and P2P unavailable. `nvidia-smi` talks
to NVML through a private interface, so it is not proxied.
Instead a wrapper runs the real `nvidia-smi` once per driver and merges what it prints.
