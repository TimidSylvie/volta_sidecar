#!/usr/bin/env python3
"""Make NVIDIA's proprietary Linux modules able to live beside the open ones.

The problem is annoyingly literal: both module flavors are called nvidia.ko and claim the same kernel names, so Linux
can only load one. The open flavor drives Turing and newer GPUs, while Maxwell, Pascal and Volta still need the
proprietary flavor. This patches a copy of the proprietary module's open interface layer—never NVIDIA's binary
core—so every colliding name becomes its own and it only claims the GPUs you asked for:

  module files         nvidia.ko, nvidia-uvm.ko         -> nvidia_<suffix>.ko, nvidia_<suffix>_uvm.ko
  exported symbols     every EXPORT_SYMBOL of nvidia.ko -> <suffix>_<name> (one -include header)
  char device          fixed major 195                  -> a major assigned by the kernel at load
  driver/proc names    "nvidia", /proc/driver/nvidia    -> "nvidia_<suffix>", /proc/driver/nvidia_<suffix>
  other /proc folders  driver/nvidia-{uvm,caps,nvlink,nvswitch} -> driver/nvidia_<suffix>-...
  memory caches        unmergeable slab caches          -> "<suffix>_" prefix (the open driver holds the plain names)
  PCI ID table         every NVIDIA display device      -> only the device IDs given

Usage: patch_driver.py --source DIR --ids 1db6[,1b80,...] [--suffix sidecar]
DIR is the "kernel" folder of the NVIDIA-Linux-x86_64-<version>.run installer (or a DKMS copy of it), edited in place.
Every expected edit must match. If NVIDIA changes the source layout, the patch stops and tells you exactly where;
the original tree is never left half-patched because all work happens on a copy.
"""
import argparse
import pathlib
import re
import shutil
import sys
import tempfile

MARK = "nvidia-sidecar"


class PatchError(Exception):
    pass


class Tree:
    def __init__(self, root: pathlib.Path):
        self.root = root

    def read(self, rel: str) -> str:
        path = self.root / rel
        if not path.exists():
            raise PatchError(f"missing file {rel} (is this the proprietary 'kernel' folder of a 580-series driver?)")
        return path.read_text()

    def write(self, rel: str, text: str) -> None:
        (self.root / rel).write_text(text)

    def sub(self, rel: str, old: str, new: str, count: int | None = 1) -> int:
        """Replace literal text, requiring an exact count unless count is None."""
        text = self.read(rel)
        n = text.count(old)
        if (count is None and n == 0) or (count is not None and n != count):
            want = "at least 1" if count is None else str(count)
            raise PatchError(f"{rel}: expected {want} of {old[:70]!r}, found {n}")
        self.write(rel, text.replace(old, new))
        return n

    def resub(self, rel: str, pattern: str, new, count: int | None = 1) -> int:
        text = self.read(rel)
        out, n = re.subn(pattern, new, text, flags=re.S | re.M)
        if (count is None and n == 0) or (count is not None and n != count):
            want = "at least 1" if count is None else str(count)
            raise PatchError(f"{rel}: expected {want} match(es) of /{pattern[:70]}/, found {n}")
        self.write(rel, out)
        return n


def exported_symbols(tree: Tree) -> list[str]:
    """Find what nvidia.ko actually exports, not symbols NV_CHECK_EXPORT_SYMBOL merely probes for."""
    names = set()
    for path in sorted((tree.root / "nvidia").rglob("*.c")):
        for m in re.finditer(r"^\s*(?:NV_)?EXPORT_SYMBOL(?:_GPL)?\s*\(\s*(\w+)\s*\)", path.read_text(), re.M):
            names.add(m.group(1))
    if not names:
        raise PatchError("found no exported symbols in nvidia/ - unexpected source layout")
    return sorted(names)


def pci_tables(ids: list[int]) -> tuple[str, str]:
    def entry(dev: int, cls: str | None) -> str:
        if cls is None:
            return (f"    {{ .vendor = PCI_VENDOR_ID_NVIDIA, .device = 0x{dev:04x}, .subvendor = PCI_ANY_ID, "
                    f".subdevice = PCI_ANY_ID, .class = 0, .class_mask = 0 }},\n")
        return (f"    {{ .vendor = PCI_VENDOR_ID_NVIDIA, .device = 0x{dev:04x}, .subvendor = PCI_ANY_ID, "
                f".subdevice = PCI_ANY_ID, .class = ({cls} << 8), .class_mask = ~0 }},\n")
    rm = "".join(entry(d, c) for d in ids for c in ("PCI_CLASS_DISPLAY_VGA", "PCI_CLASS_DISPLAY_3D"))
    mod = "".join(entry(d, None) for d in ids)
    return (f"/* Devices supported by RM ({MARK}: only the listed legacy GPUs) */\nstruct pci_device_id nv_pci_table[] = {{\n{rm}    {{ }}\n}};\n",
            f"/* Devices supported by all drivers in nvidia.ko ({MARK}: only the listed legacy GPUs) */\nstruct pci_device_id nv_module_device_table[] = {{\n{mod}    {{ }}\n}};\n")


def patch(tree: Tree, suffix: str, ids: list[int]) -> list[str]:
    s = suffix
    mod = f"nvidia_{s}"
    log = []

    # First, give the sidecar one name everywhere NVIDIA derives it from the module instance.
    tree.sub("common/inc/nv-linux.h", '#define MODULE_INSTANCE_STRING ""', f'#define MODULE_INSTANCE_STRING "_{s}"')
    tree.sub("nvidia/nv-procfs.c", "char nv_dir_name[20];", "char nv_dir_name[64];")   # "driver/nvidia_<suffix>"
    log.append(f"driver name {mod}, /proc/driver/{mod}")

    # Slab caches collide too. NVIDIA gives these a constructor, so prefix the name they register in sysfs.
    tree.sub("common/inc/nv-linux.h", "    name_unique = (char *)name;\n#endif",
             f'    name_unique = kasprintf(GFP_KERNEL, "{s}_%s", name);\n    if (!name_unique)\n        return NULL;\n#endif')
    log.append(f"memory caches prefixed {s}_")

    # Major 195 belongs to the open driver. Let Linux choose a free major for this copy instead.
    tree.sub("nvidia/nv.c", "MODULE_ALIAS_CHARDEV_MAJOR(NV_MAJOR_DEVICE_NUMBER);\n",
             f"static unsigned int nv_{s}_major;  /* {MARK}: assigned by the kernel at load */\n")
    tree.sub("nvidia/nv.c", "    rc = register_chrdev_region(MKDEV(NV_MAJOR_DEVICE_NUMBER, minor),\n            count, name);",
             f"    if (nv_{s}_major == 0)\n    {{\n        dev_t first;\n        rc = alloc_chrdev_region(&first, minor, count, name);\n"
             f"        if (rc >= 0)\n            nv_{s}_major = MAJOR(first);\n    }}\n    else\n    {{\n"
             f"        rc = register_chrdev_region(MKDEV(nv_{s}_major, minor), count, name);\n    }}")
    tree.sub("nvidia/nv.c", "MKDEV(NV_MAJOR_DEVICE_NUMBER, minor)", f"MKDEV(nv_{s}_major, minor)", count=3)
    tree.sub("nvidia/nv.c", "if (MAJOR(rdev) != NV_MAJOR_DEVICE_NUMBER)", f"if (MAJOR(rdev) != nv_{s}_major)")
    tree.sub("nvidia/nv.c", '&nv_linux_devices_cdev, "nvidia", &nvidia_fops);', f'&nv_linux_devices_cdev, "{mod}", &nvidia_fops);')
    tree.sub("nvidia/nv.c", '&nv_linux_control_device_cdev, "nvidiactl", &nvidia_fops);', f'&nv_linux_control_device_cdev, "{mod}ctl", &nvidia_fops);')
    if "NV_MAJOR_DEVICE_NUMBER" in tree.read("nvidia/nv.c"):
        raise PatchError("nvidia/nv.c still uses NV_MAJOR_DEVICE_NUMBER after the edits")
    log.append("char device: kernel-assigned major")

    # Leave every other card alone: this copy may only claim the legacy IDs selected by the user.
    rm_table, mod_table = pci_tables(ids)
    tree.resub("nvidia/nv-pci-table.c", r"/\* Devices supported by RM \*/\nstruct pci_device_id nv_pci_table\[\] = \{\n.*?\n\};\n", rm_table)
    tree.resub("nvidia/nv-pci-table.c", r"/\* Devices supported by all drivers in nvidia\.ko \*/\nstruct pci_device_id nv_module_device_table\[\d*\] = \{\n.*?\n\};\n", mod_table)
    tree.resub("nvidia/nv-pci-table.h", r"extern struct pci_device_id nv_module_device_table\[\d+\];", "extern struct pci_device_id nv_module_device_table[];")
    log.append("PCI IDs: " + ", ".join(f"10de:{d:04x}" for d in ids))

    # Rename the remaining /proc folders and character-device labels not based on MODULE_INSTANCE_STRING.
    tree.sub("nvidia/nv-caps.c", '"/driver/nvidia/capabilities', f'"/driver/{mod}/capabilities', count=None)
    tree.sub("nvidia/nv-caps.c", '"/driver/nvidia-nvlink/capabilities', f'"/driver/{mod}-nvlink/capabilities', count=None)
    tree.sub("nvidia/nv-caps.c", '#define NV_CAP_PROCFS_DIR "driver/nvidia-caps"', f'#define NV_CAP_PROCFS_DIR "driver/{mod}-caps"')
    tree.sub("nvidia/nv-caps.c", '"nvidia-caps");', f'"{mod}-caps");')
    tree.sub("nvidia/nv-caps-imex.c", '"nvidia-caps-imex-channels");', f'"{mod}-caps-imex-channels");')
    tree.sub("nvidia/nvlink_linux.c", '#define NVLINK_PROCFS_DIR "driver/nvidia-nvlink"', f'#define NVLINK_PROCFS_DIR "driver/{mod}-nvlink"')
    tree.sub("nvidia/nvlink_linux.h", '"nvidia-nvlink"', f'"{mod}-nvlink"')
    tree.sub("nvidia/export_nvswitch.h", '"nvidia-nvswitch"', f'"{mod}-nvswitch"')   # also its PCI driver name
    tree.sub("nvidia/procfs_nvswitch.c", '"driver/nvidia-nvswitch"', f'"driver/{mod}-nvswitch"')
    tree.resub("nvidia-uvm/uvm.c", r'#define NVIDIA_UVM_DEVICE_NAME(\s+)"nvidia-uvm"', rf'#define NVIDIA_UVM_DEVICE_NAME\1"{mod}-uvm"')
    tree.sub("nvidia-uvm/uvm_procfs.c", '#define UVM_PROC_DIR_NAME "driver/nvidia-uvm"', f'#define UVM_PROC_DIR_NAME "driver/{mod}-uvm"')
    log.append(f"/proc/driver/{mod}-{{uvm,caps,nvlink,nvswitch}}")

    # The open driver exports these names already, so prefix every sidecar export before the kernel sees it.
    names = exported_symbols(tree)
    header = [f"/* {MARK}: the symbols nvidia.ko exports, renamed so it can load beside the open nvidia.ko */",
              f"#ifndef NV_{s.upper()}_RENAME_H", f"#define NV_{s.upper()}_RENAME_H"]
    header += [f"#define {n} {s}_{n}" for n in names] + ["#endif", ""]
    tree.write(f"common/inc/nv-{s}-rename.h", "\n".join(header))
    tree.sub("nvidia/nvidia.Kbuild", "NVIDIA_CFLAGS += -I$(src)/nvidia\n",
             f"NVIDIA_CFLAGS += -I$(src)/nvidia\nNVIDIA_CFLAGS += -include $(src)/common/inc/nv-{s}-rename.h\n")
    tree.sub("nvidia-uvm/nvidia-uvm.Kbuild", "NVIDIA_UVM_CFLAGS += -DNVIDIA_UVM_ENABLED\n",
             f"NVIDIA_UVM_CFLAGS += -DNVIDIA_UVM_ENABLED\nNVIDIA_UVM_CFLAGS += -include $(src)/common/inc/nv-{s}-rename.h\n")
    log.append(f"{len(names)} exported symbols prefixed {s}_")

    # Finally, give the module files the names users and DKMS will see.
    tree.sub("nvidia/nvidia.Kbuild", "obj-m += nvidia.o\nnvidia-y := $(NVIDIA_OBJECTS)", f"obj-m += {mod}.o\n{mod}-y := $(NVIDIA_OBJECTS)")
    tree.sub("nvidia/nvidia.Kbuild", "nvidia-y += $(NVIDIA_BINARY_OBJECT_O)", f"{mod}-y += $(NVIDIA_BINARY_OBJECT_O)")
    tree.sub("nvidia/nvidia.Kbuild", "NVIDIA_KO = nvidia/nvidia.ko", f"NVIDIA_KO = nvidia/{mod}.ko")
    tree.sub("nvidia-uvm/nvidia-uvm.Kbuild", "obj-m += nvidia-uvm.o\nnvidia-uvm-y := $(NVIDIA_UVM_OBJECTS)", f"obj-m += {mod}_uvm.o\n{mod}_uvm-y := $(NVIDIA_UVM_OBJECTS)")
    tree.sub("nvidia-uvm/nvidia-uvm.Kbuild", "NVIDIA_UVM_KO = nvidia-uvm/nvidia-uvm.ko", f"NVIDIA_UVM_KO = nvidia-uvm/{mod}_uvm.ko")
    log.append(f"modules {mod}.ko, {mod}_uvm.ko")

    (tree.root / f".{MARK}-patched").write_text(f"suffix={s}\nids={','.join(f'{d:04x}' for d in ids)}\n")
    return log


def main(argv: list[str]) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", required=True, type=pathlib.Path, help="the proprietary 'kernel' folder (edited in place)")
    ap.add_argument("--ids", required=True, help="comma-separated PCI device IDs of the legacy GPUs, e.g. 1db6,1b80")
    ap.add_argument("--suffix", default="sidecar", help="name suffix (letters/digits, default: sidecar)")
    a = ap.parse_args(argv)
    if not re.fullmatch(r"[a-z][a-z0-9]{1,15}", a.suffix):
        ap.error("--suffix must be 2-16 lowercase letters/digits, starting with a letter")
    try:
        ids = sorted({int(x, 16) for x in a.ids.replace("10de:", "").split(",") if x.strip()})
    except ValueError:
        ap.error("--ids must be hexadecimal PCI device IDs")
    if not ids:
        ap.error("--ids is empty")
    if any(device_id < 0 or device_id > 0xFFFF for device_id in ids):
        ap.error("--ids values must be 16-bit PCI device IDs (0000..ffff)")
    src = a.source.resolve()
    if (src / f".{MARK}-patched").exists():
        print(f"{src} is already patched", file=sys.stderr)
        return 1
    if not (src / "nvidia" / "nv-kernel.o_binary").exists():
        print(f"{src} has no nvidia/nv-kernel.o_binary: use the proprietary 'kernel' folder, not 'kernel-open'", file=sys.stderr)
        return 1
    # Work on a copy and only swap it in after every expected edit matched.
    with tempfile.TemporaryDirectory(dir=src.parent) as tmp:
        work = pathlib.Path(tmp) / "kernel"
        shutil.copytree(src, work, symlinks=True)
        try:
            log = patch(Tree(work), a.suffix, ids)
        except PatchError as exc:
            print(f"patch failed, nothing changed: {exc}", file=sys.stderr)
            return 2
        backup = src.with_name(src.name + ".unpatched")
        src.rename(backup)
        work.rename(src)
        shutil.rmtree(backup)
    for line in log:
        print("patched:", line)
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
