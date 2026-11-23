#!/usr/bin/env python3
"""Minimal fresh-process acceptance check for the merged CUDA and NVML libraries."""

import argparse
import ctypes
import os
import sys


def normalize_pci(value: str) -> str:
    domain, bus, device_function = value.lower().split(":")
    device, function = device_function.split(".")
    return f"{int(domain, 16):04x}:{int(bus, 16):02x}:{int(device, 16):02x}.{int(function, 16)}"


class DlInfo(ctypes.Structure):
    _fields_ = [
        ("dli_fname", ctypes.c_char_p),
        ("dli_fbase", ctypes.c_void_p),
        ("dli_sname", ctypes.c_char_p),
        ("dli_saddr", ctypes.c_void_p),
    ]


def loaded_path(library: ctypes.CDLL, symbol: str) -> str:
    try:
        dl = ctypes.CDLL("libdl.so.2")
        dl.dladdr.argtypes = [ctypes.c_void_p, ctypes.POINTER(DlInfo)]
        dl.dladdr.restype = ctypes.c_int
        info = DlInfo()
        address = ctypes.cast(getattr(library, symbol), ctypes.c_void_p)
        if dl.dladdr(address, ctypes.byref(info)) and info.dli_fname:
            return os.path.realpath(info.dli_fname.decode(errors="replace"))
    except (AttributeError, OSError):
        pass
    return "unknown"


def check_cuda() -> tuple[int, list[tuple[str, str]], str]:
    cuda = ctypes.CDLL("libcuda.so.1")
    if cuda.cuInit(0) != 0:
        raise RuntimeError("cuInit failed")
    count = ctypes.c_int()
    if cuda.cuDeviceGetCount(ctypes.byref(count)) != 0:
        raise RuntimeError("cuDeviceGetCount failed")
    devices = []
    for ordinal in range(count.value):
        name = ctypes.create_string_buffer(256)
        if cuda.cuDeviceGetName(name, len(name), ordinal) != 0:
            raise RuntimeError(f"cuDeviceGetName failed for device {ordinal}")
        pci = ctypes.create_string_buffer(64)
        if cuda.cuDeviceGetPCIBusId(pci, len(pci), ordinal) != 0:
            raise RuntimeError(f"cuDeviceGetPCIBusId failed for device {ordinal}")
        devices.append((name.value.decode(errors="replace"), normalize_pci(pci.value.decode())))
    return count.value, devices, loaded_path(cuda, "cuInit")


def check_nvml() -> tuple[int, str]:
    nvml = ctypes.CDLL("libnvidia-ml.so.1")
    path = loaded_path(nvml, "nvmlInit_v2")
    if nvml.nvmlInit_v2() != 0:
        raise RuntimeError("nvmlInit_v2 failed")
    try:
        count = ctypes.c_uint()
        if nvml.nvmlDeviceGetCount_v2(ctypes.byref(count)) != 0:
            raise RuntimeError("nvmlDeviceGetCount_v2 failed")
        return count.value, path
    finally:
        nvml.nvmlShutdown()


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--expect", type=int, default=0, help="minimum merged GPU count")
    parser.add_argument("--require-pci", default="", help="comma-separated PCI BDFs that CUDA must enumerate")
    parser.add_argument("--show-libraries", action="store_true", help="print the actual loaded CUDA and NVML paths")
    parser.add_argument("--require-proxy-root", default="", help="require both loaded libraries to live below this directory")
    args = parser.parse_args()
    try:
        cuda_count, devices, cuda_path = check_cuda()
        nvml_count, nvml_path = check_nvml()
    except (OSError, RuntimeError) as exc:
        print(f"userspace verification failed: {exc}", file=sys.stderr)
        return 1
    print(f"CUDA devices: {cuda_count}")
    for ordinal, (name, pci) in enumerate(devices):
        print(f"  {ordinal}: {name} ({pci})")
    print(f"NVML devices: {nvml_count}")
    if args.show_libraries:
        print(f"CUDA library: {cuda_path}")
        print(f"NVML library: {nvml_path}")
    if args.require_proxy_root:
        root = os.path.realpath(args.require_proxy_root)
        escaped = [path for path in (cuda_path, nvml_path)
                   if path == "unknown" or os.path.commonpath((root, path)) != root]
        if escaped:
            print(f"fresh process bypassed the proxy root {root}: {', '.join(escaped)}", file=sys.stderr)
            return 1
    if cuda_count != nvml_count:
        print("CUDA and NVML device counts disagree", file=sys.stderr)
        return 1
    if args.expect and cuda_count < args.expect:
        print(f"expected at least {args.expect} merged devices, found {cuda_count}", file=sys.stderr)
        return 1
    required = {normalize_pci(item.strip()) for item in args.require_pci.split(",") if item.strip()}
    visible = {pci for _, pci in devices}
    missing = sorted(required - visible)
    if missing:
        print(f"CUDA is missing required PCI devices: {', '.join(missing)}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
