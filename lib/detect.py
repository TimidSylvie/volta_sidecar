#!/usr/bin/env python3
"""Figure out what this machine actually has before the installer touches anything.

This reports the NVIDIA GPUs, which driver owns each one, the open driver's version, and every build prerequisite.

Usage: detect.py report [OPTIONS]       human-readable check (exit 1 on a blocker)
       detect.py legacy-ids             candidate PCI IDs not driven by the open module
       detect.py open-version           loaded open nvidia.ko version (empty if absent)
       detect.py version-status VERSION print tested, untested, or unsupported
       detect.py installer-sha VERSION  print a pinned NVIDIA installer SHA-256, if known
       detect.py release-ids VERSION    print V100-family PCI IDs supported automatically
       detect.py validate-ids CATALOG IDS
                                        validate IDS against NVIDIA's supported-gpus.json
"""
import argparse
import json
import os
import pathlib
import platform
import re
import shutil
import subprocess
import sys

PCI = pathlib.Path("/sys/bus/pci/devices")
DISPLAY_CLASSES = (0x0300, 0x0302)  # VGA and 3D controller; NVIDIA's audio/USB functions are not GPUs.
LAST_LEGACY_BRANCH = 580            # Last branch that still drives Maxwell, Pascal and Volta.
ROOT = pathlib.Path(__file__).resolve().parent.parent
VERSION_MANIFEST = ROOT / "lib" / "supported_versions.json"
TESTED_OS = {("ubuntu", "24.04")}


def run(cmd: list[str]) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=10).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def version_manifest() -> dict:
    try:
        return json.loads(VERSION_MANIFEST.read_text())
    except (OSError, json.JSONDecodeError, TypeError):
        return {"driver_branch": LAST_LEGACY_BRANCH, "versions": {}}


def version_record(version: str) -> dict | None:
    return version_manifest().get("versions", {}).get(version)


def supported_legacy_ids() -> dict[str, str]:
    devices: dict[str, str] = {}
    for family in version_manifest().get("supported_device_families", {}).values():
        devices.update({device_id.upper(): name for device_id, name in family.get("devices", {}).items()})
    return devices


def version_status(version: str) -> str:
    if not re.fullmatch(r"580\.\d+(?:\.\d+)?", version):
        return "unsupported"
    record = version_record(version)
    return "tested" if record and record.get("status") == "tested" else "untested"


def os_release() -> dict[str, str]:
    result: dict[str, str] = {}
    try:
        for raw in pathlib.Path("/etc/os-release").read_text().splitlines():
            if "=" not in raw or raw.lstrip().startswith("#"):
                continue
            key, value = raw.split("=", 1)
            result[key] = value.strip().strip('"')
    except OSError:
        pass
    return result


def userspace_driver_version() -> str:
    smi = "/usr/bin/nvidia-smi" if pathlib.Path("/usr/bin/nvidia-smi").exists() else "nvidia-smi"
    out = run([smi, "--query-gpu=driver_version", "--format=csv,noheader"])
    versions = {line.strip() for line in out.splitlines() if re.fullmatch(r"\d+\.\d+(?:\.\d+)?", line.strip())}
    return versions.pop() if len(versions) == 1 else ""


def cuda_include() -> pathlib.Path | None:
    candidates = []
    for key in ("CUDA_INCLUDE",):
        if os.environ.get(key):
            candidates.append(pathlib.Path(os.environ[key]))
    if os.environ.get("CUDA_HOME"):
        candidates.append(pathlib.Path(os.environ["CUDA_HOME"]) / "include")
    candidates.extend(sorted(pathlib.Path("/usr/local").glob("cuda*/include"), reverse=True))
    candidates.append(pathlib.Path("/usr/include"))
    for path in candidates:
        if (path / "cuda.h").is_file() and (path / "cudaTypedefs.h").is_file():
            return path
    return None


def gpus() -> list[dict]:
    out = []
    for dev in sorted(PCI.glob("*")):
        try:
            if (dev / "vendor").read_text().strip() != "0x10de":
                continue
            cls = int((dev / "class").read_text().strip(), 16) >> 8
            if cls not in DISPLAY_CLASSES:
                continue
            device = (dev / "device").read_text().strip()[2:]
        except OSError:
            continue
        link = dev / "driver"
        driver = os.path.basename(os.readlink(link)) if link.is_symlink() else ""
        name = run(["lspci", "-s", dev.name]).split(": ", 1)[-1] or f"NVIDIA device {device}"
        out.append({"bdf": dev.name, "id": device, "driver": driver, "name": name})
    return out


def open_driver() -> tuple[str, bool]:
    """Return the loaded nvidia.ko version and whether it is the open flavor."""
    text = ""
    try:
        text = pathlib.Path("/proc/driver/nvidia/version").read_text()
    except OSError:
        pass
    m = re.search(r"Kernel Module(?: for \S+)?\s+(\d+\.\d+(?:\.\d+)?)", text)
    if not m:
        return "", False
    return m.group(1), "Open Kernel Module" in text


def legacy(gpu_list: list[dict]) -> list[dict]:
    """Return NVIDIA GPUs outside the primary open driver.

    Pre-Turing cards stay unbound or sit on vfio-pci/nouveau when the open driver refuses them. Cards already on
    nvidia_sidecar count too, which is what makes setup safe to run again.
    """
    return [g for g in gpu_list if g["driver"] != "nvidia"]


def secure_boot() -> str:
    out = run(["mokutil", "--sb-state"])
    if "enabled" in out.lower():
        return "enabled"
    if "disabled" in out.lower():
        return "disabled"
    return "unknown"


def report(mode: str = "core", allow_untested: bool = False, allow_secure_boot: bool = False) -> int:
    blockers, notes = [], []
    print(f"kernel: {platform.release()}")
    machine = platform.machine().lower()
    if machine not in ("x86_64", "amd64"):
        blockers.append(f"unsupported architecture {machine}: nvidia-sidecar currently supports Linux x86_64 only")
    release = os_release()
    distro = (release.get("ID", "").lower(), release.get("VERSION_ID", ""))
    print(f"os: {release.get('PRETTY_NAME') or 'unknown'}")
    if distro not in TESTED_OS:
        message = (f"{release.get('PRETTY_NAME') or 'this distribution'} is outside the tested release matrix "
                   "(Ubuntu 24.04)")
        (notes if allow_untested else blockers).append(message)
    if not pathlib.Path("/run/systemd/system").is_dir():
        blockers.append("systemd is not running as the host service manager")
    gl = gpus()
    if not gl:
        print("no NVIDIA GPUs found")
        return 1
    for g in gl:
        print(f"  {g['bdf']}  10de:{g['id']}  driver={g['driver'] or '(none)'}  {g['name']}")
    version, is_open = open_driver()
    if not version:
        blockers.append("no nvidia.ko is loaded: install NVIDIA's open driver (580 branch) for your newer GPU first")
    elif not is_open:
        blockers.append(f"the loaded nvidia.ko {version} is the proprietary flavor: nvidia-sidecar needs the OPEN driver "
                        "for the newer GPU(s) (the proprietary one then becomes the sidecar for the legacy GPU)")
    else:
        branch = int(version.split(".")[0])
        print(f"open driver: {version}")
        if branch != LAST_LEGACY_BRANCH:
            blockers.append(f"the open driver is {version}; the legacy GPUs need the {LAST_LEGACY_BRANCH} branch, and both "
                            f"drivers must be the SAME version (they share one CUDA userspace). Install open {LAST_LEGACY_BRANCH}.x")
        status = version_status(version)
        if status == "untested":
            notes.append(f"R580 point release {version} has not been physically tested by this project; "
                         "source patching and post-install verification remain fail-closed")
        user_version = userspace_driver_version()
        print(f"driver userspace: {user_version or 'unknown'}")
        if user_version and user_version != version:
            blockers.append(f"kernel module {version} and NVIDIA userspace {user_version} do not match exactly")
        elif not user_version:
            blockers.append("could not determine the installed NVIDIA userspace driver version with nvidia-smi")
    lg = legacy(gl)
    if not lg:
        blockers.append("no legacy GPU found (every NVIDIA GPU is already on the open driver)")
    for g in lg:
        if g["driver"] in ("vfio-pci", "nouveau"):
            blockers.append(f"{g['bdf']} is bound to {g['driver']}: release it (VM shut down / nouveau blacklisted) "
                            "before setup")
        elif g["driver"] and not g["driver"].startswith("nvidia_"):
            blockers.append(f"{g['bdf']} is bound to unexpected driver {g['driver']}")
    print("legacy GPUs: " + (", ".join(f"10de:{g['id']} ({g['bdf']})" for g in lg) or "none"))
    build = pathlib.Path(f"/lib/modules/{platform.release()}/build")
    if not build.exists():
        blockers.append(f"kernel headers missing ({build}): install linux-headers-{platform.release()}")
    required = ("awk", "curl", "depmod", "dkms", "flock", "gcc", "getent", "install", "ldconfig", "lsmod",
                "lspci", "make", "mknod", "modprobe", "mktemp", "paste", "python3", "readlink", "sed",
                "sha256sum", "systemctl", "systemd-run", "tar", "tr")
    for tool in required:
        if not shutil.which(tool):
            blockers.append(f"{tool} not found")
    if mode != "core":
        for tool in ("nm",):
            if not shutil.which(tool):
                blockers.append(f"{tool} not found (required to build the CUDA/NVML proxy)")
        include = cuda_include()
        print(f"CUDA headers: {include or 'not found'}")
        if include is None:
            blockers.append("CUDA headers missing (cuda.h and cudaTypedefs.h): install CUDA 12.8/12.9 or set CUDA_INCLUDE")
    sb = secure_boot()
    print(f"secure boot: {sb}")
    if sb != "disabled":
        message = (f"Secure Boot state is {sb}; automatic module-signing and MOK enrollment are not yet "
                   "release-tested")
        (notes if allow_secure_boot else blockers).append(message)
    for n in notes:
        print(f"note: {n}")
    for b in blockers:
        print(f"BLOCKER: {b}")
    print("ready" if not blockers else "not ready")
    return 1 if blockers else 0


def parse_ids(value: str) -> list[int]:
    try:
        ids = sorted({int(part.strip().lower().removeprefix("10de:"), 16)
                      for part in value.split(",") if part.strip()})
    except ValueError as exc:
        raise ValueError("PCI IDs must be comma-separated hexadecimal device IDs") from exc
    if not ids or any(item < 0 or item > 0xFFFF for item in ids):
        raise ValueError("PCI IDs must be in the range 0000..ffff")
    return ids


def normalized_ids(value: str) -> str:
    return ",".join(f"{device_id:04X}" for device_id in parse_ids(value))


def count_present_ids(value: str) -> int:
    selected = {f"{device_id:04x}" for device_id in parse_ids(value)}
    return sum(1 for gpu in gpus() if gpu["id"].lower() in selected)


def validate_ids(catalog_path: pathlib.Path, value: str) -> int:
    try:
        catalog = json.loads(catalog_path.read_text())
        ids = parse_ids(value)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"invalid GPU catalog or IDs: {exc}", file=sys.stderr)
        return 2
    chips = catalog.get("chips", [])
    failed = False
    for device_id in ids:
        matches = [entry for entry in chips
                   if str(entry.get("devid", "")).lower() == f"0x{device_id:04x}"]
        eligible = [entry for entry in matches
                    if "kernelopen" not in entry.get("features", []) and not entry.get("legacybranch")]
        if eligible:
            names = sorted({entry.get("name", "unknown NVIDIA GPU") for entry in eligible})
            print(f"{device_id:04x}\t{'; '.join(names)}")
            continue
        failed = True
        if any("kernelopen" in entry.get("features", []) for entry in matches):
            reason = "is a modern GPU supported by the open kernel module; refusing to give it to the sidecar"
        elif matches and any(entry.get("legacybranch") for entry in matches):
            branches = sorted({entry.get("legacybranch") for entry in matches if entry.get("legacybranch")})
            reason = f"requires an older legacy branch ({', '.join(branches)}), not R580"
        else:
            reason = "is not listed as an R580 proprietary-driver GPU"
        print(f"{device_id:04x}: {reason}", file=sys.stderr)
    return 1 if failed else 0


def main(argv: list[str]) -> int:
    cmd = argv[0] if argv else "report"
    if cmd == "report":
        ap = argparse.ArgumentParser(prog="detect.py report")
        ap.add_argument("--mode", choices=("core", "shell", "seamless"), default="core")
        ap.add_argument("--allow-untested", action="store_true")
        ap.add_argument("--allow-secure-boot", action="store_true")
        args = ap.parse_args(argv[1:])
        return report(args.mode, args.allow_untested, args.allow_secure_boot)
    if cmd == "legacy-ids":
        for g in legacy(gpus()):
            print(g["id"])
        return 0
    if cmd == "gpu-count":
        print(len(gpus()))
        return 0
    if cmd == "gpu-ids":
        for device_id in sorted({gpu["id"].upper() for gpu in gpus()}):
            print(device_id)
        return 0
    if cmd == "normalize-ids" and len(argv) == 2:
        try:
            print(normalized_ids(argv[1]))
            return 0
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
    if cmd == "count-ids" and len(argv) == 2:
        try:
            print(count_present_ids(argv[1]))
            return 0
        except ValueError as exc:
            print(exc, file=sys.stderr)
            return 1
    if cmd == "open-version":
        version, is_open = open_driver()
        print(version if is_open else "")
        return 0
    if cmd == "version-status" and len(argv) == 2:
        print(version_status(argv[1]))
        return 0 if version_status(argv[1]) != "unsupported" else 1
    if cmd == "installer-sha" and len(argv) == 2:
        record = version_record(argv[1]) or {}
        print(record.get("installer_sha256", ""))
        return 0 if record.get("installer_sha256") else 1
    if cmd == "release-ids" and len(argv) == 2:
        if version_status(argv[1]) == "unsupported":
            return 1
        devices = supported_legacy_ids()
        for device_id in sorted(devices):
            print(device_id)
        return 0 if devices else 1
    if cmd == "validate-ids" and len(argv) == 3:
        return validate_ids(pathlib.Path(argv[1]), argv[2])
    print(__doc__, file=sys.stderr)
    return 2


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
