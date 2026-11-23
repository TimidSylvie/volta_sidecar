#!/usr/bin/env bash
# Build the part applications actually see into OUT (default: ./build/userspace):
#   libnvidia-sidecar-redirect.so   path redirect: one process -> the sidecar's device files (LD_PRELOAD / loaders)
#   nvml/libnvidia-ml.so.1          NVML proxy: both drivers' GPUs in one NVML (nvidia-smi-style tools, pynvml, nvtop)
#   cuda/libcuda.so.1               CUDA proxy: both drivers' GPUs in one CUDA process (runtime, PyTorch, ...)
#   plus the small loader/helper libraries that keep both real drivers isolated
# Needs: gcc, python3, the real NVIDIA userspace (libcuda.so.1, libnvidia-ml.so.1) and CUDA headers
# (cuda.h + cudaTypedefs.h: a CUDA toolkit, or CUDA_INCLUDE=/path/to/include).
#
#   lib/build_userspace.sh [OUT] [--prefix INSTALLED_DIR]
# --prefix is where OUT will finally live because that path is baked into the proxies. It defaults to OUT itself.
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)
OUT=build/userspace PREFIX=""
while [ $# -gt 0 ]; do case "$1" in
  --prefix) PREFIX=$2; shift 2 ;;
  *) OUT=$1; shift ;;
esac; done
mkdir -p "$OUT"; OUT=$(cd "$OUT" && pwd); PREFIX=${PREFIX:-$OUT}
die() { echo "build_userspace: $*" >&2; exit 1; }

# Find the real NVIDIA libraries by soname, never one of our proxies. These paths survive same-branch updates.
real_lib() {   # soname -> absolute system path
  local p
  p=$(ldconfig -p | awk -v n="$1" '$1==n && $NF !~ /nvidia-sidecar/ {print $NF; exit}')
  [ -n "$p" ] || for d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib; do [ -e "$d/$1" ] && { p=$d/$1; break; }; done
  [ -n "$p" ] || die "$1 not found (is the NVIDIA driver's userspace installed?)"
  echo "$p"
}
REAL_CUDA=$(real_lib libcuda.so.1)
REAL_NVML=$(real_lib libnvidia-ml.so.1)
# Find a CUDA header set new enough to describe every driver entry point we wrap.
INC=${CUDA_INCLUDE:-}
if [ -z "$INC" ]; then
  for d in ${CUDA_HOME:+$CUDA_HOME/include} $(ls -d /usr/local/cuda*/include 2>/dev/null | sort -V -r) /usr/include; do
    [ -f "$d/cudaTypedefs.h" ] && [ -f "$d/cuda.h" ] && { INC=$d; break; }
  done
fi
[ -n "$INC" ] && [ -f "$INC/cudaTypedefs.h" ] || die "no CUDA headers (cuda.h + cudaTypedefs.h); install a CUDA toolkit or set CUDA_INCLUDE"
echo "build_userspace: real libcuda $REAL_CUDA, real NVML $REAL_NVML, headers $INC -> $OUT (installed at $PREFIX)"

W=$(mktemp -d); trap 'rm -rf "$W"' EXIT
CC=${CC:-gcc}
CFLAGS="-O2 -Wall -fPIC"
mkdir -p "$OUT/cuda" "$OUT/nvml"

# Build the little pieces shared by both proxies.
$CC $CFLAGS -shared -Wno-unused-result -o "$OUT/libnvidia-sidecar-redirect.so" "$HERE/sidecar_redirect.c" -ldl
$CC $CFLAGS -shared -o "$OUT/libnvidia-sidecar-nsfix.so" "$HERE/cuda-proxy/nsfix.c" -ldl
$CC $CFLAGS -shared -o "$OUT/libnvidia-sidecar-nocuda.so" "$HERE/nvml-proxy/nocuda.c" -ldl
# These empty loaders exist only to establish DT_NEEDED order before the real library enters its namespace.
echo 'int nvidia_sidecar_loader;' > "$W/loader.c"
loader() {   # NAME LIBS...; each LIB is a -l name in OUT, with PREFIX as its rpath.
  local name=$1; shift
  local libs=(); for l in "$@"; do libs+=("-l$l"); done
  $CC -shared -fPIC -o "$OUT/$name" "$W/loader.c" -Wl,--no-as-needed -L"$OUT" "${libs[@]}" -Wl,-rpath,"$PREFIX"
}
loader libnvidia-sidecar-cuda-host.so   nvidia-sidecar-nsfix
loader libnvidia-sidecar-cuda-legacy.so nvidia-sidecar-nsfix nvidia-sidecar-redirect
loader libnvidia-sidecar-host-nvml.so   nvidia-sidecar-nocuda
loader libnvidia-sidecar-legacy-nvml.so nvidia-sidecar-nocuda nvidia-sidecar-redirect

# Build the merged NVML view.
python3 "$HERE/nvml-proxy/gen_proxy.py" "$REAL_NVML" "$W/nvml"
$CC $CFLAGS -shared -Wl,-soname,libnvidia-ml.so.1 -o "$OUT/nvml/libnvidia-ml.so.1" \
  -DREAL_NVML="\"$REAL_NVML\"" -DHOST_LOADER="\"$PREFIX/libnvidia-sidecar-host-nvml.so\"" \
  -DLEGACY_LOADER="\"$PREFIX/libnvidia-sidecar-legacy-nvml.so\"" \
  "$HERE/nvml-proxy/nvml_proxy.c" "$W/nvml/names.c" "$W/nvml/stubs.S" -ldl -lpthread

# Build the merged CUDA view.
python3 "$HERE/cuda-proxy/parse_typedefs.py" "$INC/cudaTypedefs.h" > "$W/typedefs.json"
$CC -O2 -o "$W/map_procs" "$HERE/cuda-proxy/map_procs.c" -ldl
python3 -c 'import json,sys; [print(x["name"], x["ver"], x["pt"]) for x in json.load(open(sys.argv[1]))]' "$W/typedefs.json" \
  | "$W/map_procs" "$REAL_CUDA" > "$W/procmap.txt"
nm -D --defined-only "$REAL_CUDA" | awk '$2=="T"{print $3}' > "$W/exports.txt"
python3 "$HERE/cuda-proxy/gen_cuda_proxy.py" "$W/typedefs.json" "$W/procmap.txt" "$W/exports.txt" "$INC" "$W/cuda" | tail -1
$CC $CFLAGS -Wno-unused-variable -shared -Wl,-soname,libcuda.so.1 -o "$OUT/cuda/libcuda.so.1" \
  -DREAL_CUDA="\"$REAL_CUDA\"" -DHOST_LOADER="\"$PREFIX/libnvidia-sidecar-cuda-host.so\"" \
  -DLEGACY_LOADER="\"$PREFIX/libnvidia-sidecar-cuda-legacy.so\"" \
  "$HERE/cuda-proxy/cuda_proxy_core.c" "$HERE/cuda-proxy/integrity.c" "$W/cuda/cp_gen.c" "$W/cuda/cp_stubs.S" -ldl -lpthread
ln -sf libcuda.so.1 "$OUT/cuda/libcuda.so"
ln -sf libnvidia-ml.so.1 "$OUT/nvml/libnvidia-ml.so"
install -m 0755 "$HERE/nvidia-smi-merged" "$OUT/nvidia-smi"
echo "$REAL_CUDA $REAL_NVML" > "$OUT/.real"
echo "build_userspace: done"
