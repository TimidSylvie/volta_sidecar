#!/usr/bin/env bash
# Small entry point for the guarded installer. Run it from a release archive or a git checkout.
set -euo pipefail
HERE=$(cd "$(dirname "$(readlink -f "$0")")" && pwd)

if [ "${1:-}" = "--help" ] || [ "${1:-}" = "-h" ]; then
  exec "$HERE/nvidia-sidecar" setup --help
fi

if [ "$(id -u)" != 0 ]; then
  command -v sudo >/dev/null || { echo "install.sh: sudo is required" >&2; exit 1; }
  exec sudo -- "$HERE/install.sh" "$@"
fi

exec "$HERE/nvidia-sidecar" setup "$@"
