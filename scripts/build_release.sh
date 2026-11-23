#!/usr/bin/env bash
# Build a reproducible source archive from the committed tree.
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
VERSION=$(awk -F= '$1=="SIDECAR_RELEASE"{print $2; exit}' nvidia-sidecar)
[ -n "$VERSION" ] || { echo "could not read SIDECAR_RELEASE" >&2; exit 1; }
git diff --quiet && git diff --cached --quiet ||
  { echo "release archives require a clean committed tree" >&2; exit 1; }

OUT=${1:-"$ROOT/dist"}
mkdir -p "$OUT"
BASE=nvidia-sidecar-$VERSION
TMP=$(mktemp -d)
trap 'rm -rf "$TMP"' EXIT
git archive --format=tar --prefix="$BASE/" HEAD > "$TMP/$BASE.tar"
gzip -n -9 "$TMP/$BASE.tar"
mv "$TMP/$BASE.tar.gz" "$OUT/$BASE.tar.gz"
if command -v sha256sum >/dev/null; then
  (cd "$OUT" && sha256sum "$BASE.tar.gz" > "$BASE.tar.gz.sha256")
else
  (cd "$OUT" && shasum -a 256 "$BASE.tar.gz" > "$BASE.tar.gz.sha256")
fi
echo "$OUT/$BASE.tar.gz"
echo "$OUT/$BASE.tar.gz.sha256"
