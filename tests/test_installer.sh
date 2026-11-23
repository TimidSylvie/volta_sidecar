#!/usr/bin/env bash
set -euo pipefail

ROOT=$(cd "$(dirname "$0")/.." && pwd)
NVIDIA_SIDECAR_LIB_ONLY=1
source "$ROOT/nvidia-sidecar"

TEST_ROOT=$(mktemp -d)
trap 'rm -rf "$TEST_ROOT"' EXIT

STATE=$TEST_ROOT/state
CONF=$TEST_ROOT/etc/sidecar.conf
INSTALL_ROOT=$TEST_ROOT/opt/nvidia-sidecar
CLI_LINK=$TEST_ROOT/bin/nvidia-sidecar
MODPROBE_CONF=$TEST_ROOT/etc/modprobe.d/nvidia-test.conf
BOOT_SERVICE=$TEST_ROOT/etc/systemd/nvidia-sidecar.service
PERSIST_SERVICE=$TEST_ROOT/etc/systemd/nvidia-sidecar-persistenced.service
INSTALL_MANIFEST=$STATE/install.json
SOURCE_ROOT=$TEST_ROOT/usr/src
MODULES_ROOT=$TEST_ROOT/lib/modules
USERSPACE=$INSTALL_ROOT/userspace
LDCONF=$TEST_ROOT/etc/ld.so.conf.d/000-nvidia-sidecar.conf
SMI_LINK=$TEST_ROOT/bin/nvidia-smi
SMI_BACKUP=$STATE/nvidia-smi.pre-seamless
SUFFIX=test
MOD=nvidia_test
PACKAGE=nvidia-$SUFFIX
mkdir -p "$STATE" "$SOURCE_ROOT/$PACKAGE-580.178.04" "$(dirname "$CLI_LINK")"
printf '580.178.04\n' > "$STATE/version"
printf '1DB6\n' > "$STATE/ids"
printf 'suffix=test\nids=1DB6\n' > "$SOURCE_ROOT/$PACKAGE-580.178.04/.nvidia-sidecar-patched"

FAKEBIN=$TEST_ROOT/fakebin
mkdir -p "$FAKEBIN"
cat > "$FAKEBIN/dkms" <<'EOF'
#!/usr/bin/env bash
if [ "${1:-}" = status ]; then
  echo "nvidia-test/580.178.04, $(uname -r), x86_64: installed"
fi
exit 0
EOF
cat > "$FAKEBIN/systemctl" <<'EOF'
#!/usr/bin/env bash
case "${1:-}" in
  is-enabled) exit 1 ;;
  enable) [ "${FAIL_ENABLE:-0}" = 0 ] ;;
  *) exit 0 ;;
esac
EOF
cat > "$FAKEBIN/depmod" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
cat > "$FAKEBIN/flock" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
cat > "$FAKEBIN/ldconfig" <<'EOF'
#!/usr/bin/env bash
exit 0
EOF
chmod +x "$FAKEBIN/dkms" "$FAKEBIN/systemctl" "$FAKEBIN/depmod" "$FAKEBIN/flock" "$FAKEBIN/ldconfig"
PATH=$FAKEBIN:$PATH
export PATH
need_root() { :; }

# Every V100/V100S ID gets the automatic path on any R580 point release. Other old GPU families still need opt-in.
for id in 1DB1 1DB3 1DB4 1DB5 1DB6 1DB7 1DB8 1DF6; do
  id_allowed_by_setup_policy 580.178.04 "$id" 0
  id_allowed_by_setup_policy 580.196.07 "$id" 0
done
if id_allowed_by_setup_policy 580.178.04 1B80 0; then
  echo "non-V100 GPU bypassed the automatic setup policy" >&2
  exit 1
fi
id_allowed_by_setup_policy 580.178.04 1B80 1

cmd_install
test -f "$INSTALL_ROOT/nvidia-sidecar"
test "$(readlink "$CLI_LINK")" = "$INSTALL_ROOT/nvidia-sidecar"
grep -q '^# nvidia-sidecar managed' "$MODPROBE_CONF"
grep -q '"driver_version": "580.178.04"' "$INSTALL_MANIFEST"

# Running the same install again is an upgrade, not a conflict.
cmd_install

# Never overwrite a system file we cannot prove belongs to us.
cp "$BOOT_SERVICE" "$TEST_ROOT/managed-service"
printf '[Unit]\nDescription=someone else\n' > "$BOOT_SERVICE"
if (cmd_install >/dev/null 2>&1); then
  echo "unowned service file was overwritten" >&2
  exit 1
fi
cp "$TEST_ROOT/managed-service" "$BOOT_SERVICE"

# If activation fails late, put the old tree and system files back exactly.
printf 'keep me\n' > "$INSTALL_ROOT/rollback-sentinel"
before=$(sha256sum "$BOOT_SERVICE" | awk '{print $1}')
before_manifest=$(sha256sum "$INSTALL_MANIFEST" | awk '{print $1}')
if (FAIL_ENABLE=1 cmd_install >/dev/null 2>&1); then
  echo "injected systemctl failure was ignored" >&2
  exit 1
fi
test -f "$INSTALL_ROOT/rollback-sentinel"
after=$(sha256sum "$BOOT_SERVICE" | awk '{print $1}')
test "$before" = "$after"
after_manifest=$(sha256sum "$INSTALL_MANIFEST" | awk '{print $1}')
test "$before_manifest" = "$after_manifest"

# Uninstall removes only what we manage and leaves recovery/download state behind.
cmd_uninstall
test ! -e "$INSTALL_ROOT"
test ! -e "$CLI_LINK"
test ! -e "$BOOT_SERVICE"
test ! -e "$SOURCE_ROOT/$PACKAGE-580.178.04"
test -f "$STATE/version"
cmd_uninstall

# A post-install failure must restore both the previous userspace tree and activation mode.
mkdir -p "$USERSPACE/cuda" "$USERSPACE.old/cuda"
printf 'candidate\n' > "$USERSPACE/cuda/marker"
printf 'previous\n' > "$USERSPACE.old/cuda/marker"
mkdir -p "$(dirname "$LDCONF")"
printf '# nvidia-sidecar: test\n' > "$LDCONF"
ACTIVATION_LOG=$TEST_ROOT/activation.log
cmd_seamless() { printf '%s\n' "$1" >> "$ACTIVATION_LOG"; }
restore_activation_snapshot seamless 1
test "$(cat "$USERSPACE/cuda/marker")" = previous
test "$(cat "$ACTIVATION_LOG")" = on
test -d "$USERSPACE.failed."*

echo "installer transaction tests: passed"
