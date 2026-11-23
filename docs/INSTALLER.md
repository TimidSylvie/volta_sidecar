# Installer and recovery contract

`nvidia-sidecar setup` is the guarded installer. `install.sh` only obtains root and invokes it.

## Preflight

Before persistent mutation, setup verifies:

- Linux x86_64, systemd, the running kernel's headers and every required tool;
- the loaded primary module is NVIDIA's open R580 module;
- the kernel module and NVIDIA userspace versions match exactly;
- the OS is in the tested release matrix, unless the operator explicitly uses `--allow-untested`;
- every selected device is one of the supported Tesla V100/V100S PCI IDs, unless the operator explicitly uses
  `--allow-untested` for another legacy GPU family;
- Secure Boot is disabled, unless explicitly overridden;
- at least one candidate legacy NVIDIA GPU exists;
- CUDA headers are present for `shell` or `seamless` mode;
- an existing installation, if any, owns the same driver version and PCI IDs.

After downloading and internally checking NVIDIA's installer, every selected PCI ID is checked against NVIDIA's
`supported-gpus.json`. IDs marked `kernelopen`, IDs assigned to an older legacy branch, and unknown IDs are rejected.
Every syntactically valid R580 point release is eligible without an override. Releases not listed under `versions`
in `lib/supported_versions.json` are clearly reported as not yet physically tested and then pass through the same
source-patching, build and exact post-install verification gates.

## Owned persistent paths

The installer owns only:

```text
/usr/src/nvidia-<suffix>-<exact-driver-version>
/var/lib/dkms/nvidia-<suffix>/<exact-driver-version>
/etc/modprobe.d/nvidia-<suffix>.conf
/etc/systemd/system/nvidia-sidecar.service
/etc/systemd/system/nvidia-sidecar-persistenced.service
/usr/local/lib/nvidia-sidecar
/usr/local/sbin/nvidia-sidecar
```

System-wide seamless mode additionally owns:

```text
/etc/ld.so.conf.d/000-nvidia-sidecar.conf
/usr/local/bin/nvidia-smi
```

If `/usr/local/bin/nvidia-smi` already exists, it is moved to
`/var/lib/nvidia-sidecar/nvidia-smi.pre-seamless` and restored when seamless mode is disabled. Setup refuses to
overwrite an unrecognized file at any managed path.

State, downloads, the extracted NVIDIA package, the install manifest and transaction scratch space live under
`/var/lib/nvidia-sidecar`. User configuration lives at `/etc/nvidia-sidecar/sidecar.conf`.

## Transactions and interruption recovery

Kernel source patching happens on a temporary copy and is swapped into place only after every expected edit succeeds.
The main install stages the repository and generated configuration, saves the previous managed files, then switches
them as one transaction. A late systemd or manifest failure restores the old tree, configuration, symlink, manifest
and service enablement. The test suite injects such a late failure and checks the restored hashes.

Userspace upgrades build and validate a complete proxy tree before switching it into place. The previous tree remains
at `userspace.old` until a later successful upgrade. A subsequent invocation recovers it automatically if an
interruption occurred between the two renames.

Setup snapshots the effective activation mode before changing it. If the proxy build, seamless-mode transition,
fresh-process verification or final manifest update fails, it restores the previous userspace tree and activation
mode. The failed proxy candidate is retained as `userspace.failed.<pid>` for diagnosis. A service-start failure occurs
before activation changes; it leaves the safely installed core artifacts in place and reports the journal command to
inspect.

Device nodes are created in a temporary directory on devtmpfs and renamed into place only after every `mknod`
succeeds. An existing node directory must carry the management marker or consist solely of the expected legacy
sidecar character nodes.

## Failure posture

The primary open driver is never replaced. A failed DKMS build stops before configuration activation. A start failure
is reported without changing proxy activation. A later proxy or verification failure restores the prior activation,
retains diagnostic artifacts and never claims success.

Uninstall validates the recorded version and every target before removal. If the sidecar modules are busy or DKMS
removal fails, it stops and retains the files required for recovery. Downloads, state and user configuration are kept
deliberately.
