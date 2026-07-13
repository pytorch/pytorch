#!/usr/bin/env bash
# Wrapper so CMake can run arm64 Python on x86 via qemu-user.
# shellcheck source=_paths.sh
source "$(cd "$(dirname "$0")" && pwd)/_paths.sh"
sve512_paths
exec qemu-aarch64 -cpu max -L "$ROOTFS" "$ROOTFS/usr/bin/python3.13" "$@"
