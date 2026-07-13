#!/usr/bin/env bash
# Bind-mount helpers for arm64-rootfs chroot scripts.
# Source this file; do not execute directly.

rootfs_chroot_bind_mount() {
  local rootfs="$1" rel="$2" src="${3:-/$2}"
  sudo mkdir -p "$rootfs/$rel"
  if mountpoint -q "$rootfs/$rel"; then
    return 0
  fi
  sudo mount --bind "$src" "$rootfs/$rel"
}

rootfs_chroot_setup() {
  local rootfs="$1"
  shift
  sudo mkdir -p "$rootfs/proc" "$rootfs/sys" "$rootfs/dev/pts"
  for m in proc sys dev dev/pts; do
    rootfs_chroot_bind_mount "$rootfs" "$m"
  done
  for src in "$@"; do
    local rel="${src%%:*}" dest="${src#*:}"
    rootfs_chroot_bind_mount "$rootfs" "$dest" "$rel"
  done
}

rootfs_chroot_teardown() {
  local rootfs="$1"
  # Unmount deepest paths first; loop to peel stacked bind mounts from older runs.
  for m in dev/pts dev sys proc mnt/host/protoc mnt/pytorch; do
    while mountpoint -q "$rootfs/$m" 2>/dev/null; do
      sudo umount "$rootfs/$m" || break
    done
  done
}

rootfs_chroot_trap_teardown() {
  rootfs_chroot_teardown "$1"
}
