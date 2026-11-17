#!/usr/bin/env bash
set -euo pipefail

DEFAULT_SDK_URL="https://pkg.flytiger-eco.com/artifactory/generic-local/CUDA_SDK/v2.2.0/PPU_SDK_cuda-13.0.0-ubuntu2404-2.2.0-663c22.tar.gz"
SDK_URL="${SDK_URL:-$DEFAULT_SDK_URL}"
SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
ENVSETUP="$SDK_INSTALL_DIR/PPU_SDK/envsetup.sh"
if [[ -f "$ENVSETUP" ]]; then
    echo "[install_sdk] SDK already installed, skipping download: $SDK_INSTALL_DIR/PPU_SDK"
else
    TARBALL="/tmp/$(basename "$SDK_URL")"
    echo "[install_sdk] Downloading: $SDK_URL"
    echo "[install_sdk] Saving to: $TARBALL"
    if command -v wget >/dev/null 2>&1; then
        DOWNLOADER=wget
        echo "[install_sdk] Downloader: $(wget --version 2>&1 | head -1)"
    elif command -v curl >/dev/null 2>&1; then
        DOWNLOADER=curl
        echo "[install_sdk] Downloader: $(curl --version 2>&1 | head -1)"
    else
        echo "[install_sdk] Neither wget nor curl in image, cannot download SDK" >&2
        exit 1
    fi
    echo "[install_sdk] Disk free (SDK is GBs; a small tmpfs /tmp may fill up):"
    df -h /tmp "$SDK_INSTALL_DIR" 2>&1 || true

    if [[ "$DOWNLOADER" == wget ]]; then
        wget -nv -O "$TARBALL" "$SDK_URL" 2>&1
    else
        curl -fSL --retry 3 -o "$TARBALL" "$SDK_URL" 2>&1
    fi
    echo "[install_sdk] Downloaded: $(ls -lh "$TARBALL" | awk '{print $5}')"

    echo "[install_sdk] Extracting to: $SDK_INSTALL_DIR"
    tar -zxf "$TARBALL" -C "$SDK_INSTALL_DIR"
    rm -f "$TARBALL"
    [[ -f "$ENVSETUP" ]] || {
        echo "[install_sdk] $ENVSETUP not found after extraction" >&2; exit 1; }
fi

if [[ ! -e /usr/local/cuda ]]; then
    ln -s "$SDK_INSTALL_DIR/PPU_SDK/CUDA_SDK" /usr/local/cuda
    echo "[install_sdk] Created symlink: /usr/local/cuda -> PPU_SDK/CUDA_SDK"
else
    echo "[install_sdk] /usr/local/cuda exists, skipping symlink"
fi

set +u
source "$ENVSETUP"
set -u
echo "[install_sdk] envsetup.sh sourced"
command -v nvcc && nvcc --version | tail -2 || echo "[install_sdk] Warning: nvcc not in PATH" >&2

echo "[install_sdk] Done: $SDK_INSTALL_DIR/PPU_SDK"
