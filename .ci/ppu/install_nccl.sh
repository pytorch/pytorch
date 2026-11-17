#!/usr/bin/env bash
set -euo pipefail

DEFAULT_NCCL_URL="https://pkg.flytiger-eco.com/artifactory/generic-local/NCCL/v2.2.0/nccl-2.27.3-v13_ubuntu2404_sdk-v2.2.0.tar.gz"
NCCL_URL="${NCCL_URL:-$DEFAULT_NCCL_URL}"
SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
ENVSETUP="$SDK_INSTALL_DIR/pccl/envsetup.sh"
if [[ -f "$ENVSETUP" ]]; then
    echo "[install_nccl] NCCL already installed, skipping download: $SDK_INSTALL_DIR/pccl"
else
    TARBALL="/tmp/$(basename "$NCCL_URL")"
    echo "[install_nccl] Downloading: $NCCL_URL"
    echo "[install_nccl] Saving to: $TARBALL"
    if command -v wget >/dev/null 2>&1; then
        DOWNLOADER=wget
        echo "[install_nccl] Downloader: $(wget --version 2>&1 | head -1)"
    elif command -v curl >/dev/null 2>&1; then
        DOWNLOADER=curl
        echo "[install_nccl] Downloader: $(curl --version 2>&1 | head -1)"
    else
        echo "[install_nccl] Neither wget nor curl in image, cannot download NCCL" >&2
        exit 1
    fi
    echo "[install_nccl] Disk free:"
    df -h /tmp "$SDK_INSTALL_DIR" 2>&1 || true

    if [[ "$DOWNLOADER" == wget ]]; then
        wget -nv -O "$TARBALL" "$NCCL_URL" 2>&1
    else
        curl -fSL --retry 3 -o "$TARBALL" "$NCCL_URL" 2>&1
    fi
    echo "[install_nccl] Downloaded: $(ls -lh "$TARBALL" | awk '{print $5}')"

    echo "[install_nccl] Extracting to: $SDK_INSTALL_DIR"
    tar -zxf "$TARBALL" -C "$SDK_INSTALL_DIR"
    rm -f "$TARBALL"
    [[ -f "$ENVSETUP" ]] || {
        echo "[install_nccl] $ENVSETUP not found after extraction" >&2; exit 1; }
fi

set +u
source "$ENVSETUP"
set -u
echo "[install_nccl] envsetup.sh sourced"

echo "[install_nccl] Done: $SDK_INSTALL_DIR/pccl"
