#!/usr/bin/env bash
set -euo pipefail

SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
ENVSETUP="$SDK_INSTALL_DIR/PPU_SDK/envsetup.sh"
SDK_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_SDK="$SDK_ENV_SCRIPT_DIR/install_sdk.sh"

if [[ -f "$ENVSETUP" ]]; then
    echo "[sdk_env] SDK bundled in image, skipping install: $SDK_INSTALL_DIR/PPU_SDK"
else
    echo "[sdk_env] SDK not found: $ENVSETUP"
    if [[ ! -f "$INSTALL_SDK" ]]; then
        echo "[sdk_env] Install script not found: $INSTALL_SDK" >&2
        exit 1
    fi
    if [[ -z "${SDK_URL:-}" ]]; then
        echo "[sdk_env] SDK_URL not set, using default URL in install_sdk.sh"
    fi
    echo "[sdk_env] Running installer: $INSTALL_SDK"
    SDK_INSTALL_DIR="$SDK_INSTALL_DIR" SDK_URL="${SDK_URL:-}" bash "$INSTALL_SDK"
    if [[ ! -f "$ENVSETUP" ]]; then
        echo "[sdk_env] Still not found after install: $ENVSETUP" >&2
        exit 1
    fi
fi

set +u
source "$ENVSETUP"
set -u

if ! command -v nvcc >/dev/null 2>&1; then
    echo "[sdk_env] SDK env broken: nvcc unavailable" >&2
    exit 1
fi

echo "[sdk_env] SDK env ready: $(nvcc --version | tail -1)"
