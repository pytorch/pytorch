#!/usr/bin/env bash
set -euo pipefail

SDK_INSTALL_DIR="${SDK_INSTALL_DIR:-/usr/local}"
ENVSETUP="$SDK_INSTALL_DIR/pccl/envsetup.sh"
NCCL_ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
INSTALL_NCCL="$NCCL_ENV_SCRIPT_DIR/install_nccl.sh"

if [[ -f "$ENVSETUP" ]]; then
    echo "[nccl_env] NCCL bundled in image, skipping install: $SDK_INSTALL_DIR/pccl"
else
    echo "[nccl_env] NCCL not found: $ENVSETUP"
    if [[ ! -f "$INSTALL_NCCL" ]]; then
        echo "[nccl_env] Install script not found: $INSTALL_NCCL" >&2
        exit 1
    fi
    if [[ -z "${NCCL_URL:-}" ]]; then
        echo "[nccl_env] NCCL_URL not set, using default URL in install_nccl.sh"
    fi
    echo "[nccl_env] Running installer: $INSTALL_NCCL"
    SDK_INSTALL_DIR="$SDK_INSTALL_DIR" NCCL_URL="${NCCL_URL:-}" bash "$INSTALL_NCCL"
    if [[ ! -f "$ENVSETUP" ]]; then
        echo "[nccl_env] Still not found after install: $ENVSETUP" >&2
        exit 1
    fi
fi

set +u
source "$ENVSETUP"
set -u

if [[ -z "${NCCL_HOME:-}" ]]; then
    echo "[nccl_env] NCCL env broken: NCCL_HOME not set" >&2
    exit 1
fi

export NCCL_HOME
export NCCL_INCLUDE_DIR="${NCCL_HOME}/include"
export NCCL_LIB_DIR="${NCCL_HOME}/lib"

echo "[nccl_env] NCCL env ready: NCCL_HOME=$NCCL_HOME"
