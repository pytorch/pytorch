#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="${REPO_DIR:?REPO_DIR is not set}"
BUILD_ENVIRONMENT="${BUILD_ENVIRONMENT:-}"
TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0}"
BUILD_ADDITIONAL_PACKAGES="${BUILD_ADDITIONAL_PACKAGES:-}"
MAX_JOBS="${MAX_JOBS:-0}"

COMMIT=$(cat /tmp/ppu_source_commit 2>/dev/null || git -C "$REPO_DIR" rev-parse --short HEAD)
WHL_STAMP="$REPO_DIR/dist/.built_${COMMIT}_${TORCH_CUDA_ARCH_LIST// /_}"
if [[ -f "$WHL_STAMP" && "${FORCE_REBUILD:-0}" != "1" ]]; then
    echo "[build_wheel] Existing build found (stamp $WHL_STAMP), skipping build"
    ls "$REPO_DIR"/dist/torch-*.whl
    exit 0
fi

source "$(dirname "$0")/sdk_env.sh"
# Since PPU SDK 2.2, NCCL is no longer bundled in the image and must be installed separately
source "$(dirname "$0")/nccl_env.sh"
source "$(dirname "$0")/ccache_env.sh"

cd "$REPO_DIR"
echo "[build_wheel] BUILD_ENVIRONMENT=$BUILD_ENVIRONMENT"
echo "[build_wheel] TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST"

if [[ -n "$BUILD_ADDITIONAL_PACKAGES" ]]; then
    echo "[build_wheel] Warning: README build does not support build-additional-packages: $BUILD_ADDITIONAL_PACKAGES (ignored)" >&2
fi

if [[ -n "${PIP_INDEX_URL:-}" ]]; then
    mkdir -p /root/.config/pip
    printf '[global]\nindex-url = %s\n' "$PIP_INDEX_URL" > /root/.config/pip/pip.conf
fi

apt remove -y cmake >/dev/null 2>&1 || true
pip uninstall -y cmake >/dev/null 2>&1 || true
pip cache purge >/dev/null 2>&1 || true
pip install cmake==3.31.10
ln -sf /usr/local/bin/cmake /usr/bin/cmake
pip install --upgrade pybind11
echo "[build_wheel] cmake=$(cmake --version | head -1)"
pip install -r requirements.txt

if [[ -z "$MAX_JOBS" || "$MAX_JOBS" == "0" ]]; then
    MAX_JOBS=$(nproc)
    MAX_JOBS=$((MAX_JOBS > 2 ? MAX_JOBS - 2 : 1))
fi

MEM_GB=$(awk '/^MemTotal:/ {printf "%d", $2 / 1024 / 1024}' /proc/meminfo 2>/dev/null || echo 0)
MEM_PER_JOB_GB="${PPU_MEM_PER_JOB_GB:-4}"
if [[ "${MEM_GB:-0}" -gt 0 ]]; then
    MEM_CAP=$((MEM_GB / MEM_PER_JOB_GB))
    if [[ "$MEM_CAP" -lt 1 ]]; then
        MEM_CAP=1
    fi
    if [[ "$MAX_JOBS" -gt "$MEM_CAP" ]]; then
        echo "[build_wheel] MAX_JOBS=${MAX_JOBS} exceeds memory cap (${MEM_GB}GB / ${MEM_PER_JOB_GB}GB per job), lowering to ${MEM_CAP}"
        MAX_JOBS="$MEM_CAP"
    fi
fi
export MAX_JOBS
PPU_LINK_JOBS="${PPU_LINK_JOBS:-2}"
export CMAKE_JOB_POOLS="link=${PPU_LINK_JOBS}"
export CMAKE_JOB_POOL_LINK=link

TORCH_VERSION=$(sed -E 's/^([0-9]+\.[0-9]+\.[0-9]+).*/\1/' version.txt)
echo "[build_wheel] TORCH_VERSION=${TORCH_VERSION} MAX_JOBS=${MAX_JOBS} LINK_JOBS=${PPU_LINK_JOBS} MEM=${MEM_GB}GB"

echo "[build_wheel] Cleaning old build dir before build"
rm -rf "$REPO_DIR/build"

BUILD_START=$(date +%s)

env \
    NCCL_INCLUDE_DIR="${NCCL_HOME}/include" \
    NCCL_LIB_DIR="${NCCL_HOME}/lib" \
    PYTORCH_VERSION="$TORCH_VERSION" \
    PYTORCH_BUILD_VERSION="$TORCH_VERSION" \
    PYTORCH_BUILD_NUMBER=0 \
    USE_FLASH_ATTENTION=True \
    USE_MEM_EFF_ATTENTION=True \
    USE_NCCL=True \
    USE_DISTRIBUTED=True \
    USE_SYSTEM_NCCL=1 \
    BUILD_CAFFE2=False \
    BUILD_TEST=True \
    TORCH_CUDA_ARCH_LIST="$TORCH_CUDA_ARCH_LIST" \
    python3 setup.py bdist_wheel

BUILD_END=$(date +%s)
BUILD_DURATION=$((BUILD_END - BUILD_START))
echo "$BUILD_DURATION" > /tmp/ppu_build_duration_sec
echo "[build_wheel] Build took ${BUILD_DURATION}s"

WHLS=("$REPO_DIR/dist"/torch-*.whl)
if [[ ! -e "${WHLS[0]}" ]]; then
    echo "[build_wheel] Error: dist/torch-*.whl not found" >&2
    exit 1
fi
echo "[build_wheel] Artifact: $(ls "$REPO_DIR/dist"/torch-*.whl)"

if [[ "${PPU_CCACHE_ENABLED:-0}" == "1" ]]; then
    ccache -s 2>/dev/null | sed 's/^/[build_wheel] ccache /' || true
fi

echo "${WHLS[*]##*/}" > "$WHL_STAMP"
echo "[build_wheel] Done"
