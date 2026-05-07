#!/usr/bin/env bash
# Repair a PyTorch wheel: bundle libgomp, set RPATHs, retag platform.
# Uses the `wheel` Python package for unpack/pack/tags (not zip).
#
# Usage: repair_wheel.sh <input_dir> <output_dir>
#
# Environment variables:
#   DESIRED_CUDA     - cpu, cu126, cu130, etc.
#   GPU_ARCH_TYPE    - cpu, cuda, cuda-aarch64, rocm, xpu
#   GPU_ARCH_VERSION - 12.6, 13.0, 13.2, etc. (empty for CPU)
#   USE_CUDA         - "0" or "1"

set -eux

INPUT_DIR="$1"
OUTPUT_DIR="$2"

PATCHELF=/usr/local/bin/patchelf
ARCH=$(uname -m)
PLATFORM="manylinux_2_28_${ARCH}"

# Locate libgomp
OS_NAME=$(awk -F= '/^NAME/{print $2}' /etc/os-release)
if [[ "$OS_NAME" == *"Ubuntu"* ]]; then
    LIBGOMP_PATH="/usr/lib/${ARCH}-linux-gnu/libgomp.so.1"
else
    LIBGOMP_PATH="/usr/lib64/libgomp.so.1"
fi

# Determine RPATH.
# CUDA 13.x uses unified nvidia/cu13/lib; CUDA 12.x uses individual package paths.
C_SO_RPATH='$ORIGIN:$ORIGIN/lib'
LIB_SO_RPATH='$ORIGIN'
FORCE_RPATH=""
if [[ "${USE_CUDA:-0}" == "1" ]]; then
    CUDA_RPATHS='$ORIGIN/../../nvidia/cudnn/lib:$ORIGIN/../../nvidia/nvshmem/lib:$ORIGIN/../../nvidia/nccl/lib:$ORIGIN/../../nvidia/cusparselt/lib'
    CUDA_MAJOR="${GPU_ARCH_VERSION%%.*}"
    if [[ "$CUDA_MAJOR" == "13" ]]; then
        CUDA_RPATHS="${CUDA_RPATHS}:\$ORIGIN/../../nvidia/cu13/lib"
    else
        CUDA_RPATHS="${CUDA_RPATHS}:\$ORIGIN/../../nvidia/cublas/lib:\$ORIGIN/../../nvidia/cuda_cupti/lib:\$ORIGIN/../../nvidia/cuda_nvrtc/lib:\$ORIGIN/../../nvidia/cuda_runtime/lib:\$ORIGIN/../../nvidia/cufft/lib:\$ORIGIN/../../nvidia/curand/lib:\$ORIGIN/../../nvidia/cusolver/lib:\$ORIGIN/../../nvidia/cusparse/lib:\$ORIGIN/../../cusparselt/lib:\$ORIGIN/../../nvidia/nvtx/lib:\$ORIGIN/../../nvidia/cufile/lib"
    fi
    C_SO_RPATH="${CUDA_RPATHS}:\$ORIGIN:\$ORIGIN/lib"
    LIB_SO_RPATH="${CUDA_RPATHS}:\$ORIGIN"
    FORCE_RPATH="--force-rpath"
fi

# Build list of extra libraries to bundle for aarch64.
# CPU builds link against OpenBLAS/libgfortran; CUDA builds link against NVPL.
# Both use ARM Compute Library (ACL) when available.
AARCH64_DEPS=()
if [[ "$ARCH" == "aarch64" ]]; then
    [[ -f /usr/lib64/libgfortran.so.5 ]] && AARCH64_DEPS+=("/usr/lib64/libgfortran.so.5")
    if [[ -d /acl/build ]]; then
        for lib in libarm_compute.so libarm_compute_graph.so; do
            [[ -f "/acl/build/$lib" ]] && AARCH64_DEPS+=("/acl/build/$lib")
        done
    fi
    if [[ "${USE_CUDA:-0}" == "1" ]]; then
        for lib in libnvpl_blas_lp64_gomp.so.0 libnvpl_lapack_lp64_gomp.so.0 \
                   libnvpl_blas_core.so.0 libnvpl_lapack_core.so.0; do
            [[ -f "/usr/local/lib/$lib" ]] && AARCH64_DEPS+=("/usr/local/lib/$lib")
        done
    else
        [[ -f /opt/OpenBLAS/lib/libopenblas.so.0 ]] && AARCH64_DEPS+=("/opt/OpenBLAS/lib/libopenblas.so.0")
    fi
fi

mkdir -p "$OUTPUT_DIR"
for whl in "$INPUT_DIR"/*.whl; do
    WORK=$(mktemp -d)
    wheel unpack "$whl" -d "$WORK"
    UNPACKED=$(ls -d "$WORK"/torch-*)

    # Bundle libgomp
    cp "$LIBGOMP_PATH" "$UNPACKED/torch/lib/libgomp.so.1"
    find "$UNPACKED/torch" -maxdepth 1 -name '*.so*' -exec \
        $PATCHELF --replace-needed libgomp.so.1 libgomp.so.1 {} \;

    # Bundle aarch64 BLAS/LAPACK/ACL dependencies
    for dep in "${AARCH64_DEPS[@]}"; do
        cp -L "$dep" "$UNPACKED/torch/lib/$(basename "$dep")"
    done

    # Set RPATH on top-level .so files (_C.so etc.)
    find "$UNPACKED/torch" -maxdepth 1 -type f -name '*.so*' | while read sofile; do
        $PATCHELF --set-rpath "$C_SO_RPATH" $FORCE_RPATH "$sofile"
    done

    # Set RPATH on lib/ .so files
    find "$UNPACKED/torch/lib" -maxdepth 1 -type f -name '*.so*' | while read sofile; do
        $PATCHELF --set-rpath "$LIB_SO_RPATH" $FORCE_RPATH "$sofile"
    done

    # Repack (wheel pack regenerates RECORD automatically)
    wheel pack "$UNPACKED" -d "$OUTPUT_DIR"
    rm -rf "$WORK"
done

# Retag wheels with the manylinux platform tag
for whl in "$OUTPUT_DIR"/*.whl; do
    wheel tags --platform-tag "$PLATFORM" --remove "$whl"
done

echo "Repaired $(ls "$OUTPUT_DIR"/*.whl | wc -l) wheel(s) in $OUTPUT_DIR"
