#!/usr/bin/env bash
# Cross-compile PyTorch with Python bindings (aarch64) and smoke-test import under QEMU.
set -euo pipefail
# shellcheck source=_paths.sh
source "$(cd "$(dirname "$0")" && pwd)/_paths.sh"
sve512_paths

BUILD="$PT/build-sve512-cross-py"
TOOLCHAIN="$SVE512_DIR/aarch64-rootfs-toolchain.cmake"
PY_VER="3.13"
PY_INCLUDE_SHIM="$BUILD/python_cross_include"
QEMU_PY="$SVE512_DIR/qemu-python-arm64.sh"
PY_INCLUDE="$ROOTFS/usr/include/python${PY_VER}"
PY_LIB="$ROOTFS/usr/lib/aarch64-linux-gnu/libpython${PY_VER}.so"

if [[ -f "$BUILD/CMakeCache.txt" ]]; then
  if ! grep -q "aarch64-rootfs-toolchain.cmake" "$BUILD/CMakeCache.txt" 2>/dev/null \
     || ! grep -q "BUILD_PYTHON:BOOL=ON" "$BUILD/CMakeCache.txt" 2>/dev/null \
     || ! grep -q "python${PY_VER}" "$BUILD/CMakeCache.txt" 2>/dev/null; then
    echo "=== removing stale cmake cache ($BUILD) ==="
    rm -rf "$BUILD"
  fi
fi

CMAKE_COMMON=(
  -S "$PT" -B "$BUILD"
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN"
  -DAARCH64_DEBIAN_ROOTFS="$ROOTFS"
  -DPYTHON_CROSS_INCLUDE_DIR="$PY_INCLUDE_SHIM"
  -DNATIVE_BUILD_DIR="$PT/build-sleef-native"
  -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE="$PT/build_host_protoc/bin/protoc"
  -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
  -DUSE_PRIORITIZED_TEXT_FOR_LD=OFF
  -DUSE_SYSTEM_LIBS=OFF
  -DUSE_FBGEMM=OFF -DUSE_KINETO=OFF -DUSE_NNPACK=OFF
  -DUSE_XNNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF -DUSE_KLEIDIAI=OFF
  -DBUILD_PYTHON=ON -DBUILD_CAFFE2=OFF
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_NUMPY=OFF
  -DUSE_DISTRIBUTED=OFF -DUSE_MPI=OFF -DUSE_NCCL=OFF
  -DPython_EXECUTABLE="$QEMU_PY"
  -DPython_ROOT_DIR="$ROOTFS/usr"
  -DPython_INCLUDE_DIR="$PY_INCLUDE"
  -DPython_LIBRARY="$PY_LIB"
  -DCMAKE_BUILD_TYPE=Release -Wno-dev -GNinja
)

echo "=== host tools: sleef + protoc ==="
if [[ ! -x "$PT/build-sleef-native/bin/mkdisp" ]]; then
  cmake -S "$PT/third_party/sleef" -B "$PT/build-sleef-native" -GNinja -DSLEEF_BUILD_TESTS=OFF
  ninja -C "$PT/build-sleef-native" mkdisp mkrename mkalias
fi
if [[ ! -x "$PT/build_host_protoc/bin/protoc" ]]; then
  (cd "$PT" && bash scripts/build_host_protoc.sh)
fi

echo "=== python include shim ==="
bash "$SVE512_DIR/setup_python_cross_include.sh" "$ROOTFS" "$PY_INCLUDE_SHIM"

export AARCH64_DEBIAN_ROOTFS="$ROOTFS"

echo "=== cmake configure (aarch64 cross + python, rootfs sysroot) ==="
cmake "${CMAKE_COMMON[@]}"
if ! grep -q "BUILD_PYTHON:BOOL=ON" "$BUILD/CMakeCache.txt"; then
  echo "[ERROR] BUILD_PYTHON is OFF after configure; check Python Development.Module" >&2
  grep -E 'Python_|BUILD_PYTHON' "$BUILD/CMakeCache.txt" | head -20 >&2 || true
  exit 1
fi

echo "=== build torch_python + _C + torch_shm_manager ==="
ninja -C "$BUILD" torch_python _C torch_shm_manager
mkdir -p "$PT/torch/bin"
cp -f "$BUILD/bin/torch_shm_manager" "$PT/torch/bin/torch_shm_manager"

echo "=== QEMU import torch smoke (chroot + qemu-user-static) ==="
# shellcheck source=rootfs_chroot_mount.sh
source "$SVE512_DIR/rootfs_chroot_mount.sh"
rootfs_chroot_setup "$ROOTFS" "$PT:mnt/pytorch"
trap 'rootfs_chroot_trap_teardown "$ROOTFS"' EXIT

sudo chroot "$ROOTFS" env \
  PYTHONPATH="/mnt/pytorch:/mnt/pytorch/build-sve512-cross-py" \
  LD_LIBRARY_PATH="/mnt/pytorch/build-sve512-cross-py/lib" \
  python3 -c "
import torch
print('cpu_capability=', torch._C._get_cpu_capability())
print('sve_max_length=', torch.cpu.get_capabilities().get('sve_max_length'))
a = torch.ones(512)
b = torch.ones(512)
print('add_sum=', float((a + b).sum()))
"

echo "=== DONE: Python cross-build + QEMU import OK ==="
