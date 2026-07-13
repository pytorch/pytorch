#!/usr/bin/env bash
# Cross-compile PyTorch SVE512 (x86 host -> aarch64) and run smoke tests under QEMU.
set -euo pipefail
# shellcheck source=_paths.sh
source "$(cd "$(dirname "$0")" && pwd)/_paths.sh"
sve512_paths

BUILD="$PT/build-sve512-cross"
TOOLCHAIN="$SVE512_DIR/aarch64-toolchain.cmake"

CMAKE_COMMON=(
  -S "$PT" -B "$BUILD"
  -DCMAKE_TOOLCHAIN_FILE="$TOOLCHAIN"
  -DNATIVE_BUILD_DIR="$PT/build-sleef-native"
  -DCAFFE2_CUSTOM_PROTOC_EXECUTABLE="$PT/build_host_protoc/bin/protoc"
  -Dprotobuf_BUILD_PROTOC_BINARIES=OFF
  -DUSE_PRIORITIZED_TEXT_FOR_LD=OFF
  -DUSE_SYSTEM_LIBS=OFF
  -DUSE_FBGEMM=OFF -DUSE_KINETO=OFF -DUSE_NNPACK=OFF
  -DUSE_XNNPACK=OFF -DUSE_PYTORCH_QNNPACK=OFF -DUSE_KLEIDIAI=OFF
  -DBUILD_PYTHON=OFF -DBUILD_CAFFE2=OFF
  -DUSE_CUDA=OFF -DUSE_ROCM=OFF -DUSE_NUMPY=OFF
  -DUSE_DISTRIBUTED=OFF -DUSE_MPI=OFF -DUSE_NCCL=OFF
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

echo "=== cmake configure (aarch64 cross) ==="
cmake "${CMAKE_COMMON[@]}"

echo "=== build libtorch_cpu + vec tests ==="
ninja -C "$BUILD" libtorch_cpu.so vec_test_all_types_SVE512

echo "=== QEMU vec tests (360 tests) ==="
qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu \
  "$BUILD/bin/vec_test_all_types_SVE512"

echo "=== link + run ATen smoke test ==="
aarch64-linux-gnu-g++ --sysroot=/usr/aarch64-linux-gnu -std=c++17 -O2 \
  -I"$BUILD/aten/src" -I"$PT/aten/src" -I"$BUILD" -I"$PT" -I"$PT/c10/.." -I"$BUILD/include" \
  -I"$PT/third_party/cpuinfo/include" -isystem "$PT/third_party/eigen" \
  -DHAVE_SVE_CPU_DEFINITION -DAT_BUILD_ARM_VEC256_WITH_SLEEF \
  "$SVE512_DIR/sve512_smoke_test.cpp" \
  -L"$BUILD/lib" -L"$BUILD/sleef/lib" -Wl,-rpath,"$BUILD/lib" \
  -ltorch_cpu -lc10 -lsleef -lonnx -lonnx_proto -lprotobuf \
  -lCaffe2_perfkernels_sve -lcpuinfo -lgomp -lpthread -ldl -lrt -lm \
  -o "$BUILD/bin/sve512_smoke_test"

qemu-aarch64 -cpu max -L /usr/aarch64-linux-gnu "$BUILD/bin/sve512_smoke_test"

echo "=== DONE: SVE512 cross-build + QEMU validation OK ==="
