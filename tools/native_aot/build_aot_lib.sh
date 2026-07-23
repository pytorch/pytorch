#!/bin/bash
# Driver for the libtorch_native_aot_cuda.so CMake build.
#
# Prereqs (both run from the repo root with the DSL wheel installed):
#   python tools/native_aot/export.py         # kernels -> build/native_aot/<op>/
#   python tools/native_aot/gen_aot_lib.py  # embed_<op>_<key>.cpp
#
# Usage: tools/native_aot/build_aot_lib.sh [artifacts_dir] [--install]
#   --install copies the built lib into torch/lib so `import torch`
#   auto-loads it (see torch/_native/__init__.py).
set -e
REPO="$(cd "$(dirname "$0")/../.." && pwd)"
ART=${1:-$REPO/build/native_aot}
[ "$ART" = "--install" ] && { ART=$REPO/build/native_aot; INSTALL=1; }
[ "$2" = "--install" ] && INSTALL=1

PY=${PYTHON:-$REPO/.venv/bin/python}
# nvidia_cutlass_dsl is a namespace package (__file__ is None); use __path__.
DSL_ROOT=$("$PY" -c "import nvidia_cutlass_dsl; print(list(nvidia_cutlass_dsl.__path__)[0])")

cmake -S "$REPO/tools/native_aot" -B "$ART/cmake" \
  -DTORCH_ROOT="$REPO/torch" \
  -DARTIFACTS_DIR="$ART" \
  -DDSL_RUNTIME_STATIC="$DSL_ROOT/lib/libcuda_dialect_runtime_static.a" \
  -DCMAKE_BUILD_TYPE=Release
cmake --build "$ART/cmake" -j

if [ -n "$INSTALL" ]; then
  cp "$ART/libtorch_native_aot_cuda.so" "$REPO/torch/lib/"
  echo "installed to $REPO/torch/lib/libtorch_native_aot_cuda.so"
fi
echo "built $ART/libtorch_native_aot_cuda.so"
