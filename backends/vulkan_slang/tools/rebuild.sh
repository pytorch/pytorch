#!/bin/bash
# Fast incremental rebuild using ninja directly.
# Only recompiles changed .cpp files and re-links the .so.
# Usage: bash tools/rebuild.sh
#
# First time or after adding new .cpp files: automatically runs full build.
# Otherwise: incremental ninja build (~3s no-change, ~20s single-file).

set -e

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
BUILD_DIR="$ROOT/build/temp.linux-x86_64-cpython-313"
LIB_DIR="$ROOT/build/lib.linux-x86_64-cpython-313/torch_vulkan"
TARGET_DIR="$ROOT/python/torch_vulkan"

full_rebuild() {
    echo "Running full rebuild (python setup.py build_ext --inplace)..."
    cd "$ROOT"
    python setup.py build_ext --inplace
    exit $?
}

# No build.ninja? Full rebuild.
if [ ! -f "$BUILD_DIR/build.ninja" ]; then
    full_rebuild
fi

# Check if new .cpp files were added (not in ninja graph)
CPP_COUNT=$(find "$ROOT/csrc" -name "*.cpp" | wc -l)
OBJ_COUNT=$(find "$BUILD_DIR/csrc" -name "*.o" 2>/dev/null | wc -l)
if [ "$CPP_COUNT" -ne "$OBJ_COUNT" ]; then
    echo "Source count changed ($CPP_COUNT sources vs $OBJ_COUNT objects)."
    rm -rf "$BUILD_DIR"
    full_rebuild
fi

# Run ninja (incremental — only recompiles changed files)
cd "$BUILD_DIR"
ninja -j$(nproc)

# Re-link the shared library
SO_NAME="_C.cpython-313-x86_64-linux-gnu.so"
TORCH_LIB="$(python3 -c 'import torch; print(torch.__path__[0])')/lib"

mkdir -p "$LIB_DIR"
x86_64-linux-gnu-g++ -shared -Wl,-O1 -Wl,-Bsymbolic-functions \
    -Wl,-Bsymbolic-functions -Wl,-z,relro -g -fwrapv -O2 \
    "$BUILD_DIR"/csrc/autocast/*.o \
    "$BUILD_DIR"/csrc/backend/*.o \
    "$BUILD_DIR"/csrc/ops/*.o \
    "$BUILD_DIR"/csrc/vulkan/*.o \
    "$BUILD_DIR"/csrc/init.o \
    -L"$TORCH_LIB" -L/usr/lib/x86_64-linux-gnu \
    -Wl,-rpath,"$TORCH_LIB" \
    -lvulkan -lc10 -ltorch -ltorch_cpu -ltorch_python \
    -o "$LIB_DIR/$SO_NAME"

# Copy to python package dir
cp "$LIB_DIR/$SO_NAME" "$TARGET_DIR/$SO_NAME"

echo "Done. Only changed files were recompiled."
