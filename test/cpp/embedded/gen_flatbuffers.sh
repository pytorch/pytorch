#!/bin/bash
ROOT=$(pwd)
FF_LOCATION="$ROOT/third_party/flatbuffers"
cd "$FF_LOCATION" || exit
mkdir build
cd build || exit
cmake ..
cmake --build . --target flatc -j 16
mkdir -p "$ROOT/build/torch/csrc/jit/serialization"
./flatc --cpp --gen-mutable --scoped-enums \
     -o "$ROOT/test/cpp/embedded/" \
     -c "$ROOT/test/cpp/embedded/schema.fbs"
echo '// @generated' >> "$ROOT/test/cpp/embedded/schema_generated.h"
cd "$ROOT" || exit
exit
