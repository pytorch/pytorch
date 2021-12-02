#!/bin/bash
ROOT=$(pwd)
FF_LOCATION="$ROOT/third_party/flatbuffers"
cd "$FF_LOCATION" || exit
mkdir build
cd build || exit
py() { command python "$@"; }
cmake ..
cmake --build . --target flatc
mkdir -p "$ROOT/build/torch/csrc/jit/serialization"
./flatc --cpp --gen-mutable --scoped-enums \
   -o "$ROOT/build/torch/csrc/jit/serialization" \
   -c "$ROOT/torch/csrc/jit/serialization/mobile_bytecode.fbs"
cd "$ROOT" || exit
exit
