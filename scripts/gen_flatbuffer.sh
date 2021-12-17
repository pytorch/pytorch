#!/bin/bash
echo "Running gen_flatbuffers.sh OSTYPE is $OSTYPE"
ROOT=$(pwd)
FF_LOCATION="$ROOT/third_party/flatbuffers"
cd "$FF_LOCATION" || exit
mkdir build
cd build || exit
py() { command python "$@"; }
if [[ "$OSTYPE" == "darwin"* ]]; then
   echo "Current arch is $CMAKE_OSX_ARCHITECTURES"
   printenv
   echo "====================="
   cmake -G "Xcode" -DCMAKE_BUILD_TYPE=Release -DCMAKE_OSX_ARCHITECTURES=x86_64 ..
   cmake --build . --target flatc
   FLATC=./Debug/flatc
else
   cmake -DCMAKE_BUILD_TYPE=Release ..
   cmake --build . --target flatc
   FLATC=./flatc
fi
mkdir -p "$ROOT/build/torch/csrc/jit/serialization"
$FLATC --cpp --gen-mutable --scoped-enums \
   -o "$ROOT/build/torch/csrc/jit/serialization" \
   -c "$ROOT/torch/csrc/jit/serialization/mobile_bytecode.fbs"
cd "$ROOT" || exit
exit
