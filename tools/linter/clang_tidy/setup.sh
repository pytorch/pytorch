#!/usr/bin/env bash

if [ ! -d build ]; then
  git submodule update --init --recursive

  export USE_NCCL=0
  export USE_DEPLOY=1
  # We really only need compile_commands.json, so no need to build!
  time python3 setup.py --cmake-only build

  # Generate ATen files.
  time python3 -m tools.codegen.gen \
    -s aten/src/ATen \
    -d build/aten/src/ATen

  # Generate PyTorch files.
  time python3 tools/setup_helpers/generate_code.py            \
    --declarations-path build/aten/src/ATen/Declarations.yaml \
    --native-functions-path aten/src/ATen/native/native_functions.yaml \
    --nn-path aten/src
fi
