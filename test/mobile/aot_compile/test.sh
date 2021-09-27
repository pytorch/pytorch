#!/bin/bash

TORCH_INSTALL_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/torch
TORCH_BIN_DIR="$TORCH_INSTALL_DIR"/bin

BUILD_DIR="build"
BUILD_BIN_DIR="$BUILD_DIR"/bin
CURRENT_DIR="$(dirname "${BASH_SOURCE[0]}")"

COMPILED_MODEL=aot_test_model.compiled.pt
COMPILED_CODE=aot_test_model.compiled.ll

test_aot_model_compiler() {
  ls "$TORCH_BIN_DIR"/aot_model_compiler
  ls "$BUILD_BIN_DIR"/aot_model_compiler

  python "$CURRENT_DIR"/aot_test_model.py
  "$TORCH_BIN_DIR"/aot_model_compiler --model aot_test_model.pt --model_name=aot_test_model --model_version=v1 --input_dims="2,2,2"

  if [ ! -f "$COMPILED_MODEL" ] || [ ! -f "$COMPILED_CODE" ]; then
      echo "AOT model compiler failed"
      exit 1
  fi
}

echo "Running AOT compiler test.."
test_aot_model_compiler
