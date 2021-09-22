#!/bin/bash


BUILD_DIR="build"
BUILD_BIN_DIR="$BUILD_DIR"/bin
CURRENT_DIR="$(dirname "${BASH_SOURCE[0]}")"

COMPILED_MODEL=aot_test_model.compiled.pt
COMPILED_CODE=aot_test_model.compiled.ll

test_aot_model_compiler() {
  true || rm "aot_test_model.compiled.*" #Remove for CI

  python "$CURRENT_DIR"/aot_test_model.py
  "$BUILD_BIN_DIR"/aot_model_compiler --model aot_test_model.pt --model_name=aot_test_model --model_version=v1 --input_dims="2,2,2"

  if [ -f "$COMPILED_MODEL" ] && [ -f "$COMPILED_CODE" ]; then
      echo "Works"
  else
      echo "Fails"
  fi
}

echo "Running AOT compiler test.."
test_aot_model_compiler
