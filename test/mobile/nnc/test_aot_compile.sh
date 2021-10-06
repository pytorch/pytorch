#!/bin/bash

set -e -o pipefail

TORCH_INSTALL_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/torch
TORCH_BIN_DIR="$TORCH_INSTALL_DIR"/bin
CURRENT_DIR="$( cd "$( dirname "$0" )" >/dev/null 2>&1 && pwd )"

MODEL=aot_test_model.pt
COMPILED_MODEL=aot_test_model.compiled.pt
COMPILED_CODE=aot_test_model.compiled.ll

TMP_DIR=$(mktemp -d -t build_XXX)
trap 'rm -rf "$TMP_DIR"' EXIT

test_aot_model_compiler() {
  python "$CURRENT_DIR"/aot_test_model.py
  "$TORCH_BIN_DIR"/test_aot_model_compiler --model "$MODEL" --model_name=aot_test_model --model_version=v1 --input_dims="2,2,2"
  if [ ! -f "$COMPILED_MODEL" ] || [ ! -f "$COMPILED_CODE" ]; then
    echo "AOT model compiler failed to generate $COMPILED_MODEL and $COMPILED_CODE"
    exit 1
  fi
}

pushd "$TMP_DIR"
test_aot_model_compiler
popd
