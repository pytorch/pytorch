#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/Users/distiller/project/env}"
mkdir -p "$PYTORCH_FINAL_PACKAGE_DIR"

# Build
export USE_PYTORCH_METAL_EXPORT=1
export USE_COREML_DELEGATE=1
if [[ "$PACKAGE_TYPE" == conda ]]; then
  "${BUILDER_ROOT}/conda/build_pytorch.sh"
else
  export TORCH_PACKAGE_NAME="$(echo $TORCH_PACKAGE_NAME | tr '-' '_')"
  "${BUILDER_ROOT}/wheel/build_wheel.sh"
fi
