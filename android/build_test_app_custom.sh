#!/bin/bash
###############################################################################
# This script tests the custom selective build flow for PyTorch Android, which
# optimizes library size by only including ops used by a specific model.
###############################################################################

set -eux

PYTORCH_DIR="$(cd $(dirname $0)/..; pwd -P)"
PYTORCH_ANDROID_DIR="${PYTORCH_DIR}/android"
BUILD_ROOT="${PYTORCH_DIR}/build_pytorch_android_custom"

source "${PYTORCH_ANDROID_DIR}/common.sh"

prepare_model_and_dump_root_ops() {
  cd "${BUILD_ROOT}"
  MODEL="${BUILD_ROOT}/MobileNetV2.pt"
  ROOT_OPS="${BUILD_ROOT}/MobileNetV2.yaml"
  python "${PYTORCH_ANDROID_DIR}/test_app/make_assets_custom.py"
  cp "${MODEL}" "${PYTORCH_ANDROID_DIR}/test_app/app/src/main/assets/mobilenet2.pt"
}

# Start building
mkdir -p "${BUILD_ROOT}"
check_android_sdk
check_gradle
parse_abis_list "$@"
prepare_model_and_dump_root_ops
SELECTED_OP_LIST="${ROOT_OPS}" build_android

# TODO: change this to build test_app instead
$GRADLE_PATH -PABI_FILTERS=$ABIS_LIST -p $PYTORCH_ANDROID_DIR clean assembleRelease
