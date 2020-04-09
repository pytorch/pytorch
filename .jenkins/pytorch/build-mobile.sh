#!/usr/bin/env bash
# DO NOT ADD 'set -x' not to reveal CircleCI secret context environment variables
set -eu -o pipefail

# This script uses linux host toolchain + mobile build options in order to
# build & test mobile libtorch without having to setup Android/iOS
# toolchain/simulator.

COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

# Install torch & torchvision - used to download & trace test model.
retry pip install torch torchvision --progress-bar off

# Run end-to-end process of building mobile library, linking into the predictor
# binary, and running forward pass with a real model.
if [[ "$BUILD_ENVIRONMENT" == *-mobile-custom-build-static* ]]; then
  TEST_CUSTOM_BUILD_STATIC=1 test/mobile/custom_build/build.sh
elif [[ "$BUILD_ENVIRONMENT" == *-mobile-custom-build-dynamic* ]]; then
  export LLVM_DIR="$(llvm-config-5.0 --prefix)"
  echo "LLVM_DIR: ${LLVM_DIR}"
  TEST_CUSTOM_BUILD_DYNAMIC=1 test/mobile/custom_build/build.sh
else
  TEST_DEFAULT_BUILD=1 test/mobile/custom_build/build.sh
fi
