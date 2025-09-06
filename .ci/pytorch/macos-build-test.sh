#!/bin/bash

if [ -z "${BUILD_ENVIRONMENT}" ] || [[ "${BUILD_ENVIRONMENT}" == *-build* ]]; then
  # shellcheck source=./macos-build.sh
  source "$(dirname "${BASH_SOURCE[0]}")/macos-build.sh"
fi

if [ -z "${BUILD_ENVIRONMENT}" ] || [[ "${BUILD_ENVIRONMENT}" == *-test* ]]; then
  # shellcheck source=./macos-test.sh
  source "$(dirname "${BASH_SOURCE[0]}")/macos-test.sh"
fi
