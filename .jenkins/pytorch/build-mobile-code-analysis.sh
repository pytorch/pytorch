#!/usr/bin/env bash
# DO NOT ADD 'set -x' not to reveal CircleCI secret context environment variables
set -eu -o pipefail

# This script builds and runs code analyzer tool to generate aten op dependency
# graph for custom mobile build.

# shellcheck disable=SC2034
COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Clang version:"
clang --version

export LLVM_DIR="$(llvm-config-5.0 --prefix)"
echo "LLVM_DIR: ${LLVM_DIR}"

time ANALYZE_TEST=1 CHECK_RESULT=1 tools/code_analyzer/build.sh
