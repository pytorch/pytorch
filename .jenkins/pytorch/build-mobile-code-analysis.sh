#!/usr/bin/env bash
# DO NOT ADD 'set -x' not to reveal CircleCI secret context environment variables
set -eu -o pipefail

# This script builds and runs code analyzer tool to generate aten op dependency
# graph for custom mobile build.

COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Clang version:"
clang --version

export LLVM_DIR="$(llvm-config-5.0 --prefix)"
echo "LLVM_DIR: ${LLVM_DIR}"

# Run the following 2 steps together because they share the same (reusable) time
# consuming process to build LibTorch into LLVM assembly.

# 1. Run code analysis test first to fail fast
time ANALYZE_TEST=1 CHECK_RESULT=1 tools/code_analyzer/build.sh

# 2. Run code analysis on mobile LibTorch
time ANALYZE_TORCH=1 tools/code_analyzer/build.sh
