#!/bin/bash
set -ex
# Helper script to detect ROCM_PATH dynamically
# This script is meant to be sourced by other scripts
# Detect ROCM_PATH based on installation type
if command -v rocm-sdk &> /dev/null && python3 -m rocm_sdk path --root &> /dev/null; then
    # theRock/nightly installation
    export ROCM_PATH="$(python3 -m rocm_sdk path --root)"
    export ROCM_HOME="$ROCM_PATH"
    export ROCM_BIN="$(python3 -m rocm_sdk path --bin 2>/dev/null || echo ${ROCM_PATH}/bin)"
    export ROCM_CMAKE="$(python3 -m rocm_sdk path --cmake 2>/dev/null || echo ${ROCM_PATH})"
    export ROCM_DEVICE_LIB_PATH="${ROCM_PATH}/lib/llvm/amdgcn/bitcode"
    # theRock bundles system dependencies like libdrm in rocm_sysdeps
    export ROCM_SYSDEPS_INCLUDE="${ROCM_PATH}/lib/rocm_sysdeps/include"
else
    # Traditional installation
    export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export ROCM_HOME="${ROCM_HOME:-/opt/rocm}"
    export ROCM_BIN="${ROCM_PATH}/bin"
    export ROCM_CMAKE="${ROCM_PATH}"
    export ROCM_DEVICE_LIB_PATH="${ROCM_PATH}/amdgcn/bitcode"
fi


