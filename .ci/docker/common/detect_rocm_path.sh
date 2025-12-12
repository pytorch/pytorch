#!/bin/bash
# Helper script to set ROCm environment variables
# This script is meant to be sourced by other scripts
#
# The actual env vars are defined in /etc/rocm_env.sh which is created
# by install_rocm.sh during Docker image build.

if [ -f /etc/rocm_env.sh ]; then
    source /etc/rocm_env.sh
else
    # Fallback for environments where install_rocm.sh hasn't run
    export ROCM_PATH="${ROCM_PATH:-/opt/rocm}"
    export ROCM_HOME="${ROCM_HOME:-$ROCM_PATH}"
    export ROCM_DEVICE_LIB_PATH="${ROCM_PATH}/amdgcn/bitcode"
fi
