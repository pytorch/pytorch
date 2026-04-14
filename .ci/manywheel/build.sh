#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# CPU/CUDA x86 and aarch64 builds use linux-binary-manywheel.yml.
# This script handles only ROCm, XPU, and s390x.
case "${GPU_ARCH_TYPE:-BLANK}" in
    rocm)
        bash "${SCRIPTPATH}/build_rocm.sh"
        ;;
    cpu-s390x)
        bash "${SCRIPTPATH}/build_cpu.sh"
        ;;
    xpu)
        bash "${SCRIPTPATH}/build_xpu.sh"
        ;;
    *)
        echo "Un-recognized GPU_ARCH_TYPE '${GPU_ARCH_TYPE}', exiting..."
        exit 1
        ;;
esac
