#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

case "${GPU_ARCH_TYPE:-BLANK}" in
    BLANK)
        # Legacy behavior for CircleCI
        bash "${SCRIPTPATH}/build_cuda.sh"
        ;;
    cuda)
        bash "${SCRIPTPATH}/build_cuda.sh"
        ;;
    rocm)
        bash "${SCRIPTPATH}/build_rocm.sh"
        ;;
    cpu | cpu-cxx11-abi | cpu-s390x)
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
