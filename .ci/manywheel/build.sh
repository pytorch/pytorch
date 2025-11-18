#!/usr/bin/env bash

set -ex

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

# Configure ARM Compute Library (ACL) for aarch64 builds
if [[ "$BUILD_ENVIRONMENT" == *aarch64* ]]; then
  export USE_MKLDNN=1

  # ACL is required for aarch64 builds
  if [[ ! -d "/acl" ]]; then
    echo "ERROR: ARM Compute Library not found at /acl"
    echo "ACL is required for aarch64 builds. Check Docker image setup."
    exit 1
  fi

  export USE_MKLDNN_ACL=1
  export ACL_ROOT_DIR=/acl
  echo "ARM Compute Library enabled for MKLDNN: ACL_ROOT_DIR=/acl"
fi

case "${GPU_ARCH_TYPE:-BLANK}" in
    cuda | cuda-aarch64)
        bash "${SCRIPTPATH}/build_cuda.sh"
        ;;
    rocm)
        bash "${SCRIPTPATH}/build_rocm.sh"
        ;;
    cpu | cpu-cxx11-abi | cpu-aarch64 | cpu-s390x)
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
