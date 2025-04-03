#!/usr/bin/env bash
# Script used only in CD pipeline

set -eou pipefail

image="$1"
shift

if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE"
  exit 1
fi

DOCKER_IMAGE="pytorch/${image}"

TOPDIR=$(git rev-parse --show-toplevel)

GPU_ARCH_TYPE=${GPU_ARCH_TYPE:-cpu}
GPU_ARCH_VERSION=${GPU_ARCH_VERSION:-}

WITH_PUSH=${WITH_PUSH:-}

DOCKER=${DOCKER:-docker}

case ${GPU_ARCH_TYPE} in
    cpu)
        BASE_TARGET=cpu
        DOCKER_TAG=cpu
        GPU_IMAGE=ubuntu:20.04
        DOCKER_GPU_BUILD_ARG=""
        ;;
    cuda)
        BASE_TARGET=cuda${GPU_ARCH_VERSION}
        DOCKER_TAG=cuda${GPU_ARCH_VERSION}
        GPU_IMAGE=ubuntu:20.04
        DOCKER_GPU_BUILD_ARG=""
        ;;
    rocm)
        BASE_TARGET=rocm
        DOCKER_TAG=rocm${GPU_ARCH_VERSION}
        GPU_IMAGE=rocm/dev-ubuntu-22.04:${GPU_ARCH_VERSION}-complete
        PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
        DOCKER_GPU_BUILD_ARG="--build-arg PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} --build-arg ROCM_VERSION=${GPU_ARCH_VERSION}"
        ;;
    *)
        echo "ERROR: Unrecognized GPU_ARCH_TYPE: ${GPU_ARCH_TYPE}"
        exit 1
        ;;
esac


(
    set -x
    DOCKER_BUILDKIT=1 ${DOCKER} build \
         --target final \
        ${DOCKER_GPU_BUILD_ARG} \
        --build-arg "GPU_IMAGE=${GPU_IMAGE}" \
        --build-arg "BASE_TARGET=${BASE_TARGET}" \
        -t "${DOCKER_IMAGE}" \
        $@ \
        -f "${TOPDIR}/.ci/docker/libtorch/Dockerfile" \
        "${TOPDIR}/.ci/docker/"

)
