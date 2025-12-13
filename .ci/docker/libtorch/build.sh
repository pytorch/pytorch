#!/usr/bin/env bash
# Script used only in CD pipeline

set -eoux pipefail

image="$1"
shift

if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGENAME:ARCHTAG"
  exit 1
fi

TOPDIR=$(git rev-parse --show-toplevel)

DOCKER=${DOCKER:-docker}

# Go from imagename:tag to tag
DOCKER_TAG_PREFIX=$(echo "${image}" | awk -F':' '{print $2}')

GPU_ARCH_VERSION=""
if [[ "${DOCKER_TAG_PREFIX}" == cuda* ]]; then
    # extract cuda version from image name.  e.g. manylinux2_28-builder:cuda12.8 returns 12.8
    GPU_ARCH_VERSION=$(echo "${DOCKER_TAG_PREFIX}" | awk -F'cuda' '{print $2}')
elif [[ "${DOCKER_TAG_PREFIX}" == rocm* ]]; then
    # extract rocm version from image name.  e.g. manylinux2_28-builder:rocm6.2.4 returns 6.2.4
    GPU_ARCH_VERSION=$(echo "${DOCKER_TAG_PREFIX}" | awk -F'rocm' '{print $2}')
fi

case ${DOCKER_TAG_PREFIX} in
    cpu)
        BASE_TARGET=cpu
        GPU_IMAGE=ubuntu:20.04
        DOCKER_GPU_BUILD_ARG=""
        ;;
    cuda*)
        BASE_TARGET=cuda${GPU_ARCH_VERSION}
        GPU_IMAGE=ubuntu:20.04
        DOCKER_GPU_BUILD_ARG=""
        ;;
    rocm*)
        # we want the patch version of 7.0 instead
        if [[ "$GPU_ARCH_VERSION" == *"7.0"* ]]; then
            GPU_ARCH_VERSION="${GPU_ARCH_VERSION}.2"
        fi
        # we want the patch version of 6.4 instead
        if [[ "$GPU_ARCH_VERSION" == *"6.4"* ]]; then
            GPU_ARCH_VERSION="${GPU_ARCH_VERSION}.4"
        fi
        BASE_TARGET=rocm
        GPU_IMAGE=rocm/dev-ubuntu-22.04:${GPU_ARCH_VERSION}-complete
        PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201;gfx950;gfx1150;gfx1151"
        DOCKER_GPU_BUILD_ARG="--build-arg PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} --build-arg ROCM_VERSION=${GPU_ARCH_VERSION}"
        ;;
    *)
        echo "ERROR: Unrecognized DOCKER_TAG_PREFIX: ${DOCKER_TAG_PREFIX}"
        exit 1
        ;;
esac

tmp_tag=$(basename "$(mktemp -u)" | tr '[:upper:]' '[:lower:]')

DOCKER_BUILDKIT=1 ${DOCKER} build \
    --target final \
    ${DOCKER_GPU_BUILD_ARG} \
    --build-arg "GPU_IMAGE=${GPU_IMAGE}" \
    --build-arg "BASE_TARGET=${BASE_TARGET}" \
    -t "${tmp_tag}" \
    $@ \
    -f "${TOPDIR}/.ci/docker/libtorch/Dockerfile" \
    "${TOPDIR}/.ci/docker/"
