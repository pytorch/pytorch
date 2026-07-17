#!/usr/bin/env bash
# Script used only in CD pipeline

set -exou pipefail

TOPDIR=$(git rev-parse --show-toplevel)

image="$1"
shift

if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE:ARCHTAG"
  exit 1
fi

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

MANY_LINUX_VERSION=${MANY_LINUX_VERSION:-}
DOCKERFILE_SUFFIX=${DOCKERFILE_SUFFIX:-}
OPENBLAS_VERSION=${OPENBLAS_VERSION:-}
ACL_VERSION=${ACL_VERSION:-}

case ${image} in
    manylinux2_28-builder:cpu)
        TARGET=cpu_final
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG=" --build-arg DEVTOOLSET_VERSION=13"
        MANY_LINUX_VERSION="2_28"
        ;;
    manylinux2_28_aarch64-builder:cpu-aarch64)
        TARGET=final
        GPU_IMAGE=arm64v8/almalinux:8
        DOCKER_GPU_BUILD_ARG=" --build-arg DEVTOOLSET_VERSION=13"
        MANY_LINUX_VERSION="2_28_aarch64"
        ;;
    manylinuxs390x-builder:cpu-s390x)
        TARGET=final
        GPU_IMAGE=s390x/almalinux:8
        DOCKER_GPU_BUILD_ARG=""
        MANY_LINUX_VERSION="s390x"
        ;;
    manylinux2_28-builder:cuda11*)
        TARGET=cuda_final
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG="--build-arg BASE_CUDA_VERSION=${GPU_ARCH_VERSION} --build-arg DEVTOOLSET_VERSION=11"
        MANY_LINUX_VERSION="2_28"
        ;;
    manylinux2_28-builder:cuda12*)
        TARGET=cuda_final
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG="--build-arg BASE_CUDA_VERSION=${GPU_ARCH_VERSION} --build-arg DEVTOOLSET_VERSION=13"
        MANY_LINUX_VERSION="2_28"
        ;;
    manylinux2_28-builder:cuda13*)
        TARGET=cuda_final
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG="--build-arg BASE_CUDA_VERSION=${GPU_ARCH_VERSION} --build-arg DEVTOOLSET_VERSION=13"
        MANY_LINUX_VERSION="2_28"
        ;;
    manylinuxaarch64-builder:cuda*)
        TARGET=cuda_final
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG="--build-arg BASE_CUDA_VERSION=${GPU_ARCH_VERSION} --build-arg DEVTOOLSET_VERSION=13"
        MANY_LINUX_VERSION="aarch64"
        DOCKERFILE_SUFFIX="_cuda_aarch64"
        ;;
    manylinux2_28-builder:rocm*)
        # we want the patch version of 7.2 instead
        if [[ "$GPU_ARCH_VERSION" == *"7.2"* ]]; then
            GPU_ARCH_VERSION="${GPU_ARCH_VERSION}.1"
        fi
        # we want the patch version of 7.1 instead
        if [[ "$GPU_ARCH_VERSION" == *"7.1"* ]]; then
            GPU_ARCH_VERSION="${GPU_ARCH_VERSION}.1"
        fi
        # we want the patch version of 7.0 instead
        if [[ "$GPU_ARCH_VERSION" == *"7.0"* ]]; then
            GPU_ARCH_VERSION="${GPU_ARCH_VERSION}.2"
        fi
        # we want the patch version of 6.4 instead
        if [[ "$GPU_ARCH_VERSION" == *"6.4"* ]]; then
            GPU_ARCH_VERSION="${GPU_ARCH_VERSION}.4"
        fi
        TARGET=rocm_final
        MANY_LINUX_VERSION="2_28"
        DEVTOOLSET_VERSION="13"
        GPU_IMAGE=rocm/dev-almalinux-8:${GPU_ARCH_VERSION}-complete
        PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1200;gfx1201;gfx950;gfx1150;gfx1151"
        DOCKER_GPU_BUILD_ARG="--build-arg ROCM_VERSION=${GPU_ARCH_VERSION} --build-arg PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} --build-arg DEVTOOLSET_VERSION=${DEVTOOLSET_VERSION}"
        ;;
    manylinux2_28-builder:xpu)
        TARGET=xpu_final
        GPU_IMAGE=amd64/almalinux:8
        DOCKER_GPU_BUILD_ARG=" --build-arg DEVTOOLSET_VERSION=13"
        MANY_LINUX_VERSION="2_28"
        ;;
    *)
        echo "ERROR: Unrecognized image name: ${image}"
        exit 1
        ;;
esac

if [[ -n ${MANY_LINUX_VERSION} && -z ${DOCKERFILE_SUFFIX} ]]; then
    DOCKERFILE_SUFFIX=_${MANY_LINUX_VERSION}
fi
# Remote BuildKit (OSDC) pushes straight to the registry on WITH_PUSH; the local
# path (s390x) loads the built image so its workflow can tag and push it.
if [[ -n "${REMOTE_BUILDKIT:-}" && "${WITH_PUSH:-false}" == "true" ]]; then
    output_flag="--push"
elif [[ -z "${REMOTE_BUILDKIT:-}" ]]; then
    output_flag="--load"
else
    output_flag=""
fi

docker buildx build \
    ${DOCKER_GPU_BUILD_ARG} \
    --build-arg "GPU_IMAGE=${GPU_IMAGE}" \
    --build-arg "OPENBLAS_VERSION=${OPENBLAS_VERSION:-}" \
    --build-arg "ACL_VERSION=${ACL_VERSION:-}" \
    --target "${TARGET}" \
    --progress plain \
    ${output_flag} \
    "$@" \
    -f "${TOPDIR}/.ci/docker/manywheel/Dockerfile${DOCKERFILE_SUFFIX}" \
    "${TOPDIR}/.ci/docker/"
