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

BASE_IMAGE=amd64/almalinux:8
CUDA_VERSION=""
ROCM_VERSION=""
if [[ "${DOCKER_TAG_PREFIX}" == cuda* ]]; then
    # extract cuda version from image name.  e.g. manylinux2_28-builder:cuda12.8 returns 12.8
    CUDA_VERSION=$(echo "${DOCKER_TAG_PREFIX}" | awk -F'cuda' '{print $2}')
    GPU_BUILD_ARG="--build-arg CUDA_VERSION=${CUDA_VERSION}"
elif [[ "${DOCKER_TAG_PREFIX}" == rocm* ]]; then
    # extract rocm version from image name.  e.g. manylinux2_28-builder:rocm6.2.4 returns 6.2.4
    ROCM_VERSION=$(echo "${DOCKER_TAG_PREFIX}" | awk -F'rocm' '{print $2}')
    PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1200;gfx1201"
    GPU_BUILD_ARG="--build-arg ROCM_VERSION=${ROCM_VERSION} --build-arg PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
    BASE_IMAGE=rocm/dev-almalinux-8:${ROCM_VERSION}-complete
fi

case ${DOCKER_TAG_PREFIX} in
  cpu)
    BASE_TARGET=base
    ;;
  cuda*)
    BASE_TARGET=cuda${CUDA_VERSION}
    ;;
  rocm*)
    BASE_TARGET=rocm
    ;;
  *)
    echo "ERROR: Unknown docker tag ${DOCKER_TAG_PREFIX}"
    exit 1
    ;;
esac

# TODO: Remove LimitNOFILE=1048576 patch once https://github.com/pytorch/test-infra/issues/5712
# is resolved. This patch is required in order to fix timing out of Docker build on Amazon Linux 2023.
sudo sed -i s/LimitNOFILE=infinity/LimitNOFILE=1048576/ /usr/lib/systemd/system/docker.service
sudo systemctl daemon-reload
sudo systemctl restart docker

tmp_tag=$(basename "$(mktemp -u)" | tr '[:upper:]' '[:lower:]')

DOCKER_BUILDKIT=1 docker build \
  --target final \
  --build-arg "BASE_IMAGE=${BASE_IMAGE}" \
  --build-arg "BASE_TARGET=${BASE_TARGET}" \
  ${GPU_BUILD_ARG} \
  --build-arg "DEVTOOLSET_VERSION=11" \
  -t ${tmp_tag} \
  $@ \
  -f "${TOPDIR}/.ci/docker/almalinux/Dockerfile" \
  ${TOPDIR}/.ci/docker/

if [ -n "${CUDA_VERSION}" ]; then
  # Test that we're using the right CUDA compiler
  docker run --rm "${tmp_tag}" nvcc --version | grep "cuda_${CUDA_VERSION}"
fi
