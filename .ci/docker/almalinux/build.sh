#!/usr/bin/env bash
# Script used only in CD pipeline

set -eou pipefail

image="$1"
shift

if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE:ARCHTAG"
  exit 1
fi

DOCKER_IMAGE_NAME="pytorch/${image}"


export DOCKER_BUILDKIT=1
TOPDIR=$(git rev-parse --show-toplevel)

# Go from imagename:tag to tag
DOCKER_TAG_PREFIX=$(echo "${image}" | awk -F':' '{print $2}')

CUDA_VERSION=""
if [[ "${DOCKER_TAG_PREFIX}" == cuda* ]]; then
    # extract cuda version from image name and tag.  e.g. manylinux2_28-builder:cuda12.8 returns 12.8
    CUDA_VERSION=$(echo "${DOCKER_TAG_PREFIX}" | awk -F'cuda' '{print $2}')
fi

case ${DOCKER_TAG_PREFIX} in
  cpu)
    BASE_TARGET=base
    DOCKER_TAG=cpu
    ;;
  all)
    BASE_TARGET=all_cuda
    DOCKER_TAG=latest
    ;;
  cuda*)
    BASE_TARGET=cuda${CUDA_VERSION}
    DOCKER_TAG=cuda${CUDA_VERSION}
    ;;
  *)
    echo "ERROR: Unknown docker tag ${DOCKER_TAG_PREFIX}"
    exit 1
    ;;
esac


(
  set -x
  # TODO: Remove LimitNOFILE=1048576 patch once https://github.com/pytorch/test-infra/issues/5712
  # is resolved. This patch is required in order to fix timing out of Docker build on Amazon Linux 2023.
  sudo sed -i s/LimitNOFILE=infinity/LimitNOFILE=1048576/ /usr/lib/systemd/system/docker.service
  sudo systemctl daemon-reload
  sudo systemctl restart docker

  docker build \
    --target final \
    --progress plain \
    --build-arg "BASE_TARGET=${BASE_TARGET}" \
    --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
    --build-arg "DEVTOOLSET_VERSION=11" \
    -t ${DOCKER_IMAGE_NAME} \
    $@ \
    -f "${TOPDIR}/.ci/docker/almalinux/Dockerfile" \
    ${TOPDIR}/.ci/docker/
)

if [[ "${DOCKER_TAG}" =~ ^cuda* ]]; then
  # Test that we're using the right CUDA compiler
  (
    set -x
    docker run --rm "${DOCKER_IMAGE_NAME}" nvcc --version | grep "cuda_${CUDA_VERSION}"
  )
fi
