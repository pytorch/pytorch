#!/usr/bin/env bash
# Script used only in CD pipeline

set -exou pipefail

image="$1"
shift

if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGENAME:ARCHTAG"
  exit 1
fi

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
    ;;
  cuda*)
    BASE_TARGET=cuda${CUDA_VERSION}
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

export DOCKER_BUILDKIT=1
TOPDIR=$(git rev-parse --show-toplevel)
tmp_tag=$(basename "$(mktemp -u)" | tr '[:upper:]' '[:lower:]')

docker build \
  --target final \
  --progress plain \
  --build-arg "BASE_TARGET=${BASE_TARGET}" \
  --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
  --build-arg "DEVTOOLSET_VERSION=11" \
  -t ${tmp_tag} \
  $@ \
  -f "${TOPDIR}/.ci/docker/almalinux/Dockerfile" \
  ${TOPDIR}/.ci/docker/

if [ -n "${CUDA_VERSION}" ]; then
  # Test that we're using the right CUDA compiler
  docker run --rm "${tmp_tag}" nvcc --version | grep "cuda_${CUDA_VERSION}"
fi
