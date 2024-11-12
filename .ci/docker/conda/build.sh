#!/usr/bin/env bash
# Script used only in CD pipeline

set -eou pipefail

image="$1"
shift

if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE"
  exit 1
fi

DOCKER_IMAGE_NAME="pytorch/${image}"


export DOCKER_BUILDKIT=1
TOPDIR=$(git rev-parse --show-toplevel)

CUDA_VERSION=${CUDA_VERSION:-12.1}

case ${CUDA_VERSION} in
  cpu)
    BASE_TARGET=base
    DOCKER_TAG=cpu
    ;;
  all)
    BASE_TARGET=all_cuda
    DOCKER_TAG=latest
    ;;
  *)
    BASE_TARGET=cuda${CUDA_VERSION}
    DOCKER_TAG=cuda${CUDA_VERSION}
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
    --build-arg "DEVTOOLSET_VERSION=9" \
    -t ${DOCKER_IMAGE_NAME} \
    $@ \
    -f "${TOPDIR}/.ci/docker/conda/Dockerfile" \
    ${TOPDIR}/.ci/docker/
)

if [[ "${DOCKER_TAG}" =~ ^cuda* ]]; then
  # Test that we're using the right CUDA compiler
  (
    set -x
    docker run --rm "${DOCKER_IMAGE_NAME}" nvcc --version | grep "cuda_${CUDA_VERSION}"
  )
fi

GITHUB_REF=${GITHUB_REF:-$(git symbolic-ref -q HEAD || git describe --tags --exact-match)}
GIT_BRANCH_NAME=${GITHUB_REF##*/}
GIT_COMMIT_SHA=${GITHUB_SHA:-$(git rev-parse HEAD)}
DOCKER_IMAGE_BRANCH_TAG=${DOCKER_IMAGE_NAME}-${GIT_BRANCH_NAME}
DOCKER_IMAGE_SHA_TAG=${DOCKER_IMAGE_NAME}-${GIT_COMMIT_SHA}
if [[ "${WITH_PUSH:-}" == true ]]; then
  (
    set -x
    docker push "${DOCKER_IMAGE_NAME}"
    if [[ -n ${GITHUB_REF} ]]; then
        docker tag ${DOCKER_IMAGE_NAME} ${DOCKER_IMAGE_BRANCH_TAG}
        docker tag ${DOCKER_IMAGE_NAME} ${DOCKER_IMAGE_SHA_TAG}
        docker push "${DOCKER_IMAGE_BRANCH_TAG}"
        docker push "${DOCKER_IMAGE_SHA_TAG}"
    fi
  )
fi
