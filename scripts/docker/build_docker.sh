#!/usr/bin/env bash

set -eou pipefail

export DOCKER_BUILDKIT=1
TOPDIR=$(git rev-parse --show-toplevel)

CUDA_VERSION=${CUDA_VERSION:-11.7}

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
  docker build \
    --target final \
    --build-arg "BASE_TARGET=${BASE_TARGET}" \
    --build-arg "CUDA_VERSION=${CUDA_VERSION}" \
    --build-arg "DEVTOOLSET_VERSION=9" \
    -t "pytorch/conda-builder:${DOCKER_TAG}" \
    -f "${TOPDIR}/conda/Dockerfile" \
    ${TOPDIR}
)

DOCKER_IMAGE="pytorch/conda-builder:${DOCKER_TAG}"
GITHUB_REF=${GITHUB_REF:-$(git symbolic-ref -q HEAD || git describe --tags --exact-match)}
GIT_BRANCH_NAME=${GITHUB_REF##*/}
GIT_COMMIT_SHA=${GITHUB_SHA:-$(git rev-parse HEAD)}
DOCKER_IMAGE_BRANCH_TAG=${DOCKER_IMAGE}-${GIT_BRANCH_NAME}
DOCKER_IMAGE_SHA_TAG=${DOCKER_IMAGE}-${GIT_COMMIT_SHA}

if [[ "${DOCKER_TAG}" =~ ^cuda* ]]; then
  # Meant for legacy scripts since they only do the version without the "."
  # TODO: Eventually remove this
  (
    set -x
    docker tag ${DOCKER_IMAGE} "pytorch/conda-builder:cuda${CUDA_VERSION/./}"
  )
  # Test that we're using the right CUDA compiler
  (
    set -x
    docker run --rm "${DOCKER_IMAGE}" nvcc --version | grep "cuda_${CUDA_VERSION}"
  )
fi

if [[ -n ${GITHUB_REF} ]]; then
    docker tag ${DOCKER_IMAGE} ${DOCKER_IMAGE_BRANCH_TAG}
    docker tag ${DOCKER_IMAGE} ${DOCKER_IMAGE_SHA_TAG}
fi

if [[ "${WITH_PUSH:-}" == true ]]; then
  (
    set -x
    docker push "${DOCKER_IMAGE}"
    if [[ -n ${GITHUB_REF} ]]; then
        docker push "${DOCKER_IMAGE_BRANCH_TAG}"
        docker push "${DOCKER_IMAGE_SHA_TAG}"
    fi
    if [[ "${DOCKER_TAG}" =~ ^cuda* ]]; then
      docker push "pytorch/conda-builder:cuda${CUDA_VERSION/./}"
    fi
  )
fi
