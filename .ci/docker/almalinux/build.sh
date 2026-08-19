#!/usr/bin/env bash
# Script used only in the CD pipeline, on an OSDC remote BuildKit builder (there
# is no local Docker daemon). The caller sets up the buildx builder, passes the
# target tag(s) as trailing `-t ...` args ("$@"), and gates publishing via
# WITH_PUSH.

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
ROCM_VERSION=""
EXTRA_BUILD_ARGS=""
if [[ "${DOCKER_TAG_PREFIX}" == cuda* ]]; then
    # extract cuda version from image name and tag.  e.g. manylinux2_28-builder:cuda12.8 returns 12.8
    CUDA_VERSION=$(echo "${DOCKER_TAG_PREFIX}" | awk -F'cuda' '{print $2}')
    EXTRA_BUILD_ARGS="--build-arg CUDA_VERSION=${CUDA_VERSION}"
elif [[ "${DOCKER_TAG_PREFIX}" == rocm* ]]; then
    # extract rocm version from image name and tag.  e.g. manylinux2_28-builder:rocm6.2.4 returns 6.2.4
    ROCM_VERSION=$(echo "${DOCKER_TAG_PREFIX}" | awk -F'rocm' '{print $2}')
    EXTRA_BUILD_ARGS="--build-arg ROCM_IMAGE=rocm/dev-almalinux-8:${ROCM_VERSION}-complete"
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
    PYTORCH_ROCM_ARCH="gfx900;gfx906;gfx908;gfx90a;gfx942;gfx1030;gfx1100;gfx1101;gfx1102;gfx1103;gfx1200;gfx1201;gfx950;gfx1150;gfx1151"
    EXTRA_BUILD_ARGS="${EXTRA_BUILD_ARGS} --build-arg PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH}"
    ;;
  *)
    echo "ERROR: Unknown docker tag ${DOCKER_TAG_PREFIX}"
    exit 1
    ;;
esac

export DOCKER_BUILDKIT=1
TOPDIR=$(git rev-parse --show-toplevel)
DOCKERFILE="${TOPDIR}/.ci/docker/almalinux/Dockerfile"
BUILD_CONTEXT="${TOPDIR}/.ci/docker/"

# WITH_PUSH gates whether we publish: push events publish, PRs only validate the
# build (remote driver with no output keeps the result in the build cache).
output_flag=""
if [[ "${WITH_PUSH:-false}" == "true" ]]; then
  output_flag="--push"
fi

build_image() {
  docker buildx build \
    --target final \
    --progress plain \
    --build-arg "BASE_TARGET=${BASE_TARGET}" \
    --build-arg "DEVTOOLSET_VERSION=13" \
    ${EXTRA_BUILD_ARGS} \
    ${output_flag} \
    "$@" \
    -f "${DOCKERFILE}" \
    "${BUILD_CONTEXT}"
}

# The caller (binary-docker-build action) wraps this in a cold-pool
# connect-retry loop, so just build once here.
build_image "$@"
