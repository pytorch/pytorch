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

# The autoscaled buildkit pool may be cold / at capacity at start, where buildx's
# ~20s connect (gRPC default) fails before scale-up. Retry connection failures
# (not build errors) for ~2h so a capacity-limited build waits for a free pod
# instead of hard-failing — still within the job timeout. Mirrors the retry loop
# in .ci/docker/build.sh.
attempts="${REMOTE_BUILDKIT_CONNECT_ATTEMPTS:-360}"
delay="${REMOTE_BUILDKIT_CONNECT_DELAY:-15}"
for attempt in $(seq 1 "${attempts}"); do
  build_log="$(mktemp)"
  set +e
  build_image "$@" 2>&1 | tee "${build_log}"
  rc="${PIPESTATUS[0]}"
  set -e
  if [[ "${rc}" -eq 0 ]]; then
    rm -f "${build_log}"
    break
  fi
  if [[ "${attempt}" -lt "${attempts}" ]] && grep -qiE \
    "waiting for connection|context deadline exceeded|server preface|failed to (dial|list workers)|connection (refused|reset)|no such host|transport: Error|i/o timeout|use of closed network connection|EOF" \
    "${build_log}"; then
    echo "Remote BuildKit not ready yet (attempt ${attempt}/${attempts}); retrying in ${delay}s..." >&2
    rm -f "${build_log}"
    sleep "${delay}"
    continue
  fi
  rm -f "${build_log}"
  exit "${rc}"
done
