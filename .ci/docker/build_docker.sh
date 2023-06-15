#!/bin/bash

set -ex

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*)
}

# If UPSTREAM_BUILD_ID is set (see trigger job), then we can
# use it to tag this build with the same ID used to tag all other
# base image builds. Also, we can try and pull the previous
# image first, to avoid rebuilding layers that haven't changed.

#until we find a way to reliably reuse previous build, this last_tag is not in use
# last_tag="$(( CIRCLE_BUILD_NUM - 1 ))"
tag="${DOCKER_TAG}"


registry="308535385114.dkr.ecr.us-east-1.amazonaws.com"
image="${registry}/pytorch/${IMAGE_NAME}"

login() {
  aws ecr get-authorization-token --region us-east-1 --output text --query 'authorizationData[].authorizationToken' |
    base64 -d |
    cut -d: -f2 |
    docker login -u AWS --password-stdin "$1"
}


# Only run these steps if not on github actions
if [[ -z "${GITHUB_ACTIONS}" ]]; then
  # Retry on timeouts (can happen on job stampede).
  retry login "${registry}"
  # Logout on exit
  trap "docker logout ${registry}" EXIT
fi

# Try to pull the previous image (perhaps we can reuse some layers)
# if [ -n "${last_tag}" ]; then
#   docker pull "${image}:${last_tag}" || true
# fi

# Build new image
./build.sh ${IMAGE_NAME} -t "${image}:${tag}"

# Only push if `DOCKER_SKIP_PUSH` = false
if [ "${DOCKER_SKIP_PUSH:-true}" = "false" ]; then
  # Only push if docker image doesn't exist already.
  # ECR image tags are immutable so this will avoid pushing if only just testing if the docker jobs work
  # NOTE: The only workflow that should push these images should be the docker-builds.yml workflow
  if ! docker manifest inspect "${image}:${tag}" >/dev/null 2>/dev/null; then
    docker push "${image}:${tag}"
  fi
fi

if [ -z "${DOCKER_SKIP_S3_UPLOAD:-}" ]; then
  trap "rm -rf ${IMAGE_NAME}:${tag}.tar" EXIT
  docker save -o "${IMAGE_NAME}:${tag}.tar" "${image}:${tag}"
  aws s3 cp "${IMAGE_NAME}:${tag}.tar" "s3://ossci-linux-build/pytorch/base/${IMAGE_NAME}:${tag}.tar" --acl public-read
fi
