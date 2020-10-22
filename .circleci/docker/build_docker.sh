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

# Retry on timeouts (can happen on job stampede).
retry login "${registry}"

# Logout on exit
trap "docker logout ${registry}" EXIT

# export EC2=1
# export JENKINS=1

# Try to pull the previous image (perhaps we can reuse some layers)
# if [ -n "${last_tag}" ]; then
#   docker pull "${image}:${last_tag}" || true
# fi

# Build new image
./build.sh ${IMAGE_NAME} -t "${image}:${tag}"

docker push "${image}:${tag}"

docker save -o "${IMAGE_NAME}:${tag}.tar" "${image}:${tag}"
aws s3 cp "${IMAGE_NAME}:${tag}.tar" "s3://ossci-linux-build/pytorch/base/${IMAGE_NAME}:${tag}.tar" --acl public-read
