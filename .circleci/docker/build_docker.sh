#!/bin/bash

set -ex

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*)
}

tag="${DOCKER_TAG}"


registry="${DOCKER_REGISTRY:-308535385114.dkr.ecr.us-east-1.amazonaws.com}"
image="${registry}/pytorch/${IMAGE_NAME}"

# Build new image
./build.sh ${IMAGE_NAME} -t "${image}:${tag}"

docker push "${image}:${tag}"

if [ -z "${DOCKER_SKIP_S3_UPLOAD:-}" ]; then
  trap "rm -rf ${IMAGE_NAME}:${tag}.tar" EXIT
  docker save -o "${IMAGE_NAME}:${tag}.tar" "${image}:${tag}"
  aws s3 cp "${IMAGE_NAME}:${tag}.tar" "s3://ossci-linux-build/pytorch/base/${IMAGE_NAME}:${tag}.tar" --acl public-read
fi
