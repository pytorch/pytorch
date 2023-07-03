#!/bin/bash

set -ex

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*)
}

tag="${DOCKER_TAG}"
registry="308535385114.dkr.ecr.us-east-1.amazonaws.com"

# NB: The image name could now be both the short form, like pytorch-linux-bionic-py3.11-clang9, or the
# full name, like 308535385114.dkr.ecr.us-east-1.amazonaws.com/pytorch/pytorch-linux-bionic-py3.11-clang9
if [[ "${IMAGE_NAME}" == *"${registry}/pytorch/"* ]]; then
  # Extract the image name from the long name
  EXTRACTED_IMAGE_NAME=$(echo ${IMAGE_NAME#"${registry}/pytorch/"} | awk -F '[:,]' '{print $1}')
  IMAGE_NAME="${EXTRACTED_IMAGE_NAME}"
fi

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
