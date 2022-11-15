#!/usr/bin/env bash

set -xeuo pipefail

PYTORCH_DOCKER_TAG=$(git describe --tags --always)-devel
CUDA_VERSION=11.6.2

# Build PyTorch nightly docker
make -f docker.Makefile \
     DOCKER_REGISTRY=ghcr.io \
     DOCKER_ORG=pytorch \
     CUDA_VERSION=${CUDA_VERSION} \
     DOCKER_IMAGE=pytorch-nightly \
     DOCKER_TAG=${PYTORCH_DOCKER_TAG} \
     INSTALL_CHANNEL=pytorch-nightly BUILD_TYPE=official devel-image

# Get the PYTORCH_NIGHTLY_COMMIT from the docker image
PYTORCH_NIGHTLY_COMMIT=$(docker run \
       ghcr.io/pytorch/pytorch-nightly:${PYTORCH_DOCKER_TAG} \
       python -c 'import torch; print(torch.version.git_version)' | head -c 7)

docker tag ghcr.io/pytorch/pytorch-nightly:${PYTORCH_DOCKER_TAG} \
       ghcr.io/pytorch/pytorch-nightly:${PYTORCH_NIGHTLY_COMMIT}-cu${CUDA_VERSION}

docker tag ghcr.io/pytorch/pytorch-nightly:${PYTORCH_NIGHTLY_COMMIT}-cu${CUDA_VERSION} \
       ghcr.io/pytorch/pytorch-nightly:latest

if [[ ${WITH_PUSH:-} == "true" ]]; then
    # Push the nightly docker to GitHub Container Registry
    echo $GHCR_PAT | docker login ghcr.io -u pytorch --password-stdin
    make -f docker.Makefile \
         DOCKER_REGISTRY=ghcr.io \
         DOCKER_ORG=pytorch \
         DOCKER_IMAGE=pytorch-nightly \
         DOCKER_TAG=${PYTORCH_NIGHTLY_COMMIT}-cu${CUDA_VERSION} \
         devel-push

    make -f docker.Makefile \
         DOCKER_REGISTRY=ghcr.io \
         DOCKER_ORG=pytorch \
         DOCKER_IMAGE=pytorch-nightly \
         DOCKER_TAG=latest \
         devel-push
fi
