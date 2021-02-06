#!/bin/sh

set -xeuo pipefail

PYTORCH_DOCKER_TAG=$(git describe --tags --always)-devel

# Build PyTorch nightly docker
# Full name: ghcr.io/pytorch/pytorch-nightly:${PYTORCH_DOCKER_TAG}
make -f docker.Makefile \
     DOCKER_REGISTRY=ghcr.io \
     DOCKER_ORG=pytorch \
     DOCKER_IMAGE=pytorch-nightly \
     DOCKER_TAG=${PYTORCH_DOCKER_TAG} \
     INSTALL_CHANNEL=pytorch-nightly BUILD_TYPE=official devel-image

# Get the PYTORCH_NIGHTLY_COMMIT from the docker image
PYTORCH_NIGHTLY_COMMIT=$(docker run \
       pytorch/pytorch-nightly:${PYTORCH_DOCKER_TAG} \
       python -c 'import torch; print(torch.version.git_version)' | head -c 7)
docker tag pytorch/pytorch-nightly:${PYTORCH_DOCKER_TAG} \
       pytorch/pytorch-nightly:${PYTORCH_NIGHTLY_COMMIT}

# Push the nightly docker to GitHub Container Registry
echo $GHCR_PAT | docker login ghcr.io -u pytorch --password-stdin
make -f docker.Makefile \
     DOCKER_REGISTRY=ghcr.io \
     DOCKER_ORG=pytorch \
     DOCKER_IMAGE=pytorch-nightly \
     DOCKER_TAG=${PYTORCH_NIGHTLY_COMMIT} \
     devel-push
