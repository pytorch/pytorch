SHELL=/usr/bin/env bash

DOCKER_CMD ?= docker
DESIRED_CUDA ?= 12.8
DESIRED_CUDA_SHORT = $(subst .,,$(DESIRED_CUDA))
PACKAGE_NAME = magma-cuda
CUDA_ARCHITECTURES ?= 75;80;86;90;100;120

DOCKER_RUN = set -eou pipefail; ${DOCKER_CMD} run --rm -i \
	-v $(shell git rev-parse --show-toplevel)/.ci:/builder \
	-w /builder \
	-e PACKAGE_NAME=${PACKAGE_NAME}${DESIRED_CUDA_SHORT} \
	-e DESIRED_CUDA=${DESIRED_CUDA} \
	-e CUDA_ARCHITECTURES="${CUDA_ARCHITECTURES}" \
	"pytorch/almalinux-builder:cuda${DESIRED_CUDA}-main" \
	magma/build_magma.sh

.PHONY: all
all: magma-cuda132
all: magma-cuda130
all: magma-cuda129
all: magma-cuda128
all: magma-cuda126

.PHONY:
clean:
	$(RM) -r magma-*
	$(RM) -r output

.PHONY: magma-cuda132
magma-cuda132: DESIRED_CUDA := 13.2
magma-cuda132: CUDA_ARCHITECTURES := 75;80;86;90;100;120
magma-cuda132:
	$(DOCKER_RUN)

.PHONY: magma-cuda130
magma-cuda130: DESIRED_CUDA := 13.0
magma-cuda130: CUDA_ARCHITECTURES := 75;80;86;90;100;120
magma-cuda130:
	$(DOCKER_RUN)

.PHONY: magma-cuda129
magma-cuda129: DESIRED_CUDA := 12.9
magma-cuda129: CUDA_ARCHITECTURES := 75;80;86;90;100;120
magma-cuda129:
	$(DOCKER_RUN)

.PHONY: magma-cuda128
magma-cuda128: DESIRED_CUDA := 12.8
magma-cuda128: CUDA_ARCHITECTURES := 75;80;86;90;100;120
magma-cuda128:
	$(DOCKER_RUN)

.PHONY: magma-cuda126
magma-cuda126: DESIRED_CUDA := 12.6
magma-cuda126: CUDA_ARCHITECTURES := 50;60;70;80;86;90
magma-cuda126:
	$(DOCKER_RUN)
