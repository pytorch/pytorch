SHELL=/usr/bin/env bash

DOCKER_CMD ?= docker
DESIRED_CUDA ?= 11.8
DESIRED_CUDA_SHORT = $(subst .,,$(DESIRED_CUDA))
PACKAGE_NAME = magma-cuda
CUDA_ARCH_LIST ?= -gencode arch=compute_50,code=sm_50 -gencode arch=compute_60,code=sm_60 -gencode arch=compute_70,code=sm_70 -gencode arch=compute_80,code=sm_80 -gencode arch=compute_86,code=sm_86 -gencode arch=compute_90,code=sm_90

DOCKER_RUN = set -eou pipefail; ${DOCKER_CMD} run --rm -i \
	-v $(shell git rev-parse --show-toplevel)/.ci:/builder \
	-w /builder \
	-e PACKAGE_NAME=${PACKAGE_NAME}${DESIRED_CUDA_SHORT} \
	-e DESIRED_CUDA=${DESIRED_CUDA} \
	-e CUDA_ARCH_LIST="${CUDA_ARCH_LIST}" \
	"pytorch/manylinux-builder:cuda${DESIRED_CUDA}-main" \
	magma/build_magma.sh

.PHONY: all
all: magma-cuda126
all: magma-cuda124
all: magma-cuda121
all: magma-cuda118

.PHONY:
clean:
	$(RM) -r magma-*
	$(RM) -r output

.PHONY: magma-cuda126
magma-cuda126: DESIRED_CUDA := 12.6
magma-cuda126:
	$(DOCKER_RUN)

.PHONY: magma-cuda124
magma-cuda124: DESIRED_CUDA := 12.4
magma-cuda124:
	$(DOCKER_RUN)

.PHONY: magma-cuda121
magma-cuda121: DESIRED_CUDA := 12.1
magma-cuda121:
	$(DOCKER_RUN)

.PHONY: magma-cuda118
magma-cuda118: DESIRED_CUDA := 11.8
magma-cuda118: CUDA_ARCH_LIST += -gencode arch=compute_37,code=sm_37
magma-cuda118:
	$(DOCKER_RUN)
