SHELL=/usr/bin/env bash

DOCKER_CMD ?= docker
DESIRED_ROCM ?= 7.1
DESIRED_ROCM_SHORT = $(subst .,,$(DESIRED_ROCM))
PACKAGE_NAME = magma-rocm
# inherit this from underlying docker image, do not pass this env var to docker
#PYTORCH_ROCM_ARCH ?= gfx900;gfx906;gfx908;gfx90a;gfx942;gfx950;gfx1030;gfx1100;gfx1101;gfx1102;gfx1150;gfx1151;gfx1200;gfx1201

DOCKER_RUN = set -eou pipefail; ${DOCKER_CMD} run --rm -i \
	-v $(shell git rev-parse --show-toplevel)/.ci:/builder \
	-w /builder \
	-e PACKAGE_NAME=${PACKAGE_NAME}${DESIRED_ROCM_SHORT} \
	-e DESIRED_ROCM=${DESIRED_ROCM} \
	"pytorch/almalinux-builder:rocm${DESIRED_ROCM}" \
	magma-rocm/build_magma.sh

.PHONY: all
all: magma-rocm71
all: magma-rocm70
all: magma-rocm64

.PHONY:
clean:
	$(RM) -r magma-*
	$(RM) -r output

.PHONY: magma-rocm71
magma-rocm71: DESIRED_ROCM := 7.1
magma-rocm71:
	$(DOCKER_RUN)

.PHONY: magma-rocm70
magma-rocm70: DESIRED_ROCM := 7.0
magma-rocm70:
	$(DOCKER_RUN)

.PHONY: magma-rocm64
magma-rocm64: DESIRED_ROCM := 6.4
magma-rocm64:
	$(DOCKER_RUN)
