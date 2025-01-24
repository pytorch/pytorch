# Image base
IMAGE_NAME ?= 939990436136.dkr.ecr.us-east-2.amazonaws.com/pytorch/manylinux2_28-builder
IMAGE_BASE_TAG  ?= cuda12.6-main

# Build dependencies
AWS_EFA_INSTALLER_VER ?=1.34.0

# Export options
EXPORT_PATH ?= ..
ZSTD_COMPRESS_OPTIONS ?= --ultra -22

# Out output options
IMAGE_TAG=${IMAGE_BASE_TAG}-efa${AWS_EFA_INSTALLER_VER}

build:
	docker build --progress plain --rm \
		--tag "${IMAGE_NAME}:${IMAGE_TAG}" \
		--build-arg IMAGE_NAME="${IMAGE_NAME}" \
		--build-arg IMAGE_BASE_TAG="${IMAGE_BASE_TAG}" \
		--build-arg AWS_EFA_INSTALLER="${AWS_EFA_INSTALLER_VER}" \
		-f Dockerfile ..

tar-img:
	docker save \
		"${IMAGE_NAME}:${IMAGE_TAG}"  | \
		zstdmt ${ZSTD_COMPRESS_OPTIONS} -v -f -o ${EXPORT_PATH}/${IMAGE_NAME}-${IMAGE_TAG}.tar.zst

push:
	docker push "${IMAGE_NAME}:${IMAGE_TAG}"
