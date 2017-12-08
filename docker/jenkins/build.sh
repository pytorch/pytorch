#!/bin/bash

set -e

declare -a valid_images
valid_images=(
  # Primary builds
  py2-cuda8.0-cudnn7-ubuntu16.04
  py3-cuda8.0-cudnn7-ubuntu16.04
  py2-cuda9.0-cudnn7-ubuntu16.04
  py3-cuda9.0-cudnn7-ubuntu16.04
  py2-mkl-ubuntu16.04
  py3-mkl-ubuntu16.04

  # Compiler compatibility
  py2-gcc5-ubuntu16.04
  py2-gcc6-ubuntu16.04
  py2-gcc7-ubuntu16.04
  py2-clang3.8-ubuntu16.04
  py2-clang3.9-ubuntu16.04

  # Build for Android
  py2-android-ubuntu16.04
)

image="$1"
shift

if [ -z "${image}" ]; then
  echo "Usage: $0 IMAGE"
  exit 1
fi

UBUNTU_VERSION="$(echo "${image}" | perl -n -e'/ubuntu(\d+\.\d+)/ && print $1')"
DOCKERFILE="ubuntu/Dockerfile"

PYTHON_VERSION="$(echo "${image}" | perl -n -e'/py(\d+(\.\d+)?)/ && print $1')"

if [[ "$image" == *cuda* ]]; then
  CUDA_VERSION="$(echo "${image}" | perl -n -e'/cuda(\d+\.\d+)/ && print $1')"
  CUDNN_VERSION="$(echo "${image}" | perl -n -e'/cudnn(\d+)/ && print $1')"
  DOCKERFILE="ubuntu-cuda/Dockerfile"
fi

if [[ "$image" == *-mkl-* ]]; then
  MKL=yes
fi

if [[ "$image" == *-android-* ]]; then
  ANDROID=yes
fi

if [[ "$image" == *-gcc* ]]; then
  GCC_VERSION="$(echo "${image}" | perl -n -e'/gcc(\d+(\.\d+)?)/ && print $1')"
fi

if [[ "$image" == *-clang* ]]; then
  CLANG_VERSION="$(echo "${image}" | perl -n -e'/clang(\d+(\.\d+)?)/ && print $1')"
fi

# Copy over common scripts to directory containing the Dockerfile to build
cp -a bin common/* "$(dirname ${DOCKERFILE})"

# Set Jenkins UID and GID if running Jenkins
if [ -n "${JENKINS:-}" ]; then
  JENKINS_UID=$(id -u jenkins)
  JENKINS_GID=$(id -g jenkins)
fi

# Build image
docker build \
       --build-arg EC2=${EC2:-} \
       --build-arg JENKINS=${JENKINS:-} \
       --build-arg JENKINS_UID=${JENKINS_UID:-} \
       --build-arg JENKINS_GID=${JENKINS_GID:-} \
       --build-arg UBUNTU_VERSION=${UBUNTU_VERSION} \
       --build-arg PYTHON_VERSION=${PYTHON_VERSION} \
       --build-arg CUDA_VERSION=${CUDA_VERSION} \
       --build-arg CUDNN_VERSION=${CUDNN_VERSION} \
       --build-arg MKL=${MKL} \
       --build-arg ANDROID=${ANDROID} \
       --build-arg GCC_VERSION=${GCC_VERSION} \
       --build-arg CLANG_VERSION=${CLANG_VERSION} \
       "$@" \
       "$(dirname ${DOCKERFILE})"
