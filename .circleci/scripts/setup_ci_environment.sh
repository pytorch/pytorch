#!/usr/bin/env bash
set -ex

# Check if we should actually run
echo "BUILD_ENVIRONMENT: ${BUILD_ENVIRONMENT}"
echo "CIRCLE_PULL_REQUEST: ${CIRCLE_PULL_REQUEST}"
if [[ "${BUILD_ENVIRONMENT}" == *-slow-* ]]; then
  if ! [ -z "${CIRCLE_PULL_REQUEST}" ]; then
    # It's a PR; test for [slow ci] tag on the TOPMOST commit
    topmost_commit=$(git log --format='%B' -n 1 HEAD)
    if !(echo $topmost_commit | grep -q -e '\[slow ci\]' -e '\[ci slow\]' -e '\[test slow\]' -e '\[slow test\]'); then
      circleci step halt
      exit
    fi
  fi
fi
if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
  if ! [ -z "${CIRCLE_PULL_REQUEST}" ]; then
    # It's a PR; test for [xla ci] tag on the TOPMOST commit
    topmost_commit=$(git log --format='%B' -n 1 HEAD)
    if !(echo $topmost_commit | grep -q -e '\[xla ci\]' -e '\[ci xla\]' -e '\[test xla\]' -e '\[xla test\]'); then
      # NB: This doesn't halt everything, just this job.  So
      # the rest of the workflow will keep going and you need
      # to make sure you halt there too.  Blegh.
      circleci step halt
      exit
    fi
  fi
fi
if [[ "${BUILD_ENVIRONMENT}" == *namedtensor* ]]; then
  if ! [ -z "${CIRCLE_PULL_REQUEST}" ]; then
    # It's a PR; test for [namedtensor] tag on the TOPMOST commit
    topmost_commit=$(git log --format='%B' -n 1 HEAD)
    if !(echo $topmost_commit | grep -q -e '\[namedtensor\]' -e '\[ci namedtensor\]' -e '\[namedtensor ci\]'); then
      # NB: This doesn't halt everything, just this job.  So
      # the rest of the workflow will keep going and you need
      # to make sure you halt there too.  Blegh.
      circleci step halt
      exit
    fi
  fi
fi

# Set up NVIDIA docker repo
curl -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
echo "deb https://nvidia.github.io/libnvidia-container/ubuntu16.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
echo "deb https://nvidia.github.io/nvidia-container-runtime/ubuntu16.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list
echo "deb https://nvidia.github.io/nvidia-docker/ubuntu16.04/amd64 /" | sudo tee -a /etc/apt/sources.list.d/nvidia-docker.list

sudo apt-get -y update
sudo apt-get -y remove linux-image-generic linux-headers-generic linux-generic docker-ce
# WARNING: Docker version is hardcoded here; you must update the
# version number below for docker-ce and nvidia-docker2 to get newer
# versions of Docker.  We hardcode these numbers because we kept
# getting broken CI when Docker would update their docker version,
# and nvidia-docker2 would be out of date for a day until they
# released a newer version of their package.
#
# How to figure out what the correct versions of these packages are?
# My preferred method is to start a Docker instance of the correct
# Ubuntu version (e.g., docker run -it ubuntu:16.04) and then ask
# apt what the packages you need are.  Note that the CircleCI image
# comes with Docker.
sudo apt-get -y install \
  linux-headers-$(uname -r) \
  linux-image-generic \
  moreutils \
  docker-ce=5:18.09.4~3-0~ubuntu-xenial \
  nvidia-container-runtime=2.0.0+docker18.09.4-1 \
  nvidia-docker2=2.0.3+docker18.09.4-1 \
  expect-dev

sudo pkill -SIGHUP dockerd

sudo pip -q install awscli==1.16.35

if [ -n "${USE_CUDA_DOCKER_RUNTIME}" ]; then
  DRIVER_FN="NVIDIA-Linux-x86_64-410.104.run"
  wget "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"
  sudo /bin/bash "$DRIVER_FN" -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
  nvidia-smi
fi

if [[ "${BUILD_ENVIRONMENT}" == *-build ]]; then
  echo "declare -x IN_CIRCLECI=1" > /home/circleci/project/env
  echo "declare -x COMMIT_SOURCE=${CIRCLE_BRANCH}" >> /home/circleci/project/env
  echo "declare -x PYTHON_VERSION=${PYTHON_VERSION}" >> /home/circleci/project/env
  echo "declare -x SCCACHE_BUCKET=ossci-compiler-cache-circleci-v2" >> /home/circleci/project/env
  if [ -n "${USE_CUDA_DOCKER_RUNTIME}" ]; then
    echo "declare -x TORCH_CUDA_ARCH_LIST=5.2" >> /home/circleci/project/env
  fi
  export SCCACHE_MAX_JOBS=`expr $(nproc) - 1`
  export MEMORY_LIMIT_MAX_JOBS=8  # the "large" resource class on CircleCI has 32 CPU cores, if we use all of them we'll OOM
  export MAX_JOBS=$(( ${SCCACHE_MAX_JOBS} > ${MEMORY_LIMIT_MAX_JOBS} ? ${MEMORY_LIMIT_MAX_JOBS} : ${SCCACHE_MAX_JOBS} ))
  echo "declare -x MAX_JOBS=${MAX_JOBS}" >> /home/circleci/project/env

  if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
    # This IAM user allows write access to S3 bucket for sccache & bazels3cache
    set +x
    echo "declare -x AWS_ACCESS_KEY_ID=${CIRCLECI_AWS_ACCESS_KEY_FOR_SCCACHE_AND_XLA_BAZEL_S3_BUCKET_V2}" >> /home/circleci/project/env
    echo "declare -x AWS_SECRET_ACCESS_KEY=${CIRCLECI_AWS_SECRET_KEY_FOR_SCCACHE_AND_XLA_BAZEL_S3_BUCKET_V2}" >> /home/circleci/project/env
    set -x
  else
    # This IAM user allows write access to S3 bucket for sccache
    set +x
    echo "declare -x AWS_ACCESS_KEY_ID=${CIRCLECI_AWS_ACCESS_KEY_FOR_SCCACHE_S3_BUCKET_V4}" >> /home/circleci/project/env
    echo "declare -x AWS_SECRET_ACCESS_KEY=${CIRCLECI_AWS_SECRET_KEY_FOR_SCCACHE_S3_BUCKET_V4}" >> /home/circleci/project/env
    set -x
  fi
fi

# This IAM user only allows read-write access to ECR
set +x
export AWS_ACCESS_KEY_ID=${CIRCLECI_AWS_ACCESS_KEY_FOR_ECR_READ_WRITE_V4}
export AWS_SECRET_ACCESS_KEY=${CIRCLECI_AWS_SECRET_KEY_FOR_ECR_READ_WRITE_V4}
eval $(aws ecr get-login --region us-east-1 --no-include-email)
set -x
