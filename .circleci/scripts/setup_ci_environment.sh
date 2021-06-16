#!/usr/bin/env bash
set -ex -o pipefail

# Remove unnecessary sources
sudo rm -f /etc/apt/sources.list.d/google-chrome.list
sudo rm -f /etc/apt/heroku.list
sudo rm -f /etc/apt/openjdk-r-ubuntu-ppa-xenial.list
sudo rm -f /etc/apt/partner.list

retry () {
  $*  || $* || $* || $* || $*
}

# Method adapted from here: https://askubuntu.com/questions/875213/apt-get-to-retry-downloading
# (with use of tee to avoid permissions problems)
# This is better than retrying the whole apt-get command
echo "APT::Acquire::Retries \"3\";" | sudo tee /etc/apt/apt.conf.d/80-retries

retry sudo apt-get update -qq
retry sudo apt-get -y install \
  moreutils \
  expect-dev

echo "== DOCKER VERSION =="
docker version

if ! command -v aws >/dev/null; then
  retry sudo pip3 -q install awscli==1.19.64
fi

if [ -n "${USE_CUDA_DOCKER_RUNTIME:-}" ]; then
  DRIVER_FN="NVIDIA-Linux-x86_64-460.39.run"
  wget "https://s3.amazonaws.com/ossci-linux/nvidia_driver/$DRIVER_FN"
  sudo /bin/bash "$DRIVER_FN" -s --no-drm || (sudo cat /var/log/nvidia-installer.log && false)
  nvidia-smi

  # Taken directly from https://github.com/NVIDIA/nvidia-docker
  # Add the package repositories
  distribution=$(. /etc/os-release;echo "$ID$VERSION_ID")
  curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | sudo apt-key add -
  curl -s -L "https://nvidia.github.io/nvidia-docker/${distribution}/nvidia-docker.list" | sudo tee /etc/apt/sources.list.d/nvidia-docker.list

  sudo apt-get update -qq
  # Necessary to get the `--gpus` flag to function within docker
  sudo apt-get install -y nvidia-container-toolkit
  sudo systemctl restart docker
else
  # Explicitly remove nvidia docker apt repositories if not building for cuda
  sudo rm -rf /etc/apt/sources.list.d/nvidia-docker.list
fi

add_to_env_file() {
  local name=$1
  local value=$2
  case "$value" in
    *\ *)
      # BASH_ENV should be set by CircleCI
      echo "${name}='${value}'" >> "${BASH_ENV:-/tmp/env}"
      ;;
    *)
      echo "${name}=${value}" >> "${BASH_ENV:-/tmp/env}"
      ;;
  esac
}

add_to_env_file IN_CI 1
add_to_env_file COMMIT_SOURCE "${CIRCLE_BRANCH:-}"
add_to_env_file BUILD_ENVIRONMENT "${BUILD_ENVIRONMENT}"
add_to_env_file CIRCLE_PULL_REQUEST "${CIRCLE_PULL_REQUEST}"


if [[ "${BUILD_ENVIRONMENT}" == *-build ]]; then
  add_to_env_file SCCACHE_BUCKET ossci-compiler-cache-circleci-v2

  SCCACHE_MAX_JOBS=$(( $(nproc) - 1 ))
  MEMORY_LIMIT_MAX_JOBS=8  # the "large" resource class on CircleCI has 32 CPU cores, if we use all of them we'll OOM
  MAX_JOBS=$(( ${SCCACHE_MAX_JOBS} > ${MEMORY_LIMIT_MAX_JOBS} ? ${MEMORY_LIMIT_MAX_JOBS} : ${SCCACHE_MAX_JOBS} ))
  add_to_env_file MAX_JOBS "${MAX_JOBS}"

  if [ -n "${USE_CUDA_DOCKER_RUNTIME:-}" ]; then
    add_to_env_file TORCH_CUDA_ARCH_LIST 5.2
  fi

  if [[ "${BUILD_ENVIRONMENT}" == *xla* ]]; then
    # This IAM user allows write access to S3 bucket for sccache & bazels3cache
    set +x
    add_to_env_file XLA_CLANG_CACHE_S3_BUCKET_NAME "${XLA_CLANG_CACHE_S3_BUCKET_NAME:-}"
    add_to_env_file AWS_ACCESS_KEY_ID "${CIRCLECI_AWS_ACCESS_KEY_FOR_SCCACHE_AND_XLA_BAZEL_S3_BUCKET_V2:-}"
    add_to_env_file AWS_SECRET_ACCESS_KEY "${CIRCLECI_AWS_SECRET_KEY_FOR_SCCACHE_AND_XLA_BAZEL_S3_BUCKET_V2:-}"
    set -x
  else
    # This IAM user allows write access to S3 bucket for sccache
    set +x
    add_to_env_file XLA_CLANG_CACHE_S3_BUCKET_NAME "${XLA_CLANG_CACHE_S3_BUCKET_NAME:-}"
    add_to_env_file AWS_ACCESS_KEY_ID "${CIRCLECI_AWS_ACCESS_KEY_FOR_SCCACHE_S3_BUCKET_V4:-}"
    add_to_env_file AWS_SECRET_ACCESS_KEY "${CIRCLECI_AWS_SECRET_KEY_FOR_SCCACHE_S3_BUCKET_V4:-}"
    set -x
  fi
fi

# This IAM user only allows read-write access to ECR
set +x
export AWS_ACCESS_KEY_ID=${CIRCLECI_AWS_ACCESS_KEY_FOR_ECR_READ_WRITE_V4:-}
export AWS_SECRET_ACCESS_KEY=${CIRCLECI_AWS_SECRET_KEY_FOR_ECR_READ_WRITE_V4:-}
export AWS_ACCOUNT_ID=$(aws sts get-caller-identity|grep Account|cut -f4 -d\")
export AWS_REGION=us-east-1
aws ecr get-login-password --region $AWS_REGION|docker login --username AWS --password-stdin $AWS_ACCOUNT_ID.dkr.ecr.$AWS_REGION.amazonaws.com
set -x
