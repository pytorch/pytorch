#!/bin/bash
set -eux -o pipefail
# This step runs on multiple executors with different envfile locations
if [[ "$(uname)" == Darwin ]]; then
  # macos executor (builds and tests)
  workdir="/Users/distiller/project"
elif [[ -d "/home/circleci/project" ]]; then
  # machine executor (binary tests)
  workdir="/home/circleci/project"
else
  # docker executor (binary builds)
  workdir="/"
fi

# It is very important that this stays in sync with binary_populate_env.sh
export PYTORCH_ROOT="$workdir/pytorch"
export BUILDER_ROOT="$workdir/builder"

# Clone the Pytorch branch
git clone https://github.com/pytorch/pytorch.git "$PYTORCH_ROOT" -b v1.2.0
pushd "$PYTORCH_ROOT"
git submodule update --init --recursive --quiet
echo "Using Pytorch from "
git --no-pager log --max-count 1
popd

# Clone the Builder master repo
git clone -q https://github.com/pytorch/builder.git "$BUILDER_ROOT"
pushd "$BUILDER_ROOT"
echo "Using builder from "
git --no-pager log --max-count 1
popd
