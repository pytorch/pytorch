#!/bin/bash
set -eux -o pipefail

retry () {
    $*  || (sleep 1 && $*) || (sleep 2 && $*) || (sleep 4 && $*) || (sleep 8 && $*)
}


# This step runs on multiple executors with different envfile locations
if [[ "$(uname)" == Darwin ]]; then
  # macos executor (builds and tests)
  workdir="/Users/distiller/project"
elif [[ "$OSTYPE" == "msys" ]]; then
  # windows executor (builds and tests)
  rm -rf /c/w
  ln -s "/c/Users/circleci/project" /c/w
  workdir="/c/w"
elif [[ -d "/home/circleci/project" ]]; then
  # machine executor (binary tests)
  workdir="/home/circleci/project"
else
  # docker executor (binary builds)
  workdir="/"
fi

# It is very important that this stays in sync with binary_populate_env.sh
if [[ "$OSTYPE" == "msys" ]]; then
  # We need to make the paths as short as possible on Windows
  export PYTORCH_ROOT="$workdir/p"
  export BUILDER_ROOT="$workdir/b"
else
  export PYTORCH_ROOT="$workdir/pytorch"
  export BUILDER_ROOT="$workdir/builder"
fi

# Try to extract PR number from branch if not already set
if [[ -z "${CIRCLE_PR_NUMBER:-}" ]]; then
  CIRCLE_PR_NUMBER="$(echo ${CIRCLE_BRANCH} | sed -E -n 's/pull\/([0-9]*).*/\1/p')"
fi

# Clone the Pytorch branch
retry git clone https://github.com/pytorch/pytorch.git "$PYTORCH_ROOT"
pushd "$PYTORCH_ROOT"
if [[ -n "${CIRCLE_PR_NUMBER:-}" ]]; then
  # "smoke" binary build on PRs
  git fetch --force origin "pull/${CIRCLE_PR_NUMBER}/head:remotes/origin/pull/${CIRCLE_PR_NUMBER}"
  git reset --hard "$CIRCLE_SHA1"
  git checkout -q -B "$CIRCLE_BRANCH"
  git reset --hard "$CIRCLE_SHA1"
elif [[ -n "${CIRCLE_SHA1:-}" ]]; then
  # Scheduled workflows & "smoke" binary build on master on PR merges
  git reset --hard "$CIRCLE_SHA1"
  git checkout -q -B master
else
  echo "Can't tell what to checkout"
  exit 1
fi
retry git submodule update --init --recursive
echo "Using Pytorch from "
git --no-pager log --max-count 1
popd

# Clone the Builder master repo
retry git clone -q https://github.com/pytorch/builder.git "$BUILDER_ROOT"
git checkout release/1.9
pushd "$BUILDER_ROOT"
echo "Using builder from "
git --no-pager log --max-count 1
popd
