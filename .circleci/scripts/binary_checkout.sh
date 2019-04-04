#!/bin/bash

set -ex
# This step runs on multiple executors with different envfile locations
if [[ "$(uname)" == Darwin ]]; then
  source "/Users/distiller/project/env"
elif [[ -d "/home/circleci/project" ]]; then
  # machine executor (binary tests)
  source "/home/circleci/project/env"
else
  # docker executor (binary builds)
  source "/env"
fi

# Clone the Pytorch branch
git clone https://github.com/pytorch/pytorch.git "$PYTORCH_ROOT"
pushd "$PYTORCH_ROOT"
if [[ -n "$CIRCLE_PR_NUMBER" ]]; then
  # "smoke" binary build on PRs
  git fetch --force origin "pull/${CIRCLE_PR_NUMBER}/head:remotes/origin/pull/${CIRCLE_PR_NUMBER}"
  git reset --hard "$CIRCLE_SHA1"
  git checkout -q -B "$CIRCLE_BRANCH"
  git reset --hard "$CIRCLE_SHA1"
elif [[ -n "$CIRCLE_SHA1" ]]; then
  # "smoke" binary build on master on PR merges
  git reset --hard "$CIRCLE_SHA1"
  git checkout -q -B master
else
  # nightly binary builds. These run at 05:05 UTC every day. 
  last_commit="$(git rev-list --before "$(date -u +%Y-%m-%d) 05:00" --max-count 1 HEAD)"
  git checkout "$last_commit"
fi
git submodule update --init --recursive --quiet
echo "Using Pytorch from "
git --no-pager log --max-count 1
popd

# Clone the Builder master repo
git clone -q https://github.com/pytorch/builder.git "$BUILDER_ROOT"
pushd "$BUILDER_ROOT"
git fetch origin
git reset origin/master --hard
echo "Using builder from "
git --no-pager log --max-count 1
popd
