#!/bin/bash

set -ex

BASE_BRANCH=master
if [[ $SYSTEM_PULLREQUEST_TARGETBRANCH ]]; then
  git remote add upstream https://github.com/pytorch/pytorch
  git fetch upstream "$SYSTEM_PULLREQUEST_TARGETBRANCH"
  BASE_BRANCH="upstream/$SYSTEM_PULLREQUEST_TARGETBRANCH"
fi

# Run clang-format.
# Exits with non-zero status if clang-format generated changes.
time python tools/clang_format.py \
  --verbose                       \
  --diff "$BASE_BRANCH"           \
  "$@"
