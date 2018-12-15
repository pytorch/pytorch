#!/bin/bash
set -eu

# Run clang-format on whitelisted files and return non-zero if it
# introduced a diff

DIFF_AGAINST=HEAD
# From https://docs.travis-ci.com/user/environment-variables
if [[ $TRAVIS ]]; then
  git remote add upstream https://github.com/pytorch/pytorch
  git fetch upstream "$TRAVIS_BRANCH"
  DIFF_AGAINST="upstream/$TRAVIS_BRANCH"
fi

time python tools/clang_format.py --verbose --diff "$DIFF_AGAINST" --non-interactive

