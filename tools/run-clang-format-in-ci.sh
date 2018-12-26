#!/bin/bash
set -e

# Run clang-format on whitelisted files and return non-zero if it
# introduced a diff

DIFF_AGAINST=HEAD
# From https://docs.travis-ci.com/user/environment-variables
if [[ $TRAVIS ]]; then
  git remote add upstream https://github.com/pytorch/pytorch
  git fetch upstream "$TRAVIS_BRANCH"
  DIFF_AGAINST="upstream/$TRAVIS_BRANCH"
fi

CLANG_FORMAT_DIFF=$(python tools/clang_format.py --diff "$DIFF_AGAINST")
if [[ ${CLANG_FORMAT_DIFF} ]]
then
  echo "${CLANG_FORMAT_DIFF}"
  exit 1
fi

