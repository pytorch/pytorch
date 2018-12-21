#!/bin/bash

set -ex

BASE_BRANCH=master
# From https://docs.travis-ci.com/user/environment-variables
if [[ $TRAVIS ]]; then
  git remote add upstream https://github.com/pytorch/pytorch
  git fetch upstream "$TRAVIS_BRANCH"
  BASE_BRANCH="upstream/$TRAVIS_BRANCH"
fi

echo $BASE_BRANCH
echo $TRAVIS_BRANCH
git status
CHANGED_PYTHON_FILES=$(git diff --name-only $BASE_BRANCH..$TRAVIS_BRANCH | grep -E ".py\$" | tr '\n' ' ')
if [[ $CHANGED_PYTHON_FILES ]]; then
  time mypy $CHANGED_PYTHON_FILES
else
  echo "no Python files changed, skipping mypy"
fi
