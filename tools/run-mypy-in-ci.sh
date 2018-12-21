#!/bin/bash

set -ex

BASE_BRANCH=master
# From https://docs.travis-ci.com/user/environment-variables
if [[ $TRAVIS ]]; then
  git remote add upstream https://github.com/pytorch/pytorch
  git fetch upstream "$TRAVIS_BRANCH"
  BASE_BRANCH="upstream/$TRAVIS_BRANCH"
fi

echo $FETCH_HEAD
echo $BASE_BRANCH   # upstream/master
echo $TRAVIS_BRANCH # master
CHANGED_PYTHON_FILES=$(git diff --name-only $BASE_BRANCH | grep -E ".py\$" | tr '\n' ' ')
echo $(git diff --name-only $BASE_BRANCH)
if [[ $CHANGED_PYTHON_FILES ]]; then
  time mypy $CHANGED_PYTHON_FILES
else
  echo "no Python files changed, skipping mypy"
fi
