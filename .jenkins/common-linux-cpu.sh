#!/bin/bash

# Common setup for CPU Linux scripts.  Called from build.sh and test.sh
# when cuda is not in BUILD_ENVIRONMENT

if [[ "$BUILD_ENVIRONMENT" == *-py* ]]; then
  TRAVIS_PYTHON_VERSION="$(echo "${BUILD_ENVIRONMENT}" | perl -n -e'/-py([^-]+)/ && print $1')"
fi

GCC_VERSION=5
if [[ "$BUILD_ENVIRONMENT" == *-gcc* ]]; then
  GCC_VERSION="$(echo "${BUILD_ENVIRONMENT}" | perl -n -e'/-gcc([^-]+)/ && print $1')"
fi
if [[ "$GCC_VERSION" == 5.4 ]]; then
  GCC_VERSION=5
fi
if [[ "$GCC_VERSION" == 7.2 ]]; then
  GCC_VERSION=7
fi
