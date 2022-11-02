#!/bin/bash

# Common prelude for macos-build.sh and macos-test.sh

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

sysctl -a | grep machdep.cpu

# These are required for both the build job and the test job.
# In the latter to test cpp extensions.
export MACOSX_DEPLOYMENT_TARGET=10.9
export CXX=clang++
export CC=clang
