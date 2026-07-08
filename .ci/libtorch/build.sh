#!/usr/bin/env bash

# This is mostly just a shim to wheel/linux/build.sh
# TODO: Make this a dedicated script to build just libtorch

set -ex

SCRIPTPATH="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

USE_NVSHMEM=0 USE_CUSPARSELT=0 BUILD_PYTHONLESS=1 DESIRED_PYTHON="3.10" ${SCRIPTPATH}/../wheel/linux/build.sh
