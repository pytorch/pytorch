#!/bin/bash

# shellcheck disable=SC2034
COMPACT_JOB_NAME="docker-build-test"

source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

docker build -t pytorch .
