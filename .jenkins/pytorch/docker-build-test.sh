#!/bin/bash

COMPACT_JOB_NAME="docker-build-test"
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

docker build -t pytorch .
