#!/bin/bash

if [ -z "${JOB_BASE_NAME}" ] || [[ "${JOB_BASE_NAME}" == *-build* ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/macos-build.sh"
fi

if [ -z "${JOB_BASE_NAME}" ] || [[ "${JOB_BASE_NAME}" == *-test* ]]; then
  source "$(dirname "${BASH_SOURCE[0]}")/macos-test.sh"
fi
