#!/bin/bash
set -eux -o pipefail

source "${BINARY_ENV_FILE:-/c/w/env}"

pytorch/.ci/pytorch/windows/arm64/smoke_test.bat
