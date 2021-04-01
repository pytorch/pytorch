#!/bin/bash

# shellcheck disable=SC2034
COMPACT_JOB_NAME="${BUILD_ENVIRONMENT}"

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Testing pytorch docs"

cd "${SCRIPT_DIR}/../../docs"
pip_install -r requirements.txt
make doctest
