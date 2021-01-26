#!/usr/bin/env bash

set -eou pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
TOP_DIR="$(git rev-parse --show-toplevel)/"

BUILD_DIR=${BUILD_DIR:-${DIR}/manywheel}
DOCKER_COMMON=${DIR}/docker-common/
GIT_REV_PARSE="git rev-parse HEAD:"

echo $(${GIT_REV_PARSE}${BUILD_DIR/${TOP_DIR}/})-$(${GIT_REV_PARSE}${DOCKER_COMMON/${TOP_DIR}})
