#!/usr/bin/env bash

set -euo pipefail

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

CHANNEL=${CHANNEL:-pytorch-nightly}
PACKAGES=${PACKAGES:-pytorch}

for pkg in ${PACKAGES}; do
    echo "+ Attempting to prune: ${CHANNEL}/${pkg}"
    CHANNEL="${CHANNEL}" PKG="${pkg}" "${DIR}/prune.sh"
    echo
done
