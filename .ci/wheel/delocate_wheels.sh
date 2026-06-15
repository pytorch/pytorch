#!/usr/bin/env bash
# Delocate every .whl under $WHEELS_DIR in place, rewriting rpaths to
# @loader_path and bundling dependency dylibs into the wheel so it is
# self-contained. delocate is a macOS-only tool and is not tied to the
# interpreter the wheel targets, so one invocation handles every CPython
# version produced by the build loop. It is single-threaded per wheel, so
# we fan out across wheels with xargs -P.

set -eux -o pipefail

: "${WHEELS_DIR:=${RUNNER_TEMP:-/tmp}/artifacts}"

if ! command -v delocate-wheel >/dev/null; then
    pip install 'https://github.com/matthew-brett/delocate/archive/refs/tags/0.10.4.zip'
fi

find "${WHEELS_DIR}" -name '*.whl' -print0 |
    xargs -0 -n1 -P "$(sysctl -n hw.ncpu)" delocate-wheel -v
