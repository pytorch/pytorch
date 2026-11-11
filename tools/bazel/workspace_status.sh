#!/bin/bash
# Bazel runs this in the workspace root before every build.
#
# third_party/ is too large and irregular (nested BUILD files, a symlink loop)
# for Bazel to track file by file, so the native build depends on this digest
# of its file listing instead.
set -euo pipefail
listing=$(find third_party -type f -printf '%P %s %T@\n' | LC_ALL=C sort | sha256sum)
echo "STABLE_THIRD_PARTY_DIR $PWD/third_party"
echo "STABLE_THIRD_PARTY_DIGEST ${listing%% *}"
