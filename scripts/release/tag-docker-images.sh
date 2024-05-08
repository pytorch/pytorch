#!/usr/bin/env bash
#
# Step 1 after branch cut is complete.
#
# Tags latest docker images for release branch.
# In case of failure. The script can be rerun.
#
#  Before executing this script do:
#  1. Create and Check out to Release Branch
#  git checkout -b "${RELEASE_BRANCH}"
#  2. Update submodules
#  git submodule update --init --recursive
#
# Usage (run from root of project):
#  DRY_RUN=disabled ./scripts/release/tag_docker_images.sh
#

set -eou pipefail

GIT_TOP_DIR=$(git rev-parse --show-toplevel)
RELEASE_VERSION=${RELEASE_VERSION:-$(cut -d'.' -f1-2 "${GIT_TOP_DIR}/version.txt")}
DRY_RUN=${DRY_RUN:-enabled}

python3 .github/scripts/tag_docker_images_for_release.py --version ${RELEASE_VERSION} --dry-run ${DRY_RUN}
