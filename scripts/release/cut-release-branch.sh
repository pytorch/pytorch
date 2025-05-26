#!/usr/bin/env bash

: '
So you are looking to cut a release branch? Well you came
to the right script.

This script can be used to cut any branch on any repository

For `pytorch/pytorch` usage would be like:
> DRY_RUN=disabled cut-release-branch.sh

For `pytorch/builder` or domains usage would be like:
> DRY_RUN=disabled GIT_BRANCH_TO_CUT_FROM=main RELEASE_VERSION=1.11 cut-release-branch.sh
'

set -eou pipefail

GIT_TOP_DIR=$(git rev-parse --show-toplevel)
GIT_REMOTE=${GIT_REMOTE:-origin}
GIT_BRANCH_TO_CUT_FROM=${GIT_BRANCH_TO_CUT_FROM:-viable/strict}

# should output something like 1.11
RELEASE_VERSION=${RELEASE_VERSION:-$(cut -d'.' -f1-2 "${GIT_TOP_DIR}/version.txt")}

DRY_RUN_FLAG="--dry-run"
if [[ ${DRY_RUN:-enabled} == "disabled" ]]; then
    DRY_RUN_FLAG=""
fi


(
    set -x
    git fetch --all
    git checkout "${GIT_REMOTE}/${GIT_BRANCH_TO_CUT_FROM}"
)

for branch in "release/${RELEASE_VERSION}" "orig/release/${RELEASE_VERSION}"; do
    if git rev-parse --verify "${branch}" >/dev/null 2>/dev/null; then
        echo "+ Branch ${branch} already exists, skipping..."
        continue
    else
        (
            set -x
            git checkout "${GIT_REMOTE}/${GIT_BRANCH_TO_CUT_FROM}"
            git checkout -b "${branch}"
            git push -q ${DRY_RUN_FLAG} "${GIT_REMOTE}" "${branch}"
        )
    fi
done
