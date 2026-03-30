#!/bin/bash
set -eu
: "${REPO_URL:?REPO_URL must be set}"
: "${COMMIT_SHA:?COMMIT_SHA must be set}"

git clone --depth=1 "$REPO_URL" pytorch
cd pytorch
git fetch --depth=1 origin "$COMMIT_SHA"
git checkout "$COMMIT_SHA"
git submodule update --init --depth=1 --jobs=8
