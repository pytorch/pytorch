#!/bin/bash
# Git clone for RE runner
set -eu

: "${GIT_REPO:=}"
: "${CLONE_DEPTH:=1}"

if [[ -n "$GIT_REPO" ]]; then
    echo "[Runner] Cloning $GIT_REPO..."
    git clone --depth="$CLONE_DEPTH" "$GIT_REPO" repo
    cd repo
    REPO_DIR="$(pwd)"
    export REPO_DIR
    echo "[Runner] REPO_DIR=$REPO_DIR"
else
    echo "[Runner] No git repo specified, skipping clone"
fi
