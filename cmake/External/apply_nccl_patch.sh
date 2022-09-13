#!/bin/bash

# This patch is required to fix intermittent link errors when building
# NCCL. See https://github.com/pytorch/pytorch/issues/83790

TORCH_DIR=$1

# Only apply patch if "git status" is empty to avoid failing when the
# patch has already been applied
if [[ `git status --porcelain` == "" ]]; then
    git apply "${TORCH_DIR}/cmake/External/nccl.patch"
fi
