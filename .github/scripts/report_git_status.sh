#!/usr/bin/env bash

set -eux

CHANGES=$(git status --porcelain "$1")
echo "$CHANGES"
# NB: Use --no-pager here to avoid git diff asking for a prompt to continue
git --no-pager diff "$1"
[ -z "$CHANGES" ]
