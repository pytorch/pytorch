#!/usr/bin/env bash

set -eux

CHANGES=$(git status --porcelain "$1")
#echo "$CHANGES"
#git diff "$1"
[ -z "$CHANGES" ]
