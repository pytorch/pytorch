#!/usr/bin/env bash
CHANGES=$(git status --porcelain)
echo "$CHANGES"
git diff
[ -z "$CHANGES" ]
