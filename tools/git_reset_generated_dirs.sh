#!/bin/sh

# HOW TO USE:
# 1) pip install -e . --no-build-isolation
# 2) tools/git_add_generated_dirs
# 3) Edit codegen
# 4) pip install -e . --no-build-isolation
# 5) git diff to see changes
# 6) If satisfied: tools/git_reset_generated_dirs, commit, etc.
#    If not satisfied: Go to 3)

BASEDIR=$(dirname "$0")
(< $BASEDIR/generated_dirs.txt xargs -i find {} -type f) | xargs git reset HEAD
