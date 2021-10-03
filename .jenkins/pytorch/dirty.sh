#!/bin/bash
set -ex
upstream="$1"
pr="$2"
git diff --name-only "$upstream" "$pr"
# Now that PyTorch build depends on Caffe2, unconditionally trigger
# for any changes.
# TODO: Replace this with a NEGATIVE regex that allows us to skip builds when they are unnecessary
#git diff --name-only "$upstream" "$pr" | grep -Eq '^(aten/|caffe2/|.jenkins/pytorch|docs/(make.bat|Makefile|requirements.txt|source)|mypy|requirements.txt|setup.py|test/|third_party/|tools/|\.gitmodules|torch/)'
