#!/bin/bash
set -ex
upstream="$1"
pr="$2"
git diff --name-only "$upstream" "$pr" | grep -Eq '^(aten/|.jenkins/pytorch|docs/|mypy|requirements.txt|setup.py|test/|third_party/|tools/|\.gitmodules|torch/)'
