#!/bin/bash
set -ex
upstream="$1"
pr="$2"
git diff --name-only "$upstream" "$pr"
# For safety, unconditionally trigger for any changes.
#git diff --name-only "$upstream" "$pr" | grep -Eq '^(CMakeLists.txt|Makefile|.gitmodules|.jenkins/caffe2|binaries|caffe|caffe2|cmake|conda|docker|docs/caffe2|modules|scripts|third_party)'
