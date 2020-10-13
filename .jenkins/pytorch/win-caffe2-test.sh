#!/bin/bash -ex

# shellcheck disable=SC2034
COMPACT_JOB_NAME=pytorch-win-ws2019-cuda10-cudnn7-py3-test

SCRIPT_PARENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
source "$SCRIPT_PARENT_DIR/common.sh"

export IMAGE_COMMIT_ID=`git rev-parse HEAD`
export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}
if [[ ${JOB_NAME} == *"develop"* ]]; then
  export IMAGE_COMMIT_TAG=develop-${IMAGE_COMMIT_TAG}
fi

export TMP_DIR="${PWD}/build/win_tmp"
export TMP_DIR_WIN=$(cygpath -w "${TMP_DIR}")
export PYTORCH_FINAL_PACKAGE_DIR="/c/users/circleci/workspace/build-results"
export PYTORCH_FINAL_PACKAGE_DIR_WIN=$(cygpath -w "${PYTORCH_FINAL_PACKAGE_DIR}")

mkdir -p $TMP_DIR/build/torch


# This directory is used only to hold "pytorch_env_restore.bat", called via "setup_pytorch_env.bat"
CI_SCRIPTS_DIR=$TMP_DIR/ci_scripts
mkdir -p $CI_SCRIPTS_DIR

if [ -n "$(ls $CI_SCRIPTS_DIR/*)" ]; then
    rm $CI_SCRIPTS_DIR/*
fi


export SCRIPT_HELPERS_DIR=$SCRIPT_PARENT_DIR/win-test-helpers
$SCRIPT_HELPERS_DIR/setup_pytorch_env.bat

if [ -n "$CIRCLE_PULL_REQUEST" ]; then
  DETERMINE_FROM="${TMP_DIR}/determine_from"
  file_diff_from_base "$DETERMINE_FROM"
fi


pytorch_installation="$(dirname $(dirname $(cd $TMP_DIR && python -c 'import os; import torch; print(os.path.realpath(torch.__file__))')))"
python_installation="$(dirname $(dirname $(cd $TMP_DIR && python -c 'import os; import caffe2; print(os.path.realpath(caffe2.__file__))')))"
caffe2_pypath="$python_installation/caffe2"

run_tests() {
    if [[ "${JOB_BASE_NAME}" == *-test3 ]]; then
        python \
            -m pytest \
            -x \
            -v \
            --disable-warnings \
            --junit-xml="$TMP_DIR/result.xml" \
            --ignore "$caffe2_pypath/python/test/executor_test.py" \
            --ignore "$caffe2_pypath/python/operator_test/matmul_op_test.py" \
            --ignore "$caffe2_pypath/python/operator_test/pack_ops_test.py" \
            --ignore "$caffe2_pypath/python/mkl/mkl_sbn_speed_test.py" \
            --ignore "$caffe2_pypath/python/trt/test_pt_onnx_trt.py" \
            "$caffe2_pypath/python/operator_test/" 
    fi
}

run_tests && assert_git_not_dirty && echo "TEST PASSED"
