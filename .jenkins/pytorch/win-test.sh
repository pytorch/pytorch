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


mkdir -p $TMP_DIR/build/torch


# This directory is used only to hold "pytorch_env_restore.bat", called via "setup_pytorch_env.bat"
CI_SCRIPTS_DIR=$TMP_DIR/ci_scripts
mkdir -p $CI_SCRIPTS_DIR

if [ -n "$(ls $CI_SCRIPTS_DIR/*)" ]; then
    rm $CI_SCRIPTS_DIR/*
fi


export SCRIPT_HELPERS_DIR=$SCRIPT_PARENT_DIR/win-test-helpers

if [ -n "$CIRCLE_PULL_REQUEST" ]; then
  DETERMINE_FROM="${TMP_DIR}/determine_from"
  file_diff_from_base "$DETERMINE_FROM"
fi

run_tests() {
    if [ -z "${JOB_BASE_NAME}" ] || [[ "${JOB_BASE_NAME}" == *-test ]]; then
        $SCRIPT_HELPERS_DIR/test_python_nn.bat "$DETERMINE_FROM" && \
        $SCRIPT_HELPERS_DIR/test_python_all_except_nn.bat "$DETERMINE_FROM" && \
        $SCRIPT_HELPERS_DIR/test_custom_script_ops.bat && \
        $SCRIPT_HELPERS_DIR/test_libtorch.bat
    else
        if [[ "${JOB_BASE_NAME}" == *-test1 ]]; then
            $SCRIPT_HELPERS_DIR/test_python_nn.bat "$DETERMINE_FROM"
        elif [[ "${JOB_BASE_NAME}" == *-test2 ]]; then
            $SCRIPT_HELPERS_DIR/test_python_all_except_nn.bat "$DETERMINE_FROM" && \
            $SCRIPT_HELPERS_DIR/test_custom_script_ops.bat && \
            $SCRIPT_HELPERS_DIR/test_libtorch.bat
        fi
    fi
}

run_tests && assert_git_not_dirty && echo "TEST PASSED"
