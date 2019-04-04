#!/bin/bash -e

COMPACT_JOB_NAME=pytorch-win-ws2016-cuda9-cudnn7-py3-test

SCRIPT_PARENT_DIR=$(dirname "${BASH_SOURCE[0]}")
source "$SCRIPT_PARENT_DIR/common.sh"

export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}
if [[ ${JOB_NAME} == *"develop"* ]]; then
  export IMAGE_COMMIT_TAG=develop-${IMAGE_COMMIT_TAG}
fi

export TMP_DIR="${PWD}/build/win_tmp"
export TMP_DIR_WIN=$(cygpath -w "${TMP_DIR}")

CI_SCRIPTS_DIR=$TMP_DIR/ci_scripts
mkdir -p $CI_SCRIPTS_DIR
mkdir -p $TMP_DIR/build/torch

if [ ! -z "$(ls $CI_SCRIPTS_DIR/*)" ]; then
    rm $CI_SCRIPTS_DIR/*
fi


SCRIPT_HELPERS_DIR=$SCRIPT_PARENT_DIR/win-test-helpers

# Used by setup_pytorch_env.bat:
cp $SCRIPT_HELPERS_DIR/download_image.py $CI_SCRIPTS_DIR

cp $SCRIPT_HELPERS_DIR/setup_pytorch_env.bat $CI_SCRIPTS_DIR
cp $SCRIPT_HELPERS_DIR/test_python_nn.bat $CI_SCRIPTS_DIR
cp $SCRIPT_HELPERS_DIR/test_python_all_except_nn.bat $CI_SCRIPTS_DIR
cp $SCRIPT_HELPERS_DIR/test_custom_script_ops.bat $CI_SCRIPTS_DIR
cp $SCRIPT_HELPERS_DIR/test_libtorch.bat $CI_SCRIPTS_DIR


run_tests() {
    if [ -z "${JOB_BASE_NAME}" ] || [[ "${JOB_BASE_NAME}" == *-test ]]; then
        $CI_SCRIPTS_DIR/test_python_nn.bat && \
        $CI_SCRIPTS_DIR/test_python_all_except_nn.bat && \
        $CI_SCRIPTS_DIR/test_custom_script_ops.bat && \
        $CI_SCRIPTS_DIR/test_libtorch.bat
    else
        if [[ "${JOB_BASE_NAME}" == *-test1 ]]; then
            $CI_SCRIPTS_DIR/test_python_nn.bat
        elif [[ "${JOB_BASE_NAME}" == *-test2 ]]; then
            $CI_SCRIPTS_DIR/test_python_all_except_nn.bat && \
            $CI_SCRIPTS_DIR/test_custom_script_ops.bat && \
            $CI_SCRIPTS_DIR/test_libtorch.bat
        fi
    fi
}

run_tests && assert_git_not_dirty && echo "TEST PASSED"
