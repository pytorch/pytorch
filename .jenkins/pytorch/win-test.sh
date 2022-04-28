#!/bin/bash
set -ex
# shellcheck disable=SC2034
COMPACT_JOB_NAME=pytorch-win-ws2019-cuda10.1-py3-test

SCRIPT_PARENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# shellcheck source=./common.sh
source "$SCRIPT_PARENT_DIR/common.sh"

IMAGE_COMMIT_ID=$(git rev-parse HEAD)
export IMAGE_COMMIT_ID
export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}
if [[ ${JOB_NAME} == *"develop"* ]]; then
  export IMAGE_COMMIT_TAG=develop-${IMAGE_COMMIT_TAG}
fi

export TMP_DIR="${PWD}/build/win_tmp"
TMP_DIR_WIN=$(cygpath -w "${TMP_DIR}")
export TMP_DIR_WIN
export PROJECT_DIR="${PWD}"
PROJECT_DIR_WIN=$(cygpath -w "${PROJECT_DIR}")
export PROJECT_DIR_WIN
export TEST_DIR="${PWD}/test"
TEST_DIR_WIN=$(cygpath -w "${TEST_DIR}")
export TEST_DIR_WIN
export PYTORCH_FINAL_PACKAGE_DIR="${PYTORCH_FINAL_PACKAGE_DIR:-/c/users/circleci/workspace/build-results}"
PYTORCH_FINAL_PACKAGE_DIR_WIN=$(cygpath -w "${PYTORCH_FINAL_PACKAGE_DIR}")
export PYTORCH_FINAL_PACKAGE_DIR_WIN

mkdir -p "$TMP_DIR"/build/torch


# This directory is used only to hold "pytorch_env_restore.bat", called via "setup_pytorch_env.bat"
CI_SCRIPTS_DIR=$TMP_DIR/ci_scripts
mkdir -p "$CI_SCRIPTS_DIR"

if [ -n "$(ls "$CI_SCRIPTS_DIR"/*)" ]; then
    rm "$CI_SCRIPTS_DIR"/*
fi


export SCRIPT_HELPERS_DIR=$SCRIPT_PARENT_DIR/win-test-helpers

if [[ "${BUILD_ENVIRONMENT}" == *cuda11* ]]; then
  export BUILD_SPLIT_CUDA=ON
fi

if [[ "$TEST_CONFIG" = "force_on_cpu" ]]; then
  # run the full test suite for force_on_cpu test
  export USE_CUDA=0
fi

run_tests() {
    # Run nvidia-smi if available
    for path in '/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe' /c/Windows/System32/nvidia-smi.exe; do
        if [[ -x "$path" ]]; then
            "$path" || echo "true";
            break
        fi
    done

    if [[ ( -z "${JOB_BASE_NAME}" || "${JOB_BASE_NAME}" == *-test ) && $NUM_TEST_SHARDS -eq 1 ]]; then
        "$SCRIPT_HELPERS_DIR"/test_python.bat
        "$SCRIPT_HELPERS_DIR"/test_custom_script_ops.bat
        "$SCRIPT_HELPERS_DIR"/test_custom_backend.bat
        "$SCRIPT_HELPERS_DIR"/test_libtorch.bat
    else
        if [[ "${JOB_BASE_NAME}" == *-test1 || ("${SHARD_NUMBER}" == 1 && $NUM_TEST_SHARDS -gt 1) ]]; then
            "$SCRIPT_HELPERS_DIR"/test_python_first_shard.bat
            "$SCRIPT_HELPERS_DIR"/test_libtorch.bat
            if [[ "${USE_CUDA}" == "1" ]]; then
              "$SCRIPT_HELPERS_DIR"/test_python_jit_legacy.bat
            fi
        elif [[ "${JOB_BASE_NAME}" == *-test2 || ("${SHARD_NUMBER}" == 2 && $NUM_TEST_SHARDS -gt 1) ]]; then
            "$SCRIPT_HELPERS_DIR"/test_python_second_shard.bat
            "$SCRIPT_HELPERS_DIR"/test_custom_backend.bat
            "$SCRIPT_HELPERS_DIR"/test_custom_script_ops.bat
        fi
    fi
}

run_tests
assert_git_not_dirty
echo "TEST PASSED"
