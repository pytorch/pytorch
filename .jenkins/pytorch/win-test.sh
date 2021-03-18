#!/bin/bash
set -ex
# shellcheck disable=SC2034
COMPACT_JOB_NAME=pytorch-win-ws2019-cuda10-cudnn7-py3-test

SCRIPT_PARENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# shellcheck source=./common.sh
source "$SCRIPT_PARENT_DIR/common.sh"

export IMAGE_COMMIT_ID=$(git rev-parse HEAD)
export IMAGE_COMMIT_TAG=${BUILD_ENVIRONMENT}-${IMAGE_COMMIT_ID}
if [[ ${JOB_NAME} == *"develop"* ]]; then
  export IMAGE_COMMIT_TAG=develop-${IMAGE_COMMIT_TAG}
fi

export TMP_DIR="${PWD}/build/win_tmp"
export TMP_DIR_WIN=$(cygpath -w "${TMP_DIR}")
export PROJECT_DIR="${PWD}"
export PROJECT_DIR_WIN=$(cygpath -w "${PROJECT_DIR}")
export TEST_DIR="${PWD}/test"
export TEST_DIR_WIN=$(cygpath -w "${TEST_DIR}")
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

# Try to pull value from CIRCLE_PULL_REQUEST first then GITHUB_HEAD_REF second
# CIRCLE_PULL_REQUEST comes from CircleCI
# GITHUB_HEAD_REF comes from Github Actions
IN_PULL_REQUEST=${CIRCLE_PULL_REQUEST:-${GITHUB_HEAD_REF:-}}
if [ -n "$IN_PULL_REQUEST" ]; then
  DETERMINE_FROM="${TMP_DIR}/determine_from"
  file_diff_from_base "$DETERMINE_FROM"
fi

if [[ "${BUILD_ENVIRONMENT}" == *cuda11* ]]; then
  export BUILD_SPLIT_CUDA=ON
fi

run_tests() {
    # Run nvidia-smi if available
    for path in  /c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe /c/Windows/System32/nvidia-smi.exe; do
        if [ -x $path ]; then
            $path;
            break
        fi
    done

    if [ -z "${JOB_BASE_NAME}" ] || [[ "${JOB_BASE_NAME}" == *-test ]]; then
        $SCRIPT_HELPERS_DIR/test_python_nn.bat "$DETERMINE_FROM"
        $SCRIPT_HELPERS_DIR/test_python_all_except_nn.bat "$DETERMINE_FROM"
        $SCRIPT_HELPERS_DIR/test_custom_script_ops.bat
        $SCRIPT_HELPERS_DIR/test_custom_backend.bat
        $SCRIPT_HELPERS_DIR/test_libtorch.bat
    else
        export PYTORCH_COLLECT_COVERAGE=1
        if [[ "${JOB_BASE_NAME}" == *-test1 ]]; then
            $SCRIPT_HELPERS_DIR/test_python_first_shard.bat "$DETERMINE_FROM"
            $SCRIPT_HELPERS_DIR/test_libtorch.bat
            if [[ "${USE_CUDA}" == "1" ]]; then
              $SCRIPT_HELPERS_DIR/test_python_jit_legacy.bat "$DETERMINE_FROM"
            fi
        elif [[ "${JOB_BASE_NAME}" == *-test2 ]]; then
            $SCRIPT_HELPERS_DIR/test_python_second_shard.bat "$DETERMINE_FROM"
            $SCRIPT_HELPERS_DIR/test_custom_backend.bat
            $SCRIPT_HELPERS_DIR/test_custom_script_ops.bat
        fi
    fi
}

run_tests
assert_git_not_dirty
echo "TEST PASSED"

if [[ "${BUILD_ENVIRONMENT}" == "pytorch-win-vs2019-cuda10-cudnn7-py3" ]]; then
  pushd $TEST_DIR
  python -mpip install coverage
  echo "Generating XML coverage report"
  time python -mcoverage xml
  popd

  pushd $PROJECT_DIR
  python -mpip install codecov
  python -mcodecov
  popd
fi
