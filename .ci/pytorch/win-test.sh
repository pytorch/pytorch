#!/bin/bash
set -ex -o pipefail

SCRIPT_PARENT_DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )
# shellcheck source=./common.sh
source "$SCRIPT_PARENT_DIR/common.sh"

export TMP_DIR="${PWD}/build/win_tmp"
TMP_DIR_WIN=$(cygpath -w "${TMP_DIR}")
export TMP_DIR_WIN
export PROJECT_DIR="${PWD}"
PROJECT_DIR_WIN=$(cygpath -w "${PROJECT_DIR}")
export PROJECT_DIR_WIN
export TEST_DIR="${PWD}/test"
TEST_DIR_WIN=$(cygpath -w "${TEST_DIR}")
export TEST_DIR_WIN
export PYTORCH_FINAL_PACKAGE_DIR="${PYTORCH_FINAL_PACKAGE_DIR:-/c/w/build-results}"
PYTORCH_FINAL_PACKAGE_DIR_WIN=$(cygpath -w "${PYTORCH_FINAL_PACKAGE_DIR}")
export PYTORCH_FINAL_PACKAGE_DIR_WIN

# enable debug asserts in serialization
export TORCH_SERIALIZATION_DEBUG=1

mkdir -p "$TMP_DIR"/build/torch

export SCRIPT_HELPERS_DIR=$SCRIPT_PARENT_DIR/win-test-helpers

if [[ "$TEST_CONFIG" = "force_on_cpu" ]]; then
  # run the full test suite for force_on_cpu test
  export USE_CUDA=0
fi

if [[ "$BUILD_ENVIRONMENT" == *cuda* ]]; then
  # Used so that only cuda/rocm specific versions of tests are generated
  # mainly used so that we're not spending extra cycles testing cpu
  # devices on expensive gpu machines
  export PYTORCH_TESTING_DEVICE_ONLY_FOR="cuda"
fi

# TODO: Move both of them to Windows AMI
python -m pip install pytest-rerunfailures==10.3 pytest-cpp==2.3.0 tensorboard==2.13.0 protobuf==5.29.4 pytest-subtests==0.13.1

# Install Z3 optional dependency for Windows builds.
python -m pip install z3-solver==4.15.1.0

# Install tlparse for test\dynamo\test_structured_trace.py UTs.
python -m pip install tlparse==0.3.30

# Install parameterized
python -m pip install parameterized==0.8.1

# Install pulp for testing ilps under torch\distributed\_tools
python -m pip install pulp==2.9.0

# Install expecttest to merge https://github.com/pytorch/pytorch/pull/155308
python -m pip install expecttest==0.3.0

run_tests() {
    # Run nvidia-smi if available
    for path in '/c/Program Files/NVIDIA Corporation/NVSMI/nvidia-smi.exe' /c/Windows/System32/nvidia-smi.exe; do
        if [[ -x "$path" ]]; then
            "$path" || echo "true";
            break
        fi
    done

    if [[ $NUM_TEST_SHARDS -eq 1 ]]; then
        "$SCRIPT_HELPERS_DIR"/test_python_shard.bat
        "$SCRIPT_HELPERS_DIR"/test_custom_script_ops.bat
        "$SCRIPT_HELPERS_DIR"/test_custom_backend.bat
        "$SCRIPT_HELPERS_DIR"/test_libtorch.bat
    else
        "$SCRIPT_HELPERS_DIR"/test_python_shard.bat
        if [[ "${SHARD_NUMBER}" == 1 && $NUM_TEST_SHARDS -gt 1 ]]; then
            "$SCRIPT_HELPERS_DIR"/test_libtorch.bat
            if [[ "${USE_CUDA}" == "1" ]]; then
              "$SCRIPT_HELPERS_DIR"/test_python_jit_legacy.bat
            fi
        elif [[ "${SHARD_NUMBER}" == 2 && $NUM_TEST_SHARDS -gt 1 ]]; then
            "$SCRIPT_HELPERS_DIR"/test_custom_backend.bat
            "$SCRIPT_HELPERS_DIR"/test_custom_script_ops.bat
        fi
    fi
}

run_tests
assert_git_not_dirty
echo "TEST PASSED"
