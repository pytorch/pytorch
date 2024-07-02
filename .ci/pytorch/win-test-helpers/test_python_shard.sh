#!/bin/bash
set -ex
source "$SCRIPT_HELPERS_DIR/setup_pytorch_env.sh"

pushd test

export GFLAGS_EXE="/c/Program Files (x86)/Windows Kits/10/Debuggers/x64/gflags.exe"
if [[ "${SHARD_NUMBER}" == "1" ]]; then

    if [ -f "${GFLAGS_EXE}" ]; then
        echo Some smoke tests
        "$GFLAGS_EXE" -i python.exe +sls
        python "$SCRIPT_HELPERS_DIR/run_python_nn_smoketests.py"

        "$GFLAGS_EXE" -i python.exe -sls
    fi
fi

echo Copying over additional ci files

cp -r "$PYTORCH_FINAL_PACKAGE_DIR"/.additional_ci_files "$PROJECT_DIR"

time python run_test.py --exclude-jit-executor --exclude-distributed-tests --shard "$SHARD_NUMBER" "$NUM_TEST_SHARDS" --verbose
popd
