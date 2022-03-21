#!/bin/bash
set -ex

# export PYTORCH_ROCM_ARCH=gfx1030

BUILD_DIR=/tmp/pytorch

cp_to_build_dir() {
    local CUR_FILE=$1
    chmod -R 777 $CUR_FILE
    cp -rf --parents $CUR_FILE $BUILD_DIR
}

build_develop() {
    pip uninstall torch -y

    cd $BUILD_DIR
    export MAX_JOBS=16
    python tools/amd_build/build_amd.py
    VERBOSE=1 USE_ROCM=1 python3 setup.py develop | tee BUILD_DEVELOP.log
}

if true; then
    # FILE_LIST=(
    #    "test/test_cuda.py"
    # )
    # for FILE in "${FILE_LIST[@]}"; do
    #     cp_to_build_dir $FILE
    # done

    # cd $BUILD_DIR/build
    # cmake --build . --target install --config Release -- -j 16
    build_develop
else
    build_develop
fi
