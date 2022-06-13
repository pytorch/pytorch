#!/bin/bash

TORCH_INSTALL_DIR=$(python -c "import site; print(site.getsitepackages()[0])")/torch
export TORCH_INSTALL_DIR
export TORCH_BIN_DIR="$TORCH_INSTALL_DIR"/bin
export TORCH_LIB_DIR="$TORCH_INSTALL_DIR"/lib
export TORCH_TEST_DIR="$TORCH_INSTALL_DIR"/test

export BUILD_DIR="build"
export BUILD_RENAMED_DIR="build_renamed"
export BUILD_BIN_DIR="$BUILD_DIR"/bin
