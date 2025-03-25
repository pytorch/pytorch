#!/bin/sh

if [ -z "$CXX" ]; then
    CXX="clang++"
    echo "Using system default C++ compiler: $CXX"
else
    echo "Using user-provided C++ compiler: $CXX"
fi

if [ -z "$TORCH_ROOT_DIR" ]; then
    echo "Error: The TORCH_ROOT_DIR environment variable must be set." >&2
    echo "Example: export TORCH_ROOT_DIR=/home/$USER/local/pytorch" >&2
    exit 1
fi

if [ $# -lt 2 ]; then
    echo "Usage: $0 <input file path> <output file path>."
    echo "Example Usage: $0 standalone_test.cpp standalone_test.out."
    exit 1
fi

# Building the wrapper
$CXX -I$TORCH_ROOT_DIR/build/aten/src -I$TORCH_ROOT_DIR/aten/src -I$TORCH_ROOT_DIR/build -I$TORCH_ROOT_DIR -I$TORCH_ROOT_DIR/build/caffe2/aten/src -I$TORCH_ROOT_DIR/torch/csrc/api -I$TORCH_ROOT_DIR/torch/csrc/api/include -std=gnu++17 -fPIE -o $1.o -c $1

# Linking
$CXX -rdynamic -Wl,--no-as-needed,$TORCH_ROOT_DIR/build/lib/libtorch.so $1.o -Wl,--no-as-needed,$TORCH_ROOT_DIR/build/lib/libtorch_cpu.so -Wl,--no-as-needed,$TORCH_ROOT_DIR/build/lib/libc10.so -o $2
