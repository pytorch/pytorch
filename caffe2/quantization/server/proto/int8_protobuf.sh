#!/bin/bash

set -e

(
    cd "$FBCODE_DIR"/
    protoc \
        caffe2/caffe2/quantization/server/proto/fbgemm_int8.proto \
        --cpp_out="$INSTALL_DIR" \
        --python_out="$INSTALL_DIR"
)
echo "$INSTALL_DIR"/caffe2/caffe2/quantization/server/proto/* "$INSTALL_DIR"
cp "$INSTALL_DIR"/caffe2/caffe2/quantization/server/proto/* "$INSTALL_DIR"
