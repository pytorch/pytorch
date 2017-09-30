#!/bin/sh

# This script should be executed in pytorch root folder.

TEMP_DIR=tools/temp

set -ex
# Assumed to be run like tools/gen_onnx.sh
(cd torch/lib/nanopb/generator/proto && make)
# It always searches the same dir as the proto, so
# we have got to copy the option file over
mkdir -p $TEMP_DIR
cp torch/csrc/onnx/onnx.options $TEMP_DIR/onnx.options
wget https://raw.githubusercontent.com/onnx/onnx/master/onnx/onnx.proto -O $TEMP_DIR/onnx.proto
protoc --plugin=protoc-gen-nanopb=$PWD/torch/lib/nanopb/generator/protoc-gen-nanopb \
       $TEMP_DIR/onnx.proto \
       --nanopb_out=-T:.
# NB: -T suppresses timestamp. See https://github.com/nanopb/nanopb/issues/274
# nanopb generated C files are valid CPP! Yay!
cp $TEMP_DIR/onnx.pb.c torch/csrc/onnx/onnx.pb.cpp
cp $TEMP_DIR/onnx.pb.h torch/csrc/onnx/onnx.pb.h

rm -r $TEMP_DIR
