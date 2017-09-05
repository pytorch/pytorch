#!/bin/sh
set -ex
# Assumed to be run like tools/gen_onnx.sh
(cd torch/lib/nanopb/generator/proto && make)
# It always searches the same dir as the proto, so
# we have got to copy the option file over
cp torch/csrc/onnx.options torch/lib/ONNXIR/onnx/onnx.options
protoc --plugin=protoc-gen-nanopb=$PWD/torch/lib/nanopb/generator/protoc-gen-nanopb \
       torch/lib/ONNXIR/onnx/onnx.proto \
       --nanopb_out=-T:.
# NB: -T suppresses timestamp. See https://github.com/nanopb/nanopb/issues/274
# nanopb generated C files are valid CPP! Yay!
cp torch/lib/ONNXIR/onnx/onnx.pb.c torch/csrc/onnx.pb.cpp
cp torch/lib/ONNXIR/onnx/onnx.pb.h torch/csrc/onnx.pb.h
