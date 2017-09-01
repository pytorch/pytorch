#!/bin/sh
set -ex
# Assumed to be run like tools/gen_toffee.sh
(cd torch/lib/nanopb/generator/proto && make)
# It always searches the same dir as the proto, so
# we have got to copy the option file over
cp torch/csrc/toffee.options torch/lib/ToffeeIR/toffee/toffee.options
protoc --plugin=protoc-gen-nanopb=$PWD/torch/lib/nanopb/generator/protoc-gen-nanopb \
       torch/lib/ToffeeIR/toffee/toffee.proto \
       --nanopb_out=-T:.
# NB: -T suppresses timestamp. See https://github.com/nanopb/nanopb/issues/274
# nanopb generated C files are valid CPP! Yay!
cp torch/lib/ToffeeIR/toffee/toffee.pb.c torch/csrc/toffee.pb.cpp
cp torch/lib/ToffeeIR/toffee/toffee.pb.h torch/csrc/toffee.pb.h
