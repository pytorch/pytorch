#!/bin/bash
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

g++ \
  -g3 \
  -O0 \
  kineto_playground.cpp \
  -o main \
  -I/usr/local/cuda/include \
  -I../third_party/fmt/include \
  -I/usr/local/include/kineto \
  -L/usr/local/lib \
  -L/usr/local/cuda/lib64 \
  -lpthread \
  -lcuda \
  -lcudart \
  -lcupti \
  /usr/local/lib/libkineto.a \
  kplay_cu.o
