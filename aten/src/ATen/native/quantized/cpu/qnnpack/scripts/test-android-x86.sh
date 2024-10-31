#!/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

adb push build/android/x86/convolution-test /data/local/tmp/convolution-test
adb push build/android/x86/deconvolution-test /data/local/tmp/deconvolution-test
adb push build/android/x86/q8gemm-test /data/local/tmp/q8gemm-test
adb push build/android/x86/q8conv-test /data/local/tmp/q8conv-test
adb push build/android/x86/q8dw-test /data/local/tmp/q8dw-test
adb push build/android/x86/hgemm-test /data/local/tmp/hgemm-test
adb push build/android/x86/sgemm-test /data/local/tmp/sgemm-test

adb shell /data/local/tmp/convolution-test --gtest_color=yes
adb shell /data/local/tmp/deconvolution-test --gtest_color=yes
adb shell /data/local/tmp/q8gemm-test --gtest_color=yes
adb shell /data/local/tmp/q8conv-test --gtest_color=yes
adb shell /data/local/tmp/q8dw-test --gtest_color=yes
adb shell /data/local/tmp/hgemm-test --gtest_color=yes
adb shell /data/local/tmp/sgemm-test --gtest_color=yes
