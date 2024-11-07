#!/usr/bin/env bash
#
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set -e

if [ -z "$IOS_CMAKE_TOOLCHAIN_FILE" ]
then
  echo "IOS_CMAKE_TOOLCHAIN_FILE not set; please set it to path of CMake toolchain file for iOS"
  exit 1
fi

if [ ! -f "$IOS_CMAKE_TOOLCHAIN_FILE" ]
then
  echo "IOS_CMAKE_TOOLCHAIN_FILE not a file path; did you properly setup ${IOS_CMAKE_TOOLCHAIN_FILE}?"
  exit 1
fi

mkdir -p build/ios/arm64e

CMAKE_ARGS=()

# CMake-level configuration
CMAKE_ARGS+=("-DCMAKE_TOOLCHAIN_FILE=$IOS_CMAKE_TOOLCHAIN_FILE")
CMAKE_ARGS+=("-DCMAKE_BUILD_TYPE=Release")
CMAKE_ARGS+=("-DCMAKE_POSITION_INDEPENDENT_CODE=ON")

# QNNPACK-specific options
CMAKE_ARGS+=("-DPYTORCH_QNNPACK_LIBRARY_TYPE=static")
CMAKE_ARGS+=("-DPYTORCH_QNNPACK_BUILD_BENCHMARKS=OFF")
CMAKE_ARGS+=("-DPYTORCH_QNNPACK_BUILD_TESTS=OFF")

# iOS-specific options
CMAKE_ARGS+=("-DIOS_PLATFORM=OS64")
CMAKE_ARGS+=("-DIOS_ARCH=arm64e")
CMAKE_ARGS+=("-DENABLE_BITCODE=OFF")
CMAKE_ARGS+=("-DENABLE_ARC=OFF")

# Use-specified CMake arguments go last to allow overridding defaults
CMAKE_ARGS+=($@)

cd build/ios/arm64e && cmake ../../.. \
    "${CMAKE_ARGS[@]}"

# Cross-platform parallel build
if [ "$(uname)" == "Darwin" ]
then
  cmake --build . -- "-j$(sysctl -n hw.ncpu)"
else
  cmake --build . -- "-j$(nproc)"
fi
