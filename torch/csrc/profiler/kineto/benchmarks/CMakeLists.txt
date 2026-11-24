# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

# Kineto Benchmarks CMakeLists.txt
#
# Build with:
#   cd libkineto
#   mkdir build && cd build
#   cmake .. -DKINETO_BUILD_BENCHMARKS=ON
#   make json_output_benchmark

add_executable(json_output_benchmark
    ${CMAKE_CURRENT_SOURCE_DIR}/json_output_benchmark.cpp
)

target_include_directories(json_output_benchmark PRIVATE
    ${LIBKINETO_INCLUDE_DIR}
    ${LIBKINETO_SOURCE_DIR}
)

target_link_libraries(json_output_benchmark
    kineto
    fmt::fmt-header-only
)

target_compile_definitions(json_output_benchmark PRIVATE
    KINETO_NAMESPACE=libkineto
)

set_target_properties(json_output_benchmark PROPERTIES
    CXX_STANDARD 20
    CXX_STANDARD_REQUIRED YES
    CXX_EXTENSIONS NO
)
