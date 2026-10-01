# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

cmake_minimum_required(VERSION 3.22 FATAL_ERROR)
project(xpupti_compute LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 20)

include(CheckCXXCompilerFlag)
check_cxx_compiler_flag("-fsycl" has_sycl)
if(NOT has_sycl)
  message(FATAL_ERROR "xpupti_compute requires a SYCL-capable compiler (icpx -fsycl)")
endif()

# Kernel-only shared library. The SYCL compiler finalizes the device image
# inside the library, so the host-compiled (GCC/MSVC) test executables can link
# it without seeing any SYCL. ComputeOnXpu is exported via the API macro in the
# header (dllexport on Windows / default visibility on ELF) - no need for
# WINDOWS_EXPORT_ALL_SYMBOLS or an exe<->dll symbol cycle.
add_library(xpupti_compute SHARED XpuptiScopeProfilerCompute.cpp)
target_compile_options(xpupti_compute PRIVATE -fsycl)
target_link_options(xpupti_compute PRIVATE -fsycl)

# RPATH is a POSIX ELF concept; on Windows the loader finds the DLL via the
# executable directory (where it is installed next to the tests), so guard it.
if(NOT WIN32)
  set_target_properties(xpupti_compute PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

# A SHARED library yields a runtime .dll (RUNTIME) + import .lib (ARCHIVE) on
# Windows, and the .so (LIBRARY) on ELF. Install all three so the consumer can
# link and load on either platform.
install(TARGETS xpupti_compute
  RUNTIME DESTINATION .
  LIBRARY DESTINATION .
  ARCHIVE DESTINATION .)
