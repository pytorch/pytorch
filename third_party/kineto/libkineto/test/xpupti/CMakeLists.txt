# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

set(CMAKE_CXX_STANDARD 20)

set(LINK_LIBRARIES
  gtest_main
  kineto_base
  kineto_api
  $<BUILD_INTERFACE:fmt::fmt-header-only>
  ${SYCL_LIBRARY}
  ${PTI_LIBRARY}
)

# Pure host-code tests (no SYCL device code): ordinary executables.
add_executable(XpuptiScopeProfilerConfigTest XpuptiScopeProfilerConfigTest.cpp)
target_link_libraries(XpuptiScopeProfilerConfigTest PRIVATE ${LINK_LIBRARIES})
gtest_discover_tests(XpuptiScopeProfilerConfigTest)

add_executable(XpuptiActivityHandlersTest XpuptiActivityHandlersTest.cpp)
target_link_libraries(XpuptiActivityHandlersTest PRIVATE ${LINK_LIBRARIES})
gtest_discover_tests(XpuptiActivityHandlersTest)

# Hardware-free test for the CPU->GPU flow-link allowlist.
add_executable(XpuptiFlowCorrelationTest XpuptiFlowCorrelationTest.cpp)
target_link_libraries(XpuptiFlowCorrelationTest PRIVATE ${LINK_LIBRARIES})
gtest_discover_tests(XpuptiFlowCorrelationTest)

include(ExternalProject)

# The compute translation unit contains SYCL device code and must be built by
# the SYCL compiler (icpx -fsycl). CMake allows only one CXX compiler per build
# tree, so it is built here as an isolated ExternalProject producing a kernel-only
# shared library (libxpupti_compute.so / xpupti_compute.dll).
set(XPUPTI_COMPUTE_LIB
  ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_SHARED_LIBRARY_PREFIX}xpupti_compute${CMAKE_SHARED_LIBRARY_SUFFIX})

if(WIN32)
  set(XPUPTI_COMPUTE_IMPLIB
      ${CMAKE_CURRENT_BINARY_DIR}/${CMAKE_IMPORT_LIBRARY_PREFIX}xpupti_compute${CMAKE_IMPORT_LIBRARY_SUFFIX})
else()
  set(XPUPTI_COMPUTE_IMPLIB "")
endif()

# On Windows SYCL/ICX requires the Ninja generator for the inner project.
set(_xpupti_compute_generator ${CMAKE_GENERATOR})
if(WIN32)
  set(_xpupti_compute_generator Ninja)
endif()

set(_xpupti_compute_cmake_args
    -DCMAKE_CXX_COMPILER=${SYCL_COMPILER}
    -DCMAKE_INSTALL_PREFIX=${CMAKE_CURRENT_BINARY_DIR})

if(WIN32)
  get_filename_component(_sycl_compiler_dir "${SYCL_COMPILER}" DIRECTORY)
  get_filename_component(_sycl_root "${_sycl_compiler_dir}" DIRECTORY)
  list(APPEND _xpupti_compute_cmake_args
    "-DCMAKE_EXE_LINKER_FLAGS=/Qoption,link,/LIBPATH:\"${_sycl_root}/lib\""
    "-DCMAKE_SHARED_LINKER_FLAGS=/Qoption,link,/LIBPATH:\"${_sycl_root}/lib\"")
endif()

ExternalProject_Add(xpupti_compute_ep
  SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/compute
  CONFIGURE_COMMAND ${CMAKE_COMMAND}
    -G ${_xpupti_compute_generator}
    -S <SOURCE_DIR>
    -B <BINARY_DIR>
    -DCMAKE_BUILD_TYPE=$<CONFIG>
    ${_xpupti_compute_cmake_args}
  BUILD_ALWAYS TRUE
  BUILD_BYPRODUCTS ${XPUPTI_COMPUTE_LIB} ${XPUPTI_COMPUTE_IMPLIB})

add_library(xpupti_compute SHARED IMPORTED)
set_target_properties(xpupti_compute PROPERTIES
  IMPORTED_LOCATION ${XPUPTI_COMPUTE_LIB})
if(WIN32)
  set_target_properties(xpupti_compute PROPERTIES
    IMPORTED_IMPLIB ${XPUPTI_COMPUTE_IMPLIB})
endif()
add_dependencies(xpupti_compute xpupti_compute_ep)

# Runtime tests: host code linking the kernel-only compute library.
function(add_sycl_test test_file)
  get_filename_component(test_name "${test_file}" NAME_WE)
  add_executable(${test_name} ${test_file} XpuptiTestUtilities.cpp)
  target_link_libraries(${test_name} PRIVATE
    ${LINK_LIBRARIES}
    xpupti_compute)
  set_target_properties(${test_name} PROPERTIES
    BUILD_RPATH ${CMAKE_CURRENT_BINARY_DIR})
  gtest_discover_tests(${test_name})
endfunction()

add_sycl_test(XpuptiProfilerTest.cpp)
add_sycl_test(XpuptiScopeProfilerTest.cpp)
