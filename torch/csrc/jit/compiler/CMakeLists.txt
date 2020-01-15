cmake_minimum_required(VERSION 3.5)
project(nnc)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -mavx2 -march=native -Werror -Wno-deprecated")
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

set(default_build_type "Release")
if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
  set(CMAKE_BUILD_TYPE "${default_build_type}" CACHE STRING "Choose the type of build" FORCE)
endif()

find_package(LLVM REQUIRED CONFIG)

message(STATUS "Found LLVM ${LLVM_PACKAGE_VERSION}")
message(STATUS "Using LLVMConfig.cmake in: ${LLVM_DIR}")

# Set your project compile flags.
# E.g. if using the C++ header files
# you will need to enable C++11 support
# for your compiler.

include_directories(${LLVM_INCLUDE_DIRS})
add_definitions(${LLVM_DEFINITIONS})

# asmjit dependency

set(ASMJIT_EMBED TRUE)
add_definitions(-DASMJIT_STATIC)

get_filename_component(
  ASMJIT_DIR
  "../../../../third_party/fbgemm/third_party/asmjit"
  ABSOLUTE)
include("${ASMJIT_DIR}/CMakeLists.txt")
include_directories("${ASMJIT_DIR}/src")

add_library(nnc
  src/expr.cc
  src/function.cc
  src/ir.cc
  src/ir_visitor.cc
  src/asmjit_codegen.cc
  src/llvm_codegen.cc
  src/llvm_jit.cc
  src/types.cc
  src/ir_printer.cc
  src/ir_mutator.cc
  src/schedule.cc
  src/tensor.cc
  ${ASMJIT_SRC}
  )

set_source_files_properties(src/llvm_jit.cc PROPERTIES COMPILE_FLAGS -fno-rtti)

target_include_directories(nnc PUBLIC "../../../../")

llvm_map_components_to_libnames(LLVM_LINK_LIBS
  support core irreader analysis executionengine instcombine object orcJIT
  runtimedyld scalaropts transformutils native ipo orcjit)

target_link_libraries(nnc PRIVATE ${LLVM_LINK_LIBS})

add_custom_target(cpptest)
add_subdirectory(../../../../third_party/googletest/ googletest EXCLUDE_FROM_ALL)

set(TEST_SRCS
  tests/asmjit_test.cc
  tests/expr_test.cc
  tests/llvm_test.cc
  tests/type_test.cc
  tests/ir_printer_test.cc
  tests/schedule_test.cc
  )

add_library(test_lib
  tests/padded_buffer.cc
  )
target_include_directories(test_lib PUBLIC "../../../../")

foreach(test_path ${TEST_SRCS})
  get_filename_component(filename ${test_path} NAME)
  string(REPLACE ".cc" "" test_exec ${filename})
  add_executable(${test_exec} ${test_path})
  add_dependencies(cpptest ${test_exec})
  target_link_libraries(${test_exec} test_lib nnc gtest_main gtest ${ASMJIT_DEPS})
  set_target_properties(${test_exec} PROPERTIES EXCLUDE_FROM_ALL 1)
  set_target_properties(${test_exec} PROPERTIES EXCLUDE_FROM_DEFAULT_BUILD 1)
endforeach()
