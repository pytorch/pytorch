cmake_minimum_required(VERSION 3.1)

project(code_analyzer_test)

add_library(torch UNKNOWN IMPORTED)

find_package(Torch REQUIRED)
include_directories(${TORCH_INCLUDE_DIRS})
link_directories(${TORCH_LIBRARIES})

add_library(SimpleOps
  simple_ops.cpp
  utils.cpp
)

set_target_properties(SimpleOps PROPERTIES
  COMPILE_FLAGS "-S -emit-llvm -O3"
)
