# Basic CMake setup
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_op)

find_package(Torch REQUIRED)

torch_add_custom_op_library(custom_op op.cpp)

add_executable(custom_op_test test.cpp)
target_link_libraries(custom_op_test custom_op)
