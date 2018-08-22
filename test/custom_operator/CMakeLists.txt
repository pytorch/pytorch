# Basic CMake setup
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

torch_add_custom_op_library(custom_ops op.cpp)

add_executable(test_custom_ops test_custom_ops.cpp)
target_link_libraries(test_custom_ops custom_ops)
