# Basic CMake setup
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_op)

list(APPEND CMAKE_MODULE_PATH "${CMAKE_CURRENT_SOURCE_DIR}/../../cmake/Modules")
find_package(Torch REQUIRED)

torch_add_custom_op_library(custom_op op.cpp)

add_executable(custom_op_test test.cpp)
target_link_libraries(custom_op_test custom_op)
