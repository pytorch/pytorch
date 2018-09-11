# Basic CMake setup
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

# This convenience function will create a shared library target, configure
# the right include directories and link against the right libraries. It is
# exactly equivalent to the following lines:
#
# add_library(custom_ops SHARED op.cpp)
# target_include_directories(custom_ops PUBLIC "${TORCH_INCLUDE_DIRS}")
# target_link_libraries(custom_ops "${TORCH_LIBRARIES}")
# set_property(TARGET custom_ops PROPERTY CXX_STANDARD 11)
#
torch_add_custom_op_library(custom_ops op.cpp)

add_executable(test_custom_ops test_custom_ops.cpp)
target_link_libraries(test_custom_ops custom_ops)
