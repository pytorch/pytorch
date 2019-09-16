# Basic CMake setup
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_ops)

find_package(Torch REQUIRED)

add_library(custom_ops SHARED op.cpp)

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/../../third_party/pybind11/ ./pybind11)
pybind11_add_module(custom_class SHARED classes.cpp)
target_link_libraries(custom_class PRIVATE "${TORCH_LIBRARIES}")

target_compile_features(custom_ops PUBLIC cxx_range_for)
target_link_libraries(custom_ops "${TORCH_LIBRARIES}")
target_compile_definitions(custom_ops PRIVATE custom_ops_EXPORTS)

add_executable(test_custom_ops test_custom_ops.cpp)
target_link_libraries(test_custom_ops custom_ops)
