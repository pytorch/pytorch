# Basic CMake setup
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(jit_hooks)

find_package(Torch REQUIRED)

add_library(jit_hooks SHARED op.cpp)
set_property(TARGET jit_hooks PROPERTY CXX_STANDARD 14)

target_compile_features(jit_hooks PUBLIC cxx_range_for)
target_link_libraries(jit_hooks "${TORCH_LIBRARIES}")
target_compile_definitions(jit_hooks PRIVATE jit_hooks_EXPORTS)

add_executable(test_jit_hooks test_jit_hooks.cpp)
set_property(TARGET test_jit_hooks PROPERTY CXX_STANDARD 14)
target_link_libraries(test_jit_hooks jit_hooks)
