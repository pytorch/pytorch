# Basic CMake setup
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(custom_ops)

if(USE_ROCM)
  include(utils)
  include(LoadHIP)
endif()
find_package(Torch REQUIRED)

add_library(custom_ops SHARED op.cpp)
set_property(TARGET custom_ops PROPERTY CXX_STANDARD 17)

target_compile_features(custom_ops PUBLIC cxx_range_for)
target_link_libraries(custom_ops "${TORCH_LIBRARIES}")
target_compile_definitions(custom_ops PRIVATE custom_ops_EXPORTS)

add_executable(test_custom_ops test_custom_ops.cpp)
set_property(TARGET test_custom_ops PROPERTY CXX_STANDARD 17)
target_link_libraries(test_custom_ops custom_ops)
