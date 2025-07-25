# Basic CMake setup
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(jit_hooks)

if(USE_ROCM)
  include(utils)
  include(LoadHIP)
endif()
find_package(Torch REQUIRED)

add_executable(test_jit_hooks test_jit_hooks.cpp)
set_property(TARGET test_jit_hooks PROPERTY CXX_STANDARD 17)
target_link_libraries(test_jit_hooks "${TORCH_LIBRARIES}")
