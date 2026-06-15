# Basic CMake setup
cmake_minimum_required(VERSION 3.15 FATAL_ERROR)
project(jit_hooks)

# CI reuses these build artifacts across runners that may have different
# absolute workspace paths; emit $ORIGIN-relative RUNPATH so the binary finds
# its sibling .so wherever the build tree is dropped.
set(CMAKE_BUILD_RPATH_USE_ORIGIN TRUE)

if(USE_ROCM)
include(utils)
include(LoadHIP)
endif()
find_package(Torch REQUIRED)

add_executable(test_jit_hooks test_jit_hooks.cpp)
set_property(TARGET test_jit_hooks PROPERTY CXX_STANDARD 20)
target_link_libraries(test_jit_hooks "${TORCH_LIBRARIES}")
