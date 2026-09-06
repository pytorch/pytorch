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

# Note: this file intentionally doesn't set CXX_STANDARD to ensure the Torch target propagates
# it to downstream projects.
find_package(Torch REQUIRED)

add_executable(test_jit_hooks test_jit_hooks.cpp)
target_link_libraries(test_jit_hooks "${TORCH_LIBRARIES}")
