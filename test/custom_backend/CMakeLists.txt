# Basic CMake setup
cmake_minimum_required(VERSION 3.1 FATAL_ERROR)
project(custom_backend)

if(USE_ROCM)
include(utils)
include(LoadHIP)
endif()
find_package(Torch REQUIRED)

add_library(custom_backend SHARED custom_backend.cpp)
set_property(TARGET custom_backend PROPERTY CXX_STANDARD 17)
target_link_libraries(custom_backend "${TORCH_LIBRARIES}")

add_executable(test_custom_backend test_custom_backend.cpp)
set_property(TARGET test_custom_backend PROPERTY CXX_STANDARD 17)
target_link_libraries(test_custom_backend custom_backend)
