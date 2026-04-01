cmake_minimum_required(VERSION 3.18)
project(torch_vulkan_cpu_tests LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

enable_testing()

# Include generated CPU shader sources
file(GLOB GENERATED_SOURCES "generated/*.cpp")

# Trivial test
add_executable(test_trivial test_trivial.cpp)
target_include_directories(test_trivial PRIVATE generated)
add_test(NAME test_trivial COMMAND test_trivial)
