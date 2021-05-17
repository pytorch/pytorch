cmake_minimum_required(VERSION 3.5 FATAL_ERROR)
project(benchmarks CXX C)

set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_subdirectory(cpp)
add_subdirectory(operator_benchmark/pt_extension)
