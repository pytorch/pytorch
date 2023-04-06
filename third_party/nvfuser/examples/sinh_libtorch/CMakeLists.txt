cmake_minimum_required(VERSION 3.10 FATAL_ERROR)
project(sinh_example LANGUAGES CXX)
set(CMAKE_CXX_STANDARD 14)

find_package(Torch REQUIRED)

add_executable(sinh_example main.cpp)
target_link_libraries(sinh_example ${TORCH_LIBRARIES})
