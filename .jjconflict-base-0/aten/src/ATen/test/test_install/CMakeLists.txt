cmake_minimum_required(VERSION 3.10)
find_package(ATen REQUIRED)
include_directories(${ATEN_INCLUDE_DIR})

set(CMAKE_CXX_STANDARD 20 CACHE STRING "The C++ standard whose features are requested to build this target.")
add_executable(main main.cpp)
target_link_libraries(main ${ATEN_LIBRARIES})
