cmake_minimum_required(VERSION 3.0)
find_package(ATen REQUIRED)
include_directories(${ATEN_INCLUDE_DIR})

# C++14
if (not MSVC)
    set(CMAKE_CXX_FLAGS "--std=c++14 ${CMAKE_CXX_FLAGS}")
endif()
add_executable(main main.cpp)
target_link_libraries(main ${ATEN_LIBRARIES})
