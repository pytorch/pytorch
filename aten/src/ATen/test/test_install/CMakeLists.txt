cmake_minimum_required(VERSION 3.10)
find_package(ATen REQUIRED)
include_directories(${ATEN_INCLUDE_DIR})

# C++20
if(not MSVC)
    set(CMAKE_CXX_FLAGS "--std=c++20 ${CMAKE_CXX_FLAGS}")
endif()
add_executable(main main.cpp)
target_link_libraries(main ${ATEN_LIBRARIES})
