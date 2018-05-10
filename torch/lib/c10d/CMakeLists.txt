cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake ${CMAKE_MODULE_PATH})

add_library(store Store.cpp FileStore.cpp)
target_compile_options(store PUBLIC "-std=c++11")

enable_testing()
add_subdirectory(test)
