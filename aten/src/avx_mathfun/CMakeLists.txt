project(avx_mathfun C)
cmake_minimum_required(VERSION 3.0 FATAL_ERROR)

add_library(${PROJECT_NAME} INTERFACE)
target_include_directories(${PROJECT_NAME} INTERFACE ${PROJECT_SOURCE_DIR})
install(FILES avx_mathfun.h DESTINATION "TH/vector")
