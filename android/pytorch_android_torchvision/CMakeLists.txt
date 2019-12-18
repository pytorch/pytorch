cmake_minimum_required(VERSION 3.4.1)
project(pytorch_vision_jni CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(pytorch_vision_cpp_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)

file(GLOB pytorch_vision_SOURCES
  ${pytorch_vision_cpp_DIR}/pytorch_vision_jni.cpp
)

add_library(pytorch_vision_jni SHARED
    ${pytorch_vision_SOURCES}
)

target_compile_options(pytorch_vision_jni PRIVATE
  -fexceptions
)

set(BUILD_SUBDIR ${ANDROID_ABI})

set(fbjni_DIR ${CMAKE_CURRENT_LIST_DIR}/../libs/fbjni/)
set(fbjni_BUILD_DIR ${CMAKE_BINARY_DIR}/fbjni/${BUILD_SUBDIR})

add_subdirectory(${fbjni_DIR} ${fbjni_BUILD_DIR})

target_link_libraries(pytorch_vision_jni
    fbjni
)
