cmake_minimum_required(VERSION 3.5)
project(pytorch_android CXX)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(pytorch_android_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)
set(libtorch_include_DIR ${pytorch_android_DIR}/include)

set(libtorch_SO ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libtorch.so)
set(libc10_SO ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libc10.so)

message(STATUS "libtorch dir:${libtorch_DIR}")

add_library(libtorch SHARED IMPORTED)
set_property(TARGET libtorch PROPERTY IMPORTED_LOCATION ${libtorch_SO})

add_library(libc10 SHARED IMPORTED ${libc10_SO})
set_property(TARGET libc10 PROPERTY IMPORTED_LOCATION ${libc10_SO})

file(GLOB pytorch_android_SOURCES
  ${pytorch_android_DIR}/*.cpp
)

add_library(pytorch_android SHARED
    ${pytorch_android_SOURCES}
)

target_compile_options(pytorch_android PRIVATE
  -fexceptions
)

target_include_directories(pytorch_android PUBLIC
    ${pytorch_android_DIR}
    ${libtorch_include_DIR}
)

set(EXTERNAL_DIR ${CMAKE_CURRENT_LIST_DIR}/../libs/third-party/)
set(BUILD_DIR ${CMAKE_SOURCE_DIR}/build)
file(MAKE_DIRECTORY ${BUILD_DIR})

set(BUILD_DIR ${CMAKE_SOURCE_DIR}/build)
file(MAKE_DIRECTORY ${BUILD_DIR})

set(fbjni_DIR ${CMAKE_CURRENT_LIST_DIR}/../libs/fbjni/)
set(fbjni_BUILD_DIR ${BUILD_DIR}/fbjni/${ANDROID_ABI})

add_subdirectory(${fbjni_DIR} ${fbjni_BUILD_DIR})

target_link_libraries(pytorch_android
    fbjni
    libtorch
    libc10
)
