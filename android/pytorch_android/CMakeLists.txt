cmake_minimum_required(VERSION 3.4.1)
project(pytorch CXX)
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_VERBOSE_MAKEFILE ON)

set(pytorch_android_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)
set(libtorch_include_DIR ${pytorch_android_DIR}/libtorch_include/${ANDROID_ABI})

set(libtorch_FILE ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libtorch.a)
set(libc10_FILE ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libc10.a)
set(libnnpack_FILE ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libnnpack.a)
set(libqnnpack_FILE ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libqnnpack.a)
set(libpytorch_qnnpack_FILE ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libpytorch_qnnpack.a)
set(libcpuinfo_FILE ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libcpuinfo.a)
set(libclog_FILE ${CMAKE_CURRENT_LIST_DIR}/src/main/jniLibs/${ANDROID_ABI}/libclog.a)

message(STATUS "libtorch dir:${libtorch_DIR}")

add_library(libtorch STATIC IMPORTED)
set_property(TARGET libtorch PROPERTY IMPORTED_LOCATION ${libtorch_FILE})

add_library(libc10 STATIC IMPORTED)
set_property(TARGET libc10 PROPERTY IMPORTED_LOCATION ${libc10_FILE})

add_library(libnnpack STATIC IMPORTED)
set_property(TARGET libnnpack PROPERTY IMPORTED_LOCATION ${libnnpack_FILE})

add_library(libqnnpack STATIC IMPORTED)
set_property(TARGET libqnnpack PROPERTY IMPORTED_LOCATION ${libqnnpack_FILE})

add_library(libpytorch_qnnpack STATIC IMPORTED)
set_property(TARGET libpytorch_qnnpack PROPERTY IMPORTED_LOCATION ${libpytorch_qnnpack_FILE})

add_library(libcpuinfo STATIC IMPORTED)
set_property(TARGET libcpuinfo PROPERTY IMPORTED_LOCATION ${libcpuinfo_FILE})

add_library(libclog STATIC IMPORTED)
set_property(TARGET libclog PROPERTY IMPORTED_LOCATION ${libclog_FILE})

file(GLOB pytorch_android_SOURCES
  ${pytorch_android_DIR}/*.cpp
)

add_library(pytorch SHARED
    ${pytorch_android_SOURCES}
)

target_compile_options(pytorch PRIVATE
  -fexceptions
)

target_compile_definitions(pytorch PRIVATE USE_STATIC_DISPATCH)

target_include_directories(pytorch PUBLIC
    ${libtorch_include_DIR}
)

set(BUILD_DIR ${CMAKE_SOURCE_DIR}/build)
file(MAKE_DIRECTORY ${BUILD_DIR})

set(fbjni_DIR ${CMAKE_CURRENT_LIST_DIR}/../libs/fbjni/)
set(fbjni_BUILD_DIR ${BUILD_DIR}/fbjni/${ANDROID_ABI})

add_subdirectory(${fbjni_DIR} ${fbjni_BUILD_DIR})

target_link_libraries(pytorch
    fbjni
    -Wl,--gc-sections
    -Wl,--whole-archive
    libtorch
    -Wl,--no-whole-archive
    libc10
    libnnpack
    libqnnpack
    libpytorch_qnnpack
    libcpuinfo
    libclog
)
