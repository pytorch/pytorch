cmake_minimum_required(VERSION 3.4.1)
set(PROJECT_NAME pytorch_testapp_jni)
project(${PROJECT_NAME} CXX)
set(CMAKE_CXX_STANDARD 17 CACHE STRING "The C++ standard whose features are requested to build this target.")
set(CMAKE_VERBOSE_MAKEFILE ON)

set(build_DIR ${CMAKE_SOURCE_DIR}/build)

set(pytorch_testapp_cpp_DIR ${CMAKE_CURRENT_LIST_DIR}/src/main/cpp)
message(STATUS "ANDROID_STL:${ANDROID_STL}")
file(GLOB pytorch_testapp_SOURCES
  ${pytorch_testapp_cpp_DIR}/pytorch_testapp_jni.cpp
)

add_library(${PROJECT_NAME} SHARED
    ${pytorch_testapp_SOURCES}
)

file(GLOB PYTORCH_INCLUDE_DIRS "${build_DIR}/pytorch_android*.aar/headers")
file(GLOB PYTORCH_LINK_DIRS "${build_DIR}/pytorch_android*.aar/jni/${ANDROID_ABI}")

target_compile_options(${PROJECT_NAME} PRIVATE
  -fexceptions
)

set(BUILD_SUBDIR ${ANDROID_ABI})

target_include_directories(${PROJECT_NAME} PRIVATE
  ${PYTORCH_INCLUDE_DIRS}
)

find_library(PYTORCH_LIBRARY pytorch_jni
  PATHS ${PYTORCH_LINK_DIRS}
  NO_CMAKE_FIND_ROOT_PATH)

target_link_libraries(${PROJECT_NAME}
  ${PYTORCH_LIBRARY}
  log)
