cmake_minimum_required(VERSION 3.2 FATAL_ERROR)
set(CMAKE_MODULE_PATH ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake ${CMAKE_MODULE_PATH})

# Relies on CMAKE_INSTALL_PREFIX to be set to ../tmp_install.
# It then finds $PREFIX/share/cmake/ATen/ATenConfig.cmake,
# which defines ATEN_INCLUDE_DIR and ATEN_LIBRARIES.
find_package(ATen REQUIRED)
if(NOT ATen_FOUND)
  message(FATAL_ERROR "ATen not found")
endif()

find_package(Gloo REQUIRED)
if(NOT Gloo_FOUND)
  message(FATAL_ERROR "Gloo not found")
endif()

set(C10D_SRCS
  Utils.cpp
  Store.cpp
  FileStore.cpp
  TCPStore.cpp
  ProcessGroup.cpp
  )

add_library(c10d ${C10D_SRCS})
target_compile_options(c10d PUBLIC "-std=c++11")
target_include_directories(c10d PUBLIC ${ATEN_INCLUDE_DIR})
target_link_libraries(c10d PUBLIC ${ATEN_LIBRARIES})

# For torch/csrc/utils/hash.h and torch/csrc/utils/functional.h
target_include_directories(c10d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

set(C10D_GLOO_SRCS
  ProcessGroupGloo.cpp
  )

add_library(c10d_gloo ${C10D_GLOO_SRCS})
target_include_directories(c10d_gloo PUBLIC ${GLOO_INCLUDE_DIR})
target_link_libraries(c10d_gloo PUBLIC c10d ${GLOO_LIBRARIES})

add_subdirectory(example)

enable_testing()
add_subdirectory(test)
