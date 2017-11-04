if (__NNPACK_INCLUDED)
  return()
endif()
set(__NNPACK_INCLUDED TRUE)
 
if (NOT USE_NNPACK)
  return()
endif()

# try any external nnpack first
find_package(NNPACK)

if (NNPACK_FOUND)
  message(INFO "Found external NNPACK installation.")
  return()
endif()

##############################################################################
# Custom build rules to build nnpack, if external dependency is not found 
##############################################################################

set(NNPACK_PREFIX ${PROJECT_SOURCE_DIR}/third_party/NNPACK)

##############################################################################
# (1) MSVC - unsupported 
##############################################################################

if (MSVC)
  message(WARNING "NNPACK not supported on MSVC yet. Turn this warning off by USE_NNPACK=OFF.")
  set(USE_NNPACK OFF)
  return()
endif()

##############################################################################
# (2) Android, iOS, Linux, macOS - supported
##############################################################################

if (ANDROID OR IOS OR ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  message(STATUS "Brace yourself, we are building NNPACK")
  set(CAFFE2_THIRD_PARTY_ROOT ${PROJECT_SOURCE_DIR}/third_party)

  # Directory where NNPACK will download and build all dependencies
  set(CONFU_DEPENDENCIES_SOURCE_DIR ${PROJECT_BINARY_DIR}/confu-srcs
    CACHE PATH "Confu-style dependencies source directory")
  set(CONFU_DEPENDENCIES_BINARY_DIR ${PROJECT_BINARY_DIR}/confu-deps
    CACHE PATH "Confu-style dependencies binary directory")

  if(NOT TARGET nnpack)
    set(NNPACK_BUILD_TESTS OFF CACHE BOOL "")
    set(NNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
    set(NNPACK_LIBRARY_TYPE "static" CACHE STRING "")
    set(PTHREADPOOL_LIBRARY_TYPE "static" CACHE STRING "")
    add_subdirectory(
      "${NNPACK_PREFIX}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}")
  endif()

  set(NNPACK_FOUND TRUE)
  set(NNPACK_INCLUDE_DIRS
    $<TARGET_PROPERTY:nnpack,INCLUDE_DIRECTORIES>
    $<TARGET_PROPERTY:pthreadpool,INCLUDE_DIRECTORIES>)
  set(NNPACK_LIBRARIES
    $<TARGET_FILE:nnpack>
    $<TARGET_FILE:pthreadpool>)
  return()
endif()

##############################################################################
# (3) Catch-all: not supported.
##############################################################################

message(WARNING "Unknown platform - I don't know how to build NNPACK. "
                "See cmake/External/nnpack.cmake for details.")
set(USE_NNPACK OFF)
