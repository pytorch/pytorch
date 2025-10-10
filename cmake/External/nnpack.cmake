if(__NNPACK_INCLUDED)
  return()
endif()
set(__NNPACK_INCLUDED TRUE)

if(NOT USE_NNPACK)
  return()
endif()

##############################################################################
# NNPACK is built together with Caffe2
# By default, it builds code from third-party/NNPACK submodule.
# Define NNPACK_SOURCE_DIR to build with a different version.
##############################################################################

##############################################################################
# (1) MSVC - unsupported
##############################################################################

if(MSVC)
  message(WARNING "NNPACK not supported on MSVC yet. Turn this warning off by USE_NNPACK=OFF.")
  set(USE_NNPACK OFF)
  return()
endif()

##############################################################################
# (2) Anything but x86, x86-64, ARM, ARM64 - unsupported
##############################################################################
if(CMAKE_SYSTEM_PROCESSOR)
  if(NOT CMAKE_SYSTEM_PROCESSOR MATCHES "^(i686|x86_64|armv5te|armv7-a|armv7l|arm64|aarch64)$")
    message(WARNING "NNPACK is not supported on ${CMAKE_SYSTEM_PROCESSOR} processors. "
      "The only supported architectures are x86, x86-64, ARM, and ARM64. "
      "Turn this warning off by USE_NNPACK=OFF.")
    set(USE_NNPACK OFF)
    return()
  endif()
endif()

##############################################################################
# (3) Android, iOS, Linux, macOS - supported
##############################################################################

if(ANDROID OR IOS OR ${CMAKE_SYSTEM_NAME} STREQUAL "Linux" OR ${CMAKE_SYSTEM_NAME} STREQUAL "Darwin")
  message(STATUS "Brace yourself, we are building NNPACK")
  set(CAFFE2_THIRD_PARTY_ROOT ${PROJECT_SOURCE_DIR}/third_party)

  # Directories for NNPACK dependencies submoduled in Caffe2
  set(PYTHON_PEACHPY_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/python-peachpy" CACHE STRING "PeachPy (Python package) source directory")
  if(NOT DEFINED CPUINFO_SOURCE_DIR)
    set(CPUINFO_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/cpuinfo" CACHE STRING "cpuinfo source directory")
  endif()
  set(NNPACK_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/NNPACK" CACHE STRING "NNPACK source directory")
  set(FP16_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FP16" CACHE STRING "FP16 source directory")
  set(FXDIV_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/FXdiv" CACHE STRING "FXdiv source directory")
  set(PSIMD_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/psimd" CACHE STRING "PSimd source directory")
  set(PTHREADPOOL_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/pthreadpool" CACHE STRING "pthreadpool source directory")
  set(GOOGLETEST_SOURCE_DIR "${CAFFE2_THIRD_PARTY_ROOT}/googletest" CACHE STRING "Google Test source directory")

  if(NOT TARGET nnpack)
    set(NNPACK_BUILD_TESTS OFF CACHE BOOL "")
    set(NNPACK_BUILD_BENCHMARKS OFF CACHE BOOL "")
    set(NNPACK_LIBRARY_TYPE "static" CACHE STRING "")
    set(PTHREADPOOL_LIBRARY_TYPE "static" CACHE STRING "")
    set(CPUINFO_LIBRARY_TYPE "static" CACHE STRING "")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
      message(WARNING "Ancient nnpack forces CMake compatibility")
      set(CMAKE_POLICY_VERSION_MINIMUM 3.5)
    endif()
    add_subdirectory(
      "${NNPACK_SOURCE_DIR}"
      "${CONFU_DEPENDENCIES_BINARY_DIR}/NNPACK")
    if(CMAKE_VERSION VERSION_GREATER_EQUAL "4.0.0")
      unset(CMAKE_POLICY_VERSION_MINIMUM)
    endif()
    # We build static versions of nnpack and pthreadpool but link
    # them into a shared library for Caffe2, so they need PIC.
    set_property(TARGET nnpack PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET pthreadpool PROPERTY POSITION_INDEPENDENT_CODE ON)
    set_property(TARGET cpuinfo PROPERTY POSITION_INDEPENDENT_CODE ON)

  endif()

  set(NNPACK_FOUND TRUE)
  if(TARGET nnpack)
    set(NNPACK_INCLUDE_DIRS
      $<TARGET_PROPERTY:nnpack,INCLUDE_DIRECTORIES>
      $<TARGET_PROPERTY:pthreadpool,INCLUDE_DIRECTORIES>)
    set(NNPACK_LIBRARIES $<TARGET_OBJECTS:nnpack> $<TARGET_OBJECTS:cpuinfo>)
  endif()
  return()
endif()

##############################################################################
# (4) Catch-all: not supported.
##############################################################################

message(WARNING "Unknown platform - I don't know how to build NNPACK. "
                "See cmake/External/nnpack.cmake for details.")
set(USE_NNPACK OFF)
