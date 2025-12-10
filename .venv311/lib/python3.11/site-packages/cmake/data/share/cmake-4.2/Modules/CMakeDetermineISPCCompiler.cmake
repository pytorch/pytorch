# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# determine the compiler to use for ISPC programs

include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)

if( NOT (("${CMAKE_GENERATOR}" MATCHES "Make") OR ("${CMAKE_GENERATOR}" MATCHES "Ninja")) )
  message(FATAL_ERROR "ISPC language not currently supported by \"${CMAKE_GENERATOR}\" generator")
endif()

# Load system-specific compiler preferences for this language.
include(Platform/${CMAKE_SYSTEM_NAME}-Determine-ISPC OPTIONAL)
include(Platform/${CMAKE_SYSTEM_NAME}-ISPC OPTIONAL)
if(NOT CMAKE_ISPC_COMPILER_NAMES)
  set(CMAKE_ISPC_COMPILER_NAMES ispc)
endif()


if(NOT CMAKE_ISPC_COMPILER)

  set(CMAKE_ISPC_COMPILER_INIT NOTFOUND)

  # prefer the environment variable CC
  if(NOT $ENV{ISPC} STREQUAL "")
    get_filename_component(CMAKE_ISPC_COMPILER_INIT $ENV{ISPC} PROGRAM PROGRAM_ARGS CMAKE_ISPC_FLAGS_ENV_INIT)
    if(CMAKE_ISPC_FLAGS_ENV_INIT)
      set(CMAKE_ISPC_COMPILER_ARG1 "${CMAKE_ISPC_FLAGS_ENV_INIT}" CACHE STRING "First argument to ISPC compiler")
    endif()
    if(NOT EXISTS ${CMAKE_ISPC_COMPILER_INIT})
      message(FATAL_ERROR "Could not find compiler set in environment variable ISPC:\n$ENV{ISPC}.")
    endif()
  endif()

  # next try prefer the compiler specified by the generator
  if(CMAKE_GENERATOR_ISPC)
    if(NOT CMAKE_ISPC_COMPILER_INIT)
      set(CMAKE_ISPC_COMPILER_INIT ${CMAKE_GENERATOR_ISPC})
    endif()
  endif()

  # finally list compilers to try
  if(NOT CMAKE_ISPC_COMPILER_INIT)
    set(CMAKE_ISPC_COMPILER_LIST ${_CMAKE_TOOLCHAIN_PREFIX}ispc ispc)
  endif()

  # Find the compiler.
  _cmake_find_compiler(ISPC)

else()
  _cmake_find_compiler_path(ISPC)
endif()
mark_as_advanced(CMAKE_ISPC_COMPILER)

if(NOT CMAKE_ISPC_COMPILER_ID_RUN)
set(CMAKE_ISPC_COMPILER_ID_RUN 1)

  # Try to identify the compiler.
  set(CMAKE_ISPC_COMPILER_ID)
  set(CMAKE_ISPC_PLATFORM_ID)


  set(CMAKE_ISPC_COMPILER_ID_TEST_FLAGS_FIRST
  # setup logic to make sure ISPC outputs a file
  "-o cmake_ispc_output"
  )

  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerId.cmake)
  CMAKE_DETERMINE_COMPILER_ID(ISPC ISPCFLAGS CMakeISPCCompilerId.ispc)

  _cmake_find_compiler_sysroot(ISPC)
endif()

if (NOT _CMAKE_TOOLCHAIN_LOCATION)
  get_filename_component(_CMAKE_TOOLCHAIN_LOCATION "${CMAKE_ISPC_COMPILER}" PATH)
endif ()

set(_CMAKE_PROCESSING_LANGUAGE "ISPC")
include(CMakeFindBinUtils)
include(Compiler/${CMAKE_ISPC_COMPILER_ID}-FindBinUtils OPTIONAL)
unset(_CMAKE_PROCESSING_LANGUAGE)

if(CMAKE_ISPC_COMPILER_ID_VENDOR_MATCH)
  set(_SET_CMAKE_ISPC_COMPILER_ID_VENDOR_MATCH
    "set(CMAKE_ISPC_COMPILER_ID_VENDOR_MATCH [==[${CMAKE_ISPC_COMPILER_ID_VENDOR_MATCH}]==])")
else()
  set(_SET_CMAKE_ISPC_COMPILER_ID_VENDOR_MATCH "")
endif()


# configure variables set in this file for fast reload later on
configure_file(${CMAKE_ROOT}/Modules/CMakeISPCCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeISPCCompiler.cmake @ONLY)

set(CMAKE_ISPC_COMPILER_ENV_VAR "ISPC")
