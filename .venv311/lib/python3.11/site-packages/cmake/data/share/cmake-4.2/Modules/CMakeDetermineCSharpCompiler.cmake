# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

if(NOT ${CMAKE_GENERATOR} MATCHES "Visual Studio")
  message(FATAL_ERROR
    "C# is currently only supported by Visual Studio generators.")
endif()

include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)
#include(Platform/${CMAKE_SYSTEM_NAME}-Determine-CSharp OPTIONAL)
#include(Platform/${CMAKE_SYSTEM_NAME}-CSharp OPTIONAL)
if(NOT CMAKE_CSharp_COMPILER_NAMES)
  set(CMAKE_CSharp_COMPILER_NAMES csc)
endif()

# Build a small source file to identify the compiler.
if(NOT CMAKE_CSharp_COMPILER_ID_RUN)
  set(CMAKE_CSharp_COMPILER_ID_RUN 1)

  # Try to identify the compiler.
  set(CMAKE_CSharp_COMPILER_ID)
  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerId.cmake)
  CMAKE_DETERMINE_COMPILER_ID(CSharp CSFLAGS CMakeCSharpCompilerId.cs)

  execute_process(COMMAND "${CMAKE_CSharp_COMPILER}" "/help /preferreduilang:en-US" OUTPUT_VARIABLE output)
  string(REPLACE "\n" ";" output "${output}")
  foreach(line ${output})
    string(TOUPPER ${line} line)
    string(REGEX REPLACE "^.*COMPILER.*VERSION[^\\.0-9]*([\\.0-9]+).*$" "\\1" version "${line}")
    if(version AND NOT "x${line}" STREQUAL "x${version}")
      set(CMAKE_CSharp_COMPILER_VERSION ${version})
      break()
    endif()
  endforeach()
  message(STATUS "The CSharp compiler version is ${CMAKE_CSharp_COMPILER_VERSION}")
endif()

# configure variables set in this file for fast reload later on
configure_file(${CMAKE_ROOT}/Modules/CMakeCSharpCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeCSharpCompiler.cmake
  @ONLY
  )
