# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.



if(CMAKE_ISPC_COMPILER_FORCED)
  # The compiler configuration was forced by the user.
  # Assume the user has configured all compiler information.
  set(CMAKE_ISPC_COMPILER_WORKS TRUE)
  return()
endif()

include(CMakeTestCompilerCommon)

# Make sure we try to compile as a STATIC_LIBRARY
set(__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE ${CMAKE_TRY_COMPILE_TARGET_TYPE})
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

# # Try to identify the ABI and configure it into CMakeISPCCompiler.cmake
include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerABI.cmake)
CMAKE_DETERMINE_COMPILER_ABI(ISPC ${CMAKE_ROOT}/Modules/CMakeISPCCompilerABI.ispc)
if(CMAKE_ISPC_ABI_COMPILED)
#   # The compiler worked so skip dedicated test below.
  set(CMAKE_ISPC_COMPILER_WORKS TRUE)
  message(STATUS "Check for working ISPC compiler: ${CMAKE_ISPC_COMPILER} - skipped")
endif()

# Re-configure to save learned information.
configure_file(
  ${CMAKE_ROOT}/Modules/CMakeISPCCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeISPCCompiler.cmake
  @ONLY
  )
include(${CMAKE_PLATFORM_INFO_DIR}/CMakeISPCCompiler.cmake)

if(CMAKE_ISPC_SIZEOF_DATA_PTR)
  foreach(f ${CMAKE_ISPC_ABI_FILES})
    include(${f})
  endforeach()
  unset(CMAKE_ISPC_ABI_FILES)
endif()

set(CMAKE_TRY_COMPILE_TARGET_TYPE ${__CMAKE_SAVED_TRY_COMPILE_TARGET_TYPE})
