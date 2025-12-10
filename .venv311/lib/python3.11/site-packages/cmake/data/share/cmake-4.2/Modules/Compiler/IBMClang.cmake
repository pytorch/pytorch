# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
if(__COMPILER_IBMClang)
  return()
endif()
set(__COMPILER_IBMClang 1)

# Macro to set ibm-clang unique config. This should be called after common
# clang config is included and include only what isn't common.
macro(__compiler_ibmclang lang)
  # Feature flags.
  set(CMAKE_${lang}_VERBOSE_FLAG "-v")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-fPIC")
  set(CMAKE_${lang}_RESPONSE_FILE_FLAG "@")
  set(CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG "@")

  if(CMAKE_${lang}_COMPILER_TARGET AND "${lang}" STREQUAL "CXX")
    list(APPEND CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND "--target=${CMAKE_${lang}_COMPILER_TARGET}")
  endif()

  # Thin LTO is not yet supported on AIX.
  if(NOT (CMAKE_SYSTEM_NAME STREQUAL "AIX"))
    set(_CMAKE_LTO_THIN TRUE)
  endif()

  if("${lang}" STREQUAL "CXX")
    list(APPEND CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND "-w" "-dM" "-E" "${CMAKE_ROOT}/Modules/CMakeCXXCompilerABI.cpp")
  endif()
endmacro()
