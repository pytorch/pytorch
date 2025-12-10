# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__COMPILER_PGI)
  return()
endif()
set(__COMPILER_PGI 1)

include(Compiler/CMakeCommonCompilerMacros)

macro(__compiler_pgi lang)
  # Feature flags.
  set(CMAKE_${lang}_VERBOSE_FLAG "-v")

  # Initial configuration flags.
  string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
  string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -g -O0")
  string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -O2 -s")
  string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -fast -O3")
  string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -O2 -gopt")

  if(CMAKE_HOST_WIN32)
    string(APPEND CMAKE_${lang}_FLAGS_INIT " -Bdynamic")
  endif()

  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Wl,")
  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP ",")

  set(CMAKE_${lang}_LINK_MODE DRIVER)

  set(_CMAKE_${lang}_IPO_SUPPORTED_BY_CMAKE YES)
  if(NOT CMAKE_SYSTEM_PROCESSOR STREQUAL ppc64le AND (NOT CMAKE_HOST_WIN32 OR CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 16.3) AND CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 23.3)
    set(_CMAKE_${lang}_IPO_MAY_BE_SUPPORTED_BY_COMPILER YES)
    set(CMAKE_${lang}_COMPILE_OPTIONS_IPO "-Mipa=fast,inline")
  else()
    set(_CMAKE_${lang}_IPO_MAY_BE_SUPPORTED_BY_COMPILER NO)
  endif()

  # Preprocessing and assembly rules.
  set(CMAKE_${lang}_CREATE_PREPROCESSED_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -E <SOURCE> > <PREPROCESSED_SOURCE>")
  set(CMAKE_${lang}_CREATE_ASSEMBLY_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -S <SOURCE> -o <ASSEMBLY_SOURCE>")
endmacro()
