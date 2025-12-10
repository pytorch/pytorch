# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
if(__COMPILER_FUJITSU)
  return()
endif()
set(__COMPILER_FUJITSU 1)

include(Compiler/CMakeCommonCompilerMacros)

macro(__compiler_fujitsu lang)
  set(CMAKE_${lang}_VERBOSE_FLAG "-###")
  set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-cwno")

  # Initial configuration flags
  string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
  string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -g -O0")
  string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -O3 -DNDEBUG")
  string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -O2 -g -DNDEBUG")

  # PIC flags
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-fPIE")

  # Passing link options to the compiler
  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Wl,")
  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP ",")

  set(CMAKE_${lang}_LINK_MODE DRIVER)

  # IPO flag
  set(_CMAKE_${lang}_IPO_SUPPORTED_BY_CMAKE YES)
  if ("${lang}" STREQUAL "Fortran")
    # Supported by Fortran compiler only
    set(_CMAKE_${lang}_IPO_MAY_BE_SUPPORTED_BY_COMPILER YES)
    set(CMAKE_${lang}_COMPILE_OPTIONS_IPO "-Klto")
  else()
    set(_CMAKE_${lang}_IPO_MAY_BE_SUPPORTED_BY_COMPILER NO)
  endif()

  # How to actually call the compiler
  set(CMAKE_${lang}_CREATE_PREPROCESSED_SOURCE
  "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -E $<$<COMPILE_LANGUAGE:Fortran>:-Cpp> <SOURCE> > <PREPROCESSED_SOURCE>")
  set(CMAKE_${lang}_CREATE_ASSEMBLY_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -S <SOURCE> -o <ASSEMBLY_SOURCE>")
endmacro()
