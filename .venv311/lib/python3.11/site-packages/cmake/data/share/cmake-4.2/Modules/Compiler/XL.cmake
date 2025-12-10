# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__COMPILER_XL)
  return()
endif()
set(__COMPILER_XL 1)

include(Compiler/CMakeCommonCompilerMacros)

macro(__compiler_xl lang)
  # Feature flags.
  set(CMAKE_${lang}_VERBOSE_FLAG "-V")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-qpic")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-qpic")
  set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "-qhalt=i")
  set(CMAKE_${lang}_RESPONSE_FILE_FLAG "-qoptfile=")
  set(CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG "-qoptfile=")

  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-qmkshrobj")

  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Wl,")
  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP ",")

  set(CMAKE_${lang}_LINK_MODE DRIVER)

  string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -g")
  string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -O")
  string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -O")
  string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -g")
  set(CMAKE_${lang}_CREATE_PREPROCESSED_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -E <SOURCE> > <PREPROCESSED_SOURCE>")
  set(CMAKE_${lang}_CREATE_ASSEMBLY_SOURCE     "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -S <SOURCE> -o <ASSEMBLY_SOURCE>")

  set(CMAKE_DEPFILE_FLAGS_${lang} "-MF <DEP_FILE> -qmakedep=gcc")
endmacro()
