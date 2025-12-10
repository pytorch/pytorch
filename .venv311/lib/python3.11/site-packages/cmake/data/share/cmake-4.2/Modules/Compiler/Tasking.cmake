# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
if(_Tasking_CMAKE_LOADED)
  return()
endif()
set(_Tasking_CMAKE_LOADED TRUE)
include(Compiler/CMakeCommonCompilerMacros)

set(CMAKE_EXECUTABLE_SUFFIX ".elf")
set(CMAKE_STATIC_LIBRARY_SUFFIX ".a")
set(CMAKE_TRY_COMPILE_TARGET_TYPE STATIC_LIBRARY)

set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
set(BUILD_SHARED_LIBS FALSE CACHE BOOL "")
set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
set(CMAKE_LINK_SEARCH_START_STATIC TRUE)

if(NOT CMAKE_TASKING_TOOLSET)
  set(CMAKE_TASKING_TOOLSET "Standalone")
endif()

macro(__compiler_tasking lang)
  set(CMAKE_${lang}_OUTPUT_EXTENSION ".o")

  set(CMAKE_${lang}_VERBOSE_FLAG "-v")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "--pic")
  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Wl" " ")
  set(CMAKE_${lang}_RESPONSE_FILE_FLAG      "-f ")
  set(CMAKE_${lang}_RESPONSE_FILE_LINK_FLAG "-f ")
  set(CMAKE_DEPFILE_FLAGS_${lang} "--dep-file=<DEP_FILE>")
  set(CMAKE_${lang}_COMPILE_OPTIONS_WARNING_AS_ERROR "--warnings-as-errors")

  set(CMAKE_${lang}_LINK_MODE DRIVER)
  # Features for LINK_LIBRARY generator expression
  if(    CMAKE_TASKING_TOOLSET STREQUAL "SmartCode"
     OR (CMAKE_TASKING_TOOLSET STREQUAL "TriCore" AND CMAKE_CXX_COMPILER_VERSION VERSION_GREATER_EQUAL 4.2))
    ## WHOLE_ARCHIVE: Force loading all members of an archive
    set(CMAKE_${lang}_LINK_LIBRARY_USING_WHOLE_ARCHIVE "LINKER:--whole-archive=<LINK_ITEM>")
    set(CMAKE_${lang}_LINK_LIBRARY_USING_WHOLE_ARCHIVE_SUPPORTED TRUE)
    set(CMAKE_${lang}_LINK_LIBRARY_WHOLE_ARCHIVE_ATTRIBUTES LIBRARY_TYPE=STATIC DEDUPLICATION=YES OVERRIDE=DEFAULT)
  endif()

  string(APPEND CMAKE_${lang}_FLAGS_INIT " ")
  string(APPEND CMAKE_${lang}_FLAGS_DEBUG_INIT " -O0 -g")
  string(APPEND CMAKE_${lang}_FLAGS_MINSIZEREL_INIT " -O2 -t4 -DNDEBUG")
  string(APPEND CMAKE_${lang}_FLAGS_RELEASE_INIT " -O2 -t2 -DNDEBUG")
  string(APPEND CMAKE_${lang}_FLAGS_RELWITHDEBINFO_INIT " -O2 -t2 -g -DNDEBUG")

  set(CMAKE_${lang}_ARCHIVE_CREATE "\"${CMAKE_${lang}_COMPILER_AR}\" -r <TARGET> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_APPEND "\"${CMAKE_${lang}_COMPILER_AR}\" -r <TARGET> <OBJECTS>")
  set(CMAKE_${lang}_ARCHIVE_FINISH "")

  set(CMAKE_${lang}_CREATE_ASSEMBLY_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -cs <SOURCE> -o <ASSEMBLY_SOURCE>")
  set(CMAKE_${lang}_CREATE_PREPROCESSED_SOURCE "<CMAKE_${lang}_COMPILER> <DEFINES> <INCLUDES> <FLAGS> -Ep <SOURCE> > <PREPROCESSED_SOURCE>")

  if("${lang}" STREQUAL "CXX")
    set(CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND "${CMAKE_${lang}_COMPILER}")
    if(CMAKE_${lang}_COMPILER_ARG1)
      separate_arguments(_COMPILER_ARGS NATIVE_COMMAND "${CMAKE_${lang}_COMPILER_ARG1}")
      list(APPEND CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND ${_COMPILER_ARGS})
      unset(_COMPILER_ARGS)
    endif()
    list(APPEND CMAKE_${lang}_COMPILER_PREDEFINES_COMMAND "-Ep" "${CMAKE_ROOT}/Modules/CMakeCXXCompilerABI.cpp")
  endif()

endmacro()
