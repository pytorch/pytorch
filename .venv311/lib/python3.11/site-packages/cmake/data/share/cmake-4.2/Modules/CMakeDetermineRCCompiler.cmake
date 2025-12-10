# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# determine the compiler to use for RC programs
# NOTE, a generator may set CMAKE_RC_COMPILER before
# loading this file to force a compiler.
# use environment variable RC first if defined by user, next use
# the cmake variable CMAKE_GENERATOR_RC which can be defined by a generator
# as a default compiler
if(NOT CMAKE_RC_COMPILER)
  # prefer the environment variable RC
  if(NOT $ENV{RC} STREQUAL "")
    get_filename_component(CMAKE_RC_COMPILER_INIT $ENV{RC} PROGRAM PROGRAM_ARGS CMAKE_RC_FLAGS_ENV_INIT)
    if(CMAKE_RC_FLAGS_ENV_INIT)
      set(CMAKE_RC_COMPILER_ARG1 "${CMAKE_RC_FLAGS_ENV_INIT}" CACHE STRING "Arguments to RC compiler")
    endif()
    if(EXISTS ${CMAKE_RC_COMPILER_INIT})
    else()
      message(FATAL_ERROR "Could not find compiler set in environment variable RC:\n$ENV{RC}.")
    endif()
  endif()

  # next try prefer the compiler specified by the generator
  if(CMAKE_GENERATOR_RC)
    if(NOT CMAKE_RC_COMPILER_INIT)
      set(CMAKE_RC_COMPILER_INIT ${CMAKE_GENERATOR_RC})
    endif()
  endif()

  # finally list compilers to try
  if(CMAKE_RC_COMPILER_INIT)
    set(_CMAKE_RC_COMPILER_LIST     ${CMAKE_RC_COMPILER_INIT})
    set(_CMAKE_RC_COMPILER_FALLBACK ${CMAKE_RC_COMPILER_INIT})
  elseif(NOT _CMAKE_RC_COMPILER_LIST)
    set(_CMAKE_RC_COMPILER_LIST rc)
  endif()

  # Find the compiler.
  find_program(CMAKE_RC_COMPILER NAMES ${_CMAKE_RC_COMPILER_LIST} DOC "RC compiler")
  if(_CMAKE_RC_COMPILER_FALLBACK AND NOT CMAKE_RC_COMPILER)
    set(CMAKE_RC_COMPILER "${_CMAKE_RC_COMPILER_FALLBACK}" CACHE FILEPATH "RC compiler" FORCE)
  endif()
  unset(_CMAKE_RC_COMPILER_FALLBACK)
  unset(_CMAKE_RC_COMPILER_LIST)
endif()

mark_as_advanced(CMAKE_RC_COMPILER)

get_filename_component(_CMAKE_RC_COMPILER_NAME_WE ${CMAKE_RC_COMPILER} NAME_WE)
if(_CMAKE_RC_COMPILER_NAME_WE STREQUAL "windres")
  set(CMAKE_RC_OUTPUT_EXTENSION .obj)
else()
  set(CMAKE_RC_OUTPUT_EXTENSION .res)
endif()

# configure variables set in this file for fast reload later on
configure_file(${CMAKE_ROOT}/Modules/CMakeRCCompiler.cmake.in
               ${CMAKE_PLATFORM_INFO_DIR}/CMakeRCCompiler.cmake)
set(CMAKE_RC_COMPILER_ENV_VAR "RC")
