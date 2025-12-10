# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# determine the compiler to use for Objective-C programs
# NOTE, a generator may set CMAKE_OBJC_COMPILER before
# loading this file to force a compiler.
# use environment variable OBJC first if defined by user, next use
# the cmake variable CMAKE_GENERATOR_OBJC which can be defined by a generator
# as a default compiler
#
# Sets the following variables:
#   CMAKE_OBJC_COMPILER
#   CMAKE_AR
#   CMAKE_RANLIB
#
# If not already set before, it also sets
#   _CMAKE_TOOLCHAIN_PREFIX

include(${CMAKE_ROOT}/Modules/CMakeDetermineCompiler.cmake)

# Load system-specific compiler preferences for this language.
include(Platform/${CMAKE_SYSTEM_NAME}-Determine-OBJC OPTIONAL)
include(Platform/${CMAKE_SYSTEM_NAME}-OBJC OPTIONAL)
if(NOT CMAKE_OBJC_COMPILER_NAMES)
  set(CMAKE_OBJC_COMPILER_NAMES clang)
endif()

if("${CMAKE_GENERATOR}" MATCHES "Xcode")
  set(CMAKE_OBJC_COMPILER_XCODE_TYPE sourcecode.c.objc)
else()
  if(NOT CMAKE_OBJC_COMPILER)
    set(CMAKE_OBJC_COMPILER_INIT NOTFOUND)

    # prefer the environment variable OBJC or CC
    foreach(var OBJC CC)
      if($ENV{${var}} MATCHES ".+")
        get_filename_component(CMAKE_OBJC_COMPILER_INIT $ENV{${var}} PROGRAM PROGRAM_ARGS CMAKE_OBJC_FLAGS_ENV_INIT)
        if(CMAKE_OBJC_FLAGS_ENV_INIT)
          set(CMAKE_OBJC_COMPILER_ARG1 "${CMAKE_OBJC_FLAGS_ENV_INIT}" CACHE STRING "Arguments to Objective-C compiler")
        endif()
        if(NOT EXISTS ${CMAKE_OBJC_COMPILER_INIT})
          message(FATAL_ERROR "Could not find compiler set in environment variable ${var}:\n  $ENV{${var}}")
        endif()
        break()
      endif()
    endforeach()

    # next try prefer the compiler specified by the generator
    if(CMAKE_GENERATOR_OBJC)
      if(NOT CMAKE_OBJC_COMPILER_INIT)
        set(CMAKE_OBJC_COMPILER_INIT ${CMAKE_GENERATOR_OBJC})
      endif()
    endif()

    # finally list compilers to try
    if(NOT CMAKE_OBJC_COMPILER_INIT)
      set(CMAKE_OBJC_COMPILER_LIST ${_CMAKE_TOOLCHAIN_PREFIX}cc ${_CMAKE_TOOLCHAIN_PREFIX}gcc clang)
    endif()

    _cmake_find_compiler(OBJC)

  else()
    # we only get here if CMAKE_OBJC_COMPILER was specified using -D or a pre-made CMakeCache.txt
    # (e.g. via ctest) or set in CMAKE_TOOLCHAIN_FILE
    # if CMAKE_OBJC_COMPILER is a list, use the first item as
    # CMAKE_OBJC_COMPILER and the rest as CMAKE_OBJC_COMPILER_ARG1
    set(CMAKE_OBJC_COMPILER_ARG1 "${CMAKE_OBJC_COMPILER}")
    list(POP_FRONT CMAKE_OBJC_COMPILER_ARG1 CMAKE_OBJC_COMPILER)
    list(JOIN CMAKE_OBJC_COMPILER_ARG1 " " CMAKE_OBJC_COMPILER_ARG1)

    # if a compiler was specified by the user but without path,
    # now try to find it with the full path
    # if it is found, force it into the cache,
    # if not, don't overwrite the setting (which was given by the user) with "NOTFOUND"
    # if the C compiler already had a path, reuse it for searching the CXX compiler
    get_filename_component(_CMAKE_USER_OBJC_COMPILER_PATH "${CMAKE_OBJC_COMPILER}" PATH)
    if(NOT _CMAKE_USER_OBJC_COMPILER_PATH)
      find_program(CMAKE_OBJC_COMPILER_WITH_PATH NAMES ${CMAKE_OBJC_COMPILER})
      if(CMAKE_OBJC_COMPILER_WITH_PATH)
        set(CMAKE_OBJC_COMPILER ${CMAKE_OBJC_COMPILER_WITH_PATH} CACHE STRING "Objective-C compiler" FORCE)
      endif()
      unset(CMAKE_OBJC_COMPILER_WITH_PATH CACHE)
    endif()
  endif()
  mark_as_advanced(CMAKE_OBJC_COMPILER)

  # Each entry in this list is a set of extra flags to try
  # adding to the compile line to see if it helps produce
  # a valid identification file.
  set(CMAKE_OBJC_COMPILER_ID_TEST_FLAGS_FIRST)
  set(CMAKE_OBJC_COMPILER_ID_TEST_FLAGS
    # Try compiling to an object file only.
    "-c"

    )
endif()

# Build a small source file to identify the compiler.
if(NOT CMAKE_OBJC_COMPILER_ID_RUN)
  set(CMAKE_OBJC_COMPILER_ID_RUN 1)

  # Try to identify the compiler.
  set(CMAKE_OBJC_COMPILER_ID)
  file(READ ${CMAKE_ROOT}/Modules/CMakePlatformId.h.in
    CMAKE_OBJC_COMPILER_ID_PLATFORM_CONTENT)

  # Match the link line from xcodebuild output of the form
  #  Ld ...
  #      ...
  #      /path/to/cc ...CompilerIdOBJC/...
  # to extract the compiler front-end for the language.
  set(CMAKE_OBJC_COMPILER_ID_TOOL_MATCH_REGEX "\nLd[^\n]*(\n[ \t]+[^\n]*)*\n[ \t]+([^ \t\r\n]+)[^\r\n]*-o[^\r\n]*CompilerIdOBJC/(\\./)?(CompilerIdOBJC.(framework|xctest|build/[^ \t\r\n]+)/)?CompilerIdOBJC[ \t\n\\\"]")
  set(CMAKE_OBJC_COMPILER_ID_TOOL_MATCH_INDEX 2)

  include(${CMAKE_ROOT}/Modules/CMakeDetermineCompilerId.cmake)
  CMAKE_DETERMINE_COMPILER_ID(OBJC OBJCCFLAGS CMakeOBJCCompilerId.m)

  # Set old compiler and platform id variables.
  if(CMAKE_OBJC_COMPILER_ID STREQUAL "GNU")
    set(CMAKE_COMPILER_IS_GNUOBJC 1)
  endif()
  if(CMAKE_OBJC_COMPILER_ID STREQUAL "Clang")
    set(CMAKE_COMPILER_IS_CLANGOBJC 1)
  endif()
endif()

if (NOT _CMAKE_TOOLCHAIN_LOCATION)
  get_filename_component(_CMAKE_TOOLCHAIN_LOCATION "${CMAKE_OBJC_COMPILER}" PATH)
endif ()

# If we have a gcc cross compiler, they have usually some prefix, like
# e.g. powerpc-linux-gcc, arm-elf-gcc or i586-mingw32msvc-gcc, optionally
# with a 3-component version number at the end (e.g. arm-eabi-gcc-4.5.2).
# The other tools of the toolchain usually have the same prefix
# NAME_WE cannot be used since then this test will fail for names like
# "arm-unknown-nto-qnx6.3.0-gcc.exe", where BASENAME would be
# "arm-unknown-nto-qnx6" instead of the correct "arm-unknown-nto-qnx6.3.0-"
if (CMAKE_CROSSCOMPILING  AND NOT _CMAKE_TOOLCHAIN_PREFIX)

  if(CMAKE_OBJC_COMPILER_ID MATCHES "GNU|Clang|QCC")
    get_filename_component(COMPILER_BASENAME "${CMAKE_OBJC_COMPILER}" NAME)
    if (COMPILER_BASENAME MATCHES "^(.+-)(clang|g?cc)(-[0-9]+(\\.[0-9]+)*)?(-[^.]+)?(\\.exe)?$")
      set(_CMAKE_TOOLCHAIN_PREFIX ${CMAKE_MATCH_1})
      set(_CMAKE_COMPILER_SUFFIX ${CMAKE_MATCH_5})
    elseif(CMAKE_OBJC_COMPILER_ID MATCHES "Clang")
      if(CMAKE_OBJC_COMPILER_TARGET)
        set(_CMAKE_TOOLCHAIN_PREFIX ${CMAKE_OBJC_COMPILER_TARGET}-)
      endif()
    elseif(COMPILER_BASENAME MATCHES "qcc(\\.exe)?$")
      if(CMAKE_OBJC_COMPILER_TARGET MATCHES "gcc_nto([a-z0-9]+_[0-9]+|[^_le]+)(le)?")
        set(_CMAKE_TOOLCHAIN_PREFIX nto${CMAKE_MATCH_1}-)
      endif()
    endif ()

    # if "llvm-" is part of the prefix, remove it, since llvm doesn't have its own binutils
    # but uses the regular ar, objcopy, etc. (instead of llvm-objcopy etc.)
    if ("${_CMAKE_TOOLCHAIN_PREFIX}" MATCHES "(.+-)?llvm-$")
      set(_CMAKE_TOOLCHAIN_PREFIX ${CMAKE_MATCH_1})
    endif ()
  endif()

endif ()

set(_CMAKE_PROCESSING_LANGUAGE "OBJC")
include(CMakeFindBinUtils)
include(Compiler/${CMAKE_OBJC_COMPILER_ID}-FindBinUtils OPTIONAL)
unset(_CMAKE_PROCESSING_LANGUAGE)

if(CMAKE_OBJC_XCODE_ARCHS)
  set(SET_CMAKE_XCODE_ARCHS
    "set(CMAKE_XCODE_ARCHS \"${CMAKE_OBJC_XCODE_ARCHS}\")")
endif()

# configure variables set in this file for fast reload later on
configure_file(${CMAKE_ROOT}/Modules/CMakeOBJCCompiler.cmake.in
  ${CMAKE_PLATFORM_INFO_DIR}/CMakeOBJCCompiler.cmake
  @ONLY
  )
set(CMAKE_OBJC_COMPILER_ENV_VAR "OBJC")
