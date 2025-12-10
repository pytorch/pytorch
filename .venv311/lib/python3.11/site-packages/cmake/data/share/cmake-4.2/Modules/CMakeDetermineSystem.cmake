# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is used by the Makefile generator to determine the following variables:
# CMAKE_SYSTEM_NAME - on unix this is uname -s, for windows it is Windows
# CMAKE_SYSTEM_VERSION - on unix this is uname -r, for windows it is empty
# CMAKE_SYSTEM - ${CMAKE_SYSTEM}-${CMAKE_SYSTEM_VERSION}, for windows: ${CMAKE_SYSTEM}

# find out on which system cmake runs
if(CMAKE_HOST_UNIX)
  find_program(CMAKE_UNAME NAMES uname PATHS /bin /usr/bin /usr/local/bin)
  if(CMAKE_UNAME)
    if(CMAKE_HOST_SYSTEM_NAME STREQUAL "AIX")
      execute_process(COMMAND ${CMAKE_UNAME} -v
        OUTPUT_VARIABLE _CMAKE_HOST_SYSTEM_MAJOR_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      execute_process(COMMAND ${CMAKE_UNAME} -r
        OUTPUT_VARIABLE _CMAKE_HOST_SYSTEM_MINOR_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      set(CMAKE_HOST_SYSTEM_VERSION "${_CMAKE_HOST_SYSTEM_MAJOR_VERSION}.${_CMAKE_HOST_SYSTEM_MINOR_VERSION}")
      unset(_CMAKE_HOST_SYSTEM_MAJOR_VERSION)
      unset(_CMAKE_HOST_SYSTEM_MINOR_VERSION)
    elseif(CMAKE_HOST_SYSTEM_NAME STREQUAL "Android")
      execute_process(COMMAND getprop ro.build.version.sdk
        OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)

      if(NOT DEFINED CMAKE_SYSTEM_VERSION)
        set(_ANDROID_API_LEVEL_H $ENV{PREFIX}/include/android/api-level.h)
        set(_ANDROID_API_REGEX "#define __ANDROID_API__ ([0-9]+)")
        file(READ ${_ANDROID_API_LEVEL_H} _ANDROID_API_LEVEL_H_CONTENT)
        string(REGEX MATCH ${_ANDROID_API_REGEX} _ANDROID_API_LINE "${_ANDROID_API_LEVEL_H_CONTENT}")
        string(REGEX REPLACE ${_ANDROID_API_REGEX} "\\1" _ANDROID_API "${_ANDROID_API_LINE}")
        if(_ANDROID_API)
          set(CMAKE_SYSTEM_VERSION "${_ANDROID_API}")
        endif()

        unset(_ANDROID_API_LEVEL_H)
        unset(_ANDROID_API_LEVEL_H_CONTENT)
        unset(_ANDROID_API_REGEX)
        unset(_ANDROID_API_LINE)
        unset(_ANDROID_API)
      endif()
    else()
      execute_process(COMMAND ${CMAKE_UNAME} -r
        OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_VERSION
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    endif()
    if(CMAKE_HOST_SYSTEM_NAME MATCHES "Linux|CYGWIN.*|MSYS.*|^GNU$|Android")
      execute_process(COMMAND ${CMAKE_UNAME} -m
        OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_PROCESSOR
        RESULT_VARIABLE val
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    elseif(CMAKE_HOST_SYSTEM_NAME MATCHES "Darwin")
      # If we are running on Apple Silicon, honor CMAKE_APPLE_SILICON_PROCESSOR.
      if(DEFINED CMAKE_APPLE_SILICON_PROCESSOR)
        set(_CMAKE_APPLE_SILICON_PROCESSOR "${CMAKE_APPLE_SILICON_PROCESSOR}")
      elseif(DEFINED ENV{CMAKE_APPLE_SILICON_PROCESSOR})
        set(_CMAKE_APPLE_SILICON_PROCESSOR "$ENV{CMAKE_APPLE_SILICON_PROCESSOR}")
      else()
        set(_CMAKE_APPLE_SILICON_PROCESSOR "")
      endif()
      if(_CMAKE_APPLE_SILICON_PROCESSOR)
        if(";${_CMAKE_APPLE_SILICON_PROCESSOR};" MATCHES "^;(arm64|x86_64);$")
          execute_process(COMMAND sysctl -q hw.optional.arm64
            OUTPUT_VARIABLE _sysctl_stdout
            ERROR_VARIABLE _sysctl_stderr
            RESULT_VARIABLE _sysctl_result
            )
          if(NOT _sysctl_result EQUAL 0 OR NOT _sysctl_stdout MATCHES "hw.optional.arm64: 1")
            set(_CMAKE_APPLE_SILICON_PROCESSOR "")
          endif()
          unset(_sysctl_result)
          unset(_sysctl_stderr)
          unset(_sysctl_stdout)
        endif()
      endif()
      if(_CMAKE_APPLE_SILICON_PROCESSOR)
        set(CMAKE_HOST_SYSTEM_PROCESSOR "${_CMAKE_APPLE_SILICON_PROCESSOR}")
      else()
        execute_process(COMMAND ${CMAKE_UNAME} -m
          OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_PROCESSOR
          RESULT_VARIABLE val
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET)
      endif()
      unset(_CMAKE_APPLE_SILICON_PROCESSOR)
      if(CMAKE_HOST_SYSTEM_PROCESSOR STREQUAL "Power Macintosh")
        # OS X ppc 'uname -m' may report 'Power Macintosh' instead of 'powerpc'
        set(CMAKE_HOST_SYSTEM_PROCESSOR "powerpc")
      endif()
    elseif(CMAKE_HOST_SYSTEM_NAME MATCHES "OpenBSD")
      execute_process(COMMAND arch -s
        OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_PROCESSOR
        RESULT_VARIABLE val
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
    else()
      execute_process(COMMAND ${CMAKE_UNAME} -p
        OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_PROCESSOR
        RESULT_VARIABLE val
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET)
      if("${val}" GREATER 0)
        execute_process(COMMAND ${CMAKE_UNAME} -m
          OUTPUT_VARIABLE CMAKE_HOST_SYSTEM_PROCESSOR
          RESULT_VARIABLE val
          OUTPUT_STRIP_TRAILING_WHITESPACE
          ERROR_QUIET)
      endif()
    endif()
    # check the return of the last uname -m or -p
    if("${val}" GREATER 0)
        set(CMAKE_HOST_SYSTEM_PROCESSOR "unknown")
    endif()
    set(CMAKE_UNAME ${CMAKE_UNAME} CACHE INTERNAL "uname command")
    # processor may have double quote in the name, and that needs to be removed
    string(REPLACE "\"" "" CMAKE_HOST_SYSTEM_PROCESSOR "${CMAKE_HOST_SYSTEM_PROCESSOR}")
    string(REPLACE "/" "_" CMAKE_HOST_SYSTEM_PROCESSOR "${CMAKE_HOST_SYSTEM_PROCESSOR}")
  endif()
else()
  if(CMAKE_HOST_WIN32)
    if (DEFINED ENV{PROCESSOR_ARCHITEW6432})
      set (CMAKE_HOST_SYSTEM_PROCESSOR "$ENV{PROCESSOR_ARCHITEW6432}")
    else()
      set (CMAKE_HOST_SYSTEM_PROCESSOR "$ENV{PROCESSOR_ARCHITECTURE}")
    endif()
  endif()
endif()

# if a toolchain file is used, the user wants to cross compile.
# in this case read the toolchain file and keep the CMAKE_HOST_SYSTEM_*
# variables around so they can be used in CMakeLists.txt.
# In all other cases, the host and target platform are the same.
if(CMAKE_TOOLCHAIN_FILE)
  if(IS_ABSOLUTE "${CMAKE_TOOLCHAIN_FILE}" AND EXISTS "${CMAKE_TOOLCHAIN_FILE}")
    # Normalize the absolute path.
    set(CMAKE_TOOLCHAIN_FILE "${CMAKE_TOOLCHAIN_FILE}" CACHE FILEPATH "The CMake toolchain file" FORCE)
    set(CMAKE_TOOLCHAIN_FILE "$CACHE{CMAKE_TOOLCHAIN_FILE}")
    include("${CMAKE_TOOLCHAIN_FILE}" OPTIONAL RESULT_VARIABLE _INCLUDED_TOOLCHAIN_FILE)
  else()
    # at first try to load it as path relative to the directory from which cmake has been run
    include("${CMAKE_BINARY_DIR}/${CMAKE_TOOLCHAIN_FILE}" OPTIONAL RESULT_VARIABLE _INCLUDED_TOOLCHAIN_FILE)
    if(NOT _INCLUDED_TOOLCHAIN_FILE)
      # if the file isn't found there, check the default locations
      include("${CMAKE_TOOLCHAIN_FILE}" OPTIONAL RESULT_VARIABLE _INCLUDED_TOOLCHAIN_FILE)
    endif()
    if(_INCLUDED_TOOLCHAIN_FILE)
      set(CMAKE_TOOLCHAIN_FILE "${_INCLUDED_TOOLCHAIN_FILE}" CACHE FILEPATH "The CMake toolchain file" FORCE)
    endif()
  endif()
  if(NOT _INCLUDED_TOOLCHAIN_FILE)
    message(FATAL_ERROR "Could not find toolchain file:\n \"${CMAKE_TOOLCHAIN_FILE}\"")
  endif()
endif()

if(CMAKE_SYSTEM_NAME)
  # CMAKE_SYSTEM_NAME was set by a toolchain file or on the command line.
  # Assume it set CMAKE_SYSTEM_VERSION and CMAKE_SYSTEM_PROCESSOR too.
  if(NOT DEFINED CMAKE_CROSSCOMPILING)
    set(CMAKE_CROSSCOMPILING TRUE)
  endif()
elseif(CMAKE_VS_WINCE_VERSION)
  set(CMAKE_SYSTEM_NAME      "WindowsCE")
  set(CMAKE_SYSTEM_VERSION   "${CMAKE_VS_WINCE_VERSION}")
  set(CMAKE_SYSTEM_PROCESSOR "${MSVC_C_ARCHITECTURE_ID}")
  set(CMAKE_CROSSCOMPILING TRUE)
else()
  # Build for the host platform and architecture by default.
  set(CMAKE_SYSTEM_NAME      "${CMAKE_HOST_SYSTEM_NAME}")
  if(NOT DEFINED CMAKE_SYSTEM_VERSION)
    set(CMAKE_SYSTEM_VERSION "${CMAKE_HOST_SYSTEM_VERSION}")
  endif()
  set(CMAKE_SYSTEM_PROCESSOR "${CMAKE_HOST_SYSTEM_PROCESSOR}")
  if(CMAKE_CROSSCOMPILING)
    message(AUTHOR_WARNING
      "CMAKE_CROSSCOMPILING has been set by the project, toolchain file, or user.  "
      "CMake is resetting it to false because CMAKE_SYSTEM_NAME was not set.  "
      "To indicate cross compilation, only CMAKE_SYSTEM_NAME needs to be set."
      )
  endif()
  set(CMAKE_CROSSCOMPILING FALSE)
endif()

include(Platform/${CMAKE_SYSTEM_NAME}-Determine OPTIONAL)

set(CMAKE_SYSTEM ${CMAKE_SYSTEM_NAME})
if(CMAKE_SYSTEM_VERSION)
  string(APPEND CMAKE_SYSTEM -${CMAKE_SYSTEM_VERSION})
endif()
set(CMAKE_HOST_SYSTEM ${CMAKE_HOST_SYSTEM_NAME})
if(CMAKE_HOST_SYSTEM_VERSION)
  string(APPEND CMAKE_HOST_SYSTEM -${CMAKE_HOST_SYSTEM_VERSION})
endif()

# this file is also executed from cpack, then we don't need to generate these files
# in this case there is no CMAKE_BINARY_DIR
if(CMAKE_BINARY_DIR)
  # write entry to the log file
  if(CMAKE_CROSSCOMPILING)
    message(CONFIGURE_LOG
      "The target system is: ${CMAKE_SYSTEM_NAME} - ${CMAKE_SYSTEM_VERSION} - ${CMAKE_SYSTEM_PROCESSOR}\n"
      "The host system is: ${CMAKE_HOST_SYSTEM_NAME} - ${CMAKE_HOST_SYSTEM_VERSION} - ${CMAKE_HOST_SYSTEM_PROCESSOR}\n"
      )
  else()
    message(CONFIGURE_LOG
      "The system is: ${CMAKE_SYSTEM_NAME} - ${CMAKE_SYSTEM_VERSION} - ${CMAKE_SYSTEM_PROCESSOR}\n"
      )
  endif()

  # if a toolchain file is used, it needs to be included in the configured file,
  # so settings done there are also available if they don't go in the cache and in try_compile()
  set(INCLUDE_CMAKE_TOOLCHAIN_FILE_IF_REQUIRED)
  if(CMAKE_TOOLCHAIN_FILE)
    set(INCLUDE_CMAKE_TOOLCHAIN_FILE_IF_REQUIRED "include(\"${CMAKE_TOOLCHAIN_FILE}\")")
  endif()

  # configure variables set in this file for fast reload, the template file is defined at the top of this file
  configure_file(${CMAKE_ROOT}/Modules/CMakeSystem.cmake.in
                ${CMAKE_PLATFORM_INFO_DIR}/CMakeSystem.cmake
                @ONLY)

endif()
