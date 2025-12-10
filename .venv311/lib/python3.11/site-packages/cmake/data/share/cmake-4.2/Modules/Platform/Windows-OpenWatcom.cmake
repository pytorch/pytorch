# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
include_guard()

set(CMAKE_BUILD_TYPE_INIT Debug)

string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " system nt_dll")
string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT " system nt_dll")

set(CMAKE_C_COMPILE_OPTIONS_DLL "-bd") # Note: This variable is a ';' separated list
set(CMAKE_SHARED_LIBRARY_C_FLAGS "-bd") # ... while this is a space separated string.

set(CMAKE_RC_COMPILER "rc" )

cmake_policy(GET CMP0136 __WINDOWS_WATCOM_CMP0136)
if(__WINDOWS_WATCOM_CMP0136 STREQUAL "NEW")
  set(CMAKE_WATCOM_RUNTIME_LIBRARY_DEFAULT "MultiThreadedDLL")
  set(_br_bm "")
else()
  set(CMAKE_WATCOM_RUNTIME_LIBRARY_DEFAULT "")
  set(_br_bm "-br -bm")
endif()

string(APPEND CMAKE_C_FLAGS_INIT " -bt=nt -dWIN32 ${_br_bm}")
string(APPEND CMAKE_CXX_FLAGS_INIT " -bt=nt -xs -dWIN32 ${_br_bm}")

unset(__WINDOWS_WATCOM_CMP0136)
unset(_br_bm)

macro(__windows_open_watcom lang)
  if(CMAKE_CROSSCOMPILING)
    if(NOT CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES)
      set(CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES $ENV{WATCOM}/h $ENV{WATCOM}/h/nt)
    endif()
  endif()
  set(CMAKE_${lang}_CREATE_WIN32_EXE "system nt_win")
  set(CMAKE_${lang}_CREATE_CONSOLE_EXE "system nt")

  set(CMAKE_${lang}_COMPILE_OPTIONS_WATCOM_RUNTIME_LIBRARY_SingleThreaded         "")
  set(CMAKE_${lang}_COMPILE_OPTIONS_WATCOM_RUNTIME_LIBRARY_SingleThreadedDLL      -br)
  set(CMAKE_${lang}_COMPILE_OPTIONS_WATCOM_RUNTIME_LIBRARY_MultiThreaded          -bm)
  set(CMAKE_${lang}_COMPILE_OPTIONS_WATCOM_RUNTIME_LIBRARY_MultiThreadedDLL       -bm -br)
endmacro()
