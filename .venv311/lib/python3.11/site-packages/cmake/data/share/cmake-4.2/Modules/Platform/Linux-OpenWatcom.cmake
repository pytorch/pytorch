# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
include_guard()

set(CMAKE_BUILD_TYPE_INIT Debug)

string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " system linux opt noextension")
string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT " system linux")
string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " system linux")

cmake_policy(GET CMP0136 __LINUX_WATCOM_CMP0136)
if(__LINUX_WATCOM_CMP0136 STREQUAL "NEW")
  set(CMAKE_WATCOM_RUNTIME_LIBRARY_DEFAULT "SingleThreaded")
else()
  set(CMAKE_WATCOM_RUNTIME_LIBRARY_DEFAULT "")
endif()
unset(__LINUX_WATCOM_CMP0136)

# single/multi-threaded                 /-bm
# default is setup for single-threaded libraries
string(APPEND CMAKE_C_FLAGS_INIT " -bt=linux")
string(APPEND CMAKE_CXX_FLAGS_INIT " -bt=linux -xs")

macro(__linux_open_watcom lang)
  if(CMAKE_CROSSCOMPILING)
    if(NOT CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES)
      set(CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES $ENV{WATCOM}/lh)
    endif()
  endif()
  set(CMAKE_${lang}_COMPILE_OPTIONS_WATCOM_RUNTIME_LIBRARY_SingleThreaded         "")
  set(CMAKE_${lang}_COMPILE_OPTIONS_WATCOM_RUNTIME_LIBRARY_MultiThreaded          -bm)
endmacro()
