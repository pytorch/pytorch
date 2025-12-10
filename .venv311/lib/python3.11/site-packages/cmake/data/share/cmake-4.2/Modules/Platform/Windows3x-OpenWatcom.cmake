# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This module is shared by multiple languages; use include blocker.
include_guard()

set(CMAKE_BUILD_TYPE_INIT Debug)

if(DEFINED CMAKE_SYSTEM_PROCESSOR AND CMAKE_SYSTEM_PROCESSOR STREQUAL "I86")
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " system windows")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " system windows_dll")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT " system windows_dll")
else()
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " system win386")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " system win386")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT " system win386")
endif()

set(CMAKE_C_COMPILE_OPTIONS_DLL "-bd") # Note: This variable is a ';' separated list
set(CMAKE_SHARED_LIBRARY_C_FLAGS "-bd") # ... while this is a space separated string.

set(CMAKE_RC_COMPILER "rc")

set(CMAKE_WATCOM_RUNTIME_LIBRARY_DEFAULT "")

string(APPEND CMAKE_C_FLAGS_INIT " -bt=windows")
string(APPEND CMAKE_CXX_FLAGS_INIT " -bt=windows -xs")

macro(__windows3x_open_watcom lang)
  if(NOT CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES)
    set(CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES $ENV{WATCOM}/h $ENV{WATCOM}/h/win)
  endif()
endmacro()
