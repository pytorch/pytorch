# This module is shared by multiple languages; use include blocker.
include_guard()

set(CMAKE_BUILD_TYPE_INIT Debug)

if(DEFINED CMAKE_SYSTEM_PROCESSOR AND CMAKE_SYSTEM_PROCESSOR STREQUAL "I86")
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " system os2")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " system os2_dll")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT " system os2_dll")
else()
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " system os2v2")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " system os2v2_dll")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT " system os2v2_dll")
endif()

set(CMAKE_C_COMPILE_OPTIONS_DLL "-bd") # Note: This variable is a ';' separated list
set(CMAKE_SHARED_LIBRARY_C_FLAGS "-bd") # ... while this is a space separated string.

cmake_policy(GET CMP0136 __OS2_WATCOM_CMP0136)
if(__OS2_WATCOM_CMP0136 STREQUAL "NEW")
  set(CMAKE_WATCOM_RUNTIME_LIBRARY_DEFAULT "SingleThreaded")
else()
  set(CMAKE_WATCOM_RUNTIME_LIBRARY_DEFAULT "")
endif()
unset(__OS2_WATCOM_CMP0136)

string(APPEND CMAKE_C_FLAGS_INIT " -bt=os2")
string(APPEND CMAKE_CXX_FLAGS_INIT " -bt=os2 -xs")

macro(__os2_open_watcom lang)
  if(NOT CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES)
    if(DEFINED CMAKE_SYSTEM_PROCESSOR AND CMAKE_SYSTEM_PROCESSOR STREQUAL "I86")
      set(CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES $ENV{WATCOM}/h $ENV{WATCOM}/h/os21x)
    else()
      set(CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES $ENV{WATCOM}/h $ENV{WATCOM}/h/os2)
    endif()
  endif()
  set(CMAKE_${lang}_COMPILE_OPTIONS_WATCOM_RUNTIME_LIBRARY_SingleThreaded         "")
  set(CMAKE_${lang}_COMPILE_OPTIONS_WATCOM_RUNTIME_LIBRARY_SingleThreadedDLL      -br)
  set(CMAKE_${lang}_COMPILE_OPTIONS_WATCOM_RUNTIME_LIBRARY_MultiThreaded          -bm)
  set(CMAKE_${lang}_COMPILE_OPTIONS_WATCOM_RUNTIME_LIBRARY_MultiThreadedDLL       -bm -br)
endmacro()
