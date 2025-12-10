
# This module is shared by multiple languages; use include blocker.
include_guard()

set(CMAKE_BUILD_TYPE_INIT Debug)

if(DEFINED CMAKE_SYSTEM_PROCESSOR AND CMAKE_SYSTEM_PROCESSOR STREQUAL "I86")
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " system dos")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " system dos")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT " system dos")
else()
  string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " system dos4g")
  string(APPEND CMAKE_SHARED_LINKER_FLAGS_INIT " system dos4g")
  string(APPEND CMAKE_MODULE_LINKER_FLAGS_INIT " system dos4g")
endif()

set(CMAKE_C_COMPILE_OPTIONS_DLL "-bd") # Note: This variable is a ';' separated list
set(CMAKE_SHARED_LIBRARY_C_FLAGS "-bd") # ... while this is a space separated string.

string(APPEND CMAKE_C_FLAGS_INIT " -bt=dos")
string(APPEND CMAKE_CXX_FLAGS_INIT " -bt=dos -xs")

macro(__dos_open_watcom lang)
  if(NOT CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES)
    set(CMAKE_${lang}_STANDARD_INCLUDE_DIRECTORIES $ENV{WATCOM}/h)
  endif()
endmacro()
