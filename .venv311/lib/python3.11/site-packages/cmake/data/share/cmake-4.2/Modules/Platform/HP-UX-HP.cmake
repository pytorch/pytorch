# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__HPUX_COMPILER_HP)
  return()
endif()
set(__HPUX_COMPILER_HP 1)

macro(__hpux_compiler_hp lang)
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "+Z")
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "+Z")
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-Wl,-E,+nodefaultrpath -b -L/usr/lib")
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS "-Wl,-E")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG "-Wl,+b")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP ":")
  set(CMAKE_SHARED_LIBRARY_SONAME_${lang}_FLAG "-Wl,+h")

  string(APPEND CMAKE_${lang}_FLAGS_INIT " ")

  set(CMAKE_${lang}_LINK_FLAGS "-Wl,+s,+nodefaultrpath")
endmacro()
