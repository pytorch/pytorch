# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__HPUX_COMPILER_GNU)
  return()
endif()
set(__HPUX_COMPILER_GNU 1)

macro(__hpux_compiler_gnu lang)
  string(APPEND CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS " -Wl,-E,-b,+nodefaultrpath")
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS "-Wl,-E")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG "-Wl,+b")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP ":")
  set(CMAKE_SHARED_LIBRARY_SONAME_${lang}_FLAG "-Wl,+h")

  set(CMAKE_${lang}_LINK_FLAGS "-Wl,+s,+nodefaultrpath")
  unset(CMAKE_${lang}_COMPILE_OPTIONS_VISIBILITY)
endmacro()
