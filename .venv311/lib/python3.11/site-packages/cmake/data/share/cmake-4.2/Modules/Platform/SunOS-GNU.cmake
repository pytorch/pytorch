# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__SUNOS_COMPILER_GNU)
  return()
endif()
set(__SUNOS_COMPILER_GNU 1)

macro(__sunos_compiler_gnu lang)
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG "-Wl,-R")
  set(CMAKE_SHARED_LIBRARY_RUNTIME_${lang}_FLAG_SEP ":")
  set(CMAKE_SHARED_LIBRARY_RPATH_ORIGIN_TOKEN "\$ORIGIN")
  set(CMAKE_SHARED_LIBRARY_SONAME_${lang}_FLAG "-Wl,-h")

  # Initialize C link type selection flags.  These flags are used when
  # building a shared library, shared module, or executable that links
  # to other libraries to select whether to use the static or shared
  # versions of the libraries.
  foreach(type SHARED_LIBRARY SHARED_MODULE EXE)
    set(CMAKE_${type}_LINK_STATIC_${lang}_FLAGS "-Wl,-Bstatic")
    set(CMAKE_${type}_LINK_DYNAMIC_${lang}_FLAGS "-Wl,-Bdynamic")
  endforeach()
endmacro()
