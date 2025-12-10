# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__CYGWIN_COMPILER_GNU)
  return()
endif()
set(__CYGWIN_COMPILER_GNU 1)

# TODO: Is -Wl,--enable-auto-import now always default?
string(APPEND CMAKE_EXE_LINKER_FLAGS_INIT " -Wl,--enable-auto-import")

set(CMAKE_GNULD_IMAGE_VERSION
  "-Wl,--major-image-version,<TARGET_VERSION_MAJOR>,--minor-image-version,<TARGET_VERSION_MINOR>")
set(CMAKE_GENERATOR_RC windres)


macro(__cygwin_compiler_gnu lang)
  # Binary link rules.
  set(CMAKE_${lang}_CREATE_SHARED_MODULE
    "<CMAKE_${lang}_COMPILER> <LANGUAGE_COMPILE_FLAGS> <CMAKE_SHARED_MODULE_${lang}_FLAGS> <LINK_FLAGS> -o <TARGET> ${CMAKE_GNULD_IMAGE_VERSION} <OBJECTS> <LINK_LIBRARIES>")
  set(CMAKE_${lang}_CREATE_SHARED_LIBRARY
    "<CMAKE_${lang}_COMPILER> <LANGUAGE_COMPILE_FLAGS> <CMAKE_SHARED_LIBRARY_${lang}_FLAGS> <LINK_FLAGS> -o <TARGET> -Wl,--out-implib,<TARGET_IMPLIB> ${CMAKE_GNULD_IMAGE_VERSION} <OBJECTS> <LINK_LIBRARIES>")
  set(CMAKE_${lang}_LINK_EXECUTABLE
    "<CMAKE_${lang}_COMPILER> <FLAGS> <LINK_FLAGS> <OBJECTS> -o <TARGET> -Wl,--out-implib,<TARGET_IMPLIB> ${CMAKE_GNULD_IMAGE_VERSION} <LINK_LIBRARIES>")
  set(CMAKE_${lang}_CREATE_WIN32_EXE "-mwindows")

  set(CMAKE_${lang}_VERBOSE_LINK_FLAG "-Wl,-v")

   # No -fPIC on cygwin
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "")
  set(_CMAKE_${lang}_PIE_MAY_BE_SUPPORTED_BY_LINKER NO)
  set(CMAKE_${lang}_LINK_OPTIONS_PIE "")
  set(CMAKE_${lang}_LINK_OPTIONS_NO_PIE "")
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "")

  # Initialize C link type selection flags.  These flags are used when
  # building a shared library, shared module, or executable that links
  # to other libraries to select whether to use the static or shared
  # versions of the libraries.
  foreach(type SHARED_LIBRARY SHARED_MODULE EXE)
    set(CMAKE_${type}_LINK_STATIC_${lang}_FLAGS "-Wl,-Bstatic")
    set(CMAKE_${type}_LINK_DYNAMIC_${lang}_FLAGS "-Wl,-Bdynamic")
  endforeach()

  set(CMAKE_EXE_EXPORTS_${lang}_FLAG "-Wl,--export-all-symbols")
  # TODO: Is -Wl,--enable-auto-import now always default?
  string(APPEND CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS " -Wl,--enable-auto-import")
  set(CMAKE_SHARED_MODULE_CREATE_${lang}_FLAGS "${CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS}")

  if(NOT CMAKE_RC_COMPILER_INIT)
    set(CMAKE_RC_COMPILER_INIT windres)
  endif()

  enable_language(RC)
endmacro()
