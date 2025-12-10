# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This module is shared by multiple languages; use include blocker.
if(__LINUX_COMPILER_INTEL)
  return()
endif()
set(__LINUX_COMPILER_INTEL 1)

if(NOT XIAR)
  set(_intel_xiar_hints)
  foreach(lang C CXX Fortran)
    if(IS_ABSOLUTE "${CMAKE_${lang}_COMPILER}")
      get_filename_component(_hint "${CMAKE_${lang}_COMPILER}" PATH)
      list(APPEND _intel_xiar_hints ${_hint})
    endif()
  endforeach()
  find_program(XIAR NAMES xiar HINTS ${_intel_xiar_hints})
  mark_as_advanced(XIAR)
endif()

macro(__linux_compiler_intel lang)
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIC "-fPIC")
  set(CMAKE_${lang}_COMPILE_OPTIONS_PIE "-fPIE")
  set(_CMAKE_${lang}_PIE_MAY_BE_SUPPORTED_BY_LINKER NO)
  if (NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 13.0)
    set(_CMAKE_${lang}_PIE_MAY_BE_SUPPORTED_BY_LINKER YES)
    set(CMAKE_${lang}_LINK_OPTIONS_PIE ${CMAKE_${lang}_COMPILE_OPTIONS_PIE} "-pie")
    set(CMAKE_${lang}_LINK_OPTIONS_NO_PIE "-no-pie")
  endif()
  set(CMAKE_SHARED_LIBRARY_${lang}_FLAGS "-fPIC")
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-shared")

  # We pass this for historical reasons.  Projects may have
  # executables that use dlopen but do not set ENABLE_EXPORTS.
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS "-rdynamic")

  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG "-Wl,")
  set(CMAKE_${lang}_LINKER_WRAPPER_FLAG_SEP ",")

  # FIXME(#26157): compute CMAKE_<LANG>_COMPILER_LINKER* variables
  # in the meantime, enforce deactivation of push/pop state linker options
  # because xild front-end linker do not support these options even if the platform linker does...
  set(CMAKE_${lang}_LINKER_PUSHPOP_STATE_SUPPORTED FALSE)

  set(_CMAKE_${lang}_IPO_SUPPORTED_BY_CMAKE YES)

  if(XIAR)
    # INTERPROCEDURAL_OPTIMIZATION
    set(CMAKE_${lang}_COMPILE_OPTIONS_IPO -ipo)
    set(CMAKE_${lang}_CREATE_STATIC_LIBRARY_IPO
      "${XIAR} cr <TARGET> <LINK_FLAGS> <OBJECTS> "
      "${XIAR} -s <TARGET> ")
    set(_CMAKE_${lang}_IPO_MAY_BE_SUPPORTED_BY_COMPILER YES)
    set(_CMAKE_${lang}_IPO_LEGACY_BEHAVIOR YES)
  else()
    set(_CMAKE_${lang}_IPO_MAY_BE_SUPPORTED_BY_COMPILER NO)
  endif()

  if(NOT CMAKE_${lang}_COMPILER_VERSION VERSION_LESS 12.0)
    set(CMAKE_${lang}_COMPILE_OPTIONS_VISIBILITY "-fvisibility=")
  endif()
endmacro()
