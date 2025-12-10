# Guard against multiple inclusions
if(__cmake_craype_crayprgenv)
  return()
endif()
set(__cmake_craype_crayprgenv 1)

# CrayPrgEnv: loaded when compiling through the Cray compiler wrapper.
# The compiler wrapper can run on a front-end node or a compute node.

# One-time setup of the craype environment.  First, check the wrapper config.
# The wrapper's selection of a compiler (gcc, clang, intel, etc.) and
# default include/library paths is selected using the "module" command.
# The CRAYPE_LINK_TYPE environment variable partly controls if static
# or dynamic binaries are generated (see __cmake_craype_linktype below).
# Running cmake and then changing module and/or linktype configuration
# may cause build problems (since the data in the cmake cache may no
# longer be correct after the change).  We can look for this and warn
# the user about it.  Second, use the "module" provided PKG_CONFIG_PATH-like
# environment variable to add additional prefixes to the system prefix
# path.
function(__cmake_craype_setupenv)
  if(NOT DEFINED __cmake_craype_setupenv_done)  # only done once per run
    set(__cmake_craype_setupenv_done 1 PARENT_SCOPE)
    unset(__cmake_check)
    set(CMAKE_CRAYPE_LINKTYPE "$ENV{CRAYPE_LINK_TYPE}" CACHE STRING
        "saved value of CRAYPE_LINK_TYPE environment variable")
    set(CMAKE_CRAYPE_LOADEDMODULES "$ENV{LOADEDMODULES}" CACHE STRING
        "saved value of LOADEDMODULES environment variable")
    mark_as_advanced(CMAKE_CRAYPE_LINKTYPE CMAKE_CRAYPE_LOADEDMODULES)
    if (NOT "${CMAKE_CRAYPE_LINKTYPE}" STREQUAL "$ENV{CRAYPE_LINK_TYPE}")
      string(APPEND __cmake_check "CRAYPE_LINK_TYPE ")
    endif()
    if (NOT "${CMAKE_CRAYPE_LOADEDMODULES}" STREQUAL "$ENV{LOADEDMODULES}")
      string(APPEND __cmake_check "LOADEDMODULES ")
    endif()
    if(DEFINED __cmake_check)
      message(STATUS "NOTE: ${__cmake_check}changed since initial config!")
      message(STATUS "NOTE: this may cause unexpected build errors.")
    endif()
    # loop over variables of interest
    foreach(pkgcfgvar PKG_CONFIG_PATH PKG_CONFIG_PATH_DEFAULT
            PE_PKG_CONFIG_PATH)
      file(TO_CMAKE_PATH "$ENV{${pkgcfgvar}}" pkgcfg)
      foreach(path ${pkgcfg})
        string(REGEX REPLACE "(.*)/lib[^/]*/pkgconfig$" "\\1" path "${path}")
        if(NOT "${path}" STREQUAL "" AND
           NOT "${path}" IN_LIST CMAKE_SYSTEM_PREFIX_PATH)
          list(APPEND CMAKE_SYSTEM_PREFIX_PATH "${path}")
        endif()
      endforeach()
    endforeach()
    # push it up out of this function into the parent scope
    set(CMAKE_SYSTEM_PREFIX_PATH "${CMAKE_SYSTEM_PREFIX_PATH}" PARENT_SCOPE)
  endif()
endfunction()

# The wrapper disables dynamic linking by default.  Dynamic linking is
# enabled either by setting $ENV{CRAYPE_LINK_TYPE} to "dynamic" or by
# specifying "-dynamic" to the wrapper when linking.  Specifying "-static"
# to the wrapper when linking takes priority over $ENV{CRAYPE_LINK_TYPE}.
# Furthermore, if you specify multiple "-dynamic" and "-static" flags to
# the wrapper when linking, the last one will win.  In this case, the
# wrapper will also print a warning like:
#  Warning: -dynamic was already seen on command line, overriding with -static.
#
# note that cmake applies both CMAKE_${lang}_FLAGS and CMAKE_EXE_LINKER_FLAGS
# (in that order) to the linking command, so -dynamic can appear in either
# variable.
#
# Note: As of CrayPE v19.06 (which translates to the craype/2.6.0 module)
# the default has changed and is now dynamic by default.  This is handled
# accordingly
function(__cmake_craype_linktype lang rv)
  # start with ENV, but allow flags to override
  if(("$ENV{CRAYPE_VERSION}" STREQUAL "") OR
     ("$ENV{CRAYPE_VERSION}" VERSION_LESS "2.6"))
    if("$ENV{CRAYPE_LINK_TYPE}" STREQUAL "dynamic")
      set(linktype dynamic)
    else()
      set(linktype static)
    endif()
  else()
    if("$ENV{CRAYPE_LINK_TYPE}" STREQUAL "static")
      set(linktype static)
    else()
      set(linktype dynamic)
    endif()
  endif()

  # combine flags and convert to a list so we can apply the flags in order
  set(linkflags "${CMAKE_${lang}_FLAGS} ${CMAKE_EXE_LINKER_FLAGS}")
  string(REPLACE " " ";" linkflags "${linkflags}")
  foreach(flag IN LISTS linkflags)
    if("${flag}" STREQUAL "-dynamic")
      set(linktype dynamic)
    elseif("${flag}" STREQUAL "-static")
      set(linktype static)
    endif()
  endforeach()
  set(${rv} ${linktype} PARENT_SCOPE)
endfunction()

macro(__CrayPrgEnv_setup lang)
  if(DEFINED ENV{CRAYPE_VERSION})
    message(STATUS "Cray Programming Environment $ENV{CRAYPE_VERSION} ${lang}")
  elseif(DEFINED ENV{ASYNCPE_VERSION})
    message(STATUS "Cray XT Programming Environment $ENV{ASYNCPE_VERSION} ${lang}")
  else()
    message(STATUS "Cray Programming Environment (unknown version) ${lang}")
  endif()

  # setup the craype environment
  __cmake_craype_setupenv()

  # Flags for the Cray wrappers
  set(CMAKE_SHARED_LIBRARY_CREATE_${lang}_FLAGS "-shared")
  set(CMAKE_SHARED_LIBRARY_LINK_${lang}_FLAGS "-dynamic")

  # determine linktype from environment and compiler flags
  __cmake_craype_linktype(${lang} __cmake_craype_${lang}_linktype)

  # switch off shared libs if we get a static linktype
  if("${__cmake_craype_${lang}_linktype}" STREQUAL "static")
    set_property(GLOBAL PROPERTY TARGET_SUPPORTS_SHARED_LIBS FALSE)
    set(BUILD_SHARED_LIBS FALSE CACHE BOOL "")
    set(CMAKE_FIND_LIBRARY_SUFFIXES ".a")
    set(CMAKE_LINK_SEARCH_START_STATIC TRUE)
  endif()

endmacro()
