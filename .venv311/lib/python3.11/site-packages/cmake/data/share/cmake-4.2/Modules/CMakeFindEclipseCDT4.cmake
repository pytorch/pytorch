# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.


# This file is included in CMakeSystemSpecificInformation.cmake if
# the Eclipse CDT4 extra generator has been selected.

find_program(CMAKE_ECLIPSE_EXECUTABLE NAMES eclipse DOC "The Eclipse executable")

function(_FIND_ECLIPSE_VERSION)
  # This code is in a function so the variables used here have only local scope

  # Set up a map with the names of the Eclipse releases:
  set(_ECLIPSE_VERSION_NAME_    "Unknown" )
  set(_ECLIPSE_VERSION_NAME_3.2 "Callisto" )
  set(_ECLIPSE_VERSION_NAME_3.3 "Europa" )
  set(_ECLIPSE_VERSION_NAME_3.4 "Ganymede" )
  set(_ECLIPSE_VERSION_NAME_3.5 "Galileo" )
  set(_ECLIPSE_VERSION_NAME_3.6 "Helios" )
  set(_ECLIPSE_VERSION_NAME_3.7 "Indigo" )
  set(_ECLIPSE_VERSION_NAME_4.2 "Juno" )
  set(_ECLIPSE_VERSION_NAME_4.3 "Kepler" )
  set(_ECLIPSE_VERSION_NAME_4.4 "Luna" )
  set(_ECLIPSE_VERSION_NAME_4.5 "Mars" )

  if(NOT DEFINED CMAKE_ECLIPSE_VERSION)
    if(CMAKE_ECLIPSE_EXECUTABLE)
      # use REALPATH to resolve symlinks (https://gitlab.kitware.com/cmake/cmake/-/issues/13036)
      get_filename_component(_REALPATH_CMAKE_ECLIPSE_EXECUTABLE "${CMAKE_ECLIPSE_EXECUTABLE}" REALPATH)
      get_filename_component(_ECLIPSE_DIR "${_REALPATH_CMAKE_ECLIPSE_EXECUTABLE}" PATH)
      file(GLOB _ECLIPSE_FEATURE_DIR "${_ECLIPSE_DIR}/features/org.eclipse.platform*")
      if(APPLE AND NOT _ECLIPSE_FEATURE_DIR)
        file(GLOB _ECLIPSE_FEATURE_DIR "${_ECLIPSE_DIR}/../../../features/org.eclipse.platform*")
      endif()
      if("${_ECLIPSE_FEATURE_DIR}" MATCHES ".+org.eclipse.platform_([0-9]+\\.[0-9]+).+")
        set(_ECLIPSE_VERSION ${CMAKE_MATCH_1})
      endif()
    endif()

    if(_ECLIPSE_VERSION)
      message(STATUS "Found Eclipse version ${_ECLIPSE_VERSION} (${_ECLIPSE_VERSION_NAME_${_ECLIPSE_VERSION}})")
    else()
      set(_ECLIPSE_VERSION "3.6" )
      message(STATUS "Could not determine Eclipse version, assuming at least ${_ECLIPSE_VERSION} (${_ECLIPSE_VERSION_NAME_${_ECLIPSE_VERSION}}). Adjust CMAKE_ECLIPSE_VERSION if this is wrong.")
    endif()

    set(CMAKE_ECLIPSE_VERSION "${_ECLIPSE_VERSION} (${_ECLIPSE_VERSION_NAME_${_ECLIPSE_VERSION}})" CACHE STRING "The version of Eclipse. If Eclipse has not been found, 3.6 (Helios) is assumed.")
  else()
    message(STATUS "Eclipse version is set to ${CMAKE_ECLIPSE_VERSION}. Adjust CMAKE_ECLIPSE_VERSION if this is wrong.")
  endif()

  set_property(CACHE CMAKE_ECLIPSE_VERSION PROPERTY STRINGS "3.2 (${_ECLIPSE_VERSION_NAME_3.2})"
                                                            "3.3 (${_ECLIPSE_VERSION_NAME_3.3})"
                                                            "3.4 (${_ECLIPSE_VERSION_NAME_3.4})"
                                                            "3.5 (${_ECLIPSE_VERSION_NAME_3.5})"
                                                            "3.6 (${_ECLIPSE_VERSION_NAME_3.6})"
                                                            "3.7 (${_ECLIPSE_VERSION_NAME_3.7})"
                                                            "4.2 (${_ECLIPSE_VERSION_NAME_4.2})"
                                                            "4.3 (${_ECLIPSE_VERSION_NAME_4.3})"
                                                            "4.4 (${_ECLIPSE_VERSION_NAME_4.4})"
                                                            "4.5 (${_ECLIPSE_VERSION_NAME_4.5})"
              )
endfunction()

_find_eclipse_version()

# Try to find out how many CPUs we have and set the -j argument for make accordingly
set(_CMAKE_ECLIPSE_INITIAL_MAKE_ARGS "")

include(ProcessorCount)
ProcessorCount(_CMAKE_ECLIPSE_PROCESSOR_COUNT)

# Only set -j if we are under UNIX and if the make-tool used actually has "make" in the name
# (we may also get here in the future e.g. for ninja)
if("${_CMAKE_ECLIPSE_PROCESSOR_COUNT}" GREATER 1  AND  CMAKE_HOST_UNIX  AND  "${CMAKE_MAKE_PROGRAM}" MATCHES make)
  set(_CMAKE_ECLIPSE_INITIAL_MAKE_ARGS "-j${_CMAKE_ECLIPSE_PROCESSOR_COUNT}")
endif()

# This variable is used by the Eclipse generator and appended to the make invocation commands.
set(CMAKE_ECLIPSE_MAKE_ARGUMENTS "${_CMAKE_ECLIPSE_INITIAL_MAKE_ARGS}" CACHE STRING "Additional command line arguments when Eclipse invokes make. Enter e.g. -j<some_number> to get parallel builds")

set(CMAKE_ECLIPSE_GENERATE_LINKED_RESOURCES TRUE CACHE BOOL "If disabled, CMake will not generate linked resource to the subprojects and to the source files within targets")

# This variable is used by the Eclipse generator in out-of-source builds only.
set(CMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT FALSE CACHE BOOL "If enabled, CMake will generate a source project for Eclipse in CMAKE_SOURCE_DIR")
mark_as_advanced(CMAKE_ECLIPSE_GENERATE_SOURCE_PROJECT)

# Determine builtin macros and include dirs:
include(${CMAKE_CURRENT_LIST_DIR}/CMakeExtraGeneratorDetermineCompilerMacrosAndIncludeDirs.cmake)
