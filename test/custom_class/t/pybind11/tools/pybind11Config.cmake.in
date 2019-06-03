# pybind11Config.cmake
# --------------------
#
# PYBIND11 cmake module.
# This module sets the following variables in your project::
#
#   pybind11_FOUND - true if pybind11 and all required components found on the system
#   pybind11_VERSION - pybind11 version in format Major.Minor.Release
#   pybind11_INCLUDE_DIRS - Directories where pybind11 and python headers are located.
#   pybind11_INCLUDE_DIR - Directory where pybind11 headers are located.
#   pybind11_DEFINITIONS - Definitions necessary to use pybind11, namely USING_pybind11.
#   pybind11_LIBRARIES - compile flags and python libraries (as needed) to link against.
#   pybind11_LIBRARY - empty.
#   CMAKE_MODULE_PATH - appends location of accompanying FindPythonLibsNew.cmake and
#                       pybind11Tools.cmake modules.
#
#
# Available components: None
#
#
# Exported targets::
#
# If pybind11 is found, this module defines the following :prop_tgt:`IMPORTED`
# interface library targets::
#
#   pybind11::module - for extension modules
#   pybind11::embed - for embedding the Python interpreter
#
# Python headers, libraries (as needed by platform), and the C++ standard
# are attached to the target. Set PythonLibsNew variables to influence
# python detection and PYBIND11_CPP_STANDARD (-std=c++11 or -std=c++14) to
# influence standard setting. ::
#
#   find_package(pybind11 CONFIG REQUIRED)
#   message(STATUS "Found pybind11 v${pybind11_VERSION}: ${pybind11_INCLUDE_DIRS}")
#
#   # Create an extension module
#   add_library(mylib MODULE main.cpp)
#   target_link_libraries(mylib pybind11::module)
#
#   # Or embed the Python interpreter into an executable
#   add_executable(myexe main.cpp)
#   target_link_libraries(myexe pybind11::embed)
#
# Suggested usage::
#
# find_package with version info is not recommended except for release versions. ::
#
#   find_package(pybind11 CONFIG)
#   find_package(pybind11 2.0 EXACT CONFIG REQUIRED)
#
#
# The following variables can be set to guide the search for this package::
#
#   pybind11_DIR - CMake variable, set to directory containing this Config file
#   CMAKE_PREFIX_PATH - CMake variable, set to root directory of this package
#   PATH - environment variable, set to bin directory of this package
#   CMAKE_DISABLE_FIND_PACKAGE_pybind11 - CMake variable, disables
#     find_package(pybind11) when not REQUIRED, perhaps to force internal build

@PACKAGE_INIT@

set(PN pybind11)

# location of pybind11/pybind11.h
set(${PN}_INCLUDE_DIR "${PACKAGE_PREFIX_DIR}/@CMAKE_INSTALL_INCLUDEDIR@")

set(${PN}_LIBRARY "")
set(${PN}_DEFINITIONS USING_${PN})

check_required_components(${PN})

# make detectable the FindPythonLibsNew.cmake module
list(APPEND CMAKE_MODULE_PATH ${CMAKE_CURRENT_LIST_DIR})

include(pybind11Tools)

if(NOT (CMAKE_VERSION VERSION_LESS 3.0))
#-----------------------------------------------------------------------------
# Don't include targets if this file is being picked up by another
# project which has already built this as a subproject
#-----------------------------------------------------------------------------
if(NOT TARGET ${PN}::pybind11)
    include("${CMAKE_CURRENT_LIST_DIR}/${PN}Targets.cmake")

    find_package(PythonLibsNew ${PYBIND11_PYTHON_VERSION} MODULE REQUIRED)
    set_property(TARGET ${PN}::pybind11 APPEND PROPERTY INTERFACE_INCLUDE_DIRECTORIES ${PYTHON_INCLUDE_DIRS})
    set_property(TARGET ${PN}::embed APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${PYTHON_LIBRARIES})
    if(WIN32 OR CYGWIN)
      set_property(TARGET ${PN}::module APPEND PROPERTY INTERFACE_LINK_LIBRARIES ${PYTHON_LIBRARIES})
    endif()

    set_property(TARGET ${PN}::pybind11 APPEND PROPERTY INTERFACE_COMPILE_OPTIONS "${PYBIND11_CPP_STANDARD}")

    get_property(_iid TARGET ${PN}::pybind11 PROPERTY INTERFACE_INCLUDE_DIRECTORIES)
    get_property(_ill TARGET ${PN}::module PROPERTY INTERFACE_LINK_LIBRARIES)
    set(${PN}_INCLUDE_DIRS ${_iid})
    set(${PN}_LIBRARIES ${_ico} ${_ill})
endif()
endif()
