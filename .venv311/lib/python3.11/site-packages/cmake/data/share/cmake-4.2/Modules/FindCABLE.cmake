# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindCABLE
---------

.. versionchanged:: 4.1
  This module is available only if policy :policy:`CMP0191` is not set to ``NEW``.

Finds the CABLE installation and determines its include paths and libraries:

.. code-block:: cmake

  find_package(CABLE [...])

Package called CABLE (CABLE Automates Bindings for Language Extension) was
initially developed by Kitware to generate bindings to C++ classes for use in
interpreted languages, such as Tcl.  It worked in conjunction with packages like
GCC-XML.  The CABLE package has since been superseded by the ITK CableSwig
package.

.. note::

  When building wrappers for interpreted languages, these packages are no longer
  necessary.  The CastXML package now serves as the recommended tool for this
  purpose and can be found directly using the :command:`find_program` command.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``CABLE``
  Path to the ``cable`` executable.
``CABLE_INCLUDE_DIR``
  Path to the include directory.
``CABLE_TCL_LIBRARY``
  Path to the Tcl wrapper library.

Examples
^^^^^^^^

Finding CABLE to build Tcl wrapper, by linking library and adding the include
directories:

.. code-block:: cmake

  find_package(CABLE)
  target_link_libraries(tcl_wrapper_target PRIVATE ${CABLE_TCL_LIBRARY})
  target_include_directories(tcl_wrapper_target PRIVATE ${CABLE_INCLUDE_DIR})
#]=======================================================================]

cmake_policy(GET CMP0191 _FindCABLE_CMP0191)
if(_FindCABLE_CMP0191 STREQUAL "NEW")
  message(FATAL_ERROR "The FindCABLE module has been removed by policy CMP0191.")
endif()

if(_FindCABLE_testing)
  set(_FindCABLE_included TRUE)
  return()
endif()

if(NOT CABLE)
  find_path(CABLE_BUILD_DIR cableVersion.h)
endif()

if(CABLE_BUILD_DIR)
  load_cache(${CABLE_BUILD_DIR}
             EXCLUDE
               BUILD_SHARED_LIBS
               LIBRARY_OUTPUT_PATH
               EXECUTABLE_OUTPUT_PATH
               MAKECOMMAND
               CMAKE_INSTALL_PREFIX
             INCLUDE_INTERNALS
               CABLE_LIBRARY_PATH
               CABLE_EXECUTABLE_PATH)

  if(CABLE_LIBRARY_PATH)
    find_library(CABLE_TCL_LIBRARY NAMES CableTclFacility PATHS
                 ${CABLE_LIBRARY_PATH}
                 ${CABLE_LIBRARY_PATH}/*)
  else()
    find_library(CABLE_TCL_LIBRARY NAMES CableTclFacility PATHS
                 ${CABLE_BINARY_DIR}/CableTclFacility
                 ${CABLE_BINARY_DIR}/CableTclFacility/*)
  endif()

  if(CABLE_EXECUTABLE_PATH)
    find_program(CABLE NAMES cable PATHS
                 ${CABLE_EXECUTABLE_PATH}
                 ${CABLE_EXECUTABLE_PATH}/*)
  else()
    find_program(CABLE NAMES cable PATHS
                 ${CABLE_BINARY_DIR}/Executables
                 ${CABLE_BINARY_DIR}/Executables/*)
  endif()

  find_path(CABLE_INCLUDE_DIR CableTclFacility/ctCalls.h
            ${CABLE_SOURCE_DIR})
else()
  # Find the cable executable in the path.
  find_program(CABLE NAMES cable)

  # Get the path where the executable sits, but without the executable
  # name on it.
  get_filename_component(CABLE_ROOT_BIN ${CABLE} PATH)

  # Find the cable include directory in a path relative to the cable
  # executable.
  find_path(CABLE_INCLUDE_DIR CableTclFacility/ctCalls.h
            ${CABLE_ROOT_BIN}/../include/Cable)

  # Find the WrapTclFacility library in a path relative to the cable
  # executable.
  find_library(CABLE_TCL_LIBRARY NAMES CableTclFacility PATHS
               ${CABLE_ROOT_BIN}/../lib/Cable)
endif()
