# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindXalanC
-----------

.. versionadded:: 3.5

Finds the Apache Xalan-C++ XSL transform processor headers and libraries:

.. code-block:: cmake

  find_package(XalaxC [<version>] [...])

.. note::

  The Xalan-C++ library depends on the :module:`Xerces-C++ <FindXercesC>`
  library, which must be found for this module to succeed.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``XalanC::XalanC``
  Target encapsulating the Xalan-C++ library usage requirements, available only
  if Xalan-C++ is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``XalanC_FOUND``
  Boolean indicating whether (the requested version of) Xalan-C++ was found.
``XalanC_VERSION``
  The version of the found Xalan-C++ library.
``XalanC_INCLUDE_DIRS``
  Include directories needed for using Xalan-C++ library.  These contain the
  Xalan-C++ and Xerces-C++ headers.
``XalanC_LIBRARIES``
  Libraries needed to link against Xalan-C++.  These contain the Xalan-C++ and
  Xerces-C++ libraries.
``XalanC_LIBRARY``
  The path to the Xalan-C++ library (``xalan-c``), either release or debug
  variant.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``XalanC_INCLUDE_DIR``
  The directory containing the Xalan-C++ headers.
``XalanC_LIBRARY_RELEASE``
  The path to a release (optimized) variant of the Xalan-C++ library.
``XalanC_LIBRARY_DEBUG``
  The path to a debug variant of the Xalan-C++ library.

Examples
^^^^^^^^

Finding Xalan-C++ library and linking it to a project target:

.. code-block:: cmake

  find_package(XalanC)
  target_link_libraries(project_target PRIVATE XalanC::XalanC)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

# Written by Roger Leigh <rleigh@codelibre.net>

function(_XalanC_GET_VERSION  version_hdr)
    file(STRINGS ${version_hdr} _contents REGEX "^[ \t]*#define XALAN_VERSION_.*")
    if(_contents)
        string(REGEX REPLACE "[^*]*#define XALAN_VERSION_MAJOR[ \t(]+([0-9]+).*" "\\1" XalanC_MAJOR "${_contents}")
        string(REGEX REPLACE "[^*]*#define XALAN_VERSION_MINOR[ \t(]+([0-9]+).*" "\\1" XalanC_MINOR "${_contents}")
        string(REGEX REPLACE "[^*]*#define XALAN_VERSION_REVISION[ \t(]+([0-9]+).*" "\\1" XalanC_PATCH "${_contents}")

        if(NOT XalanC_MAJOR MATCHES "^[0-9]+$")
            message(FATAL_ERROR "Version parsing failed for XALAN_VERSION_MAJOR!")
        endif()
        if(NOT XalanC_MINOR MATCHES "^[0-9]+$")
            message(FATAL_ERROR "Version parsing failed for XALAN_VERSION_MINOR!")
        endif()
        if(NOT XalanC_PATCH MATCHES "^[0-9]+$")
            message(FATAL_ERROR "Version parsing failed for XALAN_VERSION_REVISION!")
        endif()

        set(XalanC_VERSION "${XalanC_MAJOR}.${XalanC_MINOR}.${XalanC_PATCH}" PARENT_SCOPE)
        set(XalanC_VERSION_MAJOR "${XalanC_MAJOR}" PARENT_SCOPE)
        set(XalanC_VERSION_MINOR "${XalanC_MINOR}" PARENT_SCOPE)
        set(XalanC_VERSION_PATCH "${XalanC_PATCH}" PARENT_SCOPE)
    else()
        message(FATAL_ERROR "Include file ${version_hdr} does not exist or does not contain expected version information")
    endif()
endfunction()

# Find include directory
find_path(XalanC_INCLUDE_DIR
          NAMES "xalanc/XalanTransformer/XalanTransformer.hpp"
          DOC "Xalan-C++ include directory")
mark_as_advanced(XalanC_INCLUDE_DIR)

if(XalanC_INCLUDE_DIR AND EXISTS "${XalanC_INCLUDE_DIR}/xalanc/Include/XalanVersion.hpp")
  _XalanC_GET_VERSION("${XalanC_INCLUDE_DIR}/xalanc/Include/XalanVersion.hpp")
endif()

if(NOT XalanC_LIBRARY)
  # Find all XalanC libraries
  find_library(XalanC_LIBRARY_RELEASE
               NAMES "Xalan-C" "xalan-c"
                     "Xalan-C_${XalanC_VERSION_MAJOR}"
                     "Xalan-C_${XalanC_VERSION_MAJOR}_${XalanC_VERSION_MINOR}"
               DOC "Xalan-C++ libraries (release)")
  find_library(XalanC_LIBRARY_DEBUG
               NAMES "Xalan-CD" "xalan-cd"
                     "Xalan-C_${XalanC_VERSION_MAJOR}D"
                     "Xalan-C_${XalanC_VERSION_MAJOR}_${XalanC_VERSION_MINOR}D"
               DOC "Xalan-C++ libraries (debug)")
  include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
  select_library_configurations(XalanC)
  mark_as_advanced(XalanC_LIBRARY_RELEASE XalanC_LIBRARY_DEBUG)
endif()

unset(XalanC_VERSION_MAJOR)
unset(XalanC_VERSION_MINOR)
unset(XalanC_VERSION_PATCH)

unset(XalanC_XERCESC_REQUIRED)
if(XalanC_FIND_REQUIRED)
  set(XalanC_XERCESC_REQUIRED REQUIRED)
endif()
find_package(XercesC ${XalanC_XERCESC_REQUIRED})
unset(XalanC_XERCESC_REQUIRED)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(XalanC
                                  REQUIRED_VARS XalanC_LIBRARY
                                                XalanC_INCLUDE_DIR
                                                XalanC_VERSION
                                                XercesC_FOUND
                                  VERSION_VAR XalanC_VERSION
                                  FAIL_MESSAGE "Failed to find XalanC")

if(XalanC_FOUND)
  set(XalanC_INCLUDE_DIRS "${XalanC_INCLUDE_DIR}" ${XercesC_INCLUDE_DIRS})
  set(XalanC_LIBRARIES "${XalanC_LIBRARY}" ${XercesC_LIBRARIES})

  # For header-only libraries
  if(NOT TARGET XalanC::XalanC)
    add_library(XalanC::XalanC UNKNOWN IMPORTED)
    if(XalanC_INCLUDE_DIRS)
      set_target_properties(XalanC::XalanC PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${XalanC_INCLUDE_DIRS}")
    endif()
    if(EXISTS "${XalanC_LIBRARY}")
      set_target_properties(XalanC::XalanC PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "CXX"
        IMPORTED_LOCATION "${XalanC_LIBRARY}")
    endif()
    if(EXISTS "${XalanC_LIBRARY_RELEASE}")
      set_property(TARGET XalanC::XalanC APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
      set_target_properties(XalanC::XalanC PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
        IMPORTED_LOCATION_RELEASE "${XalanC_LIBRARY_RELEASE}")
    endif()
    if(EXISTS "${XalanC_LIBRARY_DEBUG}")
      set_property(TARGET XalanC::XalanC APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
      set_target_properties(XalanC::XalanC PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
        IMPORTED_LOCATION_DEBUG "${XalanC_LIBRARY_DEBUG}")
    endif()
    set_target_properties(XalanC::XalanC PROPERTIES INTERFACE_LINK_LIBRARIES XercesC::XercesC)
  endif()
endif()

cmake_policy(POP)
