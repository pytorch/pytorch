# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindOpenSP
----------

.. versionadded:: 3.25

Finds the OpenSP library:

.. code-block:: cmake

  find_package(OpenSP [<version>] [...])

OpenSP is an open-source implementation of the SGML (Standard Generalized
Markup Language) parser.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``OpenSP::OpenSP``
  Target encapsulating the OpenSP library usage requirements, available only if
  the OpenSP is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``OpenSP_FOUND``
  Boolean indicating whether (the requested version of) OpenSP is available.

``OpenSP_VERSION``
  The version of found OpenSP.

``OpenSP_VERSION_MAJOR``
  The major version of OpenSP.

``OpenSP_VERSION_MINOR``
  The minor version of OpenSP.

``OpenSP_VERSION_PATCH``
  The patch version of OpenSP.

``OpenSP_INCLUDE_DIRS``
  The include directories containing headers needed to use the OpenSP library.

``OpenSP_LIBRARIES``
  Libraries required to link against to use OpenSP.  These can be passed to the
  :command:`target_link_libraries` command when not using the ``OpenSP::OpenSP``
  imported target.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OpenSP_INCLUDE_DIR``
  The OpenSP include directory.

``OpenSP_LIBRARY``
  The absolute path of the ``osp`` library.

``OpenSP_MULTI_BYTE``
  True if ``SP_MULTI_BYTE`` was found to be defined in OpenSP's ``config.h``
  header file, which indicates that the OpenSP library was compiled with support
  for multi-byte characters.  The consuming target needs to define the
  ``SP_MULTI_BYTE`` preprocessor macro to match this value in order to avoid
  issues with character decoding.

Examples
^^^^^^^^

Finding the OpenSP library and linking it to a project target:

.. code-block:: cmake

  find_package(OpenSP)
  target_link_libraries(project_target PRIVATE OpenSP::OpenSP)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_package(PkgConfig QUIET)
if (PkgConfig_FOUND)
  pkg_check_modules(PC_OpenSP QUIET opensp)
endif ()

if (NOT OpenSP_INCLUDE_DIR)
  find_path(OpenSP_INCLUDE_DIR
    NAMES ParserEventGeneratorKit.h
    HINTS
    ${PC_OpenSP_INCLUDEDIRS}
    ${PC_OpenSP_INCLUDE_DIRS}
    PATH_SUFFIXES OpenSP opensp
    DOC "The OpenSP include directory"
    )
endif ()

if (NOT OpenSP_LIBRARY)
  find_library(OpenSP_LIBRARY_RELEASE
    NAMES osp libosp opensp libopensp sp133 libsp
    HINTS
    ${PC_OpenSP_LIBDIR}
    ${PC_OpenSP_LIBRARY_DIRS}
    )

  find_library(OpenSP_LIBRARY_DEBUG
    NAMES ospd libospd openspd libopenspd sp133d libspd
    HINTS
    ${PC_OpenSP_LIBDIR}
    ${PC_OpenSP_LIBRARY_DIRS}
    )

  include(SelectLibraryConfigurations)
  select_library_configurations(OpenSP)
endif ()

if (OpenSP_INCLUDE_DIR)
  if (EXISTS "${OpenSP_INCLUDE_DIR}/config.h")
    if (NOT OpenSP_VERSION)
      file(STRINGS "${OpenSP_INCLUDE_DIR}/config.h" opensp_version_str REGEX "^#define[\t ]+SP_VERSION[\t ]+\".*\"")
      string(REGEX REPLACE "^.*SP_VERSION[\t ]+\"([^\"]*)\".*$" "\\1" OpenSP_VERSION "${opensp_version_str}")
      unset(opensp_version_str)
    endif ()

    if (OpenSP_VERSION MATCHES [=[([0-9]+)\.([0-9]+)\.([0-9]+)]=])
      set(OpenSP_VERSION_MAJOR "${CMAKE_MATCH_1}")
      set(OpenSP_VERSION_MINOR "${CMAKE_MATCH_2}")
      set(OpenSP_VERSION_PATCH "${CMAKE_MATCH_3}")
    endif ()

    include(CheckCXXSymbolExists)
    check_cxx_symbol_exists(SP_MULTI_BYTE "${OpenSP_INCLUDE_DIR}/config.h" OpenSP_MULTI_BYTE)
  endif ()
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenSP
  REQUIRED_VARS OpenSP_LIBRARY OpenSP_INCLUDE_DIR
  VERSION_VAR OpenSP_VERSION
  )

mark_as_advanced(OpenSP_INCLUDE_DIR OpenSP_LIBRARY OpenSP_MULTI_BYTE)

if (OpenSP_FOUND)
  set(OpenSP_INCLUDE_DIRS ${OpenSP_INCLUDE_DIR})
  if (NOT TARGET OpenSP::OpenSP)
    add_library(OpenSP::OpenSP UNKNOWN IMPORTED)
    if (EXISTS "${OpenSP_LIBRARY}")
      set_target_properties(OpenSP::OpenSP PROPERTIES
        IMPORTED_LOCATION "${OpenSP_LIBRARY}")
    endif ()
    set_target_properties(OpenSP::OpenSP PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${OpenSP_INCLUDE_DIRS}")

    if (OpenSP_LIBRARY_RELEASE)
      set_target_properties(OpenSP::OpenSP PROPERTIES
        IMPORTED_LOCATION_RELEASE "${OpenSP_LIBRARY_RELEASE}")
      set_property(TARGET OpenSP::OpenSP APPEND PROPERTY
        IMPORTED_CONFIGURATIONS RELEASE)
    endif ()

    if (OpenSP_LIBRARY_DEBUG)
      set_target_properties(OpenSP::OpenSP PROPERTIES
        IMPORTED_LOCATION_DEBUG "${OpenSP_LIBRARY_DEBUG}")
      set_property(TARGET OpenSP::OpenSP APPEND PROPERTY
        IMPORTED_CONFIGURATIONS DEBUG)
    endif ()
  endif ()
endif ()

include(FeatureSummary)
set_package_properties(OpenSP PROPERTIES
  URL "https://openjade.sourceforge.net/doc/index.htm"
  DESCRIPTION "An SGML System Conforming to International Standard ISO 8879"
  )

cmake_policy(POP)
