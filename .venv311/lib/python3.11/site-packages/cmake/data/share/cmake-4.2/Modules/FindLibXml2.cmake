# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLibXml2
-----------

Finds the XML processing library (libxml2):

.. code-block:: cmake

  find_package(LibXml2 [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``LibXml2::LibXml2``
  .. versionadded:: 3.12

  Target encapsulating the libxml2 library usage requirements, available only if
  library is found.

``LibXml2::xmllint``
  .. versionadded:: 3.17

  Target encapsulating the xmllint command-line executable, available only if
  xmllint executable is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``LibXml2_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) libxml2 library was
  found.

``LibXml2_VERSION``
  .. versionadded:: 4.2

  The version of the libxml2 found.

``LIBXML2_INCLUDE_DIRS``
  Include directories needed to use the libxml2 library.

``LIBXML2_LIBRARIES``
  Libraries needed to link against to use the libxml2 library.

``LIBXML2_DEFINITIONS``
  The compiler switches required for using libxml2.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``LIBXML2_INCLUDE_DIR``
  The include directory containing libxml2 headers.

``LIBXML2_LIBRARY``
  The path to the libxml2 library.

``LIBXML2_XMLLINT_EXECUTABLE``
  The path to the XML checking tool ``xmllint`` coming with libxml2.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``LIBXML2_FOUND``
  .. deprecated:: 4.2
    Use ``LibXml2_FOUND``, which has the same value.

  Boolean indicating whether the (requested version of) libxml2 library was
  found.

``LIBXML2_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``LibXml2_VERSION``.

  The version of the libxml2 found.

Examples
^^^^^^^^

Finding the libxml2 library and linking it to a project target:

.. code-block:: cmake

  find_package(LibXml2)
  target_link_libraries(project_target PRIVATE LibXml2::LibXml2)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

# use pkg-config to get the directories and then use these values
# in the find_path() and find_library() calls
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_LIBXML QUIET libxml-2.0)
endif()

find_path(LIBXML2_INCLUDE_DIR NAMES libxml/xpath.h
   HINTS
   ${PC_LIBXML_INCLUDEDIR}
   ${PC_LIBXML_INCLUDE_DIRS}
   PATH_SUFFIXES libxml2
   )

# CMake 3.9 and below used 'LIBXML2_LIBRARIES' as the name of
# the cache entry storing the find_library result.  Use the
# value if it was set by the project or user.
if(DEFINED LIBXML2_LIBRARIES AND NOT DEFINED LIBXML2_LIBRARY)
  set(LIBXML2_LIBRARY ${LIBXML2_LIBRARIES})
endif()

find_library(LIBXML2_LIBRARY NAMES xml2 libxml2 libxml2_a
   HINTS
   ${PC_LIBXML_LIBDIR}
   ${PC_LIBXML_LIBRARY_DIRS}
   )

find_program(LIBXML2_XMLLINT_EXECUTABLE xmllint)
# for backwards compat. with KDE 4.0.x:
set(XMLLINT_EXECUTABLE "${LIBXML2_XMLLINT_EXECUTABLE}")

if(LIBXML2_INCLUDE_DIR AND EXISTS "${LIBXML2_INCLUDE_DIR}/libxml/xmlversion.h")
    file(STRINGS "${LIBXML2_INCLUDE_DIR}/libxml/xmlversion.h" libxml2_version_str
         REGEX "^#define[\t ]+LIBXML_DOTTED_VERSION[\t ]+\".*\"")

    string(REGEX REPLACE "^#define[\t ]+LIBXML_DOTTED_VERSION[\t ]+\"([^\"]*)\".*" "\\1"
           LibXml2_VERSION "${libxml2_version_str}")
    set(LIBXML2_VERSION_STRING "${LibXml2_VERSION}")
    unset(libxml2_version_str)
endif()

set(LIBXML2_INCLUDE_DIRS ${LIBXML2_INCLUDE_DIR})
set(LIBXML2_LIBRARIES ${LIBXML2_LIBRARY})

# Did we find the same installation as pkg-config?
# If so, use additional information from it.
unset(LIBXML2_DEFINITIONS)
foreach(libxml2_pc_lib_dir IN LISTS PC_LIBXML_LIBDIR PC_LIBXML_LIBRARY_DIRS)
  if (LIBXML2_LIBRARY MATCHES "^${libxml2_pc_lib_dir}")
    list(APPEND LIBXML2_INCLUDE_DIRS ${PC_LIBXML_INCLUDE_DIRS})
    set(LIBXML2_DEFINITIONS ${PC_LIBXML_CFLAGS_OTHER})
    break()
  endif()
endforeach()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibXml2
                                  REQUIRED_VARS LIBXML2_LIBRARY LIBXML2_INCLUDE_DIR
                                  VERSION_VAR LibXml2_VERSION)

mark_as_advanced(LIBXML2_INCLUDE_DIR LIBXML2_LIBRARY LIBXML2_XMLLINT_EXECUTABLE)

if(LibXml2_FOUND AND NOT TARGET LibXml2::LibXml2)
  add_library(LibXml2::LibXml2 UNKNOWN IMPORTED)
  set_target_properties(LibXml2::LibXml2 PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LIBXML2_INCLUDE_DIRS}")
  set_target_properties(LibXml2::LibXml2 PROPERTIES INTERFACE_COMPILE_OPTIONS "${LIBXML2_DEFINITIONS}")
  set_property(TARGET LibXml2::LibXml2 APPEND PROPERTY IMPORTED_LOCATION "${LIBXML2_LIBRARY}")
endif()

if(LIBXML2_XMLLINT_EXECUTABLE AND NOT TARGET LibXml2::xmllint)
  add_executable(LibXml2::xmllint IMPORTED)
  set_target_properties(LibXml2::xmllint PROPERTIES IMPORTED_LOCATION "${LIBXML2_XMLLINT_EXECUTABLE}")
endif()

cmake_policy(POP)
