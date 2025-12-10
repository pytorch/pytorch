# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLibXslt
-----------

Finds the XSL Transformations, Extensible Stylesheet Language Transformations
(XSLT) library (libxslt):

.. code-block:: cmake

  find_package(LibXslt [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

.. versionadded:: 3.18

This module provides the following :ref:`Imported Targets`:

``LibXslt::LibXslt``
  Target encapsulating the usage requirements of the libxslt library.  This
  target is available only if libxslt is found.

``LibXslt::LibExslt``
  Target encapsulating the usage requirements for the libexslt library. Part of
  the libxslt package, libexslt provides optional extensions to XSLT on top of
  libxslt.  This target is available only if the main libxslt library is found.

``LibXslt::xsltproc``
  Target encapsulating the command-line XSLT processor (``xsltproc``).  This
  tool, part of the libxslt package, applies XSLT stylesheets to XML documents
  as a command-line alternative to the libxslt library.  This target is
  available only if the ``xsltproc`` executable is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``LibXslt_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) libxslt was found.

``LibXslt_VERSION``
  .. versionadded:: 4.2

  The version of libxslt found.

``LIBXSLT_LIBRARIES``
  Libraries needed to link to libxslt.

``LIBXSLT_DEFINITIONS``
  Compiler switches required for using libxslt.

``LIBXSLT_EXSLT_LIBRARIES``
  Libraries needed when linking against the exslt library.  These are available
  and needed only when using exslt library.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``LIBXSLT_INCLUDE_DIR``
  Directory containing ``libxslt/xslt.h`` and other libxslt header files.

``LIBXSLT_EXSLT_INCLUDE_DIR``
  .. versionadded:: 3.18

  Directory containing ``libexslt/exslt.h`` and other exslt-related headers.
  These are needed only when using exslt (extensions to XSLT).

``LIBXSLT_XSLTPROC_EXECUTABLE``
  Full path to the XSLT processor executable ``xsltproc`` if found.  This path
  is optional.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``LIBXSLT_FOUND``
  .. deprecated:: 4.2
    Use ``LibXslt_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) libxslt was found.

``LIBXSLT_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``LibXslt_VERSION``.

  The version of libxslt found.

Examples
^^^^^^^^

Finding libxslt library and linking it to a project target:

.. code-block:: cmake

  find_package(LibXslt)
  target_link_libraries(foo PRIVATE LibXslt::LibXslt)

When project also needs the extensions to XSLT (exslt) library, both targets
should be linked:

.. code-block:: cmake

  find_package(LibXslt)
  target_link_libraries(foo PRIVATE LibXslt::LibXslt LibXslt::LibExslt)

Example, how to use XSLT processor in a custom command build rule:

.. code-block:: cmake

  find_package(LibXslt)

  if(TARGET LibXslt::xsltproc)
    # Executed when some build rule depends on example.html.
    add_custom_command(
      OUTPUT example.html
      COMMAND LibXslt::xsltproc -o example.html transform.xslt example.xml
    )
  endif()
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

# use pkg-config to get the directories and then use these values
# in the find_path() and find_library() calls
find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(PC_LIBXSLT QUIET libxslt)
endif()
set(LIBXSLT_DEFINITIONS ${PC_LIBXSLT_CFLAGS_OTHER})

find_path(LIBXSLT_INCLUDE_DIR NAMES libxslt/xslt.h
    HINTS
   ${PC_LIBXSLT_INCLUDEDIR}
   ${PC_LIBXSLT_INCLUDE_DIRS}
  )

# CMake 3.17 and below used 'LIBXSLT_LIBRARIES' as the name of
# the cache entry storing the find_library result.  Use the
# value if it was set by the project or user.
if(DEFINED LIBXSLT_LIBRARIES AND NOT DEFINED LIBXSLT_LIBRARY)
  set(LIBXSLT_LIBRARY ${LIBXSLT_LIBRARIES})
endif()

find_library(LIBXSLT_LIBRARY NAMES xslt libxslt
    HINTS
   ${PC_LIBXSLT_LIBDIR}
   ${PC_LIBXSLT_LIBRARY_DIRS}
  )

set(LIBXSLT_LIBRARIES ${LIBXSLT_LIBRARY})

if(PkgConfig_FOUND)
  pkg_check_modules(PC_LIBXSLT_EXSLT QUIET libexslt)
endif()
set(LIBXSLT_EXSLT_DEFINITIONS ${PC_LIBXSLT_EXSLT_CFLAGS_OTHER})

find_path(LIBXSLT_EXSLT_INCLUDE_DIR NAMES libexslt/exslt.h
  HINTS
  ${PC_LIBXSLT_EXSLT_INCLUDEDIR}
  ${PC_LIBXSLT_EXSLT_INCLUDE_DIRS}
)

find_library(LIBXSLT_EXSLT_LIBRARY NAMES exslt libexslt
    HINTS
    ${PC_LIBXSLT_LIBDIR}
    ${PC_LIBXSLT_LIBRARY_DIRS}
    ${PC_LIBXSLT_EXSLT_LIBDIR}
    ${PC_LIBXSLT_EXSLT_LIBRARY_DIRS}
  )

set(LIBXSLT_EXSLT_LIBRARIES ${LIBXSLT_EXSLT_LIBRARY} )

find_program(LIBXSLT_XSLTPROC_EXECUTABLE xsltproc)

if(PC_LIBXSLT_VERSION)
    set(LibXslt_VERSION ${PC_LIBXSLT_VERSION})
    set(LIBXSLT_VERSION_STRING "${LibXslt_VERSION}")
elseif(LIBXSLT_INCLUDE_DIR AND EXISTS "${LIBXSLT_INCLUDE_DIR}/libxslt/xsltconfig.h")
    file(STRINGS "${LIBXSLT_INCLUDE_DIR}/libxslt/xsltconfig.h" libxslt_version_str
         REGEX "^#define[\t ]+LIBXSLT_DOTTED_VERSION[\t ]+\".*\"")

    string(REGEX REPLACE "^#define[\t ]+LIBXSLT_DOTTED_VERSION[\t ]+\"([^\"]*)\".*" "\\1"
           LibXslt_VERSION "${libxslt_version_str}")
    set(LIBXSLT_VERSION_STRING "${LibXslt_VERSION}")
    unset(libxslt_version_str)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibXslt
                                  REQUIRED_VARS LIBXSLT_LIBRARIES LIBXSLT_INCLUDE_DIR
                                  VERSION_VAR LibXslt_VERSION)

mark_as_advanced(LIBXSLT_INCLUDE_DIR
                 LIBXSLT_LIBRARY
                 LIBXSLT_EXSLT_INCLUDE_DIR
                 LIBXSLT_EXSLT_LIBRARY
                 LIBXSLT_XSLTPROC_EXECUTABLE)

if(LibXslt_FOUND AND NOT TARGET LibXslt::LibXslt)
  add_library(LibXslt::LibXslt UNKNOWN IMPORTED)
  set_target_properties(LibXslt::LibXslt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LIBXSLT_INCLUDE_DIR}")
  set_target_properties(LibXslt::LibXslt PROPERTIES INTERFACE_COMPILE_OPTIONS "${LIBXSLT_DEFINITIONS}")
  set_property(TARGET LibXslt::LibXslt APPEND PROPERTY IMPORTED_LOCATION "${LIBXSLT_LIBRARY}")
endif()

if(LibXslt_FOUND AND NOT TARGET LibXslt::LibExslt)
  add_library(LibXslt::LibExslt UNKNOWN IMPORTED)
  set_target_properties(LibXslt::LibExslt PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${LIBXSLT_EXSLT_INCLUDE_DIR}")
  set_target_properties(LibXslt::LibExslt PROPERTIES INTERFACE_COMPILE_OPTIONS "${LIBXSLT_EXSLT_DEFINITIONS}")
  set_property(TARGET LibXslt::LibExslt APPEND PROPERTY IMPORTED_LOCATION "${LIBXSLT_EXSLT_LIBRARY}")
endif()

if(LIBXSLT_XSLTPROC_EXECUTABLE AND NOT TARGET LibXslt::xsltproc)
  add_executable(LibXslt::xsltproc IMPORTED)
  set_target_properties(LibXslt::xsltproc PROPERTIES IMPORTED_LOCATION "${LIBXSLT_XSLTPROC_EXECUTABLE}")
endif()

cmake_policy(POP)
