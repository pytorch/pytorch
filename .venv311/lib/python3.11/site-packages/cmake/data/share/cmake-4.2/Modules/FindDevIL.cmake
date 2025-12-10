# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindDevIL
---------

Finds the Developer's Image Library, `DevIL <https://openil.sourceforge.net/>`_:

.. code-block:: cmake

  find_package(DevIL [<version>] [...])

.. versionadded:: 4.2
  Support for the ``<version>`` argument in the :command:`find_package`
  call.  Version can be also specified as a range.

The DevIL package internally consists of the following libraries, all
distributed as part of the same release:

* The core Image Library (IL)

  This library is always required when working with DevIL, as it provides the
  main image loading and manipulation functionality.

* The Image Library Utilities (ILU)

  This library depends on IL and provides image filters and effects. It is only
  required if the application uses this extended functionality.

* The Image Library Utility Toolkit (ILUT)

  This library depends on both IL and ILU, and additionally provides an
  interface to OpenGL.  It is only needed if the application uses DevIL together
  with OpenGL.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``DevIL::IL``
  .. versionadded:: 3.21

  Target encapsulating the core Image Library (IL) usage requirements, available
  if the DevIL package is found.

``DevIL::ILU``
  .. versionadded:: 3.21

  Target encapsulating the Image Library Utilities (ILU) usage requirements,
  available if the DevIL package is found.  This target also links to
  ``DevIL::IL`` for convenience, as ILU depends on the core IL library.

``DevIL::ILUT``
  .. versionadded:: 3.21

  Target encapsulating the Image Library Utility Toolkit (ILUT) usage
  requirements, available if the DevIL package and its ILUT library are found.
  This target also links to ``DevIL::ILU``, and transitively to ``DevIL::IL``,
  since ILUT depends on both.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``DevIL_FOUND``
  .. versionadded:: 3.8

  Boolean indicating whether the (requested version of) DevIL package was
  found, including the IL and ILU libraries.

``DevIL_VERSION``
  .. versionadded:: 4.2

  The version of the DevIL found.

``DevIL_ILUT_FOUND``
  .. versionadded:: 3.21

  Boolean indicating whether the ILUT library was found.  On most systems,
  ILUT is found when both IL and ILU are available.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``IL_INCLUDE_DIR``
  The directory containing the ``il.h``, ``ilu.h`` and ``ilut.h`` header files.

``IL_LIBRARIES``
  The full path to the core Image Library (IL).

``ILU_LIBRARIES``
  The full path to the ILU library.

``ILUT_LIBRARIES``
  The full path to the ILUT library.

Examples
^^^^^^^^

Finding the DevIL package and linking against the core Image Library (IL):

.. code-block:: cmake

  find_package(DevIL)
  target_link_libraries(app PRIVATE DevIL::IL)

Linking against the Image Library Utilities (ILU):

.. code-block:: cmake

  find_package(DevIL)
  target_link_libraries(app PRIVATE DevIL::ILU)

Linking against the Image Library Utility Toolkit (ILUT):

.. code-block:: cmake

  find_package(DevIL)
  target_link_libraries(app PRIVATE DevIL::ILUT)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

include(FindPackageHandleStandardArgs)

find_path(IL_INCLUDE_DIR il.h
  PATH_SUFFIXES include IL
  DOC "The path to the directory that contains il.h"
)

find_library(IL_LIBRARIES
  NAMES IL DEVIL
  PATH_SUFFIXES libx32 lib64 lib lib32
  DOC "The file that corresponds to the base il library."
)

find_library(ILUT_LIBRARIES
  NAMES ILUT
  PATH_SUFFIXES libx32 lib64 lib lib32
  DOC "The file that corresponds to the il (system?) utility library."
)

find_library(ILU_LIBRARIES
  NAMES ILU
  PATH_SUFFIXES libx32 lib64 lib lib32
  DOC "The file that corresponds to the il utility library."
)

# Get version.
block(PROPAGATE DevIL_VERSION)
  if(IL_INCLUDE_DIR AND EXISTS "${IL_INCLUDE_DIR}/il.h")
    set(regex "^[ \t]*#[ \t]*define[ \t]+IL_VERSION[ \t]+([0-9]+)[ \t]*$")

    file(STRINGS ${IL_INCLUDE_DIR}/il.h result REGEX "${regex}")

    if(result MATCHES "${regex}")
      set(DevIL_VERSION "${CMAKE_MATCH_1}")

      math(EXPR DevIL_VERSION_MAJOR "${DevIL_VERSION} / 100")
      math(EXPR DevIL_VERSION_MINOR "${DevIL_VERSION} / 10 % 10")
      math(EXPR DevIL_VERSION_PATCH "${DevIL_VERSION} % 10")

      set(DevIL_VERSION "")
      foreach(part MAJOR MINOR PATCH)
        if(DevIL_VERSION)
          string(APPEND ".${DevIL_VERSION_${part}}")
        else()
          set(DevIL_VERSION "${DevIL_VERSION_${part}}")
        endif()

        set(
          DevIL_VERSION
          "${DevIL_VERSION_MAJOR}.${DevIL_VERSION_MINOR}.${DevIL_VERSION_PATCH}"
        )
      endforeach()
    endif()
  endif()
endblock()

find_package_handle_standard_args(
  DevIL
  REQUIRED_VARS IL_LIBRARIES ILU_LIBRARIES IL_INCLUDE_DIR
  VERSION_VAR DevIL_VERSION
  HANDLE_VERSION_RANGE
)

# provide legacy variable for compatibility
set(IL_FOUND ${DevIL_FOUND})

# create imported targets ONLY if we found DevIL.
if(DevIL_FOUND)
  # Report the ILUT found if ILUT_LIBRARIES contains valid path.
  if (ILUT_LIBRARIES)
    set(DevIL_ILUT_FOUND TRUE)
  else()
    set(DevIL_ILUT_FOUND FALSE)
  endif()

  if(NOT TARGET DevIL::IL)
    add_library(DevIL::IL UNKNOWN IMPORTED)
    set_target_properties(DevIL::IL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${IL_INCLUDE_DIR}"
      IMPORTED_LOCATION "${IL_LIBRARIES}")
  endif()

  # DevIL Utilities target
  if(NOT TARGET DevIL::ILU)
    add_library(DevIL::ILU UNKNOWN IMPORTED)
    set_target_properties(DevIL::ILU PROPERTIES
      IMPORTED_LOCATION "${ILU_LIBRARIES}")
    target_link_libraries(DevIL::ILU INTERFACE DevIL::IL)
  endif()

  # ILUT (if found)
  if(NOT TARGET DevIL::ILUT AND DevIL_ILUT_FOUND)
    add_library(DevIL::ILUT UNKNOWN IMPORTED)
    set_target_properties(DevIL::ILUT PROPERTIES
      IMPORTED_LOCATION "${ILUT_LIBRARIES}")
    target_link_libraries(DevIL::ILUT INTERFACE DevIL::ILU)
  endif()
endif()

cmake_policy(POP)
