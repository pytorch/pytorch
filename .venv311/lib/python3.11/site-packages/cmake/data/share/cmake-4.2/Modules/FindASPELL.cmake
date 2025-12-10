# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindASPELL
----------

Finds the GNU Aspell spell checker library:

.. code-block:: cmake

  find_package(ASPELL [<version>] [COMPONENTS <components>] [...])

Components
^^^^^^^^^^

This module supports optional components which can be specified using the
:command:`find_package` command:

.. code-block:: cmake

  find_package(ASPELL [COMPONENTS <components>...])

Supported components include:

``ASPELL``
  .. versionadded:: 4.1

  Finds the Aspell library and its include paths.

``Executable``
  .. versionadded:: 4.1

  Finds the Aspell command-line interactive spell checker executable.

If no components are specified, the module searches for both the ``ASPELL``
and ``Executable`` components by default.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets` when
:prop_gbl:`CMAKE_ROLE` is ``PROJECT``:

``ASPELL::ASPELL``
  .. versionadded:: 4.1

  Target encapsulating the Aspell library usage requirements.  It is available
  only when the ``ASPELL`` component is found.

``ASPELL::Executable``
  .. versionadded:: 4.1

  Target encapsulating the Aspell command-line spell checker executable.  It is
  available only when the ``Executable`` component is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``ASPELL_FOUND``
  Boolean indicating whether (the requested version of) Aspell and all
  requested components were found.

``ASPELL_VERSION``
  .. versionadded:: 4.1

  Version string of the found Aspell if any.  It may be only determined if the
  ``Executable`` component is found.  If version isn't determined, version value
  is not set.

``ASPELL_INCLUDE_DIRS``
  .. versionadded:: 4.1

  Include directories needed to use Aspell.  They are available when the
  ``ASPELL`` component is found.

  The Aspell library may also provide a backward-compatible interface for Pspell
  via the ``pspell.h`` header file.  If such an interface is found, it is also
  added to the list of include directories.

``ASPELL_LIBRARIES``
  Libraries needed to link to Aspell.  They are available when the ``ASPELL``
  component is found.

  .. versionchanged:: 4.1
    This variable is now set as a regular result variable instead of being a
    cache variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``ASPELL_INCLUDE_DIR``
  The directory containing the ``aspell.h`` header file when using the
  ``Executable`` component.

``ASPELL_LIBRARY``
  .. versionadded:: 4.1

  The path to the Aspell library when using the ``ASPELL`` component.

``ASPELL_EXECUTABLE``
  The path to the ``aspell`` command-line spell checker program when using the
  ``Executable`` component.

Examples
^^^^^^^^

Finding the Aspell library with CMake 4.1 or later and linking it to a project
target:

.. code-block:: cmake

  find_package(ASPELL COMPONENTS ASPELL)
  target_link_libraries(project_target PRIVATE ASPELL::ASPELL)

When writing backward-compatible code that supports CMake 4.0 and earlier, a
local imported target can be defined directly in the project:

.. code-block:: cmake

  find_package(ASPELL COMPONENTS ASPELL)
  if(ASPELL_FOUND AND NOT TARGET ASPELL::ASPELL)
    add_library(ASPELL::ASPELL INTERFACE IMPORTED)
    set_target_properties(
      ASPELL::ASPELL
      PROPERTIES
        INTERFACE_LINK_LIBRARIES "${ASPELL_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${ASPELL_INCLUDE_DIR}"
    )
  endif()
  target_link_libraries(project_target PRIVATE ASPELL::ASPELL)

Example, how to execute the ``aspell`` command-line spell checker in a project:

.. code-block:: cmake

  find_package(ASPELL COMPONENTS Executable)
  execute_process(COMMAND ${ASPELL_EXECUTABLE} --help)
#]=======================================================================]

set(_ASPELL_REASON_FAILURE_MESSAGE "")
set(_ASPELL_REQUIRED_VARS "")

# Set default components, when 'COMPONENTS <components>...' are not specified in
# the 'find_package(ASPELL ...)' call.
if(NOT ASPELL_FIND_COMPONENTS)
  set(ASPELL_FIND_COMPONENTS "ASPELL" "Executable")
  set(ASPELL_FIND_REQUIRED_ASPELL TRUE)
  set(ASPELL_FIND_REQUIRED_Executable TRUE)
endif()

if("ASPELL" IN_LIST ASPELL_FIND_COMPONENTS)
  find_path(
    ASPELL_INCLUDE_DIR
    NAMES aspell.h
    DOC "The directory containing <aspell.h>."
  )
  mark_as_advanced(ASPELL_INCLUDE_DIR)

  if(NOT ASPELL_INCLUDE_DIR)
    string(APPEND _ASPELL_REASON_FAILURE_MESSAGE "aspell.h could not be found. ")
  endif()

  # Find backward-compatibility interface for Pspell.
  find_path(
    ASPELL_PSPELL_INCLUDE_DIR
    NAMES pspell.h
    PATH_SUFFIXES pspell
    DOC "Directory containing <pspell.h> BC interface header"
  )
  mark_as_advanced(ASPELL_PSPELL_INCLUDE_DIR)

  # For backward compatibility in projects supporting CMake 4.0 or earlier.
  # Previously the ASPELL_LIBRARIES was a cache variable storing the
  # find_library result.
  if(DEFINED ASPELL_LIBRARIES AND NOT DEFINED ASPELL_LIBRARY)
    set(ASPELL_LIBRARY ${ASPELL_LIBRARIES})
  endif()

  find_library(
    ASPELL_LIBRARY
    NAMES aspell aspell-15 libaspell-15 libaspell
    DOC "The path to the Aspell library."
  )
  mark_as_advanced(ASPELL_LIBRARY)

  if(NOT ASPELL_LIBRARY)
    string(APPEND _ASPELL_REASON_FAILURE_MESSAGE "Aspell library not found. ")
  endif()

  if(ASPELL_INCLUDE_DIR AND ASPELL_LIBRARY)
    set(ASPELL_ASPELL_FOUND TRUE)
  else()
    set(ASPELL_ASPELL_FOUND FALSE)
  endif()

  if(ASPELL_FIND_REQUIRED_ASPELL)
    list(APPEND _ASPELL_REQUIRED_VARS ASPELL_LIBRARY ASPELL_INCLUDE_DIR)
  endif()
endif()

if("Executable" IN_LIST ASPELL_FIND_COMPONENTS)
  find_program(
    ASPELL_EXECUTABLE
    NAMES aspell
    DOC "The path to the aspell command-line utility program."
  )
  mark_as_advanced(ASPELL_EXECUTABLE)

  if(NOT ASPELL_EXECUTABLE)
    string(
      APPEND
      _ASPELL_REASON_FAILURE_MESSAGE
      "Aspell command-line executable not found. "
    )
    set(ASPELL_Executable_FOUND FALSE)
  else()
    set(ASPELL_Executable_FOUND TRUE)

    block(PROPAGATE ASPELL_VERSION)
      execute_process(
        COMMAND ${ASPELL_EXECUTABLE} --version
        OUTPUT_VARIABLE output
        RESULT_VARIABLE result
        ERROR_QUIET
        OUTPUT_STRIP_TRAILING_WHITESPACE
      )
      if(result EQUAL 0 AND output MATCHES "([0-9.]+)[)]?$")
        set(ASPELL_VERSION ${CMAKE_MATCH_1})
      endif()
    endblock()
  endif()

  if(ASPELL_FIND_REQUIRED_Executable)
    list(APPEND _ASPELL_REQUIRED_VARS ASPELL_EXECUTABLE)
  endif()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  ASPELL
  REQUIRED_VARS ${_ASPELL_REQUIRED_VARS}
  HANDLE_COMPONENTS
  VERSION_VAR ASPELL_VERSION
  REASON_FAILURE_MESSAGE "${_ASPELL_REASON_FAILURE_MESSAGE}"
)

unset(_ASPELL_REASON_FAILURE_MESSAGE)
unset(_ASPELL_REQUIRED_VARS)

if(NOT ASPELL_FOUND)
  return()
endif()

get_property(_ASPELL_ROLE GLOBAL PROPERTY CMAKE_ROLE)

if("ASPELL" IN_LIST ASPELL_FIND_COMPONENTS AND ASPELL_ASPELL_FOUND)
  set(ASPELL_INCLUDE_DIRS ${ASPELL_INCLUDE_DIR})
  if(ASPELL_PSPELL_INCLUDE_DIR)
    list(APPEND ASPELL_INCLUDE_DIRS ${ASPELL_PSPELL_INCLUDE_DIR})
    list(REMOVE_DUPLICATES ASPELL_INCLUDE_DIRS)
  endif()
  set(ASPELL_LIBRARIES ${ASPELL_LIBRARY})

  if(_ASPELL_ROLE STREQUAL "PROJECT" AND NOT TARGET ASPELL::ASPELL)
    if(IS_ABSOLUTE "${ASPELL_LIBRARY}")
      add_library(ASPELL::ASPELL UNKNOWN IMPORTED)
      set_target_properties(
        ASPELL::ASPELL
        PROPERTIES
          IMPORTED_LOCATION "${ASPELL_LIBRARY}"
      )
    else()
      add_library(ASPELL::ASPELL INTERFACE IMPORTED)
      set_target_properties(
        ASPELL::ASPELL
        PROPERTIES
          IMPORTED_LIBNAME "${ASPELL_LIBRARY}"
      )
    endif()

    set_target_properties(
      ASPELL::ASPELL
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${ASPELL_INCLUDE_DIRS}"
    )
  endif()
endif()

if(
  _ASPELL_ROLE STREQUAL "PROJECT"
  AND "Executable" IN_LIST ASPELL_FIND_COMPONENTS
  AND ASPELL_Executable_FOUND
  AND NOT TARGET ASPELL::Executable
)
  add_executable(ASPELL::Executable IMPORTED)
  set_target_properties(
    ASPELL::Executable
    PROPERTIES
      IMPORTED_LOCATION "${ASPELL_EXECUTABLE}"
  )
endif()

unset(_ASPELL_ROLE)
