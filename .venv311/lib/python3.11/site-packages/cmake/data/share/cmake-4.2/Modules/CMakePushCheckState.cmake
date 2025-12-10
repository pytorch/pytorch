# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

include_guard(GLOBAL)

#[=======================================================================[.rst:
CMakePushCheckState
-------------------

This module provides commands for managing the state of variables that influence
how various CMake check commands (e.g., :command:`check_symbol_exists`, etc.)
are performed.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CMakePushCheckState)

This module provides the following commands, which are useful for scoped
configuration, for example, in CMake modules or when performing checks in a
controlled environment, ensuring that temporary modifications are isolated
to the scope of the check and do not propagate into other parts of the build
system:

* :command:`cmake_push_check_state`
* :command:`cmake_reset_check_state`
* :command:`cmake_pop_check_state`

Affected Variables
^^^^^^^^^^^^^^^^^^

The following CMake variables are saved, reset, and restored by this module's
commands:

.. include:: /module/include/CMAKE_REQUIRED_FLAGS.rst

.. include:: /module/include/CMAKE_REQUIRED_DEFINITIONS.rst

.. include:: /module/include/CMAKE_REQUIRED_INCLUDES.rst

.. include:: /module/include/CMAKE_REQUIRED_LINK_OPTIONS.rst

.. include:: /module/include/CMAKE_REQUIRED_LIBRARIES.rst

.. include:: /module/include/CMAKE_REQUIRED_LINK_DIRECTORIES.rst

.. include:: /module/include/CMAKE_REQUIRED_QUIET.rst

``CMAKE_EXTRA_INCLUDE_FILES``
  .. versionadded:: 3.6
    Previously used already by the :command:`check_type_size` command;  now
    also supported by this module.

  A :ref:`semicolon-separated list <CMake Language Lists>` of extra header
  files to include when performing the check.

.. note::

  Other CMake variables, such as :variable:`CMAKE_<LANG>_FLAGS`, propagate
  to all checks regardless of commands provided by this module, as those
  fundamental variables are designed to influence the global state of the
  build system.

Commands
^^^^^^^^

.. command:: cmake_push_check_state

  Pushes (saves) the current states of the above variables onto a stack:

  .. code-block:: cmake

    cmake_push_check_state([RESET])

  Use this command to preserve the current configuration before making
  temporary modifications for specific checks.

  ``RESET``
    When this option is specified, the command not only saves the current states
    of the listed variables but also resets them to empty, allowing them to be
    reconfigured from a clean state.

.. command:: cmake_reset_check_state

  Resets (clears) the contents of the variables listed above to empty states:

  .. code-block:: cmake

    cmake_reset_check_state()

  Use this command when performing multiple sequential checks that require
  entirely new configurations, ensuring no previous configuration
  unintentionally carries over.

.. command:: cmake_pop_check_state

  Restores the states of the variables listed above to their values at the time
  of the most recent ``cmake_push_check_state()`` call:

  .. code-block:: cmake

    cmake_pop_check_state()

  Use this command to revert temporary changes made during a check.  To
  prevent unexpected behavior, pair each ``cmake_push_check_state()`` with a
  corresponding ``cmake_pop_check_state()``.

Examples
^^^^^^^^

Example: Isolated Check With Compile Definitions
""""""""""""""""""""""""""""""""""""""""""""""""

In the following example, a check for the C symbol ``memfd_create()`` is
performed with an additional ``_GNU_SOURCE`` compile definition,  without
affecting global compile flags.  The ``RESET`` option is used to ensure
that any prior values of the check-related variables are explicitly cleared
before the check.

.. code-block:: cmake

  include(CMakePushCheckState)

  # Save and reset the current state
  cmake_push_check_state(RESET)

  # Perform check with specific compile definitions
  set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
  include(CheckSymbolExists)
  check_symbol_exists(memfd_create "sys/mman.h" HAVE_MEMFD_CREATE)

  # Restore the original state
  cmake_pop_check_state()

Example: Nested Configuration Scopes
""""""""""""""""""""""""""""""""""""

In the following example, variable states are pushed onto the stack multiple
times, allowing for sequential or nested checks.  Each
``cmake_pop_check_state()`` restores the most recent pushed states.

.. code-block:: cmake

  include(CMakePushCheckState)

  # Save and reset the current state
  cmake_push_check_state(RESET)

  # Perform the first check with additional libraries
  set(CMAKE_REQUIRED_LIBRARIES ${CMAKE_DL_LIBS})
  include(CheckSymbolExists)
  check_symbol_exists(dlopen "dlfcn.h" HAVE_DLOPEN)

  # Save current state
  cmake_push_check_state()

  # Perform the second check with libraries and additional compile definitions
  set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
  check_symbol_exists(dladdr "dlfcn.h" HAVE_DLADDR)

  message(STATUS "${CMAKE_REQUIRED_DEFINITIONS}")
  # Output: -D_GNU_SOURCE

  # Restore the previous state
  cmake_pop_check_state()

  message(STATUS "${CMAKE_REQUIRED_DEFINITIONS}")
  # Output here is empty

  # Reset variables to prepare for the next check
  cmake_reset_check_state()

  # Perform the next check only with additional compile definitions
  set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)
  check_symbol_exists(dl_iterate_phdr "link.h" HAVE_DL_ITERATE_PHDR)

  # Restore the original state
  cmake_pop_check_state()
#]=======================================================================]

macro(CMAKE_RESET_CHECK_STATE)

  set(CMAKE_EXTRA_INCLUDE_FILES)
  set(CMAKE_REQUIRED_INCLUDES)
  set(CMAKE_REQUIRED_DEFINITIONS)
  set(CMAKE_REQUIRED_LINK_OPTIONS)
  set(CMAKE_REQUIRED_LIBRARIES)
  set(CMAKE_REQUIRED_LINK_DIRECTORIES)
  set(CMAKE_REQUIRED_FLAGS)
  set(CMAKE_REQUIRED_QUIET)

endmacro()

macro(CMAKE_PUSH_CHECK_STATE)

  if(NOT DEFINED _CMAKE_PUSH_CHECK_STATE_COUNTER)
    set(_CMAKE_PUSH_CHECK_STATE_COUNTER 0)
  endif()

  math(EXPR _CMAKE_PUSH_CHECK_STATE_COUNTER "${_CMAKE_PUSH_CHECK_STATE_COUNTER}+1")

  set(_CMAKE_EXTRA_INCLUDE_FILES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}        ${CMAKE_EXTRA_INCLUDE_FILES})
  set(_CMAKE_REQUIRED_INCLUDES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}          ${CMAKE_REQUIRED_INCLUDES})
  set(_CMAKE_REQUIRED_DEFINITIONS_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}       ${CMAKE_REQUIRED_DEFINITIONS})
  set(_CMAKE_REQUIRED_LINK_OPTIONS_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}      ${CMAKE_REQUIRED_LINK_OPTIONS})
  set(_CMAKE_REQUIRED_LIBRARIES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}         ${CMAKE_REQUIRED_LIBRARIES})
  set(_CMAKE_REQUIRED_LINK_DIRECTORIES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}  ${CMAKE_REQUIRED_LINK_DIRECTORIES})
  set(_CMAKE_REQUIRED_FLAGS_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}             ${CMAKE_REQUIRED_FLAGS})
  set(_CMAKE_REQUIRED_QUIET_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}             ${CMAKE_REQUIRED_QUIET})

  if (${ARGC} GREATER 0 AND "${ARGV0}" STREQUAL "RESET")
    cmake_reset_check_state()
  endif()

endmacro()

macro(CMAKE_POP_CHECK_STATE)

# don't pop more than we pushed
  if("${_CMAKE_PUSH_CHECK_STATE_COUNTER}" GREATER "0")

    set(CMAKE_EXTRA_INCLUDE_FILES       ${_CMAKE_EXTRA_INCLUDE_FILES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})
    set(CMAKE_REQUIRED_INCLUDES         ${_CMAKE_REQUIRED_INCLUDES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})
    set(CMAKE_REQUIRED_DEFINITIONS      ${_CMAKE_REQUIRED_DEFINITIONS_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})
    set(CMAKE_REQUIRED_LINK_OPTIONS     ${_CMAKE_REQUIRED_LINK_OPTIONS_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})
    set(CMAKE_REQUIRED_LIBRARIES        ${_CMAKE_REQUIRED_LIBRARIES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})
    set(CMAKE_REQUIRED_LINK_DIRECTORIES ${_CMAKE_REQUIRED_LINK_DIRECTORIES_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})
    set(CMAKE_REQUIRED_FLAGS            ${_CMAKE_REQUIRED_FLAGS_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})
    set(CMAKE_REQUIRED_QUIET            ${_CMAKE_REQUIRED_QUIET_SAVE_${_CMAKE_PUSH_CHECK_STATE_COUNTER}})

    math(EXPR _CMAKE_PUSH_CHECK_STATE_COUNTER "${_CMAKE_PUSH_CHECK_STATE_COUNTER}-1")
  endif()

endmacro()
