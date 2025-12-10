# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindPackageMessage
------------------

This module provides a command for printing find result messages and is
intended for use in :ref:`Find Modules` implementing
:command:`find_package(<PackageName>)` calls.

Load this module in a CMake find module with:

.. code-block:: cmake
  :caption: ``FindFoo.cmake``

  include(FindPackageMessage)

Commands
^^^^^^^^

This module provides the following command:

.. command:: find_package_message

  Prints a message once for each unique find result to inform the user which
  package was found and where:

  .. code-block:: cmake

    find_package_message(<PackageName> <message> <details>)

  ``<PackageName>``
    The name of the package (for example, as used in the
    ``Find<PackageName>.cmake`` module filename).

  ``<message>``
    The message string to display.

  ``<details>``
    A unique identifier for tracking message display.  The ``<message>`` is
    printed only once per distinct ``<details>`` value.  If ``<details>`` string
    changes in a subsequent configuration phase, the message will be displayed
    again.

  If :command:`find_package` was called with the ``QUIET`` option, the
  ``<message>`` is not printed.

Examples
^^^^^^^^

Printing a result message in a custom find module:

.. code-block:: cmake
  :caption: ``FindFoo.cmake``

  find_library(Foo_LIBRARY foo)
  find_path(Foo_INCLUDE_DIR foo.h)

  include(FindPackageMessage)

  if(Foo_LIBRARY AND Foo_INCLUDE_DIR)
    find_package_message(
      Foo
      "Found Foo: ${Foo_LIBRARY}"
      "[${Foo_LIBRARY}][${Foo_INCLUDE_DIR}]"
    )
  else()
    message(STATUS "Could NOT find Foo")
  endif()

When writing standard find modules, use the
:module:`FindPackageHandleStandardArgs` module and its
:command:`find_package_handle_standard_args` command which automatically
prints the find result message based on whether the package was found:

.. code-block:: cmake
  :caption: ``FindFoo.cmake``

  # ...

  include(FindPackageHandleStandardArgs)

  find_package_handle_standard_args(
    Foo
    REQUIRED_VARS Foo_LIBRARY Foo_INCLUDE_DIR
  )
#]=======================================================================]

function(find_package_message pkg msg details)
  # Avoid printing a message repeatedly for the same find result.
  if(NOT ${pkg}_FIND_QUIETLY)
    string(REPLACE "\n" "" details "${details}")
    set(DETAILS_VAR FIND_PACKAGE_MESSAGE_DETAILS_${pkg})
    if(NOT "${details}" STREQUAL "${${DETAILS_VAR}}")
      # The message has not yet been printed.
      string(STRIP "${msg}" msg)
      message(STATUS "${msg}")

      # Save the find details in the cache to avoid printing the same
      # message again.
      set("${DETAILS_VAR}" "${details}"
        CACHE INTERNAL "Details about finding ${pkg}")
    endif()
  endif()
endfunction()
