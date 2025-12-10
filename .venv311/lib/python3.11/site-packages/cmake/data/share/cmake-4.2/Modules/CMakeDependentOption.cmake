# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CMakeDependentOption
--------------------

This module provides a command to define boolean options whose availability and
default values depend on specified conditions or other options.  This helps
maintain a clean configuration interface by only displaying options that are
relevant to the current settings.

Load this module in CMake with:

.. code-block:: cmake

  include(CMakeDependentOption)

Commands
^^^^^^^^

This module provides the following command:

.. command:: cmake_dependent_option

  Provides a boolean option that depends on a set of conditions:

  .. code-block:: cmake

    cmake_dependent_option(<variable> <help> <value> <condition> <else-value>)

  This command creates a boolean ``<variable>`` and makes it available to the
  user in the GUI (such as :manual:`cmake-gui(1)` or :manual:`ccmake(1)`), if
  a set of conditions evaluates to boolean true.

  The arguments are:

  ``<variable>``
    The name of a variable that stores the option value.

  ``<help>``
    A brief description of the option.  This string is typically a short line of
    text and is displayed in the GUI.

  ``<value>``
    Boolean value for the ``<variable>``, when ``<condition>`` evaluates to
    boolean true.

  ``<condition>``
    Specifies the conditions that determine whether ``<variable>`` is set and
    visible in the GUI.

    * If ``<condition>`` evaluates to boolean false, option is hidden from the
      user in the GUI, and a local variable ``<variable>`` is set to
      ``<else-value>``.

    * If ``<condition>`` evaluates to boolean true, a boolean cache variable
      named ``<variable>`` is created with default ``<value>``, and option is
      shown in the GUI, allowing the user to enable or disable it.

    * If ``<condition>`` later evaluates to boolean false (on consecutive
      configuration run),  option is hidden from the user in the GUI and the
      ``<variable>`` type is changed to an internal cache variable.  In that
      case a local variable of the same name is set to ``<else-value>``.

    * If ``<condition>`` becomes true again in consecutive configuration runs,
      the user's previously set value is preserved.

    The ``<condition>`` argument can be:

    * A single condition (such as a variable name).

    * A :ref:`semicolon-separated list <CMake Language Lists>` of multiple
      conditions.

    * .. versionadded:: 3.22
        A full :ref:`Condition Syntax` as used in an ``if(<condition>)`` clause.
        See policy :policy:`CMP0127`.  This enables using entire condition
        syntax (such as grouping conditions with parens and similar).

  ``<else-value>``
    The value assigned to a local variable named ``<variable>``, when
    ``<condition>`` evaluates to boolean false.

  In CMake project mode, boolean cache variables are created as explained
  above.  In CMake script mode, boolean variables are set instead.

Examples
^^^^^^^^

Example: Basic Usage
""""""""""""""""""""

Using this module in a project to conditionally set an option:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(CMakeDependentOption)

  cmake_dependent_option(USE_SSL_GNUTLS "Use GnuTLS for SSL" ON USE_SSL OFF)

Example: Enabling/Disabling Dependent Option
""""""""""""""""""""""""""""""""""""""""""""

Extending the previous example, this demonstrates how the module allows
user-configurable options based on a condition during the configuration phase:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(CMakeDependentOption)

  option(USE_SSL "Enable SSL in the project" OFF)

  cmake_dependent_option(USE_SSL_GNUTLS "Use GnuTLS for SSL" ON USE_SSL OFF)

  message(STATUS "USE_SSL: ${USE_SSL}")
  message(STATUS "USE_SSL_GNUTLS: ${USE_SSL_GNUTLS}")

On the first configuration run, a boolean cache variable ``USE_SSL`` is set to
OFF, and a local variable ``USE_SSL_GNUTLS`` is set to OFF:

.. code-block:: console

  $ cmake -B build-dir

  -- USE_SSL: OFF
  -- USE_SSL_GNUTLS: OFF

Running CMake with ``USE_SSL=ON`` sets both ``USE_SSL`` and ``USE_SSL_GNUTLS``
boolean cache variables to ON:

.. code-block:: console

  $ cmake -B build-dir -D USE_SSL=ON

  -- USE_SSL: ON
  -- USE_SSL_GNUTLS: ON

On a subsequent configuration run with ``USE_SSL=OFF``, ``USE_SSL_GNUTLS``
follows suit.  However, its value is preserved in the internal cache while being
overridden locally:

.. code-block:: console

  $ cmake -B build-dir -D USE_SSL=OFF

  -- USE_SSL: OFF
  -- USE_SSL_GNUTLS: OFF

Example: Semicolon-separated List of Conditions
"""""""""""""""""""""""""""""""""""""""""""""""

The ``<condition>`` argument can also be a semicolon-separated list of
conditions.  In the following example, if the variable ``USE_BAR`` is ON and
variable ``USE_ZOT`` is OFF, the option ``USE_FOO`` is available and defaults to
ON.  Otherwise, ``USE_FOO`` is set to OFF and hidden from the user.

If the values of ``USE_BAR`` or ``USE_ZOT`` change in the future configuration
runs, the previous value of ``USE_FOO`` is preserved so that when it becomes
available again, it retains its last set value.

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(CMakeDependentOption)

  cmake_dependent_option(USE_FOO "Use Foo" ON "USE_BAR;NOT USE_ZOT" OFF)

Example: Full Condition Syntax
""""""""""""""""""""""""""""""

As of CMake 3.22, ``cmake_dependent_option()`` supports full condition syntax.

In fhe following example, if the condition evaluates to true, the option
``USE_FOO`` is available and set to ON.  Otherwise, it is set to OFF and hidden
in the GUI.  The value of ``USE_FOO`` is preserved across configuration runs,
similar to the previous example.

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(CMakeDependentOption)

  cmake_dependent_option(USE_FOO "Use Foo" ON "USE_A AND (USE_B OR USE_C)" OFF)

Another example demonstrates how an option can be conditionally available based
on the target system:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(CMakeDependentOption)

  cmake_dependent_option(
    ENABLE_FOO
    "Enable feature Foo (this option is available when building for Windows)"
    ON
    [[CMAKE_SYSTEM_NAME STREQUAL "Windows"]]
    OFF
  )

See Also
^^^^^^^^

* The :command:`option` command to provide a boolean option that the user can
  optionally select.
#]=======================================================================]

macro(CMAKE_DEPENDENT_OPTION option doc default depends force)
  cmake_policy(GET CMP0127 _CDO_CMP0127
    PARENT_SCOPE # undocumented, do not use outside of CMake
    )
  if(${option}_ISSET MATCHES "^${option}_ISSET$")
    set(${option}_AVAILABLE 1)
    if("x${_CDO_CMP0127}x" STREQUAL "xNEWx")
      foreach(d ${depends})
        cmake_language(EVAL CODE "
          if (${d})
          else()
            set(${option}_AVAILABLE 0)
          endif()"
        )
      endforeach()
    else()
      foreach(d ${depends})
        string(REGEX REPLACE " +" ";" CMAKE_DEPENDENT_OPTION_DEP "${d}")
        if(${CMAKE_DEPENDENT_OPTION_DEP})
        else()
          set(${option}_AVAILABLE 0)
        endif()
      endforeach()
    endif()
    if(${option}_AVAILABLE)
      option(${option} "${doc}" "${default}")
      set(${option} "${${option}}" CACHE BOOL "${doc}" FORCE)
    else()
      if(${option} MATCHES "^${option}$")
      else()
        set(${option} "${${option}}" CACHE INTERNAL "${doc}")
      endif()
      set(${option} ${force})
    endif()
  else()
    set(${option} "${${option}_ISSET}")
  endif()
  if("x${_CDO_CMP0127}x" STREQUAL "xx" AND "x${depends}x" MATCHES "[^A-Za-z0-9_.; ]")
    cmake_policy(GET_WARNING CMP0127 _CDO_CMP0127_WARNING)
    message(AUTHOR_WARNING "${_CDO_CMP0127_WARNING}")
  endif()
  unset(_CDO_CMP0127)
endmacro()
