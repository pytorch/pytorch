# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckPIESupported
-----------------

.. versionadded:: 3.14

This module provides a command to check whether the linker supports Position
Independent Code (PIE) or No Position Independent Code (NO_PIE) for
executables.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckPIESupported)

When setting the :prop_tgt:`POSITION_INDEPENDENT_CODE` target property,
PIC-related compile and link options are added when building library objects,
and PIE-related compile options are added when building objects of executable
targets, regardless of this module.  Use this module to ensure that the
``POSITION_INDEPENDENT_CODE`` target property for executables is also honored
at link time.

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_pie_supported

  Checks for PIE/NO_PIE support and prepares all executables to have link
  time PIE options enabled:

  .. code-block:: cmake

    check_pie_supported([OUTPUT_VARIABLE <output>] [LANGUAGES <langs>...])

  Options are:

  ``OUTPUT_VARIABLE <output>``
    Set ``<output>`` variable with details about any error. If the check is
    bypassed because it uses cached results from a previous call, the output
    will be empty even if errors were present in the previous call.

  ``LANGUAGES <langs>...``
    Check the linkers used for each of the specified languages.
    If this option is not provided, the command checks all enabled languages.

    ``C``, ``CXX``, ``Fortran`` are supported.

    .. versionadded:: 3.23

      ``OBJC``, ``OBJCXX``, ``CUDA``, and ``HIP`` are supported.

  .. note::

    To use ``check_pie_supported()``, policy :policy:`CMP0083` must be set to
    ``NEW``; otherwise, a fatal error will occur.

Variables
^^^^^^^^^

For each language checked, the ``check_pie_supported()`` command defines two
boolean cache variables:

``CMAKE_<lang>_LINK_PIE_SUPPORTED``
  Set to true if ``PIE`` is supported by the linker and false otherwise.
``CMAKE_<lang>_LINK_NO_PIE_SUPPORTED``
  Set to true if ``NO_PIE`` is supported by the linker and false otherwise.

Examples
^^^^^^^^

To enable PIE on an executable target at link time as well, include this module
and call ``check_pie_supported()`` before setting the
``POSITION_INDEPENDENT_CODE`` target property.  This will determine whether the
linker for each checked language supports PIE-related link options.  For
example:

.. code-block:: cmake

  add_executable(foo ...)

  include(CheckPIESupported)
  check_pie_supported()
  set_property(TARGET foo PROPERTY POSITION_INDEPENDENT_CODE TRUE)

Since not all linkers require or support PIE-related link options (for example,
``MSVC``), retrieving any error messages might be useful for logging purposes:

.. code-block:: cmake

  add_executable(foo ...)

  message(CHECK_START "Checking for C linker PIE support")

  include(CheckPIESupported)
  check_pie_supported(OUTPUT_VARIABLE output LANGUAGES C)
  set_property(TARGET foo PROPERTY POSITION_INDEPENDENT_CODE TRUE)

  if(CMAKE_C_LINK_PIE_SUPPORTED)
    message(CHECK_PASS "yes")
  else()
    message(CHECK_FAIL "no")
    message(VERBOSE "PIE is not supported at link time:\n${output}"
                    "PIE link options will not be passed to linker.")
  endif()

Setting the ``POSITION_INDEPENDENT_CODE`` target property on an executable
without this module will set PIE-related compile options but not PIE-related
link options, which might not be sufficient in certain cases:

.. code-block:: cmake

  add_executable(foo ...)
  set_property(TARGET foo PROPERTY POSITION_INDEPENDENT_CODE TRUE)
#]=======================================================================]


include (Internal/CheckLinkerFlag)

function (check_pie_supported)
  cmake_policy(GET CMP0083 cmp0083)

  if (NOT cmp0083)
    message(FATAL_ERROR "check_pie_supported: Policy CMP0083 is not set")
  endif()

  if(cmp0083 STREQUAL "OLD")
    message(FATAL_ERROR "check_pie_supported: Policy CMP0083 set to OLD")
  endif()

  set(optional)
  set(one OUTPUT_VARIABLE)
  set(multiple LANGUAGES)

  cmake_parse_arguments(CHECK_PIE "${optional}" "${one}" "${multiple}" "${ARGN}")
  if(CHECK_PIE_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "check_pie_supported: Unparsed arguments: ${CHECK_PIE_UNPARSED_ARGUMENTS}")
  endif()

  if (CHECK_PIE_LANGUAGES)
    set (unsupported_languages "${CHECK_PIE_LANGUAGES}")
    list (REMOVE_ITEM unsupported_languages "C" "CXX" "OBJC" "OBJCXX" "Fortran" "CUDA" "HIP")
    if(unsupported_languages)
      message(FATAL_ERROR "check_pie_supported: language(s) '${unsupported_languages}' not supported")
    endif()
  else()
    # User did not set any languages, use defaults
    get_property (enabled_languages GLOBAL PROPERTY ENABLED_LANGUAGES)
    if (NOT enabled_languages)
      return()
    endif()

    list (FILTER enabled_languages INCLUDE REGEX "^(C|CXX|OBJC|OBJCXX|Fortran|CUDA|HIP)$")
    if (NOT enabled_languages)
      return()
    endif()

    set (CHECK_PIE_LANGUAGES ${enabled_languages})
  endif()

  set(CMAKE_REQUIRED_QUIET TRUE)
  set (outputs)

  # Isolate the checks below from the project's PIC selection.
  unset(CMAKE_POSITION_INDEPENDENT_CODE)

  foreach(lang IN LISTS CHECK_PIE_LANGUAGES)
    if(_CMAKE_${lang}_PIE_MAY_BE_SUPPORTED_BY_LINKER)
      if(NOT DEFINED CMAKE_${lang}_LINK_PIE_SUPPORTED)
        # ensure PIE compile flags are also used
        list(JOIN CMAKE_${lang}_COMPILE_OPTIONS_PIE " " CMAKE_REQUIRED_FLAGS)
        cmake_check_linker_flag(${lang}
                                "${CMAKE_${lang}_LINK_OPTIONS_PIE}"
                                CMAKE_${lang}_LINK_PIE_SUPPORTED
                                OUTPUT_VARIABLE output)
        if (NOT CMAKE_${lang}_LINK_PIE_SUPPORTED)
          string (APPEND outputs "PIE (${lang}): ${output}\n")
        endif()
        unset(CMAKE_REQUIRED_FLAGS)
      endif()

      if(NOT DEFINED CMAKE_${lang}_LINK_NO_PIE_SUPPORTED)
        cmake_check_linker_flag(${lang}
                                "${CMAKE_${lang}_LINK_OPTIONS_NO_PIE}"
                                CMAKE_${lang}_LINK_NO_PIE_SUPPORTED
                                OUTPUT_VARIABLE output)
        if (NOT CMAKE_${lang}_LINK_NO_PIE_SUPPORTED)
          string (APPEND outputs "NO_PIE (${lang}): ${output}\n")
        endif()
      endif()
    else()
      # no support at link time. Set cache variables to NO
      set(CMAKE_${lang}_LINK_PIE_SUPPORTED NO CACHE INTERNAL "PIE (${lang})")
      set(CMAKE_${lang}_LINK_NO_PIE_SUPPORTED NO CACHE INTERNAL "NO_PIE (${lang})")
      string (APPEND outputs "PIE and NO_PIE are not supported by linker for ${lang}\n")
    endif()
  endforeach()

  if (CHECK_PIE_OUTPUT_VARIABLE)
    set (${CHECK_PIE_OUTPUT_VARIABLE} "${outputs}" PARENT_SCOPE)
  endif()
endfunction()
