# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckTypeSize
-------------

This module provides a command to check the size of a C/C++ type or expression.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckTypeSize)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_type_size

  Checks once whether the C/C++ type or expression exists and determines its
  size:

  .. code-block:: cmake

    check_type_size(
      <type>
      <size-var>
      [RESULT_VARIABLE <result-var>]
      [BUILTIN_TYPES_ONLY]
      [LANGUAGE <language>]
    )

  The arguments are:

  ``<type>``
    The type or expression being checked.

  ``<size-var>``
    The name of the internal cache variable for storing the size of the type
    or expression ``<type>``.  This name is also used as a prefix as
    explained below.

  ``RESULT_VARIABLE <result-var>``
    .. versionadded:: 4.2

    The name of the internal cache variable that holds a boolean value
    indicating whether the type or expression ``<type>`` exists.  If *not*
    given, the command will by default define an internal cache variable
    named ``HAVE_<size-var>`` instead.

  ``BUILTIN_TYPES_ONLY``
    If given, only compiler-builtin types will be supported in the check.
    If *not* given, the command checks for common headers ``<sys/types.h>``,
    ``<stdint.h>``, and ``<stddef.h>``, and saves results in
    ``HAVE_SYS_TYPES_H``, ``HAVE_STDINT_H``, and ``HAVE_STDDEF_H`` internal
    cache variables.  For C++ ``std::`` types, ``<cstdint>`` and
    ``<cstddef>`` are also checked with ``HAVE_CSTDINT`` and
    ``HAVE_CSTDDEF`` defined respectively.  The command automatically
    includes the available headers in the type size check, thus supporting
    checks of types defined in the headers.

  ``LANGUAGE <language>``
    Uses the ``<language>`` compiler to perform the check.
    Acceptable values are ``C`` and ``CXX``.
    If not specified, it defaults to ``C``.

  .. rubric:: Result Variables

  Results are reported in the following variables:

  ``<size-var>``
    Internal cache variable that holds one of the following values:

    ``<size>``
      If the type or expression ``<type>`` exists, it will have a non-zero
      size ``<size>`` in bytes.

    ``0``
      When the type has an architecture-dependent size;  This may occur when
      :variable:`CMAKE_OSX_ARCHITECTURES` has multiple architectures.  In
      this case also the ``<size-var>_KEYS`` variable is defined and the
      ``<size-var>_CODE`` variable contains preprocessor tests mapping as
      explained below.

    "" (empty string)
      When the type or expression ``<type>`` does not exist.

  ``HAVE_<size-var>``
    Internal cache variable that holds a boolean value indicating whether
    the type or expression ``<type>`` exists.  This variable is defined
    when the ``RESULT_VARIABLE`` argument is not used.

  ``<result-var>``
    .. versionadded:: 4.2

    Internal cache variable defined when the ``RESULT_VARIABLE`` argument
    is used. It holds a boolean value indicating whether the type or
    expression ``<type>`` exists (same value as ``HAVE_<size-var>``). In
    this case, the ``HAVE_<size-var>`` variable is not defined.

  ``<size-var>_CODE``
    CMake variable that holds preprocessor code to define the macro
    ``<size-var>`` to the size of the type, or to leave the macro undefined
    if the type does not exist.

    When the type has an architecture-dependent size (``<size-var>`` value
    is ``0``) this variable contains preprocessor tests mapping from each
    architecture macro to the corresponding type size.

  ``<size-var>_KEYS``
    CMake variable that is defined only when the type has an
    architecture-dependent size (``<size-var>`` value is ``0``) and contains
    a list of architecture macros. The value for each key is stored in
    ``<size-var>-<key>`` variables.

  .. rubric:: Variables Affecting the Check

  The following variables may be set before calling this command to modify
  the way the check is run:

  .. include:: /module/include/CMAKE_REQUIRED_FLAGS.rst

  .. include:: /module/include/CMAKE_REQUIRED_DEFINITIONS.rst

  .. include:: /module/include/CMAKE_REQUIRED_INCLUDES.rst

  .. include:: /module/include/CMAKE_REQUIRED_LINK_OPTIONS.rst

  .. include:: /module/include/CMAKE_REQUIRED_LIBRARIES.rst

  .. include:: /module/include/CMAKE_REQUIRED_LINK_DIRECTORIES.rst

  .. include:: /module/include/CMAKE_REQUIRED_QUIET.rst

  ``CMAKE_EXTRA_INCLUDE_FILES``
    A :ref:`semicolon-separated list <CMake Language Lists>` of extra header
    files to include when performing the check.

Examples
^^^^^^^^

Consider the code:

.. code-block:: cmake

  include(CheckTypeSize)

  # Check for size of long.
  check_type_size(long SIZEOF_LONG)

  message("HAVE_SIZEOF_LONG: ${HAVE_SIZEOF_LONG}")
  message("SIZEOF_LONG: ${SIZEOF_LONG}")
  message("SIZEOF_LONG_CODE: ${SIZEOF_LONG_CODE}")

On a 64-bit architecture, the output may look something like this::

  HAVE_SIZEOF_LONG: TRUE
  SIZEOF_LONG: 8
  SIZEOF_LONG_CODE: #define SIZEOF_LONG 8

On Apple platforms, when :variable:`CMAKE_OSX_ARCHITECTURES` has multiple
architectures, types may have architecture-dependent sizes.
For example, with the code:

.. code-block:: cmake

  include(CheckTypeSize)

  check_type_size(long SIZEOF_LONG)

  message("HAVE_SIZEOF_LONG: ${HAVE_SIZEOF_LONG}")
  message("SIZEOF_LONG: ${SIZEOF_LONG}")
  foreach(key IN LISTS SIZEOF_LONG_KEYS)
    message("key: ${key}")
    message("value: ${SIZEOF_LONG-${key}}")
  endforeach()
  message("SIZEOF_LONG_CODE:\n${SIZEOF_LONG_CODE}")

the result may be::

  HAVE_SIZEOF_LONG: TRUE
  SIZEOF_LONG: 0
  key: __i386
  value: 4
  key: __x86_64
  value: 8
  SIZEOF_LONG_CODE:
  #if defined(__i386)
  # define SIZEOF_LONG 4
  #elif defined(__x86_64)
  # define SIZEOF_LONG 8
  #else
  # error SIZEOF_LONG unknown
  #endif

Example: Configuration Header
"""""""""""""""""""""""""""""

The next example demonstrates how the result variables can be used in a
configuration header:

.. code-block:: cmake

  include(CheckTypeSize)
  check_type_size(long SIZEOF_LONG)

  configure_file(config.h.in config.h @ONLY)

.. code-block:: c
  :caption: ``config.h.in``
  :force:

  /* Define whether the type 'long' exists. */
  #cmakedefine HAVE_SIZEOF_LONG

  /* The size of 'long', as computed by sizeof. */
  @SIZEOF_LONG_CODE@

Example: Checking Complex Expressions
"""""""""""""""""""""""""""""""""""""

Despite the name of this module, it may also be used to determine the size
of more complex expressions.  For example, to check the size of a struct
member:

.. code-block:: cmake

  include(CheckTypeSize)
  check_type_size("((struct something*)0)->member" SIZEOF_MEMBER)

Example: Isolated Check
"""""""""""""""""""""""

In the following example, the check is performed with temporarily modified
additional headers using the ``CMAKE_EXTRA_INCLUDE_FILES`` variable and
:module:`CMakePushCheckState` module.  The result of the check is stored in
``HAVE_SIZEOF_UNION_SEMUN``, and size is stored in ``SIZEOF_UNION_SEMUN``
internal cache variables.

.. code-block:: cmake

  include(CheckTypeSize)
  include(CMakePushCheckState)

  cmake_push_check_state(RESET)
    set(CMAKE_EXTRA_INCLUDE_FILES sys/types.h sys/ipc.h sys/sem.h)
    check_type_size("union semun" SIZEOF_UNION_SEMUN)
  cmake_pop_check_state()

Example: Customizing Result Variable
""""""""""""""""""""""""""""""""""""

Since CMake 4.2, the ``HAVE_<size-var>`` variable name can be customized
using the ``RESULT_VARIABLE`` argument. In the following example, this
module is used to check whether the ``struct flock`` exists, and the result
is stored in the ``MyProj_HAVE_STRUCT_FLOCK`` internal cache variable:

.. code-block:: cmake

  cmake_minimum_required(VERSION 4.2)

  # ...

  include(CheckTypeSize)
  include(CMakePushCheckState)

  cmake_push_check_state(RESET)
    set(CMAKE_EXTRA_INCLUDE_FILES "fcntl.h")

    check_type_size(
      "struct flock"
      MyProj_SIZEOF_STRUCT_FLOCK
      RESULT_VARIABLE MyProj_HAVE_STRUCT_FLOCK
    )
  cmake_pop_check_state()
#]=======================================================================]

include(CheckIncludeFile)
include(CheckIncludeFileCXX)

include_guard(GLOBAL)

block(SCOPE_FOR POLICIES)
cmake_policy(SET CMP0140 NEW)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>
cmake_policy(SET CMP0174 NEW)

#-----------------------------------------------------------------------------
# Helper function.  DO NOT CALL DIRECTLY.
function(__check_type_size_impl type var result_var map builtin language)
  if(NOT CMAKE_REQUIRED_QUIET)
    message(CHECK_START "Check size of ${type}")
  endif()

  # Perform language check
  string(MAKE_C_IDENTIFIER ${var} _var_escaped)
  if(language STREQUAL "C")
    set(src ${_var_escaped}.c)
  elseif(language STREQUAL "CXX")
    set(src ${_var_escaped}.cpp)
  else()
    message(FATAL_ERROR "Unknown language:\n  ${language}\nSupported languages: C, CXX.\n")
  endif()

  # Include header files.
  set(headers)
  if(NOT builtin)
    if(language STREQUAL "CXX" AND type MATCHES "^std::")
      if(HAVE_SYS_TYPES_H)
        string(APPEND headers "#include <sys/types.h>\n")
      endif()
      if(HAVE_CSTDINT)
        string(APPEND headers "#include <cstdint>\n")
      endif()
      if(HAVE_CSTDDEF)
        string(APPEND headers "#include <cstddef>\n")
      endif()
    else()
      if(HAVE_SYS_TYPES_H)
        string(APPEND headers "#include <sys/types.h>\n")
      endif()
      if(HAVE_STDINT_H)
        string(APPEND headers "#include <stdint.h>\n")
      endif()
      if(HAVE_STDDEF_H)
        string(APPEND headers "#include <stddef.h>\n")
      endif()
    endif()
  endif()
  foreach(h ${CMAKE_EXTRA_INCLUDE_FILES})
    string(APPEND headers "#include \"${h}\"\n")
  endforeach()

  if(CMAKE_REQUIRED_LINK_DIRECTORIES)
    set(_CTS_LINK_DIRECTORIES
      "-DLINK_DIRECTORIES:STRING=${CMAKE_REQUIRED_LINK_DIRECTORIES}")
  else()
    set(_CTS_LINK_DIRECTORIES)
  endif()

  # Perform the check.
  set(bin ${CMAKE_BINARY_DIR}${CMAKE_FILES_DIRECTORY}/CheckTypeSize/${var}.bin)
  file(READ ${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CheckTypeSize.c.in src_content)
  string(CONFIGURE "${src_content}" src_content @ONLY)
  try_compile(${result_var} SOURCE_FROM_VAR "${src}" src_content
    COMPILE_DEFINITIONS ${CMAKE_REQUIRED_DEFINITIONS}
    LINK_OPTIONS ${CMAKE_REQUIRED_LINK_OPTIONS}
    LINK_LIBRARIES ${CMAKE_REQUIRED_LIBRARIES}
    CMAKE_FLAGS
      "-DCOMPILE_DEFINITIONS:STRING=${CMAKE_REQUIRED_FLAGS}"
      "-DINCLUDE_DIRECTORIES:STRING=${CMAKE_REQUIRED_INCLUDES}"
      "${_CTS_LINK_DIRECTORIES}"
    COPY_FILE ${bin}
    )
  unset(_CTS_LINK_DIRECTORIES)

  if(${result_var})
    # The check compiled.  Load information from the binary.
    file(STRINGS ${bin} strings LIMIT_COUNT 10 REGEX "INFO:size")

    # Parse the information strings.
    set(regex_size ".*INFO:size\\[0*([^]]*)\\].*")
    set(regex_key " key\\[([^]]*)\\]")
    set(keys)
    set(code)
    set(mismatch)
    set(first 1)
    foreach(info ${strings})
      if("${info}" MATCHES "${regex_size}")
        # Get the type size.
        set(size "${CMAKE_MATCH_1}")
        if(first)
          set(${var} ${size})
        elseif(NOT "${size}" STREQUAL "${${var}}")
          set(mismatch 1)
        endif()
        set(first 0)

        # Get the architecture map key.
        string(REGEX MATCH   "${regex_key}"       key "${info}")
        string(REGEX REPLACE "${regex_key}" "\\1" key "${key}")
        if(key)
          string(APPEND code "\nset(${var}-${key} \"${size}\")")
          list(APPEND keys ${key})
        endif()
      endif()
    endforeach()

    # Update the architecture-to-size map.
    if(mismatch AND keys)
      configure_file(${CMAKE_CURRENT_FUNCTION_LIST_DIR}/CheckTypeSizeMap.cmake.in ${map} @ONLY)
      set(${var} 0)
    else()
      file(REMOVE ${map})
    endif()

    if(mismatch AND NOT keys)
      message(SEND_ERROR "CHECK_TYPE_SIZE found different results, consider setting CMAKE_OSX_ARCHITECTURES or CMAKE_TRY_COMPILE_OSX_ARCHITECTURES to one or no architecture !")
    endif()

    if(NOT CMAKE_REQUIRED_QUIET)
      message(CHECK_PASS "done")
    endif()
    set(${var} "${${var}}" CACHE INTERNAL "CHECK_TYPE_SIZE: sizeof(${type})")
  else()
    # The check failed to compile.
    if(NOT CMAKE_REQUIRED_QUIET)
      message(CHECK_FAIL "failed")
    endif()
    set(${var} "" CACHE INTERNAL "CHECK_TYPE_SIZE: ${type} unknown")
    file(REMOVE ${map})
  endif()
endfunction()

#-----------------------------------------------------------------------------
function(CHECK_TYPE_SIZE TYPE VARIABLE)
  cmake_parse_arguments(
    PARSE_ARGV
    2
    _CHECK_TYPE_SIZE
    "BUILTIN_TYPES_ONLY" # Options
    "RESULT_VARIABLE;LANGUAGE" # One-value arguments
    "" # Multi-value arguments
  )

  if(_CHECK_TYPE_SIZE_UNPARSED_ARGUMENTS)
    message(
      FATAL_ERROR
      "Unknown arguments:\n  ${_CHECK_TYPE_SIZE_UNPARSED_ARGUMENTS}\n"
    )
  endif()

  if(NOT DEFINED _CHECK_TYPE_SIZE_RESULT_VARIABLE)
    set(_CHECK_TYPE_SIZE_RESULT_VARIABLE HAVE_${VARIABLE})
  elseif(_CHECK_TYPE_SIZE_RESULT_VARIABLE STREQUAL "")
    message(
      FATAL_ERROR
      "Missing argument:\n  RESULT_VARIABLE argument requires a value\n"
    )
  endif()

  if(NOT DEFINED _CHECK_TYPE_SIZE_LANGUAGE)
    set(_CHECK_TYPE_SIZE_LANGUAGE C)
  elseif(_CHECK_TYPE_SIZE_LANGUAGE STREQUAL "")
    message(
      FATAL_ERROR
      "Missing argument:\n  LANGUAGE argument requires a value\n"
    )
  elseif(NOT _CHECK_TYPE_SIZE_LANGUAGE MATCHES "^(C|CXX)$")
    message(
      FATAL_ERROR
      "Unknown language:\n  ${_CHECK_TYPE_SIZE_LANGUAGE}.\n"
      "Supported languages: C, CXX.\n")
  endif()

  # Optionally check for standard headers.
  if(NOT _CHECK_TYPE_SIZE_BUILTIN_TYPES_ONLY)
    if(_CHECK_TYPE_SIZE_LANGUAGE STREQUAL "C")
      check_include_file(sys/types.h HAVE_SYS_TYPES_H)
      check_include_file(stdint.h HAVE_STDINT_H)
      check_include_file(stddef.h HAVE_STDDEF_H)
    elseif(_CHECK_TYPE_SIZE_LANGUAGE STREQUAL "CXX")
      check_include_file_cxx(sys/types.h HAVE_SYS_TYPES_H)
      if("${TYPE}" MATCHES "^std::")
        check_include_file_cxx(cstdint HAVE_CSTDINT)
        check_include_file_cxx(cstddef HAVE_CSTDDEF)
      else()
        check_include_file_cxx(stdint.h HAVE_STDINT_H)
        check_include_file_cxx(stddef.h HAVE_STDDEF_H)
      endif()
    endif()
  endif()

  # Compute or load the size or size map.
  set(${VARIABLE}_KEYS)
  set(_map_file ${CMAKE_BINARY_DIR}/${CMAKE_FILES_DIRECTORY}/CheckTypeSize/${VARIABLE}.cmake)
  if(NOT DEFINED ${_CHECK_TYPE_SIZE_RESULT_VARIABLE})
    __check_type_size_impl(
      ${TYPE}
      ${VARIABLE}
      ${_CHECK_TYPE_SIZE_RESULT_VARIABLE}
      ${_map_file}
      ${_CHECK_TYPE_SIZE_BUILTIN_TYPES_ONLY}
      ${_CHECK_TYPE_SIZE_LANGUAGE}
    )
  endif()
  include(${_map_file} OPTIONAL)

  set(_propagated_vars "")

  # Create preprocessor code.
  if(${VARIABLE}_KEYS)
    set(${VARIABLE}_CODE)
    set(_if if)
    foreach(key ${${VARIABLE}_KEYS})
      string(APPEND ${VARIABLE}_CODE "#${_if} defined(${key})\n# define ${VARIABLE} ${${VARIABLE}-${key}}\n")
      set(_if elif)
      list(APPEND _propagated_vars ${VARIABLE}-${key})
    endforeach()
    string(APPEND ${VARIABLE}_CODE "#else\n# error ${VARIABLE} unknown\n#endif")
  elseif(${VARIABLE})
    set(${VARIABLE}_CODE "#define ${VARIABLE} ${${VARIABLE}}")
  else()
    set(${VARIABLE}_CODE "/* #undef ${VARIABLE} */")
  endif()

  return(PROPAGATE ${VARIABLE}_CODE ${VARIABLE}_KEYS ${_propagated_vars})
endfunction()

#-----------------------------------------------------------------------------
endblock()
