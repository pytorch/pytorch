# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CheckStructHasMember
--------------------

This module provides a command to check whether a struct or class has a
specified member variable.

Load this module in a CMake project with:

.. code-block:: cmake

  include(CheckStructHasMember)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_struct_has_member

  Checks once if the given struct or class has the specified member variable:

  .. code-block:: cmake

    check_struct_has_member(
      <struct>
      <member>
      <headers>
      <variable>
      [LANGUAGE <language>]
    )

  This command checks once whether the struct or class ``<struct>`` contains
  the specified ``<member>`` after including the given header(s) ``<headers>``
  where the prototype should be declared.  Multiple header files can be
  specified in one argument as a string using a :ref:`semicolon-separated list
  <CMake Language Lists>`.  The result is stored in an internal cache variable
  ``<variable>``.

  The options are:

  ``LANGUAGE <language>``
    Use the ``<language>`` compiler to perform the check.
    Acceptable values are ``C`` and ``CXX``.
    If not specified, it defaults to ``C``.

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

Examples
^^^^^^^^

Example: Checking C Struct Member
"""""""""""""""""""""""""""""""""

In the following example, this module checks if the C struct ``timeval`` has
a member variable ``tv_sec`` after including the ``<sys/select.h>`` header.
The result of the check is stored in the internal cache variable
``HAVE_TIMEVAL_TV_SEC``.

.. code-block:: cmake

  include(CheckStructHasMember)

  check_struct_has_member(
    "struct timeval"
    tv_sec
    sys/select.h
    HAVE_TIMEVAL_TV_SEC
  )

Example: Checking C++ Struct Member
"""""""""""""""""""""""""""""""""""

In the following example, this module checks if the C++ struct ``std::tm``
has a member variable ``tm_gmtoff`` after including the ``<ctime>`` header.
The result of the check is stored in the internal cache variable
``HAVE_TM_GMTOFF``.

.. code-block:: cmake

  include(CheckStructHasMember)

  check_struct_has_member(
    std::tm
    tm_gmtoff
    ctime
    HAVE_TM_GMTOFF
    LANGUAGE CXX
  )

Example: Isolated Check With Compile Definitions
""""""""""""""""""""""""""""""""""""""""""""""""

In the following example, the check is performed with temporarily modified
compile definitions using the :module:`CMakePushCheckState` module:

.. code-block:: cmake

  include(CheckStructHasMember)
  include(CMakePushCheckState)

  cmake_push_check_state(RESET)
    set(CMAKE_REQUIRED_DEFINITIONS -D_GNU_SOURCE)

    check_struct_has_member(
      "struct utsname"
      domainname
      sys/utsname.h
      HAVE_UTSNAME_DOMAINNAME
    )
  cmake_pop_check_state()
#]=======================================================================]

include_guard(GLOBAL)
include(CheckSourceCompiles)

macro (CHECK_STRUCT_HAS_MEMBER _STRUCT _MEMBER _HEADER _RESULT)
  set(_INCLUDE_FILES)
  foreach (it ${_HEADER})
    string(APPEND _INCLUDE_FILES "#include <${it}>\n")
  endforeach ()

  if("x${ARGN}" STREQUAL "x")
    set(_lang C)
  elseif("x${ARGN}" MATCHES "^xLANGUAGE;([a-zA-Z]+)$")
    set(_lang "${CMAKE_MATCH_1}")
  else()
    message(FATAL_ERROR "Unknown arguments:\n  ${ARGN}\n")
  endif()

  set(_CHECK_STRUCT_MEMBER_SOURCE_CODE "
${_INCLUDE_FILES}
int main(void)
{
  (void)sizeof(((${_STRUCT} *)0)->${_MEMBER});
  return 0;
}
")

  if("${_lang}" STREQUAL "C")
    check_source_compiles(C "${_CHECK_STRUCT_MEMBER_SOURCE_CODE}" ${_RESULT})
  elseif("${_lang}" STREQUAL "CXX")
    check_source_compiles(CXX "${_CHECK_STRUCT_MEMBER_SOURCE_CODE}" ${_RESULT})
  else()
    message(FATAL_ERROR "Unknown language:\n  ${_lang}\nSupported languages: C, CXX.\n")
  endif()
endmacro ()

# FIXME(#24994): The following modules are included only for compatibility
# with projects that accidentally relied on them with CMake 3.26 and below.
include(CheckCSourceCompiles)
include(CheckCXXSourceCompiles)
