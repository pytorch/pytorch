# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
TestCXXAcceptsFlag
------------------

.. deprecated:: 3.0

  This module should no longer be used.  It has been superseded by the
  :module:`CheckCXXCompilerFlag` module.  As of CMake 3.19, the
  :module:`CheckCompilerFlag` module is also available for checking flags across
  multiple languages.

This module provides a command to test whether the C++ (CXX) compiler supports
specific flags.

Load this module in a CMake project with:

.. code-block:: cmake

  include(TestCXXAcceptsFlag)

Commands
^^^^^^^^

This module provides the following command:

.. command:: check_cxx_accepts_flag

  Checks whether the CXX compiler accepts the specified flags:

  .. code-block:: cmake

    check_cxx_accepts_flag(<flags> <result-variable>)

  ``<flags>``
    One or more compiler flags to test.  For multiple flags, provide them as a
    space-separated string.

  ``<result-variable>``
    Name of an internal cache variable that stores the result.  It is set to
    boolean true if the compiler accepts the flags and false otherwise.

Examples
^^^^^^^^

Checking if the C++ compiler supports specific flags:

.. code-block:: cmake

  include(TestCXXAcceptsFlag)
  check_cxx_accepts_flag("-fno-common -fstack-clash-protection" HAVE_FLAGS)

Migrating to the :module:`CheckCompilerFlag` module:

.. code-block:: cmake

  include(CheckCompilerFlag)
  check_compiler_flag(CXX "-fno-common;-fstack-clash-protection" HAVE_FLAGS)
#]=======================================================================]

macro(CHECK_CXX_ACCEPTS_FLAG FLAGS  VARIABLE)
  if(NOT DEFINED ${VARIABLE})
    message(CHECK_START "Checking to see if CXX compiler accepts flag ${FLAGS}")
    try_compile(${VARIABLE}
      SOURCES ${CMAKE_ROOT}/Modules/DummyCXXFile.cxx
      CMAKE_FLAGS -DCOMPILE_DEFINITIONS:STRING=${FLAGS}
      )
    if(${VARIABLE})
      message(CHECK_PASS "yes")
    else()
      message(CHECK_FAIL "no")
    endif()
  endif()
endmacro()
