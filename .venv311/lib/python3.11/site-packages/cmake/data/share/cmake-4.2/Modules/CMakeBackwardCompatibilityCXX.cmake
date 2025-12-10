# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CMakeBackwardCompatibilityCXX
-----------------------------

This module defines several backward compatibility cache variables for the
``CXX`` language to support early C++ (pre-C++98, ANSI C++).

Load this module in a CMake project with:

.. code-block:: cmake

  include(CMakeBackwardCompatibilityCXX)

The following modules are included by this module:

* :module:`TestForANSIForScope`
* :module:`TestForANSIStreamHeaders`
* :module:`TestForSSTREAM`
* :module:`TestForSTDNamespace`

Additionally, the following cache variable may be defined:

``CMAKE_ANSI_CXXFLAGS``
  A space-separated string of compiler options for enabling ANSI C++ mode, if
  available.

.. note::

  This module is intended for C++ code written before ``C++ 98``.  As of the
  ``C++ 98`` standard, these issues have been formally addressed, making such
  checks obsolete.

Examples
^^^^^^^^

Including this module provides backward compatibility cache variables, which
can be used in C++.  For example:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(CMakeBackwardCompatibilityCXX)
  file(
    CONFIGURE
    OUTPUT config.h
    CONTENT [[
      #cmakedefine CMAKE_NO_ANSI_FOR_SCOPE
      #cmakedefine CMAKE_NO_ANSI_STRING_STREAM
      #cmakedefine CMAKE_NO_ANSI_STREAM_HEADERS
      #cmakedefine CMAKE_NO_STD_NAMESPACE
    ]]
  )
#]=======================================================================]

if(NOT CMAKE_SKIP_COMPATIBILITY_TESTS)
  # check for some ANSI flags in the CXX compiler if it is not gnu
  if(NOT CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    include(TestCXXAcceptsFlag)
    set(CMAKE_TRY_ANSI_CXX_FLAGS "")
    if(CMAKE_SYSTEM_NAME MATCHES "OSF")
      set(CMAKE_TRY_ANSI_CXX_FLAGS "-std strict_ansi -nopure_cname")
    endif()
    # if CMAKE_TRY_ANSI_CXX_FLAGS has something in it, see
    # if the compiler accepts it
    if(NOT CMAKE_TRY_ANSI_CXX_FLAGS STREQUAL "")
      check_cxx_accepts_flag(${CMAKE_TRY_ANSI_CXX_FLAGS} CMAKE_CXX_ACCEPTS_FLAGS)
      # if the compiler liked the flag then set CMAKE_ANSI_CXXFLAGS
      # to the flag
      if(CMAKE_CXX_ACCEPTS_FLAGS)
        set(CMAKE_ANSI_CXXFLAGS ${CMAKE_TRY_ANSI_CXX_FLAGS} CACHE INTERNAL
        "What flags are required by the c++ compiler to make it ansi." )
      endif()
    endif()
  endif()
  set(CMAKE_CXX_FLAGS_SAVE ${CMAKE_CXX_FLAGS})
  string(APPEND CMAKE_CXX_FLAGS " ${CMAKE_ANSI_CXXFLAGS}")
  include(TestForANSIStreamHeaders)
  include(CheckIncludeFileCXX)
  include(TestForSTDNamespace)
  include(TestForANSIForScope)
  include(TestForSSTREAM)
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS_SAVE}")
endif()
