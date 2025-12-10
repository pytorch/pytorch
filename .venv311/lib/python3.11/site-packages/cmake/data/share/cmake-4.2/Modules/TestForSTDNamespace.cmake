# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
TestForSTDNamespace
-------------------

This module checks whether the ``CXX`` compiler supports the ``std`` namespace
for the C++ Standard Library.

Load this module in a CMake project with:

.. code-block:: cmake

  include(TestForSTDNamespace)

Early versions of C++ (pre-C++98) did not have a requirement for a dedicated
namespace of C++ Standard Template Library (STL) components (e.g. ``list``,
etc.) and other parts of the C++ Standard Library (such as I/O streams
``cout``, ``endl``, etc), so they were available globally.

This module defines the following cache variable:

``CMAKE_NO_STD_NAMESPACE``
  A cache variable containing the result of the check.  It will be set to value
  ``0`` if the ``std`` namespace is supported (``C++ 98`` and newer), and to
  value ``1`` if not (``ANSI C++``).

.. note::

  The ``std`` namespace got formally introduced in ``C++ 98`` standard, making
  this issue obsolete.

Examples
^^^^^^^^

Including this module will check for the ``std`` namespace support and define
the ``CMAKE_NO_STD_NAMESPACE`` cache variable:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(TestForSTDNamespace)
  file(
    CONFIGURE
    OUTPUT config.h
    CONTENT "#cmakedefine CMAKE_NO_STD_NAMESPACE"
  )

which can be then used in a C++ program to define the missing namespace:

.. code-block:: c++
  :caption: ``example.cxx``

  #include "config.h"

  #ifdef CMAKE_NO_STD_NAMESPACE
  #  define std
  #endif

See Also
^^^^^^^^

* The :module:`CMakeBackwardCompatibilityCXX` module.
#]=======================================================================]

if(NOT DEFINED CMAKE_STD_NAMESPACE)
  message(CHECK_START "Check for STD namespace")
  try_compile(CMAKE_STD_NAMESPACE
    SOURCES ${CMAKE_ROOT}/Modules/TestForSTDNamespace.cxx
    )
  if (CMAKE_STD_NAMESPACE)
    message(CHECK_PASS "found")
    set (CMAKE_NO_STD_NAMESPACE 0 CACHE INTERNAL
         "Does the compiler support std::.")
  else ()
    message(CHECK_FAIL "not found")
    set (CMAKE_NO_STD_NAMESPACE 1 CACHE INTERNAL
       "Does the compiler support std::.")
  endif ()
endif()
