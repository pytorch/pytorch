# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
TestForSSTREAM
--------------

This module checks whether the C++ standard header ``<sstream>`` exists and
functions correctly.

Load this module in a CMake project with:

.. code-block:: cmake

  include(TestForSSTREAM)

In early versions of C++ (pre-C++98), the ``<sstream>`` header was not
formally standardized and may not have been available.

This module defines the following cache variables:

``CMAKE_NO_ANSI_STRING_STREAM``
  A cache variable indicating whether the ``<sstream>`` header is available. It
  will be set to value ``0`` if ``<sstream>`` is available (``C++ 98`` and
  newer), and to value ``1`` if ``<sstream>`` is missing (``ANSI C++``).

``CMAKE_HAS_ANSI_STRING_STREAM``
  A cache variable that is the opposite of ``CMAKE_NO_ANSI_STRING_STREAM``
  (true if ``<sstream>`` is available and false if ``<sstream>`` is missing).

.. note::

  The ``<sstream>`` header was formally introduced in the ``C++ 98`` standard,
  making this check obsolete for modern compilers.

Examples
^^^^^^^^

Including this module will check for ``<sstream>`` support and define the
``CMAKE_NO_ANSI_STRING_STREAM`` cache variable:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(TestForSSTREAM)
  file(
    CONFIGURE
    OUTPUT config.h
    CONTENT "#cmakedefine CMAKE_NO_ANSI_STRING_STREAM"
  )

Then it can be used in a C++ program:

.. code-block:: c++
  :caption: ``example.cxx``

  #include "config.h"

  #ifndef CMAKE_NO_ANSI_STRING_STREAM
  #  include <sstream>
  #endif

  int main() { ... }

See Also
^^^^^^^^

* The :module:`CMakeBackwardCompatibilityCXX` module.
#]=======================================================================]

if(NOT DEFINED CMAKE_HAS_ANSI_STRING_STREAM)
  message(CHECK_START "Check for sstream")
  try_compile(CMAKE_HAS_ANSI_STRING_STREAM
    SOURCES ${CMAKE_ROOT}/Modules/TestForSSTREAM.cxx
    )
  if (CMAKE_HAS_ANSI_STRING_STREAM)
    message(CHECK_PASS "found")
    set (CMAKE_NO_ANSI_STRING_STREAM 0 CACHE INTERNAL
         "Does the compiler support sstream")
  else ()
    message(CHECK_FAIL "not found")
    set (CMAKE_NO_ANSI_STRING_STREAM 1 CACHE INTERNAL
       "Does the compiler support sstream")
  endif ()
endif()
