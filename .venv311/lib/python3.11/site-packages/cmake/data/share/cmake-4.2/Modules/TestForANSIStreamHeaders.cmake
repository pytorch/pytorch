# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
TestForANSIStreamHeaders
------------------------

This module checks whether the ``CXX`` compiler supports standard library
headers without the ``.h`` extension (e.g. ``<iostream>``).

Load this module in a CMake project with:

.. code-block:: cmake

  include(TestForANSIStreamHeaders)

Early versions of C++ (pre-C++98) didn't support including standard headers
without extensions.

This module defines the following cache variable:

``CMAKE_NO_ANSI_STREAM_HEADERS``
  A cache variable containing the result of the check.  It will be set to value
  ``0`` if the standard headers can be included without the ``.h`` extension
  (``C++ 98`` and newer), and to value ``1`` if ``.h`` is required
  (``ANSI C++``).

.. note::

  The C++ standard headers without extensions got formally introduced in the
  ``C++ 98`` standard, making this issue obsolete.

Examples
^^^^^^^^

Including this module will check how the C++ standard headers can be included
and define the ``CMAKE_NO_ANSI_STREAM_HEADERS`` cache variable:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(TestForANSIStreamHeaders)
  file(
    CONFIGURE
    OUTPUT config.h
    CONTENT "#cmakedefine CMAKE_NO_ANSI_STREAM_HEADERS"
  )

C++ program can then include the available header conditionally:

.. code-block:: c++
  :caption: ``example.cxx``

  #include "config.h"

  #ifdef CMAKE_NO_ANSI_STREAM_HEADERS
  #  include <iostream.h>
  #else
  #  include <iostream>
  #endif

  int main() { ... }

See Also
^^^^^^^^

* The :module:`CMakeBackwardCompatibilityCXX` module.
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/CheckIncludeFileCXX.cmake)

if(NOT CMAKE_NO_ANSI_STREAM_HEADERS)
  check_include_file_cxx(iostream CMAKE_ANSI_STREAM_HEADERS)
  if (CMAKE_ANSI_STREAM_HEADERS)
    set (CMAKE_NO_ANSI_STREAM_HEADERS 0 CACHE INTERNAL
         "Does the compiler support headers like iostream.")
  else ()
    set (CMAKE_NO_ANSI_STREAM_HEADERS 1 CACHE INTERNAL
       "Does the compiler support headers like iostream.")
  endif ()

  mark_as_advanced(CMAKE_NO_ANSI_STREAM_HEADERS)
endif()
