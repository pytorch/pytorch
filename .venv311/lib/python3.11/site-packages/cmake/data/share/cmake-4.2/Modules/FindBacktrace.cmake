# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindBacktrace
-------------

Finds `backtrace(3) <https://man7.org/linux/man-pages/man3/backtrace.3.html>`_,
a library that provides functions for application self-debugging:

.. code-block:: cmake

  find_package(Backtrace [...])

This module checks whether ``backtrace(3)`` is supported, either through the
standard C library (``libc``), or a separate library.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``Backtrace::Backtrace``
  .. versionadded:: 3.30

  An interface library encapsulating the usage requirements of Backtrace.  This
  target is available only when Backtrace is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Backtrace_FOUND``
  Boolean indicating whether the ``backtrace(3)`` support is available.

``Backtrace_INCLUDE_DIRS``
  The include directories needed to use ``backtrace(3)`` header.

``Backtrace_LIBRARIES``
  The libraries (linker flags) needed to use ``backtrace(3)``, if any.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables are also available to set or use:

``Backtrace_HEADER``
  The header file needed for ``backtrace(3)``.  This variable allows dynamic
  usage of the header in the project code.  It can also be overridden by the
  user.

``Backtrace_INCLUDE_DIR``
  The directory holding the ``backtrace(3)`` header.

``Backtrace_LIBRARY``
  The external library providing backtrace, if any.

Examples
^^^^^^^^

Finding Backtrace and linking it to a project target as of CMake 3.30:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  find_package(Backtrace)
  target_link_libraries(app PRIVATE Backtrace::Backtrace)

The ``Backtrace_HEADER`` variable can be used, for example, in a configuration
header file created by :command:`configure_file`:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  add_library(app app.c)

  find_package(Backtrace)
  target_link_libraries(app PRIVATE Backtrace::Backtrace)

  configure_file(config.h.in config.h)

.. code-block:: c
  :caption: ``config.h.in``

  #cmakedefine01 Backtrace_FOUND
  #if Backtrace_FOUND
  #  include <@Backtrace_HEADER@>
  #endif

.. code-block:: c
  :caption: ``app.c``

  #include "config.h"

If the project needs to support CMake 3.29 or earlier, the imported target can
be defined manually:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  find_package(Backtrace)
  if(Backtrace_FOUND AND NOT TARGET Backtrace::Backtrace)
    add_library(Backtrace::Backtrace INTERFACE IMPORTED)
    set_target_properties(
      Backtrace::Backtrace
      PROPERTIES
        INTERFACE_LINK_LIBRARIES "${Backtrace_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${Backtrace_INCLUDE_DIRS}"
    )
  endif()
  target_link_libraries(app PRIVATE Backtrace::Backtrace)
#]=======================================================================]

include(CMakePushCheckState)
include(CheckSymbolExists)
include(FindPackageHandleStandardArgs)

# List of variables to be provided to find_package_handle_standard_args()
set(_Backtrace_STD_ARGS Backtrace_INCLUDE_DIR)

if(Backtrace_HEADER)
  set(_Backtrace_HEADER_TRY "${Backtrace_HEADER}")
else()
  set(_Backtrace_HEADER_TRY "execinfo.h")
endif()

find_path(Backtrace_INCLUDE_DIR "${_Backtrace_HEADER_TRY}")
set(Backtrace_INCLUDE_DIRS ${Backtrace_INCLUDE_DIR})

if (NOT DEFINED Backtrace_LIBRARY)
  # First, check if we already have backtrace(), e.g., in libc
  cmake_push_check_state(RESET)
  set(CMAKE_REQUIRED_INCLUDES ${Backtrace_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_QUIET ${Backtrace_FIND_QUIETLY})
  check_symbol_exists("backtrace" "${_Backtrace_HEADER_TRY}" _Backtrace_SYM_FOUND)
  cmake_pop_check_state()
endif()

if(_Backtrace_SYM_FOUND)
  # Avoid repeating the message() call below each time CMake is run.
  if(NOT Backtrace_FIND_QUIETLY AND NOT DEFINED Backtrace_LIBRARY)
    message(STATUS "backtrace facility detected in default set of libraries")
  endif()
  set(Backtrace_LIBRARY "" CACHE FILEPATH "Library providing backtrace(3), empty for default set of libraries")
else()
  # Check for external library, for non-glibc systems
  if(Backtrace_INCLUDE_DIR)
    # OpenBSD has libbacktrace renamed to libexecinfo
    find_library(Backtrace_LIBRARY "execinfo")
  else()     # respect user wishes
    set(_Backtrace_HEADER_TRY "backtrace.h")
    find_path(Backtrace_INCLUDE_DIR ${_Backtrace_HEADER_TRY})
    find_library(Backtrace_LIBRARY "backtrace")
  endif()

  # Prepend list with library path as it's more common practice
  set(_Backtrace_STD_ARGS Backtrace_LIBRARY ${_Backtrace_STD_ARGS})
endif()

set(Backtrace_LIBRARIES ${Backtrace_LIBRARY})
set(Backtrace_HEADER "${_Backtrace_HEADER_TRY}" CACHE STRING "Header providing backtrace(3) facility")

find_package_handle_standard_args(Backtrace REQUIRED_VARS ${_Backtrace_STD_ARGS})
mark_as_advanced(Backtrace_HEADER Backtrace_INCLUDE_DIR Backtrace_LIBRARY)

if(Backtrace_FOUND AND NOT TARGET Backtrace::Backtrace)
  if(Backtrace_LIBRARY)
    add_library(Backtrace::Backtrace UNKNOWN IMPORTED)
    set_property(TARGET Backtrace::Backtrace PROPERTY IMPORTED_LOCATION "${Backtrace_LIBRARY}")
  else()
    add_library(Backtrace::Backtrace INTERFACE IMPORTED)
    target_link_libraries(Backtrace::Backtrace INTERFACE ${Backtrace_LIBRARIES})
  endif()
  target_include_directories(Backtrace::Backtrace INTERFACE ${Backtrace_INCLUDE_DIRS})
endif()
