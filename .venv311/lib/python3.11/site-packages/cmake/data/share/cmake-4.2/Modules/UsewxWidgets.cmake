# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
UsewxWidgets
------------

This module serves as a convenience wrapper for using the wxWidgets
library (formerly known as wxWindows) and propagates its usage requirements,
such as library directories, include directories, and compiler flags, into
the current directory scope for use by targets.

Load this module in a CMake project with:

.. code-block:: cmake

  include(UsewxWidgets)

This module calls :command:`include_directories` and
:command:`link_directories`, sets compile definitions for the current
directory and appends some compile flags to use wxWidgets library after
calling the :module:`find_package(wxWidgets) <FindwxWidgets>`.

Examples
^^^^^^^^

Include this module in a project after finding wxWidgets to configure its
usage requirements:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  # Note that for MinGW users the order of libraries is important.
  find_package(wxWidgets COMPONENTS net gl core base)

  add_library(example example.cxx)

  if(wxWidgets_FOUND)
    include(UsewxWidgets)

    # Link wxWidgets libraries for each dependent executable/library target.
    target_link_libraries(example PRIVATE ${wxWidgets_LIBRARIES})
  endif()

As of CMake 3.27, a better approach is to link only the
:module:`wxWidgets::wxWidgets <FindwxWidgets>` imported target to specific
targets that require it, rather than including this module.  Imported
targets provide better control of the package usage properties, such as
include directories and compile flags, by applying them only to the targets
they are linked to, avoiding unnecessary propagation to all targets in the
current directory.

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  find_package(wxWidgets COMPONENTS net gl core base)

  add_library(example example.cxx)

  # Link the imported target for each dependent executable/library target.
  target_link_libraries(example PRIVATE wxWidgets::wxWidgets)

See Also
^^^^^^^^

* The :module:`FindwxWidgets` module to find wxWidgets.
#]=======================================================================]

# Author: Jan Woetzel <jw -at- mip.informatik.uni-kiel.de>

if   (wxWidgets_FOUND)
  if   (wxWidgets_INCLUDE_DIRS)
    if(wxWidgets_INCLUDE_DIRS_NO_SYSTEM)
      include_directories(${wxWidgets_INCLUDE_DIRS})
    else()
      include_directories(SYSTEM ${wxWidgets_INCLUDE_DIRS})
    endif()
  endif()

  if   (wxWidgets_LIBRARY_DIRS)
    link_directories(${wxWidgets_LIBRARY_DIRS})
  endif()

  if   (wxWidgets_DEFINITIONS)
    set_property(DIRECTORY APPEND
      PROPERTY COMPILE_DEFINITIONS ${wxWidgets_DEFINITIONS})
  endif()

  if   (wxWidgets_DEFINITIONS_DEBUG)
    set_property(DIRECTORY APPEND
      PROPERTY COMPILE_DEFINITIONS_DEBUG ${wxWidgets_DEFINITIONS_DEBUG})
  endif()

  if   (wxWidgets_CXX_FLAGS)
    # Flags are expected to be a string here, not a list.
    string(REPLACE ";" " " wxWidgets_CXX_FLAGS_str "${wxWidgets_CXX_FLAGS}")
    string(APPEND CMAKE_CXX_FLAGS " ${wxWidgets_CXX_FLAGS_str}")
    unset(wxWidgets_CXX_FLAGS_str)
  endif()
else ()
  message("wxWidgets requested but not found.")
endif()
