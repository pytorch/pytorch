# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
Use_wxWindows
-------------

.. deprecated:: 2.8.10

  This module should no longer be used.  Use :module:`find_package(wxWidgets)
  <FindwxWidgets>` instead.

This module serves as a convenience wrapper for finding the wxWidgets library
(formerly known as wxWindows) and propagates its usage requirements, such as
libraries, include directories, and compiler flags, into the current directory
scope for use by targets.

Load this module in a CMake project with:

.. code-block:: cmake

  include(Use_wxWindows)

Examples
^^^^^^^^

In earlier versions of CMake, wxWidgets (wxWindows) could be found and used in
the current directory like this:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  include(Use_wxWindows)

To request OpenGL support, the ``WXWINDOWS_USE_GL`` variable could be set before
including this module:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  set(WXWINDOWS_USE_GL ON)
  include(Use_wxWindows)

  add_library(example example.cxx)

Starting with CMake 3.0, wxWidgets can be found using the
:module:`FindwxWidgets` module, which provides the wxWidgets usage requirements
either using result variables or imported target as of CMake 3.27:

.. code-block:: cmake
  :caption: ``CMakeLists.txt``

  find_package(wxWidgets)

  add_library(example example.cxx)

  target_link_libraries(example PRIVATE wxWidgets::wxWidgets)
#]=======================================================================]

# Author: Jan Woetzel <jw -at- mip.informatik.uni-kiel.de> (07/2003)

# -----------------------------------------------------
# 16.Feb.2004: changed INCLUDE to FIND_PACKAGE to read from users own non-system CMAKE_MODULE_PATH (Jan Woetzel JW)
# 07/2006: rewrite as FindwxWidgets.cmake, kept for backward compatibility JW

message(
  DEPRECATION
  "Use_wxWindows module is DEPRECATED.\n"
  "Please use find_package(wxWidgets) instead. (JW)"
)

# ------------------------

find_package( wxWindows )

if(WXWINDOWS_FOUND)

#message("DBG Use_wxWindows.cmake:  WXWINDOWS_INCLUDE_DIR=${WXWINDOWS_INCLUDE_DIR} WXWINDOWS_LINK_DIRECTORIES=${WXWINDOWS_LINK_DIRECTORIES}     WXWINDOWS_LIBRARIES=${WXWINDOWS_LIBRARIES}  CMAKE_WXWINDOWS_CXX_FLAGS=${CMAKE_WXWINDOWS_CXX_FLAGS} WXWINDOWS_DEFINITIONS=${WXWINDOWS_DEFINITIONS}")

 if(WXWINDOWS_INCLUDE_DIR)
    include_directories(${WXWINDOWS_INCLUDE_DIR})
  endif()
 if(WXWINDOWS_LINK_DIRECTORIES)
    link_directories(${WXWINDOWS_LINK_DIRECTORIES})
  endif()
  if(WXWINDOWS_LIBRARIES)
    link_libraries(${WXWINDOWS_LIBRARIES})
  endif()
  if (CMAKE_WXWINDOWS_CXX_FLAGS)
    string(APPEND CMAKE_CXX_FLAGS " ${CMAKE_WXWINDOWS_CXX_FLAGS}")
  endif()
  if(WXWINDOWS_DEFINITIONS)
    add_definitions(${WXWINDOWS_DEFINITIONS})
  endif()
else()
  message(SEND_ERROR "wxWindows not found by Use_wxWindows.cmake")
endif()
