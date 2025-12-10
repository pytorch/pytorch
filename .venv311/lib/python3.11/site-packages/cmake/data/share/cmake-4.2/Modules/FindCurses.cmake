# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindCurses
----------

Finds the curses or ncurses library:

.. code-block:: cmake

  find_package(Curses [...])

Curses is a terminal control library for Unix-like systems, used to build text
user interface (TUI) applications.  Originally developed in 1978, it has since
evolved into multiple implementations, most notably ncurses (new curses), BSD
curses, and PDCurses (a public domain curses library for non-Unix platforms).

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Curses_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether Curses was found.

``CURSES_INCLUDE_DIRS``
  .. versionadded:: 3.1

  The include directories needed to use Curses.

``CURSES_LIBRARIES``
  The libraries needed to use Curses.

``CURSES_CFLAGS``
  .. versionadded:: 3.16

  Compiler flags which ought be given to C/C++ compilers when using Curses.

``CURSES_HAVE_CURSES_H``
  Boolean indicating whether ``curses.h`` is available.

``CURSES_HAVE_NCURSES_H``
  Boolean indicating whether ``ncurses.h`` is available.

``CURSES_HAVE_NCURSES_NCURSES_H``
  Boolean indicating whether ``ncurses/ncurses.h`` is available.

``CURSES_HAVE_NCURSES_CURSES_H``
  Boolean indicating whether ``ncurses/curses.h`` is available.

Hints
^^^^^

This module accepts the following variables:

``CURSES_NEED_NCURSES``
  Set this variable to ``TRUE`` before calling ``find_package(Curses)`` if the
  the ncurses implementation functionality is specifically required.

``CURSES_NEED_WIDE``
  .. versionadded:: 3.10

  Set this variable to ``TRUE`` before calling ``find_package(Curses)`` if
  Unicode (wide character) support is required.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``CURSES_FOUND``
  .. deprecated:: 4.2
    Use ``Curses_FOUND``, which has the same value.

  Boolean indicating whether Curses was found.

``CURSES_INCLUDE_DIR``
  .. deprecated:: 3.1
    Use the ``CURSES_INCLUDE_DIRS`` variable instead.

  Path to a Curses include directory.

``CURSES_LIBRARY``
  .. deprecated:: 2.4
    Use the ``CURSES_LIBRARIES`` variable instead.

  Path to Curses library.

Examples
^^^^^^^^

Finding Curses and creating an imported interface target for linking it to a
project target:

.. code-block:: cmake

  find_package(Curses)
  if(Curses_FOUND AND NOT TARGET Curses::Curses)
    add_library(Curses::Curses INTERFACE IMPORTED)
    set_target_properties(
      Curses::Curses
      PROPERTIES
        INTERFACE_LINK_LIBRARIES "${CURSES_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${CURSES_INCLUDE_DIRS}"
    )
  endif()

  add_executable(app app.c)
  target_link_libraries(app PRIVATE Curses::Curses)

When ncurses is specifically required:

.. code-block:: cmake

  set(CURSES_NEED_NCURSES TRUE)
  find_package(Curses)
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/CheckLibraryExists.cmake)

# we don't know anything about cursesw, so only ncurses
# may be ncursesw
if(NOT CURSES_NEED_WIDE)
  set(NCURSES_LIBRARY_NAME "ncurses")
  set(CURSES_FORM_LIBRARY_NAME "form")
else()
  set(NCURSES_LIBRARY_NAME "ncursesw")
  set(CURSES_FORM_LIBRARY_NAME "formw")
  # Also, if we are searching for wide curses - we are actually searching
  # for ncurses, we don't know about any other unicode version.
  set(CURSES_NEED_NCURSES TRUE)
endif()

find_library(CURSES_CURSES_LIBRARY NAMES curses)

find_library(CURSES_NCURSES_LIBRARY NAMES "${NCURSES_LIBRARY_NAME}" )
set(CURSES_USE_NCURSES FALSE)

if(CURSES_NCURSES_LIBRARY  AND ((NOT CURSES_CURSES_LIBRARY) OR CURSES_NEED_NCURSES))
  set(CURSES_USE_NCURSES TRUE)
endif()
# http://cygwin.com/ml/cygwin-announce/2010-01/msg00002.html
# cygwin ncurses stopped providing curses.h symlinks see above
# message.  Cygwin is an ncurses package, so force ncurses on
# cygwin if the curses.h is missing
if(CURSES_NCURSES_LIBRARY AND CYGWIN)
  if (CURSES_NEED_WIDE)
    if(NOT EXISTS /usr/include/ncursesw/curses.h)
      set(CURSES_USE_NCURSES TRUE)
    endif()
  else()
    if(NOT EXISTS /usr/include/curses.h)
      set(CURSES_USE_NCURSES TRUE)
    endif()
  endif()
endif()


# Not sure the logic is correct here.
# If NCurses is required, use the function wsyncup() to check if the library
# has NCurses functionality (at least this is where it breaks on NetBSD).
# If wsyncup is in curses, use this one.
# If not, try to find ncurses and check if this has the symbol.
# Once the ncurses library is found, search the ncurses.h header first, but
# some web pages also say that even with ncurses there is not always a ncurses.h:
# http://osdir.com/ml/gnome.apps.mc.devel/2002-06/msg00029.html
# So at first try ncurses.h, if not found, try to find curses.h under the same
# prefix as the library was found, if still not found, try curses.h with the
# default search paths.
if(CURSES_CURSES_LIBRARY  AND  CURSES_NEED_NCURSES)
  include(${CMAKE_CURRENT_LIST_DIR}/CMakePushCheckState.cmake)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_QUIET ${Curses_FIND_QUIETLY})
  check_library_exists("${CURSES_CURSES_LIBRARY}"
    wsyncup "" CURSES_CURSES_HAS_WSYNCUP)

  if(CURSES_NCURSES_LIBRARY  AND NOT  CURSES_CURSES_HAS_WSYNCUP)
    check_library_exists("${CURSES_NCURSES_LIBRARY}"
      wsyncup "" CURSES_NCURSES_HAS_WSYNCUP)
    if( CURSES_NCURSES_HAS_WSYNCUP)
      set(CURSES_USE_NCURSES TRUE)
    endif()
  endif()
  cmake_pop_check_state()

endif()

if(CURSES_USE_NCURSES)
  get_filename_component(_cursesLibDir "${CURSES_NCURSES_LIBRARY}" PATH)
  get_filename_component(_cursesParentDir "${_cursesLibDir}" PATH)

  # Use CURSES_NCURSES_INCLUDE_PATH if set, for compatibility.
  if(CURSES_NCURSES_INCLUDE_PATH)
    if (CURSES_NEED_WIDE)
      find_path(CURSES_INCLUDE_PATH
        NAMES ncursesw/ncurses.h ncursesw/curses.h ncursesw.h cursesw.h
        PATHS ${CURSES_NCURSES_INCLUDE_PATH}
        NO_DEFAULT_PATH
        )
    else()
      find_path(CURSES_INCLUDE_PATH
        NAMES ncurses/ncurses.h ncurses/curses.h ncurses.h curses.h
        PATHS ${CURSES_NCURSES_INCLUDE_PATH}
        NO_DEFAULT_PATH
        )
    endif()
  endif()

  if (CURSES_NEED_WIDE)
    set(CURSES_TINFO_LIBRARY_NAME tinfow)
    find_path(CURSES_INCLUDE_PATH
      NAMES ncursesw/ncurses.h ncursesw/curses.h ncursesw.h cursesw.h
      HINTS "${_cursesParentDir}/include"
      )
  else()
    set(CURSES_TINFO_LIBRARY_NAME tinfo)
    find_path(CURSES_INCLUDE_PATH
      NAMES ncurses/ncurses.h ncurses/curses.h ncurses.h curses.h
      HINTS "${_cursesParentDir}/include"
      )
  endif()

  # Previous versions of FindCurses provided these values.
  if(NOT DEFINED CURSES_LIBRARY)
    set(CURSES_LIBRARY "${CURSES_NCURSES_LIBRARY}")
  endif()

  check_library_exists("${CURSES_NCURSES_LIBRARY}"
    cbreak "" CURSES_NCURSES_HAS_CBREAK)
  check_library_exists("${CURSES_NCURSES_LIBRARY}"
    nodelay "" CURSES_NCURSES_HAS_NODELAY)
  if(NOT CURSES_NCURSES_HAS_CBREAK OR NOT CURSES_NCURSES_HAS_NODELAY)
    find_library(CURSES_EXTRA_LIBRARY "${CURSES_TINFO_LIBRARY_NAME}" HINTS "${_cursesLibDir}")
    find_library(CURSES_EXTRA_LIBRARY "${CURSES_TINFO_LIBRARY_NAME}" )

    mark_as_advanced(
      CURSES_EXTRA_LIBRARY
      )
  endif()
else()
  get_filename_component(_cursesLibDir "${CURSES_CURSES_LIBRARY}" PATH)
  get_filename_component(_cursesParentDir "${_cursesLibDir}" PATH)

  # We can't find anything with CURSES_NEED_WIDE because we know
  # only about ncursesw unicode curses version
  if(NOT CURSES_NEED_WIDE)
    find_path(CURSES_INCLUDE_PATH
      NAMES curses.h
      HINTS "${_cursesParentDir}/include"
      )
  endif()

  # Previous versions of FindCurses provided these values.
  if(NOT DEFINED CURSES_CURSES_H_PATH)
    set(CURSES_CURSES_H_PATH "${CURSES_INCLUDE_PATH}")
  endif()
  if(NOT DEFINED CURSES_LIBRARY)
    set(CURSES_LIBRARY "${CURSES_CURSES_LIBRARY}")
  endif()
endif()

# Report whether each possible header name exists in the include directory.
if(NOT DEFINED CURSES_HAVE_NCURSES_NCURSES_H)
  if(CURSES_NEED_WIDE)
    if(EXISTS "${CURSES_INCLUDE_PATH}/ncursesw/ncurses.h")
      set(CURSES_HAVE_NCURSES_NCURSES_H "${CURSES_INCLUDE_PATH}/ncursesw/ncurses.h")
    endif()
  elseif(EXISTS "${CURSES_INCLUDE_PATH}/ncurses/ncurses.h")
    set(CURSES_HAVE_NCURSES_NCURSES_H "${CURSES_INCLUDE_PATH}/ncurses/ncurses.h")
  endif()
  if(NOT DEFINED CURSES_HAVE_NCURSES_NCURSES_H)
    set(CURSES_HAVE_NCURSES_NCURSES_H "CURSES_HAVE_NCURSES_NCURSES_H-NOTFOUND")
  endif()
endif()
if(NOT DEFINED CURSES_HAVE_NCURSES_CURSES_H)
  if(CURSES_NEED_WIDE)
    if(EXISTS "${CURSES_INCLUDE_PATH}/ncursesw/curses.h")
      set(CURSES_HAVE_NCURSES_CURSES_H "${CURSES_INCLUDE_PATH}/ncursesw/curses.h")
    endif()
  elseif(EXISTS "${CURSES_INCLUDE_PATH}/ncurses/curses.h")
    set(CURSES_HAVE_NCURSES_CURSES_H "${CURSES_INCLUDE_PATH}/ncurses/curses.h")
  endif()
  if(NOT DEFINED CURSES_HAVE_NCURSES_CURSES_H)
    set(CURSES_HAVE_NCURSES_CURSES_H "CURSES_HAVE_NCURSES_CURSES_H-NOTFOUND")
  endif()
endif()
if(NOT CURSES_NEED_WIDE)
  # ncursesw can't be found for this paths
  if(NOT DEFINED CURSES_HAVE_NCURSES_H)
    if(EXISTS "${CURSES_INCLUDE_PATH}/ncurses.h")
      set(CURSES_HAVE_NCURSES_H "${CURSES_INCLUDE_PATH}/ncurses.h")
    else()
      set(CURSES_HAVE_NCURSES_H "CURSES_HAVE_NCURSES_H-NOTFOUND")
    endif()
  endif()
  if(NOT DEFINED CURSES_HAVE_CURSES_H)
    if(EXISTS "${CURSES_INCLUDE_PATH}/curses.h")
      set(CURSES_HAVE_CURSES_H "${CURSES_INCLUDE_PATH}/curses.h")
    else()
      set(CURSES_HAVE_CURSES_H "CURSES_HAVE_CURSES_H-NOTFOUND")
    endif()
  endif()
endif()

find_library(CURSES_FORM_LIBRARY "${CURSES_FORM_LIBRARY_NAME}" HINTS "${_cursesLibDir}")
find_library(CURSES_FORM_LIBRARY "${CURSES_FORM_LIBRARY_NAME}" )

# Previous versions of FindCurses provided these values.
if(NOT DEFINED FORM_LIBRARY)
  set(FORM_LIBRARY "${CURSES_FORM_LIBRARY}")
endif()

# Need to provide the *_LIBRARIES
set(CURSES_LIBRARIES ${CURSES_LIBRARY})

if(CURSES_EXTRA_LIBRARY)
  set(CURSES_LIBRARIES ${CURSES_LIBRARIES} ${CURSES_EXTRA_LIBRARY})
endif()

if(CURSES_FORM_LIBRARY)
  set(CURSES_LIBRARIES ${CURSES_LIBRARIES} ${CURSES_FORM_LIBRARY})
endif()

# Provide the *_INCLUDE_DIRS and *_CFLAGS results.
set(CURSES_INCLUDE_DIRS ${CURSES_INCLUDE_PATH})
set(CURSES_INCLUDE_DIR ${CURSES_INCLUDE_PATH}) # compatibility

find_package(PkgConfig QUIET)
if(PkgConfig_FOUND)
  pkg_check_modules(NCURSES QUIET ${NCURSES_LIBRARY_NAME})
  set(CURSES_CFLAGS ${NCURSES_CFLAGS_OTHER})
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Curses DEFAULT_MSG
  CURSES_LIBRARY CURSES_INCLUDE_PATH)

mark_as_advanced(
  CURSES_INCLUDE_PATH
  CURSES_CURSES_LIBRARY
  CURSES_NCURSES_LIBRARY
  CURSES_FORM_LIBRARY
  )
