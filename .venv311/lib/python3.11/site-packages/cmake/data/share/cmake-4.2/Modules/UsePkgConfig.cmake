# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
UsePkgConfig
------------

.. deprecated:: 3.0

  This module should no longer be used.  Instead, use the
  :module:`FindPkgConfig` module or the :command:`cmake_pkg_config` command.

  This module provided a command for finding external packages using
  ``pkg-config`` command-line utility.  It has been replaced by the more
  convenient ``FindPkgConfig`` module, which is commonly used in
  :ref:`Find Modules`.

  As of CMake 3.31, the built-in :command:`cmake_pkg_config` command provides
  even more features to extract package information.

Load this module in a CMake project with:

.. code-block:: cmake

  include(UsePkgConfig)

Commands
^^^^^^^^

This module provides the following command:

.. command:: pkgconfig

  Finds external package using ``pkg-config`` and sets result variables:

  .. code-block:: cmake

    pkgconfig(<package> <includedir> <libdir> <linkflags> <cflags>)

  This command invokes ``pkg-config`` command-line utility to retrieve the
  package information into specified variables.  If ``pkg-config`` or the
  specified package ``<package>`` is NOT found, the result variables remain
  empty.

  The arguments are:

  ``<package>``
    Name of the package as defined in its PC metadata file (``<package>.pc``).

  ``<includedir>``
    Variable name to store the package's include directory.

  ``<libdir>``
    Variable name to store the directory containing the package library.

  ``<linkflags>``
    Variable name to store the linker flags for the package.

  ``<cflags>``
    Variable name to store the compiler flags for the package.

Examples
^^^^^^^^

Using this module fills the desired information into the four given variables:

.. code-block:: cmake

  include(UsePkgConfig)
  pkgconfig(
    libart-2.0
    LIBART_INCLUDEDIR
    LIBART_LIBDIR
    LIBART_LDFLAGS
    LIBART_CFLAGS
  )

Migrating to the :module:`FindPkgConfig`  would look something like this:

.. code-block:: cmake

  find_package(PkgConfig QUIET)
  if(PkgConfig_FOUND)
    pkg_check_modules(LIBART QUIET libart-2.0)
  endif()

  message(STATUS "LIBART_INCLUDEDIR=${LIBART_INCLUDEDIR}")
  message(STATUS "LIBART_LIBDIR=${LIBART_LIBDIR}")
  message(STATUS "LIBART_LDFLAGS=${LIBART_LDFLAGS}")
  message(STATUS "LIBART_CFLAGS=${LIBART_CFLAGS}")
#]=======================================================================]

find_program(PKGCONFIG_EXECUTABLE NAMES pkg-config )

macro(PKGCONFIG _package _include_DIR _link_DIR _link_FLAGS _cflags)
  message(STATUS
    "WARNING: you are using the obsolete 'PKGCONFIG' macro, use FindPkgConfig")
# reset the variables at the beginning
  set(${_include_DIR})
  set(${_link_DIR})
  set(${_link_FLAGS})
  set(${_cflags})

  # if pkg-config has been found
  if(PKGCONFIG_EXECUTABLE)

    execute_process(COMMAND ${PKGCONFIG_EXECUTABLE} ${_package} --exists RESULT_VARIABLE _return_VALUE OUTPUT_VARIABLE _pkgconfigDevNull )

    # and if the package of interest also exists for pkg-config, then get the information
    if(NOT _return_VALUE)

      execute_process(COMMAND ${PKGCONFIG_EXECUTABLE} ${_package} --variable=includedir
        OUTPUT_VARIABLE ${_include_DIR} OUTPUT_STRIP_TRAILING_WHITESPACE )
      string(REGEX REPLACE "[\r\n]" " " ${_include_DIR} "${${_include_DIR}}")

      execute_process(COMMAND ${PKGCONFIG_EXECUTABLE} ${_package} --variable=libdir
        OUTPUT_VARIABLE ${_link_DIR} OUTPUT_STRIP_TRAILING_WHITESPACE )
      string(REGEX REPLACE "[\r\n]" " " ${_link_DIR} "${${_link_DIR}}")

      execute_process(COMMAND ${PKGCONFIG_EXECUTABLE} ${_package} --libs
        OUTPUT_VARIABLE ${_link_FLAGS} OUTPUT_STRIP_TRAILING_WHITESPACE )
      string(REGEX REPLACE "[\r\n]" " " ${_link_FLAGS} "${${_link_FLAGS}}")

      execute_process(COMMAND ${PKGCONFIG_EXECUTABLE} ${_package} --cflags
        OUTPUT_VARIABLE ${_cflags} OUTPUT_STRIP_TRAILING_WHITESPACE )
      string(REGEX REPLACE "[\r\n]" " " ${_cflags} "${${_cflags}}")

    else()

      message(STATUS "PKGCONFIG() indicates that ${_package} is not installed (install the package which contains ${_package}.pc if you want to support this feature)")

    endif()

  # if pkg-config has NOT been found, INFORM the user
  else()

    message(STATUS "WARNING: PKGCONFIG() indicates that the tool pkg-config has not been found on your system. You should install it.")

  endif()

endmacro()

mark_as_advanced(PKGCONFIG_EXECUTABLE)
