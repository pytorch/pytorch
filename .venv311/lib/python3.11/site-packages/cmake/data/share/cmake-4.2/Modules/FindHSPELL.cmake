# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindHSPELL
----------

Finds the Hebrew spell-checker and morphology engine (Hspell):

.. code-block:: cmake

  find_package(HSPELL [<version>] [...])

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``HSPELL_FOUND``
  Boolean indicating whether (the requested version of) Hspell was found.

``HSPELL_VERSION``
  .. versionadded:: 4.2

  The version of Hspell found (x.y).

``HSPELL_VERSION_MAJOR``
  The major version of Hspell found.

``HSPELL_VERSION_MINOR``
  The minor version of Hspell found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``HSPELL_INCLUDE_DIR``
  The Hspell include directory.

``HSPELL_LIBRARIES``
  The libraries needed to use Hspell.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``HSPELL_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``HSPELL_VERSION``, which has the same value.

  The version of Hspell found (x.y).

Examples
^^^^^^^^

Finding Hspell:

.. code-block:: cmake

  find_package(HSPELL)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(HSPELL_INCLUDE_DIR hspell.h)

find_library(HSPELL_LIBRARIES NAMES hspell)

if (HSPELL_INCLUDE_DIR)
    file(STRINGS "${HSPELL_INCLUDE_DIR}/hspell.h" HSPELL_H REGEX "#define HSPELL_VERSION_M(AJO|INO)R [0-9]+")
    string(REGEX REPLACE ".*#define HSPELL_VERSION_MAJOR ([0-9]+).*" "\\1" HSPELL_VERSION_MAJOR "${HSPELL_H}")
    string(REGEX REPLACE ".*#define HSPELL_VERSION_MINOR ([0-9]+).*" "\\1" HSPELL_VERSION_MINOR "${HSPELL_H}")
    set(HSPELL_VERSION "${HSPELL_VERSION_MAJOR}.${HSPELL_VERSION_MINOR}")
    set(HSPELL_VERSION_STRING "${HSPELL_VERSION}")
    unset(HSPELL_H)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(HSPELL
                                  REQUIRED_VARS HSPELL_LIBRARIES HSPELL_INCLUDE_DIR
                                  VERSION_VAR HSPELL_VERSION)

mark_as_advanced(HSPELL_INCLUDE_DIR HSPELL_LIBRARIES)

cmake_policy(POP)
