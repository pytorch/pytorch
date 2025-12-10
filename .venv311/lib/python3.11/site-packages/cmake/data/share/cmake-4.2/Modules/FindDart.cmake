# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindDart
--------

.. deprecated:: 3.27
  This module is available only if policy :policy:`CMP0145` is not set to ``NEW``.

Finds DART:

.. code-block:: cmake

  find_package(DART [...])

This module looks for the dart testing software and sets ``DART_ROOT`` to
point to where it found it.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Dart_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether DART was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``DART_ROOT``
  The path to the Dart root installation.
#]=======================================================================]

if(_FindDart_testing)
  set(_FindDart_included TRUE)
  return()
endif()

find_path(DART_ROOT README.INSTALL
    HINTS
      ENV DART_ROOT
    PATHS
      ${PROJECT_SOURCE_DIR}
      /usr/share
      C:/
      "C:/Program Files"
      ${PROJECT_SOURCE_DIR}/..
      [HKEY_LOCAL_MACHINE\\SOFTWARE\\Dart\\InstallPath]
      ENV ProgramFiles
    PATH_SUFFIXES
      Dart
    DOC "If you have Dart installed, where is it located?"
    )

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Dart DEFAULT_MSG DART_ROOT)

mark_as_advanced(DART_ROOT)
