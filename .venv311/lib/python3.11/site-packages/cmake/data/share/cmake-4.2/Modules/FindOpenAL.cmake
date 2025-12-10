# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindOpenAL
----------

Finds the Open Audio Library (OpenAL):

.. code-block:: cmake

  find_package(OpenAL [...])

OpenAL is a cross-platform 3D audio API designed for efficient rendering of
multichannel three-dimensional positional audio.  It is commonly used in games
and multimedia applications to provide immersive and spatialized sound.

Projects using this module should include the OpenAL header file using
``#include <al.h>``, and **not** ``#include <AL/al.h>``.  The reason for this is
that the latter is not portable.  For example, Windows/Creative Labs does not by
default put OpenAL headers in ``AL/`` and macOS uses the convention of
``<OpenAL/al.h>``.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``OpenAL::OpenAL``
  .. versionadded:: 3.25

  Target encapsulating the OpenAL library usage requirements, available only if
  the OpenAL library is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``OpenAL_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether OpenAL was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OPENAL_INCLUDE_DIR``
  The include directory containing headers needed to use the OpenAL library.
``OPENAL_LIBRARY``
  The path to the OpenAL library.

Hints
^^^^^

This module accepts the following variables:

``OPENALDIR``
  Environment variable which can be used to set the installation prefix of
  OpenAL to be found in non-standard locations.

  OpenAL is searched in the following order:

  1. By default on macOS, system framework is searched first:
     ``/System/Library/Frameworks``, whose priority can be changed by setting
     the :variable:`CMAKE_FIND_FRAMEWORK` variable.
  2. Environment variable ``ENV{OPENALDIR}``.
  3. System paths.
  4. User-compiled framework: ``~/Library/Frameworks``.
  5. Manually compiled framework: ``/Library/Frameworks``.
  6. Add-on package: ``/opt``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OPENAL_FOUND``
  .. deprecated:: 4.2
    Use ``OpenAL_FOUND``, which has the same value.

  Boolean indicating whether OpenAL was found.

Examples
^^^^^^^^

Finding the OpenAL library and linking it to a project target:

.. code-block:: cmake

  find_package(OpenAL)
  target_link_libraries(project_target PRIVATE OpenAL::OpenAL)
#]=======================================================================]

# For Windows, Creative Labs seems to have added a registry key for their
# OpenAL 1.1 installer. I have added that key to the list of search paths,
# however, the key looks like it could be a little fragile depending on
# if they decide to change the 1.00.0000 number for bug fix releases.
# Also, they seem to have laid down groundwork for multiple library platforms
# which puts the library in an extra subdirectory. Currently there is only
# Win32 and I have hardcoded that here. This may need to be adjusted as
# platforms are introduced.
# The OpenAL 1.0 installer doesn't seem to have a useful key I can use.
# I do not know if the Nvidia OpenAL SDK has a registry key.

find_path(OPENAL_INCLUDE_DIR al.h
  HINTS
    ENV OPENALDIR
  PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /opt
    [HKEY_LOCAL_MACHINE\\SOFTWARE\\Creative\ Labs\\OpenAL\ 1.1\ Software\ Development\ Kit\\1.00.0000;InstallDir]
  PATH_SUFFIXES include/AL include/OpenAL include AL OpenAL
  )

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(_OpenAL_ARCH_DIR libs/Win64)
else()
  set(_OpenAL_ARCH_DIR libs/Win32)
endif()

find_library(OPENAL_LIBRARY
  NAMES OpenAL al openal OpenAL32
  HINTS
    ENV OPENALDIR
  PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /opt
    [HKEY_LOCAL_MACHINE\\SOFTWARE\\Creative\ Labs\\OpenAL\ 1.1\ Software\ Development\ Kit\\1.00.0000;InstallDir]
  PATH_SUFFIXES libx32 lib64 lib libs64 libs ${_OpenAL_ARCH_DIR}
  )

unset(_OpenAL_ARCH_DIR)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  OpenAL
  REQUIRED_VARS OPENAL_LIBRARY OPENAL_INCLUDE_DIR
  )

mark_as_advanced(OPENAL_LIBRARY OPENAL_INCLUDE_DIR)

if(OpenAL_FOUND AND NOT TARGET OpenAL::OpenAL)
  add_library(OpenAL::OpenAL UNKNOWN IMPORTED)
  set_target_properties(OpenAL::OpenAL PROPERTIES
    IMPORTED_LOCATION "${OPENAL_LIBRARY}")
  set_target_properties(OpenAL::OpenAL PROPERTIES
    INTERFACE_INCLUDE_DIRECTORIES "${OPENAL_INCLUDE_DIR}")
endif()
