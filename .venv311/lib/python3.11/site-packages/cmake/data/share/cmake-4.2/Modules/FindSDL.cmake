# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSDL
-------

Finds the SDL (Simple DirectMedia Layer) library:

.. code-block:: cmake

  find_package(SDL [<version>] [...])

SDL is a cross-platform library for developing multimedia software, such as
games and emulators.

.. note::

  This module is specifically intended for SDL version 1.  Starting with version
  2, SDL provides a CMake package configuration file when built with CMake and
  should be found using ``find_package(SDL2)``.  Similarly, SDL version 3 can be
  found using ``find_package(SDL3)``.  These newer versions provide separate
  :ref:`Imported Targets` that encapsulate usage requirements.  Refer to the
  official SDL documentation for more information.

Note that the include path for the SDL header has changed in recent SDL 1
versions from ``SDL/SDL.h`` to simply ``SDL.h``.  This change aligns with SDL's
convention of using ``#include "SDL.h"`` for portability, as not all systems
install the headers in a ``SDL/`` subdirectory (e.g., FreeBSD).

When targeting macOS and using the SDL framework, be sure to include both
``SDLmain.h`` and ``SDLmain.m`` in the project.  For other platforms, the
``SDLmain`` library is typically linked using ``-lSDLmain``, which this module
will attempt to locate automatically.  Additionally, for macOS, this module will
add the ``-framework Cocoa`` flag as needed.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``SDL::SDL``
  .. versionadded:: 3.19

  Target encapsulating the SDL library usage requirements, available if SDL is
  found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``SDL_FOUND``
  Boolean indicating whether (the requested version of) SDL was found.

``SDL_VERSION``
  .. versionadded:: 3.19

  The human-readable string containing the version of SDL found.

``SDL_VERSION_MAJOR``
  .. versionadded:: 3.19

  The major version of SDL found.

``SDL_VERSION_MINOR``
  .. versionadded:: 3.19

  The minor version of SDL found.

``SDL_VERSION_PATCH``
  .. versionadded:: 3.19

  The patch version of SDL found.

``SDL_INCLUDE_DIRS``
  .. versionadded:: 3.19

  Include directories needed to use SDL.

``SDL_LIBRARIES``
  .. versionadded:: 3.19

  Libraries needed to link against to use SDL.

Cache Variables
^^^^^^^^^^^^^^^

These variables may optionally be set to help this module find the correct
files:

``SDL_INCLUDE_DIR``
  The directory containing the ``SDL.h`` header file.
``SDL_LIBRARY``
  A list of libraries containing the path to the SDL library and libraries
  needed to link against to use SDL.

Hints
^^^^^

This module accepts the following variables:

``SDL_BUILDING_LIBRARY``
  When set to boolean true, the ``SDL_main`` library will be excluded from
  linking, as it is not required when building the SDL library itself (only
  applications need ``main()`` function).  If not set, this module assumes an
  application is being built and attempts to locate and include the appropriate
  ``SDL_main`` link flags in the returned ``SDL_LIBRARY`` variable.

``SDLDIR``
  Environment variable that can be set to help locate an SDL library installed
  in a custom location.  It should point to the installation destination that
  was used when configuring, building, and installing SDL library:
  ``./configure --prefix=$SDLDIR``.

  On macOS, setting this variable will prefer the Framework version (if found)
  over others.  In this case, the cache value of ``SDL_LIBRARY`` would need to
  be manually changed to override this selection or set the
  :variable:`CMAKE_INCLUDE_PATH` variable to modify the search paths.

Troubleshooting
^^^^^^^^^^^^^^^

In case the SDL library is not found automatically, the ``SDL_LIBRARY_TEMP``
variable may be empty, and ``SDL_LIBRARY`` will not be set.  This typically
means that CMake could not locate the SDL library (e.g., ``SDL.dll``,
``libSDL.so``, ``SDL.framework``, etc.).  To resolve this, manually set
``SDL_LIBRARY_TEMP`` to the correct path and reconfigure the project.
Similarly, if ``SDLMAIN_LIBRARY`` is unset, it may also need to be specified
manually.  These variables are used to construct the final ``SDL_LIBRARY``
value.  If they are not set, ``SDL_LIBRARY`` will remain undefined.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``SDL_VERSION_STRING``
  .. deprecated:: 3.19
    Superseded by the ``SDL_VERSION`` with the same value.

  The human-readable string containing the version of SDL if found.

Examples
^^^^^^^^

Finding SDL library and linking it to a project target:

.. code-block:: cmake

  find_package(SDL)
  target_link_libraries(project_target PRIVATE SDL::SDL)

When working with SDL version 2, the upstream package provides the
``SDL2::SDL2`` imported target directly.  It can be used in a project without
using this module:

.. code-block:: cmake

  find_package(SDL2)
  target_link_libraries(project_target PRIVATE SDL2::SDL2)

Similarly, for SDL version 3:

.. code-block:: cmake

  find_package(SDL3)
  target_link_libraries(project_target PRIVATE SDL3::SDL3)

See Also
^^^^^^^^

* The :module:`FindSDL_gfx` module to find the SDL_gfx library.
* The :module:`FindSDL_image` module to find the SDL_image library.
* The :module:`FindSDL_mixer` module to find the SDL_mixer library.
* The :module:`FindSDL_net` module to find the SDL_net library.
* The :module:`FindSDL_sound` module to find the SDL_sound library.
* The :module:`FindSDL_ttf` module to find the SDL_ttf library.
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(SDL_INCLUDE_DIR SDL.h
  HINTS
    ENV SDLDIR
  PATH_SUFFIXES SDL SDL12 SDL11
                # path suffixes to search inside ENV{SDLDIR}
                include/SDL include/SDL12 include/SDL11 include
)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(VC_LIB_PATH_SUFFIX lib/x64)
else()
  set(VC_LIB_PATH_SUFFIX lib/x86)
endif()

# SDL-1.1 is the name used by FreeBSD ports...
# don't confuse it for the version number.
find_library(SDL_LIBRARY_TEMP
  NAMES SDL SDL-1.1
  HINTS
    ENV SDLDIR
  PATH_SUFFIXES lib ${VC_LIB_PATH_SUFFIX}
)

# Hide this cache variable from the user, it's an internal implementation
# detail. The documented library variable for the user is SDL_LIBRARY
# which is derived from SDL_LIBRARY_TEMP further below.
set_property(CACHE SDL_LIBRARY_TEMP PROPERTY TYPE INTERNAL)

if(NOT SDL_BUILDING_LIBRARY)
  if(NOT SDL_INCLUDE_DIR MATCHES ".framework")
    # Non-OS X framework versions expect you to also dynamically link to
    # SDLmain. This is mainly for Windows and OS X. Other (Unix) platforms
    # seem to provide SDLmain for compatibility even though they don't
    # necessarily need it.
    find_library(SDLMAIN_LIBRARY
      NAMES SDLmain SDLmain-1.1
      HINTS
        ENV SDLDIR
      PATH_SUFFIXES lib ${VC_LIB_PATH_SUFFIX}
      PATHS
      /opt
    )
  endif()
endif()

# SDL may require threads on your system.
# The Apple build may not need an explicit flag because one of the
# frameworks may already provide it.
# But for non-OSX systems, I will use the CMake Threads package.
if(NOT APPLE)
  find_package(Threads)
endif()

# MinGW needs an additional link flag, -mwindows
# It's total link flags should look like -lmingw32 -lSDLmain -lSDL -mwindows
if(MINGW)
  set(MINGW32_LIBRARY mingw32 "-mwindows" CACHE STRING "link flags for MinGW")
endif()

if(SDL_LIBRARY_TEMP)
  # For SDLmain
  if(SDLMAIN_LIBRARY AND NOT SDL_BUILDING_LIBRARY)
    list(FIND SDL_LIBRARY_TEMP "${SDLMAIN_LIBRARY}" _SDL_MAIN_INDEX)
    if(_SDL_MAIN_INDEX EQUAL -1)
      set(SDL_LIBRARY_TEMP "${SDLMAIN_LIBRARY}" ${SDL_LIBRARY_TEMP})
    endif()
    unset(_SDL_MAIN_INDEX)
  endif()

  # For OS X, SDL uses Cocoa as a backend so it must link to Cocoa.
  # CMake doesn't display the -framework Cocoa string in the UI even
  # though it actually is there if I modify a preused variable.
  # I think it has something to do with the CACHE STRING.
  # So I use a temporary variable until the end so I can set the
  # "real" variable in one-shot.
  if(APPLE)
    set(SDL_LIBRARY_TEMP ${SDL_LIBRARY_TEMP} "-framework Cocoa")
  endif()

  # For threads, as mentioned Apple doesn't need this.
  # In fact, there seems to be a problem if I used the Threads package
  # and try using this line, so I'm just skipping it entirely for OS X.
  if(NOT APPLE)
    set(SDL_LIBRARY_TEMP ${SDL_LIBRARY_TEMP} ${CMAKE_THREAD_LIBS_INIT})
  endif()

  # For MinGW library
  if(MINGW)
    set(SDL_LIBRARY_TEMP ${MINGW32_LIBRARY} ${SDL_LIBRARY_TEMP})
  endif()

  # Set the final string here so the GUI reflects the final state.
  set(SDL_LIBRARY ${SDL_LIBRARY_TEMP} CACHE STRING "Where the SDL Library can be found")
endif()

if(SDL_INCLUDE_DIR AND EXISTS "${SDL_INCLUDE_DIR}/SDL_version.h")
  file(STRINGS "${SDL_INCLUDE_DIR}/SDL_version.h" SDL_VERSION_MAJOR_LINE REGEX "^#define[ \t]+SDL_MAJOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL_INCLUDE_DIR}/SDL_version.h" SDL_VERSION_MINOR_LINE REGEX "^#define[ \t]+SDL_MINOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL_INCLUDE_DIR}/SDL_version.h" SDL_VERSION_PATCH_LINE REGEX "^#define[ \t]+SDL_PATCHLEVEL[ \t]+[0-9]+$")
  string(REGEX REPLACE "^#define[ \t]+SDL_MAJOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL_VERSION_MAJOR "${SDL_VERSION_MAJOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_MINOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL_VERSION_MINOR "${SDL_VERSION_MINOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_PATCHLEVEL[ \t]+([0-9]+)$" "\\1" SDL_VERSION_PATCH "${SDL_VERSION_PATCH_LINE}")
  unset(SDL_VERSION_MAJOR_LINE)
  unset(SDL_VERSION_MINOR_LINE)
  unset(SDL_VERSION_PATCH_LINE)
  set(SDL_VERSION ${SDL_VERSION_MAJOR}.${SDL_VERSION_MINOR}.${SDL_VERSION_PATCH})
  set(SDL_VERSION_STRING ${SDL_VERSION})
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SDL
                                  REQUIRED_VARS SDL_LIBRARY SDL_INCLUDE_DIR
                                  VERSION_VAR SDL_VERSION)

if(SDL_FOUND)
  set(SDL_LIBRARIES ${SDL_LIBRARY})
  set(SDL_INCLUDE_DIRS ${SDL_INCLUDE_DIR})
  if(NOT TARGET SDL::SDL)
    add_library(SDL::SDL INTERFACE IMPORTED)
    set_target_properties(SDL::SDL PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${SDL_INCLUDE_DIR}"
      INTERFACE_LINK_LIBRARIES "${SDL_LIBRARY}")
  endif()
endif()

cmake_policy(POP)
