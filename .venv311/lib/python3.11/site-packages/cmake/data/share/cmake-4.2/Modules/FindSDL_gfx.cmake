# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSDL_gfx
-----------

.. versionadded:: 3.25

Finds the SDL_gfx library that provides graphics support in SDL (Simple
DirectMedia Layer) applications:

.. code-block:: cmake

  find_package(SDL_gfx [<version>] [...])

.. note::

  This module is for SDL_gfx version 1.  For version 2 or newer usage refer to
  the upstream documentation.

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``SDL::SDL_gfx``
  Target encapsulating the SDL_gfx library usage requirements, available if
  SDL_gfx is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``SDL_gfx_FOUND``
  Boolean indicating whether the (requested version of) SDL_gfx library was
  found.

``SDL_gfx_VERSION``
  .. versionadded:: 4.2

  The human-readable string containing the version of SDL_gfx found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``SDL_GFX_INCLUDE_DIRS``
  The directory containing the headers needed to use SDL_gfx.

``SDL_GFX_LIBRARIES``
  The path to the SDL_gfx library needed to link against to use SDL_gfx.

Hints
^^^^^

This module accepts the following variables:

``SDLDIR``
  Environment variable that can be set to help locate an SDL library installed
  in a custom location.  It should point to the installation destination that
  was used when configuring, building, and installing SDL library:
  ``./configure --prefix=$SDLDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``SDL_GFX_FOUND``
  .. deprecated:: 4.2
    Use ``SDL_gfx_FOUND``, which has the same value.

  Boolean indicating whether the (requested version of) SDL_gfx library was
  found.

``SDL_GFX_VERSION_STRING``
  .. deprecated:: 4.2
    Use the ``SDL_gfx_VERSION``.

  The human-readable string containing the version of SDL_gfx found.

Examples
^^^^^^^^

Finding SDL_gfx library and linking it to a project target:

.. code-block:: cmake

  find_package(SDL_gfx)
  target_link_libraries(project_target PRIVATE SDL::SDL_gfx)

See Also
^^^^^^^^

* The :module:`FindSDL` module to find the main SDL library.
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(SDL_GFX_INCLUDE_DIRS
  NAMES
    SDL_framerate.h
    SDL_gfxBlitFunc.h
    SDL_gfxPrimitives.h
    SDL_gfxPrimitives_font.h
    SDL_imageFilter.h
    SDL_rotozoom.h
  HINTS
    ENV SDLGFXDIR
    ENV SDLDIR
  PATH_SUFFIXES SDL
                # path suffixes to search inside ENV{SDLDIR}
                include/SDL include/SDL12 include/SDL11 include
)

if(CMAKE_SIZEOF_VOID_P EQUAL 8)
  set(VC_LIB_PATH_SUFFIX lib/x64)
else()
  set(VC_LIB_PATH_SUFFIX lib/x86)
endif()

find_library(SDL_GFX_LIBRARIES
  NAMES SDL_gfx
  HINTS
    ENV SDLGFXDIR
    ENV SDLDIR
  PATH_SUFFIXES lib ${VC_LIB_PATH_SUFFIX}
)

if(SDL_GFX_INCLUDE_DIRS AND EXISTS "${SDL_GFX_INCLUDE_DIRS}/SDL_gfxPrimitives.h")
  file(STRINGS "${SDL_GFX_INCLUDE_DIRS}/SDL_gfxPrimitives.h" SDL_GFX_VERSION_MAJOR_LINE REGEX "^#define[ \t]+SDL_GFXPRIMITIVES_MAJOR[ \t]+[0-9]+$")
  file(STRINGS "${SDL_GFX_INCLUDE_DIRS}/SDL_gfxPrimitives.h" SDL_GFX_VERSION_MINOR_LINE REGEX "^#define[ \t]+SDL_GFXPRIMITIVES_MINOR[ \t]+[0-9]+$")
  file(STRINGS "${SDL_GFX_INCLUDE_DIRS}/SDL_gfxPrimitives.h" SDL_GFX_VERSION_PATCH_LINE REGEX "^#define[ \t]+SDL_GFXPRIMITIVES_MICRO[ \t]+[0-9]+$")
  string(REGEX REPLACE "^#define[ \t]+SDL_GFXPRIMITIVES_MAJOR[ \t]+([0-9]+)$" "\\1" SDL_GFX_VERSION_MAJOR "${SDL_GFX_VERSION_MAJOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_GFXPRIMITIVES_MINOR[ \t]+([0-9]+)$" "\\1" SDL_GFX_VERSION_MINOR "${SDL_GFX_VERSION_MINOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_GFXPRIMITIVES_MICRO[ \t]+([0-9]+)$" "\\1" SDL_GFX_VERSION_PATCH "${SDL_GFX_VERSION_PATCH_LINE}")
  set(SDL_gfx_VERSION ${SDL_GFX_VERSION_MAJOR}.${SDL_GFX_VERSION_MINOR}.${SDL_GFX_VERSION_PATCH})
  set(SDL_GFX_VERSION_STRING "${SDL_gfx_VERSION}")
  unset(SDL_GFX_VERSION_MAJOR_LINE)
  unset(SDL_GFX_VERSION_MINOR_LINE)
  unset(SDL_GFX_VERSION_PATCH_LINE)
  unset(SDL_GFX_VERSION_MAJOR)
  unset(SDL_GFX_VERSION_MINOR)
  unset(SDL_GFX_VERSION_PATCH)
endif()

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SDL_gfx
                                  REQUIRED_VARS SDL_GFX_LIBRARIES SDL_GFX_INCLUDE_DIRS
                                  VERSION_VAR SDL_gfx_VERSION)

if(SDL_gfx_FOUND)
  if(NOT TARGET SDL::SDL_gfx)
    add_library(SDL::SDL_gfx INTERFACE IMPORTED)
    set_target_properties(SDL::SDL_gfx PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${SDL_GFX_INCLUDE_DIRS}"
      INTERFACE_LINK_LIBRARIES "${SDL_GFX_LIBRARIES}")
  endif()
endif()

cmake_policy(POP)
