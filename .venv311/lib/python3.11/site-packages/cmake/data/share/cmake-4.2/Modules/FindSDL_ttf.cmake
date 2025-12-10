# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSDL_ttf
-----------

Finds the SDL_ttf library that provides support for rendering text with TrueType
fonts in SDL (Simple DirectMedia Layer) applications:

.. code-block:: cmake

  find_package(SDL_ttf [<version>] [...])

.. note::

  This module is specifically intended for SDL_ttf version 1.  Starting with
  version 2.0.15, SDL_ttf provides a CMake package configuration file when built
  with CMake and should be found using ``find_package(SDL2_ttf)``.  Similarly,
  SDL_ttf version 3 can be found using ``find_package(SDL3_ttf)``.  These newer
  versions provide :ref:`Imported Targets` that encapsulate usage requirements.
  Refer to the official SDL documentation for more information.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``SDL_ttf_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) SDL_ttf library was
  found.

``SDL_ttf_VERSION``
  .. versionadded:: 4.2

  The human-readable string containing the version of SDL_ttf found.

``SDL_TTF_INCLUDE_DIRS``
  Include directories containing headers needed to use SDL_ttf library.

``SDL_TTF_LIBRARIES``
  Libraries needed to link against to use SDL_ttf.

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

``SDL_TTF_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``SDL_ttf_VERSION``, which has the same value.

  The human-readable string containing the version of SDL_ttf found.

``SDL_TTF_FOUND``
  .. deprecated:: 4.2
    Use ``SDL_ttf_FOUND``, which has the same value.

``SDLTTF_FOUND``
  .. deprecated:: 2.8.10
    Replaced with ``SDL_ttf_FOUND``, which has the same value.

``SDLTTF_INCLUDE_DIR``
  .. deprecated:: 2.8.10
    Replaced with ``SDL_TTF_INCLUDE_DIRS``, which has the same value.

``SDLTTF_LIBRARY``
  .. deprecated:: 2.8.10
    Replaced with ``SDL_TTF_LIBRARIES``, which has the same value.

Examples
^^^^^^^^

Finding SDL_ttf library and creating an imported interface target for linking
it to a project target:

.. code-block:: cmake

  find_package(SDL_ttf)

  if(SDL_ttf_FOUND AND NOT TARGET SDL::SDL_ttf)
    add_library(SDL::SDL_ttf INTERFACE IMPORTED)
    set_target_properties(
      SDL::SDL_ttf
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${SDL_TTF_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${SDL_TTF_LIBRARIES}"
    )
  endif()

  target_link_libraries(project_target PRIVATE SDL::SDL_ttf)

When working with SDL_ttf version 2, the upstream package provides the
``SDL2_ttf::SDL2_ttf`` imported target directly.  It can be used in a project
without using this module:

.. code-block:: cmake

  find_package(SDL2_ttf)
  target_link_libraries(project_target PRIVATE SDL2_ttf::SDL2_ttf)

Similarly, for SDL_ttf version 3:

.. code-block:: cmake

  find_package(SDL3_ttf)
  target_link_libraries(project_target PRIVATE SDL3_ttf::SDL3_ttf)

See Also
^^^^^^^^

* The :module:`FindSDL` module to find the main SDL library.
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

if(NOT SDL_TTF_INCLUDE_DIR AND SDLTTF_INCLUDE_DIR)
  set(SDL_TTF_INCLUDE_DIR ${SDLTTF_INCLUDE_DIR} CACHE PATH "directory cache
entry initialized from old variable name")
endif()
find_path(SDL_TTF_INCLUDE_DIR SDL_ttf.h
  HINTS
    ENV SDLTTFDIR
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

if(NOT SDL_TTF_LIBRARY AND SDLTTF_LIBRARY)
  set(SDL_TTF_LIBRARY ${SDLTTF_LIBRARY} CACHE FILEPATH "file cache entry
initialized from old variable name")
endif()
find_library(SDL_TTF_LIBRARY
  NAMES SDL_ttf
  HINTS
    ENV SDLTTFDIR
    ENV SDLDIR
  PATH_SUFFIXES lib ${VC_LIB_PATH_SUFFIX}
)

if(SDL_TTF_INCLUDE_DIR AND EXISTS "${SDL_TTF_INCLUDE_DIR}/SDL_ttf.h")
  file(STRINGS "${SDL_TTF_INCLUDE_DIR}/SDL_ttf.h" SDL_TTF_VERSION_MAJOR_LINE REGEX "^#define[ \t]+SDL_TTF_MAJOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL_TTF_INCLUDE_DIR}/SDL_ttf.h" SDL_TTF_VERSION_MINOR_LINE REGEX "^#define[ \t]+SDL_TTF_MINOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL_TTF_INCLUDE_DIR}/SDL_ttf.h" SDL_TTF_VERSION_PATCH_LINE REGEX "^#define[ \t]+SDL_TTF_PATCHLEVEL[ \t]+[0-9]+$")
  string(REGEX REPLACE "^#define[ \t]+SDL_TTF_MAJOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL_TTF_VERSION_MAJOR "${SDL_TTF_VERSION_MAJOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_TTF_MINOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL_TTF_VERSION_MINOR "${SDL_TTF_VERSION_MINOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_TTF_PATCHLEVEL[ \t]+([0-9]+)$" "\\1" SDL_TTF_VERSION_PATCH "${SDL_TTF_VERSION_PATCH_LINE}")
  set(SDL_ttf_VERSION ${SDL_TTF_VERSION_MAJOR}.${SDL_TTF_VERSION_MINOR}.${SDL_TTF_VERSION_PATCH})
  set(SDL_TTF_VERSION_STRING "${SDL_ttf_VERSION}")
  unset(SDL_TTF_VERSION_MAJOR_LINE)
  unset(SDL_TTF_VERSION_MINOR_LINE)
  unset(SDL_TTF_VERSION_PATCH_LINE)
  unset(SDL_TTF_VERSION_MAJOR)
  unset(SDL_TTF_VERSION_MINOR)
  unset(SDL_TTF_VERSION_PATCH)
endif()

set(SDL_TTF_LIBRARIES ${SDL_TTF_LIBRARY})
set(SDL_TTF_INCLUDE_DIRS ${SDL_TTF_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SDL_ttf
                                  REQUIRED_VARS SDL_TTF_LIBRARIES SDL_TTF_INCLUDE_DIRS
                                  VERSION_VAR SDL_ttf_VERSION)

# for backward compatibility
set(SDLTTF_LIBRARY ${SDL_TTF_LIBRARIES})
set(SDLTTF_INCLUDE_DIR ${SDL_TTF_INCLUDE_DIRS})
set(SDLTTF_FOUND ${SDL_TTF_FOUND})

mark_as_advanced(SDL_TTF_LIBRARY SDL_TTF_INCLUDE_DIR)

cmake_policy(POP)
