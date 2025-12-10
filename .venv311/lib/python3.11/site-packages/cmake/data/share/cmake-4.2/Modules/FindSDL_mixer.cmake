# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSDL_mixer
-------------

Finds the SDL_mixer library that provides an audio mixer with support for
various file formats in SDL (Simple DirectMedia Layer) applications:

.. code-block:: cmake

  find_package(SDL_mixer [<version>] [...])

.. note::

  This module is specifically intended for SDL_mixer version 1.  Starting with
  version 2.5, SDL_mixer provides a CMake package configuration file when built
  with CMake and should be found using ``find_package(SDL2_mixer)``.  These
  newer versions provide :ref:`Imported Targets` that encapsulate usage
  requirements.  Refer to the official SDL documentation for more information.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``SDL_mixer_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) SDL_mixer library
  was found.

``SDL_mixer_VERSION``
  .. versionadded:: 4.2

  The human-readable string containing the version of SDL_mixer found.

``SDL_MIXER_INCLUDE_DIRS``
  Include directories containing headers needed to use the SDL_mixer library.

``SDL_MIXER_LIBRARIES``
  Libraries needed to link against to use SDL_mixer.

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

``SDL_MIXER_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``SDL_mixer_VERSION``, which has the same value.

  The human-readable string containing the version of SDL_mixer found.

``SDL_MIXER_FOUND``
  .. deprecated:: 4.2
    Use ``SDL_mixer_FOUND``, which has the same value.

``SDLMIXER_FOUND``
  .. deprecated:: 2.8.10
    Use ``SDL_mixer_FOUND``, which has the same value.

``SDLMIXER_INCLUDE_DIR``
  .. deprecated:: 2.8.10
    Use ``SDL_MIXER_INCLUDE_DIRS``, which has the same value.

``SDLMIXER_LIBRARY``
  .. deprecated:: 2.8.10
    Use ``SDL_MIXER_LIBRARIES``, which has the same value.

Examples
^^^^^^^^

Finding SDL_mixer library and creating an imported interface target for linking
it to a project target:

.. code-block:: cmake

  find_package(SDL_mixer)

  if(SDL_mixer_FOUND AND NOT TARGET SDL::SDL_mixer)
    add_library(SDL::SDL_mixer INTERFACE IMPORTED)
    set_target_properties(
      SDL::SDL_mixer
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${SDL_MIXER_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${SDL_MIXER_LIBRARIES}"
    )
  endif()

  target_link_libraries(project_target PRIVATE SDL::SDL_mixer)

When working with SDL_mixer version 2, the upstream package provides the
``SDL2_mixer::SDL2_mixer`` imported target directly.  It can be used in a
project without using this module:

.. code-block:: cmake

  find_package(SDL2_mixer)
  target_link_libraries(project_target PRIVATE SDL2_mixer::SDL2_mixer)

See Also
^^^^^^^^

* The :module:`FindSDL` module to find the main SDL library.
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

if(NOT SDL_MIXER_INCLUDE_DIR AND SDLMIXER_INCLUDE_DIR)
  set(SDL_MIXER_INCLUDE_DIR ${SDLMIXER_INCLUDE_DIR} CACHE PATH "directory cache
entry initialized from old variable name")
endif()
find_path(SDL_MIXER_INCLUDE_DIR SDL_mixer.h
  HINTS
    ENV SDLMIXERDIR
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

if(NOT SDL_MIXER_LIBRARY AND SDLMIXER_LIBRARY)
  set(SDL_MIXER_LIBRARY ${SDLMIXER_LIBRARY} CACHE FILEPATH "file cache entry
initialized from old variable name")
endif()
find_library(SDL_MIXER_LIBRARY
  NAMES SDL_mixer
  HINTS
    ENV SDLMIXERDIR
    ENV SDLDIR
  PATH_SUFFIXES lib ${VC_LIB_PATH_SUFFIX}
)

if(SDL_MIXER_INCLUDE_DIR AND EXISTS "${SDL_MIXER_INCLUDE_DIR}/SDL_mixer.h")
  file(STRINGS "${SDL_MIXER_INCLUDE_DIR}/SDL_mixer.h" SDL_MIXER_VERSION_MAJOR_LINE REGEX "^#define[ \t]+SDL_MIXER_MAJOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL_MIXER_INCLUDE_DIR}/SDL_mixer.h" SDL_MIXER_VERSION_MINOR_LINE REGEX "^#define[ \t]+SDL_MIXER_MINOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL_MIXER_INCLUDE_DIR}/SDL_mixer.h" SDL_MIXER_VERSION_PATCH_LINE REGEX "^#define[ \t]+SDL_MIXER_PATCHLEVEL[ \t]+[0-9]+$")
  string(REGEX REPLACE "^#define[ \t]+SDL_MIXER_MAJOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL_MIXER_VERSION_MAJOR "${SDL_MIXER_VERSION_MAJOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_MIXER_MINOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL_MIXER_VERSION_MINOR "${SDL_MIXER_VERSION_MINOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_MIXER_PATCHLEVEL[ \t]+([0-9]+)$" "\\1" SDL_MIXER_VERSION_PATCH "${SDL_MIXER_VERSION_PATCH_LINE}")
  set(SDL_mixer_VERSION ${SDL_MIXER_VERSION_MAJOR}.${SDL_MIXER_VERSION_MINOR}.${SDL_MIXER_VERSION_PATCH})
  set(SDL_MIXER_VERSION_STRING "${SDL_mixer_VERSION}")
  unset(SDL_MIXER_VERSION_MAJOR_LINE)
  unset(SDL_MIXER_VERSION_MINOR_LINE)
  unset(SDL_MIXER_VERSION_PATCH_LINE)
  unset(SDL_MIXER_VERSION_MAJOR)
  unset(SDL_MIXER_VERSION_MINOR)
  unset(SDL_MIXER_VERSION_PATCH)
endif()

set(SDL_MIXER_LIBRARIES ${SDL_MIXER_LIBRARY})
set(SDL_MIXER_INCLUDE_DIRS ${SDL_MIXER_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SDL_mixer
                                  REQUIRED_VARS SDL_MIXER_LIBRARIES SDL_MIXER_INCLUDE_DIRS
                                  VERSION_VAR SDL_mixer_VERSION)

# for backward compatibility
set(SDLMIXER_LIBRARY ${SDL_MIXER_LIBRARIES})
set(SDLMIXER_INCLUDE_DIR ${SDL_MIXER_INCLUDE_DIRS})
set(SDLMIXER_FOUND ${SDL_mixer_FOUND})

mark_as_advanced(SDL_MIXER_LIBRARY SDL_MIXER_INCLUDE_DIR)

cmake_policy(POP)
