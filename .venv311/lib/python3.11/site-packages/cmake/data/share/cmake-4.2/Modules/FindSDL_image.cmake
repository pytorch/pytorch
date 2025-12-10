# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSDL_image
-------------

Finds the SDL_image library that loads images of various formats as SDL (Simple
DirectMedia Layer) surfaces:

.. code-block:: cmake

  find_package(SDL_image [<version>] [...])

.. note::

  This module is specifically intended for SDL_image version 1.  Starting with
  version 2.6, SDL_image provides a CMake package configuration file when built
  with CMake and should be found using ``find_package(SDL2_image)``.  Similarly,
  SDL_image version 3 can be found using ``find_package(SDL3_image)``.  These
  newer versions provide :ref:`Imported Targets` that encapsulate usage
  requirements.  Refer to the official SDL documentation for more information.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``SDL_image_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) SDL_image library
  was found.

``SDL_image_VERSION``
  .. versionadded:: 4.2

  The human-readable string containing the version of SDL_image found.

``SDL_IMAGE_INCLUDE_DIRS``
  Include directories containing headers needed to use the SDL_image library.

``SDL_IMAGE_LIBRARIES``
  Libraries needed to link against to use the SDL_image library.

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

``SDL_IMAGE_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``SDL_image_VERSION``, which has the same value.

  The human-readable string containing the version of SDL_image found.

``SDL_IMAGE_FOUND``
  .. deprecated:: 4.2
    Use ``SDL_image_FOUND``, which has the same value.

``SDLIMAGE_FOUND``
  .. deprecated:: 2.8.10
    Use ``SDL_image_FOUND``, which has the same value.

``SDLIMAGE_INCLUDE_DIR``
  .. deprecated:: 2.8.10
    Use ``SDL_IMAGE_INCLUDE_DIRS``, which has the same value.

``SDLIMAGE_LIBRARY``
  .. deprecated:: 2.8.10
    Use ``SDL_IMAGE_LIBRARIES``, which has the same value.

Examples
^^^^^^^^

Finding SDL_image library and creating an imported interface target for linking
it to a project target:

.. code-block:: cmake

  find_package(SDL_image)

  if(SDL_image_FOUND AND NOT TARGET SDL::SDL_image)
    add_library(SDL::SDL_image INTERFACE IMPORTED)
    set_target_properties(
      SDL::SDL_image
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${SDL_IMAGE_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${SDL_IMAGE_LIBRARIES}"
    )
  endif()

  target_link_libraries(project_target PRIVATE SDL::SDL_image)

When working with SDL_image version 2, the upstream package provides the
``SDL2_image::SDL2_image`` imported target directly.  It can be used in a
project without using this module:

.. code-block:: cmake

  find_package(SDL2_image)
  target_link_libraries(project_target PRIVATE SDL2_image::SDL2_image)

Similarly, for SDL_image version 3:

.. code-block:: cmake

  find_package(SDL3_image)
  target_link_libraries(project_target PRIVATE SDL3_image::SDL3_image)

See Also
^^^^^^^^

* The :module:`FindSDL` module to find the main SDL library.
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

if(NOT SDL_IMAGE_INCLUDE_DIR AND SDLIMAGE_INCLUDE_DIR)
  set(SDL_IMAGE_INCLUDE_DIR ${SDLIMAGE_INCLUDE_DIR} CACHE PATH "directory cache
entry initialized from old variable name")
endif()
find_path(SDL_IMAGE_INCLUDE_DIR SDL_image.h
  HINTS
    ENV SDLIMAGEDIR
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

if(NOT SDL_IMAGE_LIBRARY AND SDLIMAGE_LIBRARY)
  set(SDL_IMAGE_LIBRARY ${SDLIMAGE_LIBRARY} CACHE FILEPATH "file cache entry
initialized from old variable name")
endif()
find_library(SDL_IMAGE_LIBRARY
  NAMES SDL_image
  HINTS
    ENV SDLIMAGEDIR
    ENV SDLDIR
  PATH_SUFFIXES lib ${VC_LIB_PATH_SUFFIX}
)

if(SDL_IMAGE_INCLUDE_DIR AND EXISTS "${SDL_IMAGE_INCLUDE_DIR}/SDL_image.h")
  file(STRINGS "${SDL_IMAGE_INCLUDE_DIR}/SDL_image.h" SDL_IMAGE_VERSION_MAJOR_LINE REGEX "^#define[ \t]+SDL_IMAGE_MAJOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL_IMAGE_INCLUDE_DIR}/SDL_image.h" SDL_IMAGE_VERSION_MINOR_LINE REGEX "^#define[ \t]+SDL_IMAGE_MINOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL_IMAGE_INCLUDE_DIR}/SDL_image.h" SDL_IMAGE_VERSION_PATCH_LINE REGEX "^#define[ \t]+SDL_IMAGE_PATCHLEVEL[ \t]+[0-9]+$")
  string(REGEX REPLACE "^#define[ \t]+SDL_IMAGE_MAJOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL_IMAGE_VERSION_MAJOR "${SDL_IMAGE_VERSION_MAJOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_IMAGE_MINOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL_IMAGE_VERSION_MINOR "${SDL_IMAGE_VERSION_MINOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_IMAGE_PATCHLEVEL[ \t]+([0-9]+)$" "\\1" SDL_IMAGE_VERSION_PATCH "${SDL_IMAGE_VERSION_PATCH_LINE}")
  set(SDL_image_VERSION ${SDL_IMAGE_VERSION_MAJOR}.${SDL_IMAGE_VERSION_MINOR}.${SDL_IMAGE_VERSION_PATCH})
  set(SDL_IMAGE_VERSION_STRING "${SDL_image_VERSION}")
  unset(SDL_IMAGE_VERSION_MAJOR_LINE)
  unset(SDL_IMAGE_VERSION_MINOR_LINE)
  unset(SDL_IMAGE_VERSION_PATCH_LINE)
  unset(SDL_IMAGE_VERSION_MAJOR)
  unset(SDL_IMAGE_VERSION_MINOR)
  unset(SDL_IMAGE_VERSION_PATCH)
endif()

set(SDL_IMAGE_LIBRARIES ${SDL_IMAGE_LIBRARY})
set(SDL_IMAGE_INCLUDE_DIRS ${SDL_IMAGE_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SDL_image
                                  REQUIRED_VARS SDL_IMAGE_LIBRARIES SDL_IMAGE_INCLUDE_DIRS
                                  VERSION_VAR SDL_image_VERSION)

# for backward compatibility
set(SDLIMAGE_LIBRARY ${SDL_IMAGE_LIBRARIES})
set(SDLIMAGE_INCLUDE_DIR ${SDL_IMAGE_INCLUDE_DIRS})
set(SDLIMAGE_FOUND ${SDL_image_FOUND})

mark_as_advanced(SDL_IMAGE_LIBRARY SDL_IMAGE_INCLUDE_DIR)

cmake_policy(POP)
