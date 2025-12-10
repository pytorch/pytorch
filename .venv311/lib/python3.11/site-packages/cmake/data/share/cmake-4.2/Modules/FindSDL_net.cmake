# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSDL_net
-----------

Finds the SDL_net library, a cross-platform network library for use with the
SDL (Simple DirectMedia Layer) applications:

.. code-block:: cmake

  find_package(SDL_net [<version>] [...])

.. note::

  This module is specifically intended for SDL_net version 1.  Starting with
  version 2.1, SDL_net provides a CMake package configuration file when built
  with CMake and should be found using ``find_package(SDL2_net)``.  These
  newer versions provide :ref:`Imported Targets` that encapsulate usage
  requirements.  Refer to the official SDL documentation for more information.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``SDL_net_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) SDL_net library was
  found.

``SDL_net_VERSION``
  .. versionadded:: 4.2

  The human-readable string containing the version of SDL_net found.

``SDL_NET_INCLUDE_DIRS``
  Include directories containing headers needed to use the SDL_net library.

``SDL_NET_LIBRARIES``
  Libraries needed to link against to use the SDL_net library.

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

``SDL_NET_VERSION_STRING``
  .. deprecated:: 4.2
    Use ``SDL_net_VERSION``, which has the same value.

  The human-readable string containing the version of SDL_net found.

``SDL_NET_FOUND``
  .. deprecated:: 4.2
    Use ``SDL_net_FOUND``, which has the same value.

``SDLNET_FOUND``
  .. deprecated:: 2.8.10
    Use ``SDL_net_FOUND``, which has the same value.

``SDLNET_INCLUDE_DIR``
  .. deprecated:: 2.8.10
    Use ``SDL_NET_INCLUDE_DIRS``, which has the same value.

``SDLNET_LIBRARY``
  .. deprecated:: 2.8.10
    Use ``SDL_NET_LIBRARIES``, which has the same value.

Examples
^^^^^^^^

Finding SDL_net library and creating an imported interface target for linking it
to a project target:

.. code-block:: cmake

  find_package(SDL_net)

  if(SDL_net_FOUND AND NOT TARGET SDL::SDL_net)
    add_library(SDL::SDL_net INTERFACE IMPORTED)
    set_target_properties(
      SDL::SDL_net
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${SDL_NET_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${SDL_NET_LIBRARIES}"
    )
  endif()

  target_link_libraries(project_target PRIVATE SDL::SDL_net)

When working with SDL_net version 2, the upstream package provides the
``SDL2_net::SDL2_net`` imported target directly.  It can be used in a project
without using this module:

.. code-block:: cmake

  find_package(SDL2_net)
  target_link_libraries(project_target PRIVATE SDL2_net::SDL2_net)

See Also
^^^^^^^^

* The :module:`FindSDL` module to find the main SDL library.
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

if(NOT SDL_NET_INCLUDE_DIR AND SDLNET_INCLUDE_DIR)
  set(SDL_NET_INCLUDE_DIR ${SDLNET_INCLUDE_DIR} CACHE PATH "directory cache
entry initialized from old variable name")
endif()
find_path(SDL_NET_INCLUDE_DIR SDL_net.h
  HINTS
    ENV SDLNETDIR
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

if(NOT SDL_NET_LIBRARY AND SDLNET_LIBRARY)
  set(SDL_NET_LIBRARY ${SDLNET_LIBRARY} CACHE FILEPATH "file cache entry
initialized from old variable name")
endif()
find_library(SDL_NET_LIBRARY
  NAMES SDL_net
  HINTS
    ENV SDLNETDIR
    ENV SDLDIR
  PATH_SUFFIXES lib ${VC_LIB_PATH_SUFFIX}
)

if(SDL_NET_INCLUDE_DIR AND EXISTS "${SDL_NET_INCLUDE_DIR}/SDL_net.h")
  file(STRINGS "${SDL_NET_INCLUDE_DIR}/SDL_net.h" SDL_NET_VERSION_MAJOR_LINE REGEX "^#define[ \t]+SDL_NET_MAJOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL_NET_INCLUDE_DIR}/SDL_net.h" SDL_NET_VERSION_MINOR_LINE REGEX "^#define[ \t]+SDL_NET_MINOR_VERSION[ \t]+[0-9]+$")
  file(STRINGS "${SDL_NET_INCLUDE_DIR}/SDL_net.h" SDL_NET_VERSION_PATCH_LINE REGEX "^#define[ \t]+SDL_NET_PATCHLEVEL[ \t]+[0-9]+$")
  string(REGEX REPLACE "^#define[ \t]+SDL_NET_MAJOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL_NET_VERSION_MAJOR "${SDL_NET_VERSION_MAJOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_NET_MINOR_VERSION[ \t]+([0-9]+)$" "\\1" SDL_NET_VERSION_MINOR "${SDL_NET_VERSION_MINOR_LINE}")
  string(REGEX REPLACE "^#define[ \t]+SDL_NET_PATCHLEVEL[ \t]+([0-9]+)$" "\\1" SDL_NET_VERSION_PATCH "${SDL_NET_VERSION_PATCH_LINE}")
  set(SDL_net_VERSION ${SDL_NET_VERSION_MAJOR}.${SDL_NET_VERSION_MINOR}.${SDL_NET_VERSION_PATCH})
  set(SDL_NET_VERSION_STRING "${SDL_net_VERSION}")
  unset(SDL_NET_VERSION_MAJOR_LINE)
  unset(SDL_NET_VERSION_MINOR_LINE)
  unset(SDL_NET_VERSION_PATCH_LINE)
  unset(SDL_NET_VERSION_MAJOR)
  unset(SDL_NET_VERSION_MINOR)
  unset(SDL_NET_VERSION_PATCH)
endif()

set(SDL_NET_LIBRARIES ${SDL_NET_LIBRARY})
set(SDL_NET_INCLUDE_DIRS ${SDL_NET_INCLUDE_DIR})

include(FindPackageHandleStandardArgs)

find_package_handle_standard_args(SDL_net
                                  REQUIRED_VARS SDL_NET_LIBRARIES SDL_NET_INCLUDE_DIRS
                                  VERSION_VAR SDL_net_VERSION)

# for backward compatibility
set(SDLNET_LIBRARY ${SDL_NET_LIBRARIES})
set(SDLNET_INCLUDE_DIR ${SDL_NET_INCLUDE_DIRS})
set(SDLNET_FOUND ${SDL_net_FOUND})

mark_as_advanced(SDL_NET_LIBRARY SDL_NET_INCLUDE_DIR)

cmake_policy(POP)
