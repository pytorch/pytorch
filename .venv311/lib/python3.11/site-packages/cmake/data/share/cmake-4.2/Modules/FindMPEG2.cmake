# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindMPEG2
---------

Finds the native MPEG2 library (libmpeg2):

.. code-block:: cmake

  find_package(MPEG2 [...])

.. note::

  Depending on how the native libmpeg2 library is built and installed, it may
  depend on the SDL (Simple DirectMedia Layer) library.  If SDL is found, this
  module includes it in its usage requirements when used.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``MPEG2_FOUND``
  Boolean indicating whether the libmpeg2 library was found.
``MPEG2_LIBRARIES``
  Libraries needed to link against to use libmpeg2.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``MPEG2_INCLUDE_DIR``
  The directory containing the ``mpeg2.h`` and related headers needed to use
  libmpeg2 library.
``MPEG2_mpeg2_LIBRARY``
  The path to the libmpeg2 library.
``MPEG2_vo_LIBRARY``
  The path to the vo (Video Out) library.

Examples
^^^^^^^^

Finding libmpeg2 library and creating an imported interface target for linking
it to a project target:

.. code-block:: cmake

  find_package(MPEG2)

  if(MPEG2_FOUND AND NOT TARGET MPEG2::MPEG2)
    add_library(MPEG2::MPEG2 INTERFACE IMPORTED)
    set_target_properties(
      MPEG2::MPEG2
      PROPERTIES
        INTERFACE_LINK_LIBRARIES "${MPEG2_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${MPEG2_INCLUDE_DIR}"
    )
  endif()

  target_link_libraries(project_target PRIVATE MPEG2::MPEG2)
#]=======================================================================]

find_path(MPEG2_INCLUDE_DIR
  NAMES mpeg2.h mpeg2dec/mpeg2.h)

find_library(MPEG2_mpeg2_LIBRARY mpeg2)

find_library(MPEG2_vo_LIBRARY vo)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPEG2 DEFAULT_MSG MPEG2_mpeg2_LIBRARY MPEG2_INCLUDE_DIR)

if(MPEG2_FOUND)
  set(MPEG2_LIBRARIES ${MPEG2_mpeg2_LIBRARY})
  if(MPEG2_vo_LIBRARY)
    list(APPEND MPEG2_LIBRARIES ${MPEG2_vo_LIBRARY})
  endif()

  # Some native mpeg2 installations will depend on libSDL. If found, add it in.
  find_package(SDL)
  if(SDL_FOUND)
    set( MPEG2_LIBRARIES ${MPEG2_LIBRARIES} ${SDL_LIBRARY})
  endif()
endif()

mark_as_advanced(MPEG2_INCLUDE_DIR MPEG2_mpeg2_LIBRARY MPEG2_vo_LIBRARY)
