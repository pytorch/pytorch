# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindMPEG
--------

Finds the native MPEG library (libmpeg2):

.. code-block:: cmake

  find_package(MPEG [...])

.. note::

  This module is functionally identical to the :module:`FindMPEG2` module, which
  also finds the libmpeg2 library.  Both modules were introduced in the past to
  provide flexibility in handling potential differences in future versions of
  the MPEG library and to maintain backward compatibility across CMake releases.

  The ``FindMPEG2`` module additionally checks for the SDL dependency and
  includes it in its usage requirements.  For working with libmpeg2, it is
  recommended to use the :module:`FindMPEG2` module instead of this one.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``MPEG_FOUND``
  Boolean indicating whether the libmpeg2 library was found.
``MPEG_LIBRARIES``
  Libraries needed to link against to use libmpeg2.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``MPEG_INCLUDE_DIR``
  The directory containing the ``mpeg2.h`` and related headers needed to use
  libmpeg2 library.
``MPEG_mpeg2_LIBRARY``
  The path to the libmpeg2 library.
``MPEG_vo_LIBRARY``
  The path to the vo (Video Out) library.

Examples
^^^^^^^^

Finding libmpeg2 library and creating an imported interface target for linking
it to a project target:

.. code-block:: cmake

  find_package(MPEG)

  if(MPEG_FOUND AND NOT TARGET MPEG::MPEG)
    add_library(MPEG::MPEG INTERFACE IMPORTED)
    set_target_properties(
      MPEG::MPEG
      PROPERTIES
        INTERFACE_LINK_LIBRARIES "${MPEG_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${MPEG_INCLUDE_DIR}"
    )
  endif()

  target_link_libraries(project_target PRIVATE MPEG::MPEG)
#]=======================================================================]

find_path(MPEG_INCLUDE_DIR
  NAMES mpeg2.h mpeg2dec/mpeg2.h mpeg2dec/include/video_out.h)

find_library(MPEG_mpeg2_LIBRARY mpeg2)

find_library(MPEG_vo_LIBRARY vo)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MPEG DEFAULT_MSG MPEG_mpeg2_LIBRARY MPEG_INCLUDE_DIR)

if(MPEG_FOUND)
  set( MPEG_LIBRARIES ${MPEG_mpeg2_LIBRARY} )
  if(MPEG_vo_LIBRARY)
    list(APPEND MPEG2_LIBRARIES ${MPEG_vo_LIBRARY})
  endif()
endif()

mark_as_advanced(MPEG_INCLUDE_DIR MPEG_mpeg2_LIBRARY MPEG_vo_LIBRARY)
