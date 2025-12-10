# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindGIF
-------

Finds the Graphics Interchange Format (GIF) library (``giflib``):

.. code-block:: cmake

  find_package(GIF [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``GIF::GIF``
  .. versionadded:: 3.14

  Target that encapsulates the usage requirements of the GIF library, available
  when the library is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``GIF_FOUND``
  Boolean indicating whether the (requested version of) GIF library was found.

``GIF_VERSION``
  Version string of the GIF library found (for example, ``5.1.4``).  For GIF
  library versions prior to 4.1.6, version string will be set only to ``3`` or
  ``4`` as these versions did not provide version information in their headers.

``GIF_INCLUDE_DIRS``
  Include directories needed to use the GIF library.

``GIF_LIBRARIES``
  Libraries needed to link to the GIF library.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``GIF_INCLUDE_DIR``
  Directory containing the ``gif_lib.h`` and other GIF library headers.

``GIF_LIBRARY``
  Path to the GIF library.

Hints
^^^^^

This module accepts the following variables:

``GIF_DIR``
  Environment variable that can be set to help locate a GIF library installed in
  a custom location.  It should point to the installation destination that was
  used when configuring, building, and installing GIF library:
  ``./configure --prefix=$GIF_DIR``.

Examples
^^^^^^^^

Finding GIF library and linking it to a project target:

.. code-block:: cmake

  find_package(GIF)
  target_link_libraries(project_target PRIVATE GIF::GIF)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

# Created by Eric Wing.
# Modifications by Alexander Neundorf, Ben Campbell

find_path(GIF_INCLUDE_DIR gif_lib.h
  HINTS
    ENV GIF_DIR
  PATH_SUFFIXES include
)

# the gif library can have many names :-/
set(POTENTIAL_GIF_LIBS gif libgif ungif libungif giflib giflib4)

find_library(GIF_LIBRARY
  NAMES ${POTENTIAL_GIF_LIBS}
  NAMES_PER_DIR
  HINTS
    ENV GIF_DIR
  PATH_SUFFIXES lib
)

# Very basic version detection.
# The GIF_LIB_VERSION string in gif_lib.h seems to be unreliable, since it seems
# to be always " Version 2.0, " in versions 3.x of giflib.
# In version 4 the member UserData was added to GifFileType, so we check for this
# one.
# Versions after 4.1.6 define GIFLIB_MAJOR, GIFLIB_MINOR, and GIFLIB_RELEASE
# see http://giflib.sourceforge.net/gif_lib.html#compatibility
if(GIF_INCLUDE_DIR)
  include(${CMAKE_CURRENT_LIST_DIR}/CMakePushCheckState.cmake)
  include(${CMAKE_CURRENT_LIST_DIR}/CheckStructHasMember.cmake)
  cmake_push_check_state()
  set(CMAKE_REQUIRED_QUIET ${GIF_FIND_QUIETLY})
  set(CMAKE_REQUIRED_INCLUDES "${GIF_INCLUDE_DIR}")

  # Check for the specific version defines (>=4.1.6 only)
  file(STRINGS ${GIF_INCLUDE_DIR}/gif_lib.h _GIF_DEFS REGEX "^[ \t]*#define[ \t]+GIFLIB_(MAJOR|MINOR|RELEASE)")
  if(_GIF_DEFS)
    # yay - got exact version info
    string(REGEX REPLACE ".*GIFLIB_MAJOR ([0-9]+).*" "\\1" _GIF_MAJ "${_GIF_DEFS}")
    string(REGEX REPLACE ".*GIFLIB_MINOR ([0-9]+).*" "\\1" _GIF_MIN "${_GIF_DEFS}")
    string(REGEX REPLACE ".*GIFLIB_RELEASE ([0-9]+).*" "\\1" _GIF_REL "${_GIF_DEFS}")
    set(GIF_VERSION "${_GIF_MAJ}.${_GIF_MIN}.${_GIF_REL}")
  else()
    # use UserData field to sniff version instead
    check_struct_has_member(GifFileType UserData gif_lib.h GIF_GifFileType_UserData )
    if(GIF_GifFileType_UserData)
      set(GIF_VERSION 4)
    else()
      set(GIF_VERSION 3)
    endif()
  endif()

  unset(_GIF_MAJ)
  unset(_GIF_MIN)
  unset(_GIF_REL)
  unset(_GIF_DEFS)
  cmake_pop_check_state()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(GIF  REQUIRED_VARS  GIF_LIBRARY  GIF_INCLUDE_DIR
                                       VERSION_VAR GIF_VERSION )

if(GIF_FOUND)
  set(GIF_INCLUDE_DIRS "${GIF_INCLUDE_DIR}")
  set(GIF_LIBRARIES ${GIF_LIBRARY})

  if(NOT TARGET GIF::GIF)
    add_library(GIF::GIF UNKNOWN IMPORTED)
    set_target_properties(GIF::GIF PROPERTIES
      INTERFACE_INCLUDE_DIRECTORIES "${GIF_INCLUDE_DIRS}")
    if(EXISTS "${GIF_LIBRARY}")
      set_target_properties(GIF::GIF PROPERTIES
        IMPORTED_LINK_INTERFACE_LANGUAGES "C"
        IMPORTED_LOCATION "${GIF_LIBRARY}")
    endif()
  endif()
endif()

mark_as_advanced(GIF_INCLUDE_DIR GIF_LIBRARY)

cmake_policy(POP)
