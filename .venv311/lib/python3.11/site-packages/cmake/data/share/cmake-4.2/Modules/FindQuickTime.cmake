# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindQuickTime
-------------

Finds the QuickTime multimedia framework, which provides support for video,
audio, and interactive media:

.. code-block:: cmake

  find_package(QuickTime [...])

.. note::

  This module is for the QuickTime framework, which has been deprecated by Apple
  and is no longer supported.  On Apple systems, use AVFoundation and AVKit
  instead.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``QuickTime_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether QuickTime was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``QUICKTIME_LIBRARY``
  The path to the QuickTime library.

``QUICKTIME_INCLUDE_DIR``
  Directory containing QuickTime headers.

Hints
^^^^^

This module accepts the following variables:

``QUICKTIME_DIR``
  Environment variable that can be set to help locate a QuickTime library
  installed in a custom location.  It should point to the installation
  destination that was used when configuring, building, and installing QuickTime
  library: ``./configure --prefix=$QUICKTIME_DIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``QUICKTIME_FOUND``
  .. deprecated:: 4.2
    Use ``QuickTime_FOUND``, which has the same value.

  Boolean indicating whether QuickTime was found.

Examples
^^^^^^^^

Finding QuickTime library and creating an imported interface target for
linking it to a project target:

.. code-block:: cmake

  find_package(QuickTime)

  if(QuickTime_FOUND AND NOT TARGET QuickTime::QuickTime)
    add_library(QuickTime::QuickTime INTERFACE IMPORTED)
    set_target_properties(
      QuickTime::QuickTime
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${QUICKTIME_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${QUICKTIME_LIBRARY}"
    )
  endif()

  target_link_libraries(example PRIVATE QuickTime::QuickTime)
#]=======================================================================]

find_path(QUICKTIME_INCLUDE_DIR QuickTime/QuickTime.h QuickTime.h
  HINTS
    ENV QUICKTIME_DIR
  PATH_SUFFIXES
    include
)
find_library(QUICKTIME_LIBRARY QuickTime
  HINTS
    ENV QUICKTIME_DIR
  PATH_SUFFIXES
    lib
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(QuickTime DEFAULT_MSG QUICKTIME_LIBRARY QUICKTIME_INCLUDE_DIR)
