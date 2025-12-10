# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindPhysFS
----------

Finds the PhysicsFS library (PhysFS) for file I/O abstraction:

.. code-block:: cmake

  find_package(PhysFS [...])

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``PhysFS_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the PhysicsFS library was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``PHYSFS_INCLUDE_DIR``
  Directory containing the ``<physfs.h>`` and related headers needed for using
  the library.
``PHYSFS_LIBRARY``
  Path to the PhysicsFS library needed to link against.

Hints
^^^^^

This module accepts the following variables:

``PHYSFSDIR``
  Environment variable that can be set to help locate a PhysicsFS library
  installed in a custom location.  It should point to the installation
  destination that was used when configuring, building, and installing PhysicsFS
  library: ``./configure --prefix=$PHYSFSDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``PHYSFS_FOUND``
  .. deprecated:: 4.2
    Use the ``PhysFS_FOUND``, which has the same value.

  Boolean indicating whether the PhysicsFS library was found.

Examples
^^^^^^^^

Finding the PhysicsFS library:

.. code-block:: cmake

  find_package(PhysFS)
#]=======================================================================]

find_path(PHYSFS_INCLUDE_DIR physfs.h
  HINTS
    ENV PHYSFSDIR
  PATH_SUFFIXES include/physfs include
  PATHS
  ~/Library/Frameworks
  /Library/Frameworks
  /opt
)

find_library(PHYSFS_LIBRARY
  NAMES physfs
  HINTS
    ENV PHYSFSDIR
  PATH_SUFFIXES lib
  PATHS
  ~/Library/Frameworks
  /Library/Frameworks
  /opt
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(PhysFS DEFAULT_MSG PHYSFS_LIBRARY PHYSFS_INCLUDE_DIR)
