# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindProducer
------------

.. note::

  Producer (also known as *Open Producer*) library originated from the
  osgProducer utility library in early versions of the OpenSceneGraph toolkit
  and was later developed into a standalone library.  The osgProducer was
  eventually replaced by the osgViewer library, and the standalone Producer
  library became obsolete and is no longer maintained.  For details about
  OpenSceneGraph usage, refer to the :module:`FindOpenSceneGraph` module.

Finds the Producer library, a windowing and event handling library designed
primarily for real-time graphics applications:

.. code-block:: cmake

  find_package(Producer [...])

Producer library headers are intended to be included in C++ project source code
as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <Producer/CameraGroup>

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Producer_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether Producer was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``PRODUCER_INCLUDE_DIR``
  The include directory containing headers needed to use Producer.

``PRODUCER_LIBRARY``
  The path to the Producer library needed to link against for usage.

Hints
^^^^^

This module accepts the following variables:

``PRODUCER_DIR``
  Environment variable that can be set to help locate a custom installation of
  the Producer library.  It should point to the root directory where the
  Producer library was installed.  This should match the installation prefix
  used when configuring and building Producer, such as with
  ``./configure --prefix=$PRODUCER_DIR``.

Because Producer was historically tightly integrated with OpenSceneGraph, this
module also accepts the following environment variables as equivalents to
``PRODUCER_DIR`` for convenience to specify common installation root for
multiple OpenSceneGraph-related libraries at once:

``OSGDIR``
  Environment variable treated the same as ``PRODUCER_DIR``.

``OSG_DIR``
  Environment variable treated the same as ``PRODUCER_DIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``PRODUCER_FOUND``
  .. deprecated:: 4.2
    Use ``Producer_FOUND``, which has the same value.

  Boolean indicating whether Producer was found.

Examples
^^^^^^^^

Finding the Producer library and creating an :ref:`imported target
<Imported Targets>` that encapsulates its usage requirements for linking to a
project target:

.. code-block:: cmake

  find_package(Producer)

  if(Producer_FOUND AND NOT TARGET Producer::Producer)
    add_library(Producer::Producer INTERFACE IMPORTED)
    set_target_properties(
      Producer::Producer
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${PRODUCER_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${PRODUCER_LIBRARY}"
    )
  endif()

  target_link_libraries(example PRIVATE Producer::Producer)
#]=======================================================================]

# Try the user's environment request before anything else.
find_path(PRODUCER_INCLUDE_DIR Producer/CameraGroup
  HINTS
    ENV PRODUCER_DIR
    ENV OSG_DIR
    ENV OSGDIR
  PATH_SUFFIXES include
  PATHS
    ~/Library/Frameworks
    /Library/Frameworks
    /opt
    [HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session\ Manager\\Environment;OpenThreads_ROOT]
    [HKEY_LOCAL_MACHINE\\SYSTEM\\CurrentControlSet\\Control\\Session\ Manager\\Environment;OSG_ROOT]
)

find_library(PRODUCER_LIBRARY
  NAMES Producer
  HINTS
    ENV PRODUCER_DIR
    ENV OSG_DIR
    ENV OSGDIR
  PATH_SUFFIXES lib
  PATHS
  /opt
)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Producer DEFAULT_MSG
    PRODUCER_LIBRARY PRODUCER_INCLUDE_DIR)
