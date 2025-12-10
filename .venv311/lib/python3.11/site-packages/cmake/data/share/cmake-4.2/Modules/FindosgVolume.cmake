# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgVolume
-------------

Finds the osgVolume NodeKit from the OpenSceneGraph toolkit.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgVolume as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgVolume)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgVolume NodeKit.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgVolume explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgVolume)

OpenSceneGraph and osgVolume headers are intended to be included in C++ project
source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgVolume/Volume>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgVolume_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgVolume NodeKit of the OpenSceneGraph
  toolkit was found.

``OSGVOLUME_LIBRARIES``
  The libraries needed to link against to use osgVolume.

``OSGVOLUME_LIBRARY``
  A result variable that is set to the same value as the ``OSGVOLUME_LIBRARIES``
  variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGVOLUME_INCLUDE_DIR``
  The include directory containing headers needed to use osgVolume.

``OSGVOLUME_LIBRARY_DEBUG``
  The path to the osgVolume debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgVolume NodeKit, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGVOLUME_FOUND``
  .. deprecated:: 4.2
    Use ``osgVolume_FOUND``, which has the same value.

  Boolean indicating whether the osgVolume NodeKit of the OpenSceneGraph
  toolkit was found.

Examples
^^^^^^^^

Finding osgVolume explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgVolume)

  if(osgVolume_FOUND AND NOT TARGET osgVolume::osgVolume)
    add_library(osgVolume::osgVolume INTERFACE IMPORTED)
    set_target_properties(
      osgVolume::osgVolume
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGVOLUME_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGVOLUME_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgVolume::osgVolume)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGVOLUME osgVolume/Volume)
OSG_FIND_LIBRARY(OSGVOLUME osgVolume)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgVolume DEFAULT_MSG
    OSGVOLUME_LIBRARY OSGVOLUME_INCLUDE_DIR)
