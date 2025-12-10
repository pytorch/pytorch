# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgTerrain
--------------

Finds the osgTerrain NodeKit from the OpenSceneGraph toolkit.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgTerrain as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgTerrain)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgTerrain NodeKit.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgTerrain explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgTerrain)

OpenSceneGraph and osgTerrain headers are intended to be included in C++ project
source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgTerrain/Terrain>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgTerrain_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgTerrain NodeKit of the OpenSceneGraph
  toolkit was found.

``OSGTERRAIN_LIBRARIES``
  The libraries needed to link against to use osgTerrain.

``OSGTERRAIN_LIBRARY``
  A result variable that is set to the same value as the
  ``OSGTERRAIN_LIBRARIES`` variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGTERRAIN_INCLUDE_DIR``
  The include directory containing headers needed to use osgTerrain.

``OSGTERRAIN_LIBRARY_DEBUG``
  The path to the osgTerrain debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgTerrain NodeKit, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGTERRAIN_FOUND``
  .. deprecated:: 4.2
    Use ``osgTerrain_FOUND``, which has the same value.

  Boolean indicating whether the osgTerrain NodeKit of the OpenSceneGraph
  toolkit was found.

Examples
^^^^^^^^

Finding osgTerrain explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgTerrain)

  if(osgTerrain_FOUND AND NOT TARGET osgTerrain::osgTerrain)
    add_library(osgTerrain::osgTerrain INTERFACE IMPORTED)
    set_target_properties(
      osgTerrain::osgTerrain
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGTERRAIN_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGTERRAIN_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgTerrain::osgTerrain)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGTERRAIN osgTerrain/Terrain)
OSG_FIND_LIBRARY(OSGTERRAIN osgTerrain)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgTerrain DEFAULT_MSG
    OSGTERRAIN_LIBRARY OSGTERRAIN_INCLUDE_DIR)
