# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgUtil
-----------

Finds the osgUtil library from the OpenSceneGraph toolkit.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgUtil as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgUtil)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgUtil library.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgUtil explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgUtil)

OpenSceneGraph and osgUtil headers are intended to be included in C++ project
source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgUtil/SceneView>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgUtil_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgUtil library of the OpenSceneGraph
  toolkit was found.

``OSGUTIL_LIBRARIES``
  The libraries needed to link against to use osgUtil.

``OSGUTIL_LIBRARY``
  A result variable that is set to the same value as the ``OSGUTIL_LIBRARIES``
  variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGUTIL_INCLUDE_DIR``
  The include directory containing headers needed to use osgUtil.

``OSGUTIL_LIBRARY_DEBUG``
  The path to the osgUtil debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgUtil library, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGUTIL_FOUND``
  .. deprecated:: 4.2
    Use ``osgUtil_FOUND``, which has the same value.

  Boolean indicating whether the osgUtil library of the OpenSceneGraph
  toolkit was found.

Examples
^^^^^^^^

Finding osgUtil explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgUtil)

  if(osgUtil_FOUND AND NOT TARGET osgUtil::osgUtil)
    add_library(osgUtil::osgUtil INTERFACE IMPORTED)
    set_target_properties(
      osgUtil::osgUtil
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGUTIL_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGUTIL_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgUtil::osgUtil)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGUTIL osgUtil/SceneView)
OSG_FIND_LIBRARY(OSGUTIL osgUtil)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgUtil DEFAULT_MSG
    OSGUTIL_LIBRARY OSGUTIL_INCLUDE_DIR)
