# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgViewer
-------------

Finds the osgViewer library from the OpenSceneGraph toolkit.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgViewer as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgViewer)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgViewer library.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgViewer explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgViewer)

OpenSceneGraph and osgViewer headers are intended to be included in C++ project
source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgViewer/Viewer>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgViewer_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgViewer library of the OpenSceneGraph
  toolkit was found.

``OSGVIEWER_LIBRARIES``
  The libraries needed to link against to use osgViewer.

``OSGVIEWER_LIBRARY``
  A result variable that is set to the same value as the ``OSGVIEWER_LIBRARIES``
  variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGVIEWER_INCLUDE_DIR``
  The include directory containing headers needed to use osgViewer.

``OSGVIEWER_LIBRARY_DEBUG``
  The path to the osgViewer debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgViewer library, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGVIEWER_FOUND``
  .. deprecated:: 4.2
    Use ``osgViewer_FOUND``, which has the same value.

  Boolean indicating whether the osgViewer library of the OpenSceneGraph
  toolkit was found.

Examples
^^^^^^^^

Finding osgViewer explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgViewer)

  if(osgViewer_FOUND AND NOT TARGET osgViewer::osgViewer)
    add_library(osgViewer::osgViewer INTERFACE IMPORTED)
    set_target_properties(
      osgViewer::osgViewer
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGVIEWER_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGVIEWER_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgViewer::osgViewer)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGVIEWER osgViewer/Viewer)
OSG_FIND_LIBRARY(OSGVIEWER osgViewer)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgViewer DEFAULT_MSG
    OSGVIEWER_LIBRARY OSGVIEWER_INCLUDE_DIR)
