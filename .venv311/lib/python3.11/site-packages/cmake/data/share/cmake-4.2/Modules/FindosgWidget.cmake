# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgWidget
-------------

Finds the osgWidget NodeKit from the OpenSceneGraph toolkit.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgWidget as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgWidget)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgWidget NodeKit.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgWidget explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgWidget)

OpenSceneGraph and osgWidget headers are intended to be included in C++ project
source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgWidget/Widget>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgWidget_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgWidget NodeKit of the OpenSceneGraph
  toolkit was found.

``OSGWIDGET_LIBRARIES``
  The libraries needed to link against to use osgWidget.

``OSGWIDGET_LIBRARY``
  A result variable that is set to the same value as the ``OSGWIDGET_LIBRARIES``
  variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGWIDGET_INCLUDE_DIR``
  The include directory containing headers needed to use osgWidget.

``OSGWIDGET_LIBRARY_DEBUG``
  The path to the osgWidget debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgWidget NodeKit, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGWIDGET_FOUND``
  .. deprecated:: 4.2
    Use ``osgWidget_FOUND``, which has the same value.

  Boolean indicating whether the osgWidget NodeKit of the OpenSceneGraph
  toolkit was found.

Examples
^^^^^^^^

Finding osgWidget explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgWidget)

  if(osgWidget_FOUND AND NOT TARGET osgWidget::osgWidget)
    add_library(osgWidget::osgWidget INTERFACE IMPORTED)
    set_target_properties(
      osgWidget::osgWidget
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGWIDGET_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGWIDGET_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgWidget::osgWidget)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# FindosgWidget.cmake tweaked from Findosg* suite as created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGWIDGET osgWidget/Widget)
OSG_FIND_LIBRARY(OSGWIDGET osgWidget)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgWidget DEFAULT_MSG
    OSGWIDGET_LIBRARY OSGWIDGET_INCLUDE_DIR)
