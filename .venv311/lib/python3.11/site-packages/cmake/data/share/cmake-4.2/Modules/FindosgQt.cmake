# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgQt
---------

Finds the osgQt utility library from the OpenSceneGraph toolkit.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgQt as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgQt)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgQt library.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgQt explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgQt)

OpenSceneGraph and osgQt headers are intended to be included in C++ project
source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgQt/GraphicsWindowQt>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgQt_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgQt library of the OpenSceneGraph toolkit
  was found.

``OSGQT_LIBRARIES``
  The libraries needed to link against to use osgQt.

``OSGQT_LIBRARY``
  A result variable that is set to the same value as the ``OSGQT_LIBRARIES``
  variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGQT_INCLUDE_DIR``
  The include directory containing headers needed to use osgQt.

``OSGQT_LIBRARY_DEBUG``
  The path to the osgQt debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgQt library, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGQT_FOUND``
  .. deprecated:: 4.2
    Use ``osgQt_FOUND``, which has the same value.

  Boolean indicating whether the osgQt library of the OpenSceneGraph toolkit
  was found.

Examples
^^^^^^^^

Finding osgQt explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgQt)

  if(osgQt_FOUND AND NOT TARGET osgQt::osgQt)
    add_library(osgQt::osgQt INTERFACE IMPORTED)
    set_target_properties(
      osgQt::osgQt
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGQT_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGQT_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgQt::osgQt)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.
# Modified to work with osgQt by Robert Osfield, January 2012.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGQT osgQt/GraphicsWindowQt)
OSG_FIND_LIBRARY(OSGQT osgQt)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgQt DEFAULT_MSG
    OSGQT_LIBRARY OSGQT_INCLUDE_DIR)
