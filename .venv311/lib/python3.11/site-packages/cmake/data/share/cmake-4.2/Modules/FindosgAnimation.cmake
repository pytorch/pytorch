# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgAnimation
----------------

Finds the osgAnimation library from the OpenSceneGraph toolkit.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgAnimation as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgAnimation)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgAnimation library.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgAnimation explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgAnimation)

OpenSceneGraph and osgAnimation headers are intended to be included in C++
project source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgAnimation/Animation>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgAnimation_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgAnimation library of the OpenSceneGraph
  toolkit was found.

``OSGANIMATION_LIBRARIES``
  The libraries needed to link against to use osgAnimation.

``OSGANIMATION_LIBRARY``
  A result variable that is set to the same value as the
  ``OSGANIMATION_LIBRARIES`` variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGANIMATION_INCLUDE_DIR``
  The include directory containing headers needed to use osgAnimation.

``OSGANIMATION_LIBRARY_DEBUG``
  The path to the osgAnimation debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgAnimation library, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGANIMATION_FOUND``
  .. deprecated:: 4.2
    Use ``osgAnimation_FOUND``, which has the same value.

  Boolean indicating whether the osgAnimation library of the OpenSceneGraph
  toolkit was found.

Examples
^^^^^^^^

Finding the osgAnimation library explicitly with this module and creating an
interface :ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgAnimation)

  if(osgAnimation_FOUND AND NOT TARGET osgAnimation::osgAnimation)
    add_library(osgAnimation::osgAnimation INTERFACE IMPORTED)
    set_target_properties(
      osgAnimation::osgAnimation
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGANIMATION_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGANIMATION_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgAnimation::osgAnimation)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGANIMATION osgAnimation/Animation)
OSG_FIND_LIBRARY(OSGANIMATION osgAnimation)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgAnimation DEFAULT_MSG
    OSGANIMATION_LIBRARY OSGANIMATION_INCLUDE_DIR)
