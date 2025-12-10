# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgSim
----------

Finds the osgSim NodeKit from the OpenSceneGraph toolkit.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgSim as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgSim)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgSim NodeKit.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgSim explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgSim)

OpenSceneGraph and osgSim headers are intended to be included in C++ project
source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgSim/ImpostorSprite>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgSim_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgSim NodeKit of the OpenSceneGraph
  toolkit was found.

``OSGSIM_LIBRARIES``
  The libraries needed to link against to use osgSim.

``OSGSIM_LIBRARY``
  A result variable that is set to the same value as the ``OSGSIM_LIBRARIES``
  variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGSIM_INCLUDE_DIR``
  The include directory containing headers needed to use osgSim.

``OSGSIM_LIBRARY_DEBUG``
  The path to the osgSim debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgSim NodeKit, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGSIM_FOUND``
  .. deprecated:: 4.2
    Use ``osgSim_FOUND``, which has the same value.

  Boolean indicating whether the osgSim NodeKit of the OpenSceneGraph
  toolkit was found.

Examples
^^^^^^^^

Finding osgSim explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgSim)

  if(osgSim_FOUND AND NOT TARGET osgSim::osgSim)
    add_library(osgSim::osgSim INTERFACE IMPORTED)
    set_target_properties(
      osgSim::osgSim
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGSIM_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGSIM_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgSim::osgSim)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGSIM osgSim/ImpostorSprite)
OSG_FIND_LIBRARY(OSGSIM osgSim)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgSim DEFAULT_MSG
    OSGSIM_LIBRARY OSGSIM_INCLUDE_DIR)
