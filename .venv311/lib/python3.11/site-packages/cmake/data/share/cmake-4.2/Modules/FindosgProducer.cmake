# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgProducer
---------------

Finds the osgProducer utility library from the OpenSceneGraph toolkit.

.. note::

  The osgProducer library has been removed from the OpenSceneGraph toolkit in
  early OpenSceneGraph versions (pre 1.0 release) and replaced with osgViewer.
  Its development has shifted at time to a standalone project and repository
  Producer, which can be found with :module:`FindProducer` module.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgProducer as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgProducer)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgProducer library.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgProducer explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgProducer)

OpenSceneGraph and osgProducer headers are intended to be included in C++
project source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgProducer/OsgSceneHandler>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgProducer_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgProducer library of the OpenSceneGraph
  toolkit was found.

``OSGPRODUCER_LIBRARIES``
  The libraries needed to link against to use osgProducer.

``OSGPRODUCER_LIBRARY``
  A result variable that is set to the same value as the
  ``OSGPRODUCER_LIBRARIES`` variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGPRODUCER_INCLUDE_DIR``
  The include directory containing headers needed to use osgProducer.

``OSGPRODUCER_LIBRARY_DEBUG``
  The path to the osgProducer debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgProducer library, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGPRODUCER_FOUND``
  .. deprecated:: 4.2
    Use ``osgProducer_FOUND``, which has the same value.

  Boolean indicating whether the osgProducer library of the OpenSceneGraph
  toolkit was found.

Examples
^^^^^^^^

Finding osgProducer explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgProducer)

  if(osgProducer_FOUND AND NOT TARGET osgProducer::osgProducer)
    add_library(osgProducer::osgProducer INTERFACE IMPORTED)
    set_target_properties(
      osgProducer::osgProducer
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGPRODUCER_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGPRODUCER_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgProducer::osgProducer)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
* The :module:`FindProducer` module, which finds the standalone Producer library
  that evolved from the legacy osgProducer.
#]=======================================================================]

# Created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGPRODUCER osgProducer/OsgSceneHandler)
OSG_FIND_LIBRARY(OSGPRODUCER osgProducer)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgProducer DEFAULT_MSG
    OSGPRODUCER_LIBRARY OSGPRODUCER_INCLUDE_DIR)
