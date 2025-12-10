# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgParticle
---------------

Finds the osgParticle NodeKit from the OpenSceneGraph toolkit.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgParticle as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgParticle)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgParticle NodeKit.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgParticle explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgParticle)

OpenSceneGraph and osgParticle headers are intended to be included in C++
project source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgParticle/FireEffect>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgParticle_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgParticle NodeKit of the OpenSceneGraph
  toolkit was found.

``OSGPARTICLE_LIBRARIES``
  The libraries needed to link against to use the osgParticle NodeKit

``OSGPARTICLE_LIBRARY``
  A result variable that is set to the same value as the
  ``OSGPARTICLE_LIBRARIES`` variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGPARTICLE_INCLUDE_DIR``
  The include directory containing headers needed to use osgParticle NodeKit.

``OSGPARTICLE_LIBRARY_DEBUG``
  The path to the osgParticle debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgParticle NodeKit, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGPARTICLE_FOUND``
  .. deprecated:: 4.2
    Use ``osgParticle_FOUND``, which has the same value.

  Boolean indicating whether the osgParticle NodeKit of the OpenSceneGraph
  toolkit was found.

Examples
^^^^^^^^

Finding osgParticle explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgParticle)

  if(osgParticle_FOUND AND NOT TARGET osgParticle::osgParticle)
    add_library(osgParticle::osgParticle INTERFACE IMPORTED)
    set_target_properties(
      osgParticle::osgParticle
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGPARTICLE_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGPARTICLE_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgParticle::osgParticle)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGPARTICLE osgParticle/FireEffect)
OSG_FIND_LIBRARY(OSGPARTICLE osgParticle)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgParticle DEFAULT_MSG
    OSGPARTICLE_LIBRARY OSGPARTICLE_INCLUDE_DIR)
