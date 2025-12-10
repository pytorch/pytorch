# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgPresentation
-------------------

Finds the osgPresentation NodeKit from the OpenSceneGraph toolkit, available
since OpenSceneGraph version 3.0.0.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgPresentation as a component.  This will automatically
  handle dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgPresentation)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgPresentation NodeKit.  It is not intended to be included directly during
typical use of the :command:`find_package` command.  However, it is available as
a standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgPresentation explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgPresentation)

OpenSceneGraph and osgPresentation headers are intended to be included in C++
project source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgPresentation/SlideEventHandler>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgPresentation_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgPresentation NodeKit of the
  OpenSceneGraph toolkit was found.

``OSGPRESENTATION_LIBRARIES``
  The libraries needed to link against to use osgPresentation.

``OSGPRESENTATION_LIBRARY``
  A result variable that is set to the same value as the
  ``OSGPRESENTATION_LIBRARIES`` variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGPRESENTATION_INCLUDE_DIR``
  The include directory containing headers needed to use osgPresentation.

``OSGPRESENTATION_LIBRARY_DEBUG``
  The path to the osgPresentation debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgPresentation NodeKit, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGPRESENTATION_FOUND``
  .. deprecated:: 4.2
    Use ``osgPresentation_FOUND``, which has the same value.

  Boolean indicating whether the osgPresentation NodeKit of the
  OpenSceneGraph toolkit was found.

Examples
^^^^^^^^

Finding osgPresentation explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgPresentation)

  if(osgPresentation_FOUND AND NOT TARGET osgPresentation::osgPresentation)
    add_library(osgPresentation::osgPresentation INTERFACE IMPORTED)
    set_target_properties(
      osgPresentation::osgPresentation
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGPRESENTATION_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGPRESENTATION_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgPresentation::osgPresentation)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.
# Modified to work with osgPresentation by Robert Osfield, January 2012.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGPRESENTATION osgPresentation/SlideEventHandler)
OSG_FIND_LIBRARY(OSGPRESENTATION osgPresentation)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgPresentation DEFAULT_MSG
    OSGPRESENTATION_LIBRARY OSGPRESENTATION_INCLUDE_DIR)
