# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgGA
---------

Finds the osgGA library from the OpenSceneGraph toolkit.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgGA as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgGA)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgGA library.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgGA explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgGA)

OpenSceneGraph and osgGA headers are intended to be included in C++ project
source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgGA/FlightManipulator>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgGA_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgGA library of the OpenSceneGraph toolkit
  was found.

``OSGGA_LIBRARIES``
  The libraries needed to link against to use osgGA.

``OSGGA_LIBRARY``
  A result variable that is set to the same value as the ``OSGGA_LIBRARIES``
  variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGGA_INCLUDE_DIR``
  The include directory containing headers needed to use osgGA.

``OSGGA_LIBRARY_DEBUG``
  The path to the osgGA debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgGA library, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGGA_FOUND``
  .. deprecated:: 4.2
    Use ``osgGA_FOUND``, which has the same value.

  Boolean indicating whether the osgGA library of the OpenSceneGraph toolkit
  was found.

Examples
^^^^^^^^

Finding osgGA explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgGA)

  if(osgGA_FOUND AND NOT TARGET osgGA::osgGA)
    add_library(osgGA::osgGA INTERFACE IMPORTED)
    set_target_properties(
      osgGA::osgGA
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGGA_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGGA_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgGA::osgGA)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGGA osgGA/FlightManipulator)
OSG_FIND_LIBRARY(OSGGA osgGA)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgGA DEFAULT_MSG
    OSGGA_LIBRARY OSGGA_INCLUDE_DIR)
