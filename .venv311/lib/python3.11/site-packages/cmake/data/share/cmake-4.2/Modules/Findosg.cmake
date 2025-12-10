# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
Findosg
-------

Finds the core OpenSceneGraph osg library (``libosg``).

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead, which automatically finds the osg core library along with its
  required dependencies like OpenThreads:

  .. code-block:: cmake

    find_package(OpenSceneGraph)

This module is used internally by :module:`FindOpenSceneGraph` to find the osg
library.  It is not intended to be included directly during typical use of the
:command:`find_package` command.  However, it is available as a standalone
module for advanced use cases where finer control over detection is needed, such
as explicitly finding osg library or bypassing automatic component detection:

.. code-block:: cmake

  find_package(osg)

OpenSceneGraph core library headers are intended to be included in C++ project
source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osg_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osg library was found.

``OSG_LIBRARIES``
  The libraries needed to link against to use osg library.

``OSG_LIBRARY``
  A result variable that is set to the same value as the ``OSG_LIBRARIES``
  variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSG_INCLUDE_DIR``
  The include directory containing headers needed to use osg library.

``OSG_LIBRARY_DEBUG``
  The path to the osg debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osg library, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSG_FOUND``
  .. deprecated:: 4.2
    Use ``osg_FOUND``, which has the same value.

  Boolean indicating whether the osg library was found.

Examples
^^^^^^^^

Finding the osg library explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osg)

  if(osg_FOUND AND NOT TARGET osg::osg)
    add_library(osg::osg INTERFACE IMPORTED)
    set_target_properties(
      osg::osg
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSG_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSG_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osg::osg)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSG osg/PositionAttitudeTransform)
OSG_FIND_LIBRARY(OSG osg)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osg DEFAULT_MSG OSG_LIBRARY OSG_INCLUDE_DIR)
