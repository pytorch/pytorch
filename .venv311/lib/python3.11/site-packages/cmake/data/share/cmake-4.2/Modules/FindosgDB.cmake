# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgDB
---------

Finds the osgDB library from the OpenSceneGraph toolkit.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgDB as a component.  This will automatically handle
  dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgDB)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgDB library.  It is not intended to be included directly during typical
use of the :command:`find_package` command.  However, it is available as a
standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgDB explicitly or bypass automatic
component detection:

.. code-block:: cmake

  find_package(osgDB)

OpenSceneGraph and osgDB headers are intended to be included in C++ project
source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgDB/DatabasePager>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgDB_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgDB library of the OpenSceneGraph toolkit
  was found.

``OSGDB_LIBRARIES``
  The libraries needed to link against to use osgDB.

``OSGDB_LIBRARY``
  A result variable that is set to the same value as the ``OSGDB_LIBRARIES``
  variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGDB_INCLUDE_DIR``
  The include directory containing headers needed to use osgDB.

``OSGDB_LIBRARY_DEBUG``
  The path to the osgDB debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgDB library, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGDB_FOUND``
  .. deprecated:: 4.2
    Use ``osgDB_FOUND``, which has the same value.

  Boolean indicating whether the osgDB library of the OpenSceneGraph toolkit
  was found.

Examples
^^^^^^^^

Finding osgDB explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgDB)

  if(osgDB_FOUND AND NOT TARGET osgDB::osgDB)
    add_library(osgDB::osgDB INTERFACE IMPORTED)
    set_target_properties(
      osgDB::osgDB
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGDB_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGDB_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgDB::osgDB)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGDB osgDB/DatabasePager)
OSG_FIND_LIBRARY(OSGDB osgDB)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgDB DEFAULT_MSG
    OSGDB_LIBRARY OSGDB_INCLUDE_DIR)
