# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindosgIntrospection
--------------------

Finds the osgIntrospection library from the OpenSceneGraph toolkit.

.. note::

  The osgIntrospection library has been removed from the OpenSceneGraph toolkit
  as of OpenSceneGraph version 3.0.

.. note::

  In most cases, it's recommended to use the :module:`FindOpenSceneGraph` module
  instead and list osgIntrospection as a component.  This will automatically
  handle dependencies such as the OpenThreads and core osg libraries:

  .. code-block:: cmake

    find_package(OpenSceneGraph COMPONENTS osgIntrospection)

This module is used internally by :module:`FindOpenSceneGraph` to find the
osgIntrospection library.  It is not intended to be included directly during
typical use of the :command:`find_package` command.  However, it is available as
a standalone module for advanced use cases where finer control over detection is
needed.  For example, to find the osgIntrospection explicitly or bypass
automatic component detection:

.. code-block:: cmake

  find_package(osgIntrospection)

OpenSceneGraph and osgIntrospection headers are intended to be included in C++
project source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>
  #include <osgIntrospection/Reflection>
  // ...

When working with the OpenSceneGraph toolkit, other libraries such as OpenGL may
also be required.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``osgIntrospection_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the osgIntrospection library of the OpenSceneGraph
  toolkit was found.

``OSGINTROSPECTION_LIBRARIES``
  The libraries needed to link against to use osgIntrospection.

``OSGINTROSPECTION_LIBRARY``
  A result variable that is set to the same value as the
  ``OSGINTROSPECTION_LIBRARIES`` variable.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``OSGINTROSPECTION_INCLUDE_DIR``
  The include directory containing headers needed to use osgIntrospection.

``OSGINTROSPECTION_LIBRARY_DEBUG``
  The path to the osgIntrospection debug library.

Hints
^^^^^

This module accepts the following variables:

``OSGDIR``
  Environment variable that can be set to help locate the OpenSceneGraph
  toolkit, including its osgIntrospection library, when installed in a custom
  location.  It should point to the OpenSceneGraph installation prefix used when
  it was configured, built, and installed: ``./configure --prefix=$OSGDIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OSGINTROSPECTION_FOUND``
  .. deprecated:: 4.2
    Use ``osgIntrospection_FOUND``, which has the same value.

  Boolean indicating whether the osgIntrospection library of the OpenSceneGraph
  toolkit was found.

Examples
^^^^^^^^

Finding osgIntrospection explicitly with this module and creating an interface
:ref:`imported target <Imported Targets>` that encapsulates its usage
requirements for linking it to a project target:

.. code-block:: cmake

  find_package(osgIntrospection)

  if(osgIntrospection_FOUND AND NOT TARGET osgIntrospection::osgIntrospection)
    add_library(osgIntrospection::osgIntrospection INTERFACE IMPORTED)
    set_target_properties(
      osgIntrospection::osgIntrospection
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OSGINTROSPECTION_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${OSGINTROSPECTION_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE osgIntrospection::osgIntrospection)

See Also
^^^^^^^^

* The :module:`FindOpenSceneGraph` module to find OpenSceneGraph toolkit.
#]=======================================================================]

# Created by Eric Wing.

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)
OSG_FIND_PATH   (OSGINTROSPECTION osgIntrospection/Reflection)
OSG_FIND_LIBRARY(OSGINTROSPECTION osgIntrospection)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(osgIntrospection DEFAULT_MSG
    OSGINTROSPECTION_LIBRARY OSGINTROSPECTION_INCLUDE_DIR)
