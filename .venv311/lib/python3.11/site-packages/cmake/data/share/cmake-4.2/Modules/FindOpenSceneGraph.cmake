# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindOpenSceneGraph
------------------

Finds `OpenSceneGraph`_ (OSG), a 3D graphics application programming interface:

.. code-block:: cmake

  find_package(OpenSceneGraph [<version>] [COMPONENTS <components>...] [...])

.. note::

  OpenSceneGraph development has largely transitioned to its successor project,
  VulkanSceneGraph, which should be preferred for new code.  Refer to the
  upstream documentation for guidance on using VulkanSceneGraph with CMake.

This module searches for the OpenSceneGraph core osg library, its dependency
OpenThreads, and additional OpenSceneGraph libraries, some of which are also
known as *NodeKits*, if specified.

When working with OpenSceneGraph, its core library headers are intended to be
included in C++ project source code as:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osg/PositionAttitudeTransform>

Headers for the OpenSceneGraph libraries and NodeKits follow a similar inclusion
structure, for example:

.. code-block:: c++
  :caption: ``example.cxx``

  #include <osgAnimation/Animation>
  #include <osgDB/DatabasePager>
  #include <osgFX/BumpMapping>
  // ...

.. _`OpenSceneGraph`: https://openscenegraph.github.io/openscenegraph.io/

Components
^^^^^^^^^^

OpenSceneGraph toolkit consists of the core library osg, and additional
libraries, which can be optionally specified as components with the
:command:`find_package` command:

.. code-block:: cmake

  find_package(OpenSceneGraph [COMPONENTS <components>...])

Supported components include:

``osg``
  Finds the core osg library (``libosg``), required to use OpenSceneGraph.
  This component is always automatically implied.

``OpenThreads``
  Finds the dependent OpenThreads library (``libOpenThreads``) via the
  :module:`FindOpenThreads` module.  This component is always automatically
  implied as it is required to use OpenSceneGraph.

``osgAnimation``
  Finds the osgAnimation library, which provides general purpose utility classes
  for animation.

``osgDB``
  Finds the osgDB library for reading and writing scene graphs support.

``osgFX``
  Finds the osgFX NodeKit, which provides a framework for implementing special
  effects.

``osgGA``
  Finds the osgGA (GUI Abstraction) library, which provides facilities to work
  with varying window systems.

``osgIntrospection``
  Finds the osgIntrospection library, which provides a reflection framework for
  accessing and invoking class properties and methods at runtime without
  modifying the classes.

  .. note::
    The osgIntrospection library has been removed from the OpenSceneGraph
    toolkit as of OpenSceneGraph version 3.0.

``osgManipulator``
  Finds the osgManipulator NodeKit, which provides support for 3D interactive
  manipulators.

``osgParticle``
  Finds the osgParticle NodeKit, which provides support for particle effects.

``osgPresentation``
  Finds the osgPresentation NodeKit, which provides support for 3D scene graph
  based presentations.

  .. note::
    This NodeKit has been added in OpenSceneGraph 3.0.0.

``osgProducer``
  Finds the osgProducer utility library, which provides functionality for window
  management and event handling.

  .. note::
    The osgProducer has been removed from early versions of OpenSceneGraph
    toolkit 1.x, and has been superseded by the osgViewer library.

``osgQt``
  Finds the osgQt utility library, which provides various classes to aid the
  integration of Qt.

  .. note::
    As of OpenSceneGraph version 3.6, this library has been moved to its own
    repository.

``osgShadow``
  Finds the osgShadow NodeKit, which provides support for a range of shadow
  techniques.

``osgSim``
  Finds the osgSim NodeKit, which adds support for simulation features like
  navigation lights and OpenFlight-style movement controls.

``osgTerrain``
  Finds the osgTerrain NodeKit, which provides geospecifc terrain rendering
  support.

``osgText``
  Finds the osgText NodeKit, which provides high quality text support.

``osgUtil``
  Finds the osgUtil library, which provides general-purpose utilities like
  update, cull, and draw traversals, as well as scene graph tools such as
  optimization, triangle stripping, and tessellation.

``osgViewer``
  Finds the osgViewer library, which provides high level viewer functionality.

``osgVolume``
  Finds the osgVolume NodeKit, which provides volume rendering support.

``osgWidget``
  Finds the osgWidget NodeKit, which provides support for 2D and 3D GUI widget
  sets.

If no components are specified, this module searches for the ``osg`` and
``OpenThreads`` components by default.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``OpenSceneGraph_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) OpenSceneGraph with
  all specified components was found.

``OpenSceneGraph_VERSION``
  .. versionadded:: 4.2

  The version of OpenSceneGraph found.

``OPENSCENEGRAPH_INCLUDE_DIRS``
  Include directories containing headers needed to use OpenSceneGraph.

``OPENSCENEGRAPH_LIBRARIES``
  Libraries needed to link against to use OpenSceneGraph.

Hints
^^^^^

This module accepts the following variables:

``OpenSceneGraph_DEBUG``
  Set this variable to boolean true to enable debugging output by this module.

``OpenSceneGraph_MARK_AS_ADVANCED``
  Set this variable to boolean true to mark cache variables of this module as
  advanced automatically.

To help this module find OpenSceneGraph and its various components installed in
custom location, :variable:`CMAKE_PREFIX_PATH` variable can be used.
Additionally, the following variables are also respected:

``<COMPONENT>_DIR``
  Environment or CMake variable that can be set to the root of the OSG common
  installation, where ``<COMPONENT>`` is the uppercase form of component listed
  above.  For example, ``OSGVOLUME_DIR`` to find the ``osgVolume`` component.

``OSG_DIR``
  Environment or CMake variable that can be set to influence detection of
  OpenSceneGraph installation root location as a whole.

``OSGDIR``
  Environment variable treated the same as ``OSG_DIR``.

``OSG_ROOT``
  Environment variable treated the same as ``OSG_DIR``.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``OPENSCENEGRAPH_FOUND``
  .. deprecated:: 4.2
    Use ``OpenSceneGraph_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) OpenSceneGraph with
  all specified components was found.

``OPENSCENEGRAPH_VERSION``
  .. deprecated:: 4.2
    Superseded by the ``OpenSceneGraph_VERSION``.

  The version of OpenSceneGraph found.

Examples
^^^^^^^^

Finding the OpenSceneGraph with ``osgDB`` and ``osgUtil`` libraries specified as
components and creating an interface :ref:`imported target <Imported Targets>`
that encapsulates its usage requirements for linking to a project target:

.. code-block:: cmake

  find_package(OpenSceneGraph 2.0.0 REQUIRED COMPONENTS osgDB osgUtil)

  if(OpenSceneGraph_FOUND AND NOT TARGET OpenSceneGraph::OpenSceneGraph)
    add_library(OpenSceneGraph::OpenSceneGraph INTERFACE IMPORTED)
    set_target_properties(
      OpenSceneGraph::OpenSceneGraph
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${OPENSCENEGRAPH_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${OPENSCENEGRAPH_LIBRARIES}"
    )
  endif()

  add_executable(example example.cxx)

  target_link_libraries(example PRIVATE OpenSceneGraph::OpenSceneGraph)

See Also
^^^^^^^^

The following OpenSceneGraph-related helper find modules are used internally by
this module when finding specific OpenSceneGraph components.  These modules are
not intended to be included or invoked directly by project code during typical
use of ``find_package(OpenSceneGraph)``.  However, they can be useful for
advanced scenarios where finer control over component detection is needed.  For
example, to find them explicitly and override or bypass detection of specific
OpenSceneGraph components:

* The :module:`Findosg` module to find the core osg library.
* The :module:`FindosgAnimation` module to find osgAnimation.
* The :module:`FindosgDB` module to find osgDB.
* The :module:`FindosgFX` module to find osgDB.
* The :module:`FindosgGA` module to find osgGA.
* The :module:`FindosgIntrospection` module to find osgIntrospection.
* The :module:`FindosgManipulator` module to find osgManipulator.
* The :module:`FindosgParticle` module to find osgParticle.
* The :module:`FindosgPresentation` module to find osgPresentation.
* The :module:`FindosgProducer` module to find osgProducer.
* The :module:`FindosgQt` module to find osgQt.
* The :module:`FindosgShadow` module to find osgShadow.
* The :module:`FindosgSim` module to find osgSim.
* The :module:`FindosgTerrain` module to find osgTerrain.
* The :module:`FindosgText` module to find osgText.
* The :module:`FindosgUtil` module to find osgUtil.
* The :module:`FindosgViewer` module to find osgViewer.
* The :module:`FindosgVolume` module to find osgVolume.
* The :module:`FindosgWidget` module to find osgWidget.
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

include(${CMAKE_CURRENT_LIST_DIR}/Findosg_functions.cmake)

set(_osg_modules_to_process)
foreach(_osg_component ${OpenSceneGraph_FIND_COMPONENTS})
    list(APPEND _osg_modules_to_process ${_osg_component})
endforeach()
list(APPEND _osg_modules_to_process "osg" "OpenThreads")
list(REMOVE_DUPLICATES _osg_modules_to_process)

if(OpenSceneGraph_DEBUG)
    message(STATUS "[ FindOpenSceneGraph.cmake:${CMAKE_CURRENT_LIST_LINE} ] "
        "Components = ${_osg_modules_to_process}")
endif()

#
# First we need to find and parse osg/Version
#
OSG_FIND_PATH(OSG osg/Version)
if(OpenSceneGraph_MARK_AS_ADVANCED)
    OSG_MARK_AS_ADVANCED(OSG)
endif()

# Try to ascertain the version...
if(OSG_INCLUDE_DIR)
    if(OpenSceneGraph_DEBUG)
        message(STATUS "[ FindOpenSceneGraph.cmake:${CMAKE_CURRENT_LIST_LINE} ] "
            "Detected OSG_INCLUDE_DIR = ${OSG_INCLUDE_DIR}")
    endif()

    set(_osg_Version_file "${OSG_INCLUDE_DIR}/osg/Version")
    if("${OSG_INCLUDE_DIR}" MATCHES "\\.framework$" AND NOT EXISTS "${_osg_Version_file}")
        set(_osg_Version_file "${OSG_INCLUDE_DIR}/Headers/Version")
    endif()

    if(EXISTS "${_osg_Version_file}")
      file(STRINGS "${_osg_Version_file}" _osg_Version_contents
           REGEX "#define (OSG_VERSION_[A-Z]+|OPENSCENEGRAPH_[A-Z]+_VERSION)[ \t]+[0-9]+")
    else()
      set(_osg_Version_contents "unknown")
    endif()

    string(REGEX MATCH ".*#define OSG_VERSION_MAJOR[ \t]+[0-9]+.*"
        _osg_old_defines "${_osg_Version_contents}")
    string(REGEX MATCH ".*#define OPENSCENEGRAPH_MAJOR_VERSION[ \t]+[0-9]+.*"
        _osg_new_defines "${_osg_Version_contents}")
    if(_osg_old_defines)
        string(REGEX REPLACE ".*#define OSG_VERSION_MAJOR[ \t]+([0-9]+).*"
            "\\1" _osg_VERSION_MAJOR ${_osg_Version_contents})
        string(REGEX REPLACE ".*#define OSG_VERSION_MINOR[ \t]+([0-9]+).*"
            "\\1" _osg_VERSION_MINOR ${_osg_Version_contents})
        string(REGEX REPLACE ".*#define OSG_VERSION_PATCH[ \t]+([0-9]+).*"
            "\\1" _osg_VERSION_PATCH ${_osg_Version_contents})
    elseif(_osg_new_defines)
        string(REGEX REPLACE ".*#define OPENSCENEGRAPH_MAJOR_VERSION[ \t]+([0-9]+).*"
            "\\1" _osg_VERSION_MAJOR ${_osg_Version_contents})
        string(REGEX REPLACE ".*#define OPENSCENEGRAPH_MINOR_VERSION[ \t]+([0-9]+).*"
            "\\1" _osg_VERSION_MINOR ${_osg_Version_contents})
        string(REGEX REPLACE ".*#define OPENSCENEGRAPH_PATCH_VERSION[ \t]+([0-9]+).*"
            "\\1" _osg_VERSION_PATCH ${_osg_Version_contents})
    else()
        message(WARNING "[ FindOpenSceneGraph.cmake:${CMAKE_CURRENT_LIST_LINE} ] "
            "Failed to parse version number, please report this as a bug")
    endif()
    unset(_osg_Version_contents)

    set(OPENSCENEGRAPH_VERSION "${_osg_VERSION_MAJOR}.${_osg_VERSION_MINOR}.${_osg_VERSION_PATCH}"
                                CACHE INTERNAL "The version of OSG which was detected")
    set(OpenSceneGraph_VERSION "${OPENSCENEGRAPH_VERSION}")

    if(OpenSceneGraph_DEBUG)
        message(STATUS "[ FindOpenSceneGraph.cmake:${CMAKE_CURRENT_LIST_LINE} ] "
            "Detected version ${OpenSceneGraph_VERSION}")
    endif()
endif()

set(_osg_quiet)
if(OpenSceneGraph_FIND_QUIETLY)
    set(_osg_quiet "QUIET")
endif()
#
# Here we call find_package() on all of the components
#
foreach(_osg_module ${_osg_modules_to_process})
    if(OpenSceneGraph_DEBUG)
        message(STATUS "[ FindOpenSceneGraph.cmake:${CMAKE_CURRENT_LIST_LINE} ] "
            "Calling find_package(${_osg_module} ${_osg_required} ${_osg_quiet})")
    endif()
    find_package(${_osg_module} ${_osg_quiet})

    string(TOUPPER ${_osg_module} _osg_module_UC)
    # append to list if module was found OR is required
    if( ${_osg_module_UC}_FOUND OR OpenSceneGraph_FIND_REQUIRED )
      list(APPEND OPENSCENEGRAPH_INCLUDE_DIR ${${_osg_module_UC}_INCLUDE_DIR})
      list(APPEND OPENSCENEGRAPH_LIBRARIES ${${_osg_module_UC}_LIBRARIES})
    endif()

    if(OpenSceneGraph_MARK_AS_ADVANCED)
        OSG_MARK_AS_ADVANCED(${_osg_module})
    endif()
endforeach()

if(OPENSCENEGRAPH_INCLUDE_DIR)
    list(REMOVE_DUPLICATES OPENSCENEGRAPH_INCLUDE_DIR)
endif()

#
# Check each module to see if it's found
#
set(_osg_component_founds)
if(OpenSceneGraph_FIND_REQUIRED)
    foreach(_osg_module ${_osg_modules_to_process})
        string(TOUPPER ${_osg_module} _osg_module_UC)
        list(APPEND _osg_component_founds ${_osg_module_UC}_FOUND)
    endforeach()
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(OpenSceneGraph
                                  REQUIRED_VARS OPENSCENEGRAPH_LIBRARIES OPENSCENEGRAPH_INCLUDE_DIR ${_osg_component_founds}
                                  VERSION_VAR OpenSceneGraph_VERSION)

unset(_osg_component_founds)

set(OPENSCENEGRAPH_INCLUDE_DIRS ${OPENSCENEGRAPH_INCLUDE_DIR})

cmake_policy(POP)
