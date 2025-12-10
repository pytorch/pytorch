# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindBullet
----------

Finds the Bullet physics engine:

.. code-block:: cmake

  find_package(Bullet [...])

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Bullet_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether Bullet was found.

``BULLET_INCLUDE_DIRS``
  The Bullet include directories.

``BULLET_LIBRARIES``
  Libraries needed to link to Bullet.  By default, all Bullet components
  (Dynamics, Collision, LinearMath, and SoftBody) are added.

Hints
^^^^^

This module accepts the following variables:

``BULLET_ROOT``
  Can be set to Bullet install path or Windows build path to specify where to
  find Bullet.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``BULLET_FOUND``
  .. deprecated:: 4.2
    Use ``Bullet_FOUND``, which has the same value.

  Boolean indicating whether Bullet was found.

Examples
^^^^^^^^

Finding Bullet and conditionally creating an interface :ref:`imported target
<Imported Targets>` that encapsulates its usage requirements for linking to a
project target:

.. code-block:: cmake

  find_package(Bullet)

  if(Bullet_FOUND AND NOT TARGET Bullet::Bullet)
    add_library(Bullet::Bullet INTERFACE IMPORTED)
    set_target_properties(
      Bullet::Bullet
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${BULLET_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${BULLET_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE Bullet::Bullet)
#]=======================================================================]

macro(_FIND_BULLET_LIBRARY _var)
  find_library(${_var}
     NAMES
        ${ARGN}
     HINTS
        ${BULLET_ROOT}
        ${BULLET_ROOT}/lib/Release
        ${BULLET_ROOT}/lib/Debug
        ${BULLET_ROOT}/out/release8/libs
        ${BULLET_ROOT}/out/debug8/libs
     PATH_SUFFIXES lib
  )
  mark_as_advanced(${_var})
endmacro()

macro(_BULLET_APPEND_LIBRARIES _list _release)
  set(_debug ${_release}_DEBUG)
  if(${_debug})
    set(${_list} ${${_list}} optimized ${${_release}} debug ${${_debug}})
  else()
    set(${_list} ${${_list}} ${${_release}})
  endif()
endmacro()

find_path(BULLET_INCLUDE_DIR NAMES btBulletCollisionCommon.h
  HINTS
    ${BULLET_ROOT}/include
    ${BULLET_ROOT}/src
  PATH_SUFFIXES bullet
)

# Find the libraries

_FIND_BULLET_LIBRARY(BULLET_DYNAMICS_LIBRARY        BulletDynamics)
_FIND_BULLET_LIBRARY(BULLET_DYNAMICS_LIBRARY_DEBUG  BulletDynamics_Debug BulletDynamics_d)
_FIND_BULLET_LIBRARY(BULLET_COLLISION_LIBRARY       BulletCollision)
_FIND_BULLET_LIBRARY(BULLET_COLLISION_LIBRARY_DEBUG BulletCollision_Debug BulletCollision_d)
_FIND_BULLET_LIBRARY(BULLET_MATH_LIBRARY            BulletMath LinearMath)
_FIND_BULLET_LIBRARY(BULLET_MATH_LIBRARY_DEBUG      BulletMath_Debug BulletMath_d LinearMath_Debug LinearMath_d)
_FIND_BULLET_LIBRARY(BULLET_SOFTBODY_LIBRARY        BulletSoftBody)
_FIND_BULLET_LIBRARY(BULLET_SOFTBODY_LIBRARY_DEBUG  BulletSoftBody_Debug BulletSoftBody_d)


include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Bullet DEFAULT_MSG
    BULLET_DYNAMICS_LIBRARY BULLET_COLLISION_LIBRARY BULLET_MATH_LIBRARY
    BULLET_SOFTBODY_LIBRARY BULLET_INCLUDE_DIR)

set(BULLET_INCLUDE_DIRS ${BULLET_INCLUDE_DIR})
if(Bullet_FOUND)
   _BULLET_APPEND_LIBRARIES(BULLET_LIBRARIES BULLET_DYNAMICS_LIBRARY)
   _BULLET_APPEND_LIBRARIES(BULLET_LIBRARIES BULLET_COLLISION_LIBRARY)
   _BULLET_APPEND_LIBRARIES(BULLET_LIBRARIES BULLET_MATH_LIBRARY)
   _BULLET_APPEND_LIBRARIES(BULLET_LIBRARIES BULLET_SOFTBODY_LIBRARY)
endif()
