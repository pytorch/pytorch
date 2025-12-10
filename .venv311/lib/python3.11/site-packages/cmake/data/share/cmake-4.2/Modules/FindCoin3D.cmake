# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindCoin3D
----------

Finds Coin3D (Open Inventor):

.. code-block:: cmake

  find_package(Coin3D [...])

Coin3D is an implementation of the Open Inventor API.  It provides
data structures and algorithms for 3D visualization.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Coin3D_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether Coin3D, Open Inventor was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``COIN3D_INCLUDE_DIRS``
  Directory containing the Open Inventor header files (``Inventor/So.h``).
``COIN3D_LIBRARIES``
  Coin3D libraries required for linking.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``COIN3D_FOUND``
  .. deprecated:: 4.2
    Use ``Coin3D_FOUND``, which has the same value.

  Boolean indicating whether Coin3D, Open Inventor was found.

Examples
^^^^^^^^

Finding Coin3D and conditionally creating an interface :ref:`imported target
<Imported Targets>` that encapsulates its usage requirements for linking to a
project target:

.. code-block:: cmake

  find_package(Coin3D)

  if(Coin3D_FOUND AND NOT TARGET Coin3D::Coin3D)
    add_library(Coin3D::Coin3D INTERFACE IMPORTED)
    set_target_properties(
      Coin3D::Coin3D
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${COIN3D_INCLUDE_DIRS}"
        INTERFACE_LINK_LIBRARIES "${COIN3D_LIBRARIES}"
    )
  endif()

  target_link_libraries(example PRIVATE Coin3D::Coin3D)
#]=======================================================================]

if (WIN32)
  if (CYGWIN)

    find_path(COIN3D_INCLUDE_DIRS Inventor/So.h)
    find_library(COIN3D_LIBRARIES Coin)

  else ()

    find_path(COIN3D_INCLUDE_DIRS Inventor/So.h
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\SIM\\Coin3D\\2;Installation Path]/include"
    )

    find_library(COIN3D_LIBRARY_DEBUG NAMES coin2d coin4d
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\SIM\\Coin3D\\2;Installation Path]/lib"
    )

    find_library(COIN3D_LIBRARY_RELEASE NAMES coin2 coin4
      "[HKEY_LOCAL_MACHINE\\SOFTWARE\\SIM\\Coin3D\\2;Installation Path]/lib"
    )

    if (COIN3D_LIBRARY_DEBUG AND COIN3D_LIBRARY_RELEASE)
      set(COIN3D_LIBRARIES optimized ${COIN3D_LIBRARY_RELEASE}
                           debug ${COIN3D_LIBRARY_DEBUG})
    else ()
      if (COIN3D_LIBRARY_DEBUG)
        set (COIN3D_LIBRARIES ${COIN3D_LIBRARY_DEBUG})
      endif ()
      if (COIN3D_LIBRARY_RELEASE)
        set (COIN3D_LIBRARIES ${COIN3D_LIBRARY_RELEASE})
      endif ()
    endif ()

  endif ()

else ()
  if(APPLE)
    find_path(COIN3D_INCLUDE_DIRS Inventor/So.h
     /Library/Frameworks/Inventor.framework/Headers
    )
    find_library(COIN3D_LIBRARIES Coin
      /Library/Frameworks/Inventor.framework/Libraries
    )
    set(COIN3D_LIBRARIES "-framework Coin3d" CACHE STRING "Coin3D library for OSX")
  else()

    find_path(COIN3D_INCLUDE_DIRS Inventor/So.h)
    find_library(COIN3D_LIBRARIES Coin)

  endif()

endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Coin3D DEFAULT_MSG COIN3D_LIBRARIES COIN3D_INCLUDE_DIRS)

mark_as_advanced(COIN3D_INCLUDE_DIRS COIN3D_LIBRARIES )
