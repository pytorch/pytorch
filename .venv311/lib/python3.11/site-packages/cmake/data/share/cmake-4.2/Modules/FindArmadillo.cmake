# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindArmadillo
-------------

Finds the Armadillo C++ library:

.. code-block:: cmake

  find_package(Armadillo [<version>] [...])

Armadillo is a library for linear algebra and scientific computing.

.. versionadded:: 3.18
  Support for linking wrapped libraries directly (see the
  ``ARMA_DONT_USE_WRAPPER`` preprocessor macro that needs to be defined before
  including the ``<armadillo>`` header).

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Armadillo_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether the (requested version of) Armadillo library
  was found.

``Armadillo_VERSION``
  .. versionadded:: 4.2

  The version of Armadillo found (e.g., ``14.90.0``).

``Armadillo_VERSION_NAME``
  .. versionadded:: 4.2

  The version name of Armadillo found (e.g., ``Antipodean Antileech``).

``ARMADILLO_INCLUDE_DIRS``
  List of required include directories.

``ARMADILLO_LIBRARIES``
  List of libraries to be linked.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``ARMADILLO_FOUND``
  .. deprecated:: 4.2
    Use ``Armadillo_FOUND``, which has the same value.

  Boolean indicating whether the (requested version of) Armadillo library
  was found.

``ARMADILLO_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``Armadillo_VERSION``.

  The version of Armadillo found.

``ARMADILLO_VERSION_MAJOR``
  .. deprecated:: 4.2
    Superseded by the ``Armadillo_VERSION``.

  Major version number.

``ARMADILLO_VERSION_MINOR``
  .. deprecated:: 4.2
    Superseded by the ``Armadillo_VERSION``.

  Minor version number.

``ARMADILLO_VERSION_PATCH``
  .. deprecated:: 4.2
    Superseded by the ``Armadillo_VERSION``.

  Patch version number.

``ARMADILLO_VERSION_NAME``
  .. deprecated:: 4.2
    Superseded by the ``Armadillo_VERSION_NAME``.

  The version name of Armadillo found (e.g., ``Antipodean Antileech``).

Examples
^^^^^^^^

Finding Armadillo and creating an imported target:

.. code-block:: cmake

  find_package(Armadillo REQUIRED)

  if(Armadillo_FOUND AND NOT TARGET Armadillo::Armadillo)
    add_library(Armadillo::Armadillo INTERFACE IMPORTED)
    set_target_properties(
      Armadillo::Armadillo
      PROPERTIES
        INTERFACE_LINK_LIBRARIES "${ARMADILLO_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${ARMADILLO_INCLUDE_DIRS}"
    )
  endif()

  add_executable(foo foo.cc)
  target_link_libraries(foo PRIVATE Armadillo::Armadillo)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(ARMADILLO_INCLUDE_DIR
  NAMES armadillo
  PATHS "$ENV{ProgramFiles}/Armadillo/include"
  )
mark_as_advanced(ARMADILLO_INCLUDE_DIR)

if(ARMADILLO_INCLUDE_DIR)
  # ------------------------------------------------------------------------
  #  Extract version information from <armadillo>
  # ------------------------------------------------------------------------

  # WARNING: Early releases of Armadillo didn't have the arma_version.hpp file.
  # (e.g. v.0.9.8-1 in ubuntu maverick packages (2001-03-15))
  # If the file is missing, set all values to 0
  set(ARMADILLO_VERSION_MAJOR 0)
  set(ARMADILLO_VERSION_MINOR 0)
  set(ARMADILLO_VERSION_PATCH 0)
  set(Armadillo_VERSION_NAME "EARLY RELEASE")

  if(EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/arma_version.hpp")
    # Read and parse armdillo version header file for version number
    file(STRINGS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/arma_version.hpp" _ARMA_HEADER_CONTENTS REGEX "#define ARMA_VERSION_[A-Z]+ ")
    string(REGEX REPLACE ".*#define ARMA_VERSION_MAJOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MAJOR "${_ARMA_HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define ARMA_VERSION_MINOR ([0-9]+).*" "\\1" ARMADILLO_VERSION_MINOR "${_ARMA_HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define ARMA_VERSION_PATCH ([0-9]+).*" "\\1" ARMADILLO_VERSION_PATCH "${_ARMA_HEADER_CONTENTS}")

    # WARNING: The number of spaces before the version name is not one.
    string(REGEX REPLACE ".*#define ARMA_VERSION_NAME\ +\"([0-9a-zA-Z\ _-]+)\".*" "\\1" Armadillo_VERSION_NAME "${_ARMA_HEADER_CONTENTS}")

    set(ARMADILLO_VERSION_NAME "${Armadillo_VERSION_NAME}")
  endif()

  set(Armadillo_VERSION "${ARMADILLO_VERSION_MAJOR}.${ARMADILLO_VERSION_MINOR}.${ARMADILLO_VERSION_PATCH}")
  set(ARMADILLO_VERSION_STRING "${Armadillo_VERSION}")
endif ()

if(EXISTS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp")
  file(STRINGS "${ARMADILLO_INCLUDE_DIR}/armadillo_bits/config.hpp" _ARMA_CONFIG_CONTENTS REGEX "^#define ARMA_USE_[A-Z]+")
  string(REGEX MATCH "ARMA_USE_WRAPPER" _ARMA_USE_WRAPPER "${_ARMA_CONFIG_CONTENTS}")
  string(REGEX MATCH "ARMA_USE_LAPACK" _ARMA_USE_LAPACK "${_ARMA_CONFIG_CONTENTS}")
  string(REGEX MATCH "ARMA_USE_BLAS" _ARMA_USE_BLAS "${_ARMA_CONFIG_CONTENTS}")
  string(REGEX MATCH "ARMA_USE_ARPACK" _ARMA_USE_ARPACK "${_ARMA_CONFIG_CONTENTS}")
  string(REGEX MATCH "ARMA_USE_HDF5" _ARMA_USE_HDF5 "${_ARMA_CONFIG_CONTENTS}")
endif()

include(FindPackageHandleStandardArgs)

# If _ARMA_USE_WRAPPER is set, then we just link to armadillo, but if it's not
# then we need support libraries instead.
set(_ARMA_SUPPORT_LIBRARIES)

if(_ARMA_USE_WRAPPER)
  # Link to the armadillo wrapper library.
  find_library(ARMADILLO_LIBRARY
    NAMES armadillo
    NAMES_PER_DIR
    PATHS
      "$ENV{ProgramFiles}/Armadillo/lib"
      "$ENV{ProgramFiles}/Armadillo/lib64"
      "$ENV{ProgramFiles}/Armadillo"
    )
  mark_as_advanced(ARMADILLO_LIBRARY)
  set(_ARMA_REQUIRED_VARS ARMADILLO_LIBRARY)
else()
  set(ARMADILLO_LIBRARY "")
endif()

# Transitive linking with the wrapper does not work with MSVC,
# so we must *also* link against Armadillo's dependencies.
if(NOT _ARMA_USE_WRAPPER OR MSVC)
  # Link directly to individual components.
  foreach(pkg
      LAPACK
      BLAS
      ARPACK
      HDF5
      )
    if(_ARMA_USE_${pkg})
      find_package(${pkg} QUIET)
      list(APPEND _ARMA_REQUIRED_VARS "${pkg}_FOUND")
      if(${pkg}_FOUND)
        list(APPEND _ARMA_SUPPORT_LIBRARIES ${${pkg}_LIBRARIES})
      endif()
    endif()
  endforeach()
endif()

find_package_handle_standard_args(Armadillo
  REQUIRED_VARS ARMADILLO_INCLUDE_DIR ${_ARMA_REQUIRED_VARS}
  VERSION_VAR Armadillo_VERSION)

if (Armadillo_FOUND)
  set(ARMADILLO_INCLUDE_DIRS ${ARMADILLO_INCLUDE_DIR})
  set(ARMADILLO_LIBRARIES ${ARMADILLO_LIBRARY} ${_ARMA_SUPPORT_LIBRARIES})
endif ()

# Clean up internal variables
unset(_ARMA_REQUIRED_VARS)
unset(_ARMA_SUPPORT_LIBRARIES)
unset(_ARMA_USE_WRAPPER)
unset(_ARMA_USE_LAPACK)
unset(_ARMA_USE_BLAS)
unset(_ARMA_USE_ARPACK)
unset(_ARMA_USE_HDF5)
unset(_ARMA_CONFIG_CONTENTS)
unset(_ARMA_HEADER_CONTENTS)

cmake_policy(POP)
