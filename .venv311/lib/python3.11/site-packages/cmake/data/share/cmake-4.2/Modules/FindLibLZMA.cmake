# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindLibLZMA
-----------

Finds the liblzma, a data compression library that implements the LZMA
(Lempel–Ziv–Markov chain algorithm):

.. code-block:: cmake

  find_package(LibLZMA [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``LibLZMA::LibLZMA``
  .. versionadded:: 3.14

  Target encapsulating the liblzma library usage requirements, available only if
  liblzma is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``LibLZMA_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) liblzma was found.

``LibLZMA_VERSION``
  .. versionadded:: 4.2

  The version of liblzma found (available as a string, for example, ``5.0.3``).

``LIBLZMA_INCLUDE_DIRS``
  Include directories containing headers needed to use liblzma.

``LIBLZMA_LIBRARIES``
  Libraries needed to link against to use liblzma.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``LIBLZMA_HAS_AUTO_DECODER``
  Boolean sanity check result indicating whether the ``lzma_auto_decoder()``
  function (automatic decoder functionality) is found in liblzma (required).

``LIBLZMA_HAS_EASY_ENCODER``
  Boolean sanity check result indicating whether the ``lzma_easy_encoder()``
  function (basic encoder API) is found in liblzma (required).

``LIBLZMA_HAS_LZMA_PRESET``
  Boolean sanity check result indicating whether the ``lzma_lzma_preset()``
  function (preset compression configuration) is found in liblzma (required).

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``LIBLZMA_FOUND``
  .. deprecated:: 4.2
    Use ``LibLZMA_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) liblzma was found.

``LIBLZMA_VERSION``
  .. versionadded:: 3.26
  .. deprecated:: 4.2
    Superseded by the ``LibLZMA_VERSION``.

  The version of liblzma found.

``LIBLZMA_VERSION_STRING``
  .. deprecated:: 3.26
    Superseded by the ``LIBLZMA_VERSION`` (and ``LibLZMA_VERSION``).

  The version of liblzma found.

``LIBLZMA_VERSION_MAJOR``
  .. deprecated:: 3.26

  The major version of liblzma found.

``LIBLZMA_VERSION_MINOR``
  .. deprecated:: 3.26

  The minor version of liblzma found.

``LIBLZMA_VERSION_PATCH``
  .. deprecated:: 3.26

  The patch version of liblzma found.

Examples
^^^^^^^^

Finding the liblzma library and linking it to a project target:

.. code-block:: cmake

  find_package(LibLZMA)
  target_link_libraries(project_target PRIVATE LibLZMA::LibLZMA)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(LIBLZMA_INCLUDE_DIR lzma.h )
if(NOT LIBLZMA_LIBRARY)
  find_library(LIBLZMA_LIBRARY_RELEASE NAMES lzma liblzma NAMES_PER_DIR PATH_SUFFIXES lib)
  find_library(LIBLZMA_LIBRARY_DEBUG NAMES lzmad liblzmad NAMES_PER_DIR PATH_SUFFIXES lib)
  include(${CMAKE_CURRENT_LIST_DIR}/SelectLibraryConfigurations.cmake)
  select_library_configurations(LIBLZMA)
else()
  file(TO_CMAKE_PATH "${LIBLZMA_LIBRARY}" LIBLZMA_LIBRARY)
endif()

if(LIBLZMA_INCLUDE_DIR AND EXISTS "${LIBLZMA_INCLUDE_DIR}/lzma/version.h")
    file(STRINGS "${LIBLZMA_INCLUDE_DIR}/lzma/version.h" LIBLZMA_HEADER_CONTENTS REGEX "#define LZMA_VERSION_[A-Z]+ [0-9]+")

    string(REGEX REPLACE ".*#define LZMA_VERSION_MAJOR ([0-9]+).*" "\\1" LIBLZMA_VERSION_MAJOR "${LIBLZMA_HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define LZMA_VERSION_MINOR ([0-9]+).*" "\\1" LIBLZMA_VERSION_MINOR "${LIBLZMA_HEADER_CONTENTS}")
    string(REGEX REPLACE ".*#define LZMA_VERSION_PATCH ([0-9]+).*" "\\1" LIBLZMA_VERSION_PATCH "${LIBLZMA_HEADER_CONTENTS}")

    set(LibLZMA_VERSION "${LIBLZMA_VERSION_MAJOR}.${LIBLZMA_VERSION_MINOR}.${LIBLZMA_VERSION_PATCH}")
    set(LIBLZMA_VERSION "${LibLZMA_VERSION}")
    set(LIBLZMA_VERSION_STRING "${LibLZMA_VERSION}")
    unset(LIBLZMA_HEADER_CONTENTS)
endif()

# We're using new code known now as XZ, even library still been called LZMA
# it can be found in http://tukaani.org/xz/
# Avoid using old codebase
if (LIBLZMA_LIBRARY)
  include(${CMAKE_CURRENT_LIST_DIR}/CheckLibraryExists.cmake)
  set(CMAKE_REQUIRED_QUIET_SAVE ${CMAKE_REQUIRED_QUIET})
  set(CMAKE_REQUIRED_QUIET ${LibLZMA_FIND_QUIETLY})
  if(NOT LIBLZMA_LIBRARY_RELEASE AND NOT LIBLZMA_LIBRARY_DEBUG)
    set(LIBLZMA_LIBRARY_check ${LIBLZMA_LIBRARY})
  elseif(LIBLZMA_LIBRARY_RELEASE)
    set(LIBLZMA_LIBRARY_check ${LIBLZMA_LIBRARY_RELEASE})
  elseif(LIBLZMA_LIBRARY_DEBUG)
    set(LIBLZMA_LIBRARY_check ${LIBLZMA_LIBRARY_DEBUG})
  endif()
  check_library_exists(${LIBLZMA_LIBRARY_check} lzma_auto_decoder "" LIBLZMA_HAS_AUTO_DECODER)
  check_library_exists(${LIBLZMA_LIBRARY_check} lzma_easy_encoder "" LIBLZMA_HAS_EASY_ENCODER)
  check_library_exists(${LIBLZMA_LIBRARY_check} lzma_lzma_preset "" LIBLZMA_HAS_LZMA_PRESET)
  unset(LIBLZMA_LIBRARY_check)
  set(CMAKE_REQUIRED_QUIET ${CMAKE_REQUIRED_QUIET_SAVE})
endif ()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(LibLZMA  REQUIRED_VARS  LIBLZMA_LIBRARY
                                                          LIBLZMA_INCLUDE_DIR
                                                          LIBLZMA_HAS_AUTO_DECODER
                                                          LIBLZMA_HAS_EASY_ENCODER
                                                          LIBLZMA_HAS_LZMA_PRESET
                                           VERSION_VAR    LibLZMA_VERSION
                                 )
mark_as_advanced( LIBLZMA_INCLUDE_DIR LIBLZMA_LIBRARY )

if (LibLZMA_FOUND)
    set(LIBLZMA_LIBRARIES ${LIBLZMA_LIBRARY})
    set(LIBLZMA_INCLUDE_DIRS ${LIBLZMA_INCLUDE_DIR})
    if(NOT TARGET LibLZMA::LibLZMA)
        add_library(LibLZMA::LibLZMA UNKNOWN IMPORTED)
        set_target_properties(LibLZMA::LibLZMA PROPERTIES
                              INTERFACE_INCLUDE_DIRECTORIES "${LIBLZMA_INCLUDE_DIR}"
                              IMPORTED_LINK_INTERFACE_LANGUAGES C)

        if(LIBLZMA_LIBRARY_RELEASE)
            set_property(TARGET LibLZMA::LibLZMA APPEND PROPERTY
                IMPORTED_CONFIGURATIONS RELEASE)
            set_target_properties(LibLZMA::LibLZMA PROPERTIES
                IMPORTED_LOCATION_RELEASE "${LIBLZMA_LIBRARY_RELEASE}")
        endif()

        if(LIBLZMA_LIBRARY_DEBUG)
            set_property(TARGET LibLZMA::LibLZMA APPEND PROPERTY
                IMPORTED_CONFIGURATIONS DEBUG)
            set_target_properties(LibLZMA::LibLZMA PROPERTIES
                IMPORTED_LOCATION_DEBUG "${LIBLZMA_LIBRARY_DEBUG}")
        endif()

        if(NOT LIBLZMA_LIBRARY_RELEASE AND NOT LIBLZMA_LIBRARY_DEBUG)
            set_target_properties(LibLZMA::LibLZMA PROPERTIES
                IMPORTED_LOCATION "${LIBLZMA_LIBRARY}")
        endif()
    endif()
endif ()

cmake_policy(POP)
