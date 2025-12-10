# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindALSA
--------

Finds the Advanced Linux Sound Architecture (ALSA) library (``asound``):

.. code-block:: cmake

  find_package(ALSA [<version>] [...])

Imported Targets
^^^^^^^^^^^^^^^^

This module provides the following :ref:`Imported Targets`:

``ALSA::ALSA``
  .. versionadded:: 3.12

  Target encapsulating the ALSA library usage requirements.  This target is
  available only if ALSA is found.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``ALSA_FOUND``
  Boolean indicating whether the (requested version of) ALSA library was
  found.

``ALSA_VERSION``
  .. versionadded:: 4.2

  The version of ALSA found.

``ALSA_LIBRARIES``
  List of libraries needed for linking to use ALSA library.

``ALSA_INCLUDE_DIRS``
  Include directories containing headers needed to use ALSA library.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``ALSA_INCLUDE_DIR``
  The ALSA include directory.

``ALSA_LIBRARY``
  The absolute path of the asound library.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``ALSA_VERSION_STRING``
  .. deprecated:: 4.2
    Superseded by the ``ALSA_VERSION``.

  The version of ALSA found.

Examples
^^^^^^^^

Finding the ALSA library and linking it to a project target:

.. code-block:: cmake

  find_package(ALSA)
  target_link_libraries(project_target PRIVATE ALSA::ALSA)
#]=======================================================================]

cmake_policy(PUSH)
cmake_policy(SET CMP0159 NEW) # file(STRINGS) with REGEX updates CMAKE_MATCH_<n>

find_path(ALSA_INCLUDE_DIR NAMES alsa/asoundlib.h
          DOC "The ALSA (asound) include directory"
)

find_library(ALSA_LIBRARY NAMES asound
          DOC "The ALSA (asound) library"
)

if(ALSA_INCLUDE_DIR AND EXISTS "${ALSA_INCLUDE_DIR}/alsa/version.h")
  file(STRINGS "${ALSA_INCLUDE_DIR}/alsa/version.h" alsa_version_str REGEX "^#define[\t ]+SND_LIB_VERSION_STR[\t ]+\".*\"")

  string(REGEX REPLACE "^.*SND_LIB_VERSION_STR[\t ]+\"([^\"]*)\".*$" "\\1" ALSA_VERSION "${alsa_version_str}")
  set(ALSA_VERSION_STRING "${ALSA_VERSION}")
  unset(alsa_version_str)
endif()

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ALSA
                                  REQUIRED_VARS ALSA_LIBRARY ALSA_INCLUDE_DIR
                                  VERSION_VAR ALSA_VERSION)

if(ALSA_FOUND)
  set( ALSA_LIBRARIES ${ALSA_LIBRARY} )
  set( ALSA_INCLUDE_DIRS ${ALSA_INCLUDE_DIR} )
  if(NOT TARGET ALSA::ALSA)
    add_library(ALSA::ALSA UNKNOWN IMPORTED)
    set_target_properties(ALSA::ALSA PROPERTIES INTERFACE_INCLUDE_DIRECTORIES "${ALSA_INCLUDE_DIRS}")
    set_property(TARGET ALSA::ALSA APPEND PROPERTY IMPORTED_LOCATION "${ALSA_LIBRARY}")
  endif()
endif()

mark_as_advanced(ALSA_INCLUDE_DIR ALSA_LIBRARY)

cmake_policy(POP)
