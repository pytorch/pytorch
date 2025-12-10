# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSelfPackers
---------------

Finds `UPX <https://upx.github.io/>`_, the Ultimate Packer for eXecutables:

.. code-block:: cmake

  find_package(SelfPackers [...])

This module searches for executable packers-tools that compress executables or
shared libraries into on-the-fly, self-extracting versions.  It currently
supports ``UPX``.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``SelfPackers_FOUND``
  .. versionadded:: 4.2

  Boolean indicating whether packer tools were found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``SELF_PACKER_FOR_EXECUTABLE``
  Path to the executable packer for compressing executables.

``SELF_PACKER_FOR_SHARED_LIB``
  Path to the executable packer for compressing shared libraries.

``SELF_PACKER_FOR_EXECUTABLE_FLAGS``
  Command-line options to use when compressing executables.

``SELF_PACKER_FOR_SHARED_LIB_FLAGS``
  Command-line options to use when compressing shared libraries.

Examples
^^^^^^^^

Finding UPX:

.. code-block:: cmake

  find_package(SelfPackers)
#]=======================================================================]

include(${CMAKE_CURRENT_LIST_DIR}/FindCygwin.cmake)
include(${CMAKE_CURRENT_LIST_DIR}/FindMsys.cmake)

find_program(SELF_PACKER_FOR_EXECUTABLE
  upx
  ${CYGWIN_INSTALL_PATH}/bin
  ${MSYS_INSTALL_PATH}/usr/bin
)

find_program(SELF_PACKER_FOR_SHARED_LIB
  upx
  ${CYGWIN_INSTALL_PATH}/bin
  ${MSYS_INSTALL_PATH}/usr/bin
)

mark_as_advanced(
  SELF_PACKER_FOR_EXECUTABLE
  SELF_PACKER_FOR_SHARED_LIB
)

#
# Set flags
#
if (SELF_PACKER_FOR_EXECUTABLE MATCHES "upx")
  set (SELF_PACKER_FOR_EXECUTABLE_FLAGS "-q" CACHE STRING
       "Flags for the executable self-packer.")
else ()
  set (SELF_PACKER_FOR_EXECUTABLE_FLAGS "" CACHE STRING
       "Flags for the executable self-packer.")
endif ()

if (SELF_PACKER_FOR_SHARED_LIB MATCHES "upx")
  set (SELF_PACKER_FOR_SHARED_LIB_FLAGS "-q" CACHE STRING
       "Flags for the shared lib self-packer.")
else ()
  set (SELF_PACKER_FOR_SHARED_LIB_FLAGS "" CACHE STRING
       "Flags for the shared lib self-packer.")
endif ()

mark_as_advanced(
  SELF_PACKER_FOR_EXECUTABLE_FLAGS
  SELF_PACKER_FOR_SHARED_LIB_FLAGS
)

if(SELF_PACKER_FOR_EXECUTABLE AND SELF_PACKER_FOR_SHARED_LIB)
  set(SelfPackers_FOUND TRUE)
else()
  set(SelfPackers_FOUND FALSE)
endif()
