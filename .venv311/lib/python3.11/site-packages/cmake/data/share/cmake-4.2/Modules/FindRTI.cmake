# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindRTI
-------

Finds HLA RTI standard libraries and their include directories:

.. code-block:: cmake

  find_package(RTI [...])

`RTI <https://en.wikipedia.org/wiki/Run-time_infrastructure_(simulation)>`_
(Run-Time Infrastructure) is a simulation infrastructure standardized by IEEE
and SISO, required when implementing HLA (High Level Architecture).  It provides
a well-defined C++ API, ensuring that M&S (Modeling and Simulation) applications
remain independent of a particular RTI implementation.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``RTI_FOUND``
  Boolean indicating whether HLA RTI was found.
``RTI_LIBRARIES``
  The libraries to link against to use RTI.
``RTI_DEFINITIONS``
  Compile definitions for using RTI.  Default value is set to
  ``-DRTI_USES_STD_FSTREAM``.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``RTI_INCLUDE_DIR``
  Directory where RTI include files are found.

Examples
^^^^^^^^

Finding RTI and creating an imported interface target for linking it to a
project target:

.. code-block:: cmake

  find_package(RTI)

  if(RTI_FOUND AND NOT TARGET RTI::RTI)
    add_library(RTI::RTI INTERFACE IMPORTED)
    set_target_properties(
      RTI::RTI
      PROPERTIES
        INTERFACE_INCLUDE_DIRECTORIES "${RTI_INCLUDE_DIR}"
        INTERFACE_LINK_LIBRARIES "${RTI_LIBRARIES}"
        INTERFACE_COMPILE_DEFINITIONS "${RTI_DEFINITIONS}"
    )
  endif()

  target_link_libraries(example PRIVATE RTI::RTI)
#]=======================================================================]

macro(RTI_MESSAGE_QUIETLY QUIET TYPE MSG)
  if(NOT ${QUIET})
    message(${TYPE} "${MSG}")
  endif()
endmacro()

set(RTI_DEFINITIONS "-DRTI_USES_STD_FSTREAM")

# noqa: spellcheck off
# Detect the CERTI installation:
#   - https://www.nongnu.org/certi/
#   - Mailing list for reporting issues and development discussions:
#     <certi-devel@nongnu.org>
# Detect the MAK Technologies RTI installation:
#   - https://www.mak.com/mak-one/tools/mak-rti
# note: the following list is ordered to find the most recent version first
set(RTI_POSSIBLE_DIRS
  ENV CERTI_HOME
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MAK Technologies\\MAK RTI 3.2 MSVC++ 8.0;Location]"
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MAK RTI 3.2-win32-msvc++8.0;InstallLocation]"
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\MAK Technologies\\MAK RTI 2.2;Location]"
  "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Microsoft\\Windows\\CurrentVersion\\Uninstall\\MAK RTI 2.2;InstallLocation]")

set(RTI_OLD_FIND_LIBRARY_PREFIXES "${CMAKE_FIND_LIBRARY_PREFIXES}")
# The MAK RTI has the "lib" prefix even on Windows.
set(CMAKE_FIND_LIBRARY_PREFIXES "lib" "")
# noqa: spellcheck on

find_library(RTI_LIBRARY
  NAMES RTI RTI-NG
  PATHS ${RTI_POSSIBLE_DIRS}
  PATH_SUFFIXES lib
  DOC "The RTI Library")

if (RTI_LIBRARY)
  set(RTI_LIBRARIES ${RTI_LIBRARY})
  RTI_MESSAGE_QUIETLY(RTI_FIND_QUIETLY STATUS "RTI library found: ${RTI_LIBRARY}")
else ()
  RTI_MESSAGE_QUIETLY(RTI_FIND_QUIETLY STATUS "RTI library NOT found")
endif ()

find_library(RTI_FEDTIME_LIBRARY
  NAMES FedTime
  PATHS ${RTI_POSSIBLE_DIRS}
  PATH_SUFFIXES lib
  DOC "The FedTime Library")

if (RTI_FEDTIME_LIBRARY)
  set(RTI_LIBRARIES ${RTI_LIBRARIES} ${RTI_FEDTIME_LIBRARY})
  RTI_MESSAGE_QUIETLY(RTI_FIND_QUIETLY STATUS "RTI FedTime found: ${RTI_FEDTIME_LIBRARY}")
endif ()

find_path(RTI_INCLUDE_DIR
  NAMES RTI.hh
  PATHS ${RTI_POSSIBLE_DIRS}
  PATH_SUFFIXES include
  DOC "The RTI Include Files")

if (RTI_INCLUDE_DIR)
  RTI_MESSAGE_QUIETLY(RTI_FIND_QUIETLY STATUS "RTI headers found: ${RTI_INCLUDE_DIR}")
else ()
  RTI_MESSAGE_QUIETLY(RTI_FIND_QUIETLY STATUS "RTI headers NOT found")
endif ()

# Set the modified system variables back to the original value.
set(CMAKE_FIND_LIBRARY_PREFIXES "${RTI_OLD_FIND_LIBRARY_PREFIXES}")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(RTI DEFAULT_MSG
  RTI_LIBRARY RTI_INCLUDE_DIR)
