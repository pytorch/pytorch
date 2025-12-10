# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindDCMTK
---------

Finds the DICOM ToolKit (DCMTK) libraries and applications:

.. code-block:: cmake

  find_package(DCMTK [...])

DCMTK is a set of libraries and applications implementing large parts of
the DICOM Standard (Digital Imaging and Communications in Medicine).

.. versionadded:: 3.5
  This module is now able to find a version of DCMTK that does or does not
  export a ``DCMTKConfig.cmake`` file.

  DCMTK since its version `3.6.1_20140617
  <https://git.dcmtk.org/?p=dcmtk.git;a=commit;h=662ae187c493c6b9a73dd5e3875372cebd0c11fe>`_
  supports and installs :ref:`package configuration file
  <Config File Packages>` (``DCMTKConfig.cmake``) for use with the
  :command:`find_package` command in *config mode*.

  This module now applies a two-step process:

  * Step 1: Attempts to find DCMTK version providing a ``DCMTKConfig.cmake``
    file and, if found, returns the results without further action.
  * Step 2: If step 1 failed, this module falls back to *module mode*
    (it searches standard locations) and sets the ``DCMTK_*`` result
    variables.

  Until all clients update to the more recent DCMTK, build systems will need
  to support different versions of DCMTK.

  On any given system, the following combinations of DCMTK versions could
  be considered for the DCMTK installed on the system (for example, via a
  system package manager), or locally (for example, a custom installation,
  or through the :module:`FetchContent` module):

  ===== ================== =================== ============
  Case  System DCMTK       Local DCMTK         Supported?
  ===== ================== =================== ============
  A     N/A                [ ] DCMTKConfig     YES
  B     N/A                [X] DCMTKConfig     YES
  C     [ ] DCMTKConfig    N/A                 YES
  D     [X] DCMTKConfig    N/A                 YES
  E     [ ] DCMTKConfig    [ ] DCMTKConfig     YES (*)
  F     [X] DCMTKConfig    [ ] DCMTKConfig     NO
  G     [ ] DCMTKConfig    [X] DCMTKConfig     YES
  H     [X] DCMTKConfig    [X] DCMTKConfig     YES
  ===== ================== =================== ============

  Legend:

    (*)
      See the `Troubleshooting`_ section.

    N/A
      DCMTK is not available.

    [ ] DCMTKConfig
      DCMTK does NOT export a ``DCMTKConfig.cmake`` file.

    [X] DCMTKConfig
      DCMTK exports a ``DCMTKConfig.cmake`` file.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``DCMTK_FOUND``
  Boolean indicating whether DCMTK was found.

``DCMTK_INCLUDE_DIRS``
  Include directories containing headers needed to use DCMTK.

``DCMTK_LIBRARIES``
  Libraries needed to link against to use DCMTK.

Hints
^^^^^

This module accepts the following variables:

``DCMTK_DIR``
  (optional) Source directory for DCMTK.

Troubleshooting
^^^^^^^^^^^^^^^

.. rubric:: What to do if project finds a different version of DCMTK?

Remove DCMTK entry from the CMake cache per :command:`find_package`
documentation, and re-run configuration.  To find DCMTK on custom location
use variables such as :variable:`CMAKE_PREFIX_PATH`, or ``DCMTK_DIR``.

Examples
^^^^^^^^

Example: Finding DCMTK
""""""""""""""""""""""

Finding DCMTK with this module:

.. code-block:: cmake

  find_package(DCMTK)

Example: Finding DCMTK Without This Module
""""""""""""""""""""""""""""""""""""""""""

To explicitly use the ``DCMTKConfig.cmake`` package configuration file
(recommended when possible) and find DCMTK in *config mode* without using
this module, the ``NO_MODULE`` option can be given to
:command:`find_package`:

.. code-block:: cmake

  find_package(DCMTK NO_MODULE)

Example: Creating Imported Target
"""""""""""""""""""""""""""""""""

In the following example, DCMTK is searched with this module and
an :ref:`imported target <Imported Targets>` is conditionally created to
provide DCMTK usage requirements which can be easily linked to project
targets.  For example, if DCMTK is found in *config mode*, the
``DCMTK::DCMTK`` imported target will be available through the found config
files instead:

.. code-block:: cmake

  find_package(DCMTK)

  # Upstream DCMTKConfig.cmake already provides DCMTK::DCMTK imported target
  if(DCMTK_FOUND AND NOT TARGET DCMTK::DCMTK)
    add_library(DCMTK::DCMTK INTERFACE IMPORTED)
    set_target_properties(
      DCMTK:DCMTK
      PROPERTIES
        INTERFACE_LINK_LIBRARIES "${DCMTK_LIBRARIES}"
        INTERFACE_INCLUDE_DIRECTORIES "${DCMTK_INCLUDE_DIRS}"
    )
  endif()

  target_link_libraries(example PRIVATE DCMTK::DCMTK)
#]=======================================================================]

#
# Written for VXL by Amitha Perera.
# Upgraded for GDCM by Mathieu Malaterre.
# Modified for EasyViz by Thomas Sondergaard.
#

set(_dcmtk_dir_description "The directory of DCMTK build or install tree.")

# Ensure that DCMTK_DIR is set to a reasonable default value
# so that DCMTK libraries can be found on a standard Unix distribution.
# It also overwrite the value of DCMTK_DIR after this one has been
# set by a successful discovery of DCMTK by the unpatched FindDCMTK.cmake module
# distributed with CMake (as of 0167cea)
if(NOT DCMTK_DIR OR DCMTK_DIR STREQUAL "/usr/include/dcmtk")
  set(DCMTK_DIR "/usr" CACHE PATH ${_dcmtk_dir_description} FORCE)
endif()

set(_SAVED_DCMTK_DIR ${DCMTK_DIR})

#
# Step1: Attempt to find a version of DCMTK providing a DCMTKConfig.cmake file.
#
if(NOT DCMTK_FIND_QUIETLY)
  message(CHECK_START "Trying to find DCMTK expecting DCMTKConfig.cmake")
endif()
find_package(DCMTK QUIET NO_MODULE)
if(DCMTK_FOUND
    AND NOT "x" STREQUAL "x${DCMTK_LIBRARIES}"
    AND NOT "x" STREQUAL "x${DCMTK_INCLUDE_DIRS}")

  if(NOT DCMTK_FIND_QUIETLY)
    message(CHECK_PASS "ok")
  endif()
  return()
else()
  if(NOT DCMTK_FIND_QUIETLY)
    message(CHECK_FAIL "failed")
  endif()
endif()

if(NOT DCMTK_FIND_QUIETLY)
  message(STATUS "Trying to find DCMTK relying on FindDCMTK.cmake")
endif()

# Restore the value reset by the previous call to 'find_package(DCMTK QUIET NO_MODULE)'
set(DCMTK_DIR ${_SAVED_DCMTK_DIR} CACHE PATH ${_dcmtk_dir_description} FORCE)


#
# Step2: Attempt to find a version of DCMTK that does NOT provide a DCMTKConfig.cmake file.
#

# prefer DCMTK_DIR over default system paths like /usr/lib
if(DCMTK_DIR)
  set(CMAKE_PREFIX_PATH ${DCMTK_DIR}/lib ${CMAKE_PREFIX_PATH}) # this is given to FIND_LIBRARY or FIND_PATH
endif()

# Find all libraries, store debug and release separately
foreach(lib
    dcmpstat
    dcmsr
    dcmsign
    dcmtls
    dcmqrdb
    dcmnet
    dcmjpeg
    dcmimage
    dcmimgle
    dcmdata
    oflog
    ofstd
    ijg12
    ijg16
    ijg8
    )

  # Find Release libraries
  find_library(DCMTK_${lib}_LIBRARY_RELEASE
    ${lib}
    PATHS
    ${DCMTK_DIR}/${lib}/libsrc
    ${DCMTK_DIR}/${lib}/libsrc/Release
    ${DCMTK_DIR}/${lib}/Release
    ${DCMTK_DIR}/lib
    ${DCMTK_DIR}/lib/Release
    ${DCMTK_DIR}/dcmjpeg/lib${lib}/Release
    NO_DEFAULT_PATH
    )

  # Find Debug libraries
  find_library(DCMTK_${lib}_LIBRARY_DEBUG
    ${lib}${DCMTK_CMAKE_DEBUG_POSTFIX}
    PATHS
    ${DCMTK_DIR}/${lib}/libsrc
    ${DCMTK_DIR}/${lib}/libsrc/Debug
    ${DCMTK_DIR}/${lib}/Debug
    ${DCMTK_DIR}/lib
    ${DCMTK_DIR}/lib/Debug
    ${DCMTK_DIR}/dcmjpeg/lib${lib}/Debug
    NO_DEFAULT_PATH
    )

  mark_as_advanced(DCMTK_${lib}_LIBRARY_RELEASE)
  mark_as_advanced(DCMTK_${lib}_LIBRARY_DEBUG)

  # Add libraries to variable according to build type
  if(DCMTK_${lib}_LIBRARY_RELEASE)
    list(APPEND DCMTK_LIBRARIES optimized ${DCMTK_${lib}_LIBRARY_RELEASE})
  endif()

  if(DCMTK_${lib}_LIBRARY_DEBUG)
    list(APPEND DCMTK_LIBRARIES debug ${DCMTK_${lib}_LIBRARY_DEBUG})
  endif()

endforeach()

set(CMAKE_THREAD_LIBS_INIT)
if(DCMTK_oflog_LIBRARY_RELEASE OR DCMTK_oflog_LIBRARY_DEBUG)
  # Hack - Not having a DCMTKConfig.cmake file to read the settings from, we will attempt to
  # find the library in all cases.
  # Ideally, pthread library should be discovered only if DCMTK_WITH_THREADS is enabled.
  find_package(Threads)
endif()

if(CMAKE_THREAD_LIBS_INIT)
  list(APPEND DCMTK_LIBRARIES ${CMAKE_THREAD_LIBS_INIT})
endif()

#
# SPECIFIC CASE FOR DCMTK BUILD DIR as DCMTK_DIR
# (as opposed to a DCMTK install dir)
# Have to find the source directory.
if(EXISTS ${DCMTK_DIR}/CMakeCache.txt)
          load_cache(${DCMTK_DIR} READ_WITH_PREFIX "EXT"
          DCMTK_SOURCE_DIR)
  if(NOT EXISTS ${EXTDCMTK_SOURCE_DIR})
    message(FATAL_ERROR
      "DCMTK build directory references
nonexistent DCMTK source directory ${EXTDCMTK_SOURCE_DIR}")
  endif()
endif()

set(DCMTK_config_TEST_HEADER osconfig.h)
set(DCMTK_dcmdata_TEST_HEADER dctypes.h)
set(DCMTK_dcmimage_TEST_HEADER dicoimg.h)
set(DCMTK_dcmimgle_TEST_HEADER dcmimage.h)
set(DCMTK_dcmjpeg_TEST_HEADER djdecode.h)
set(DCMTK_dcmnet_TEST_HEADER assoc.h)
set(DCMTK_dcmpstat_TEST_HEADER dcmpstat.h)
set(DCMTK_dcmqrdb_TEST_HEADER dcmqrdba.h)
set(DCMTK_dcmsign_TEST_HEADER sicert.h)
set(DCMTK_dcmsr_TEST_HEADER dsrtree.h)
set(DCMTK_dcmtls_TEST_HEADER tlslayer.h)
set(DCMTK_ofstd_TEST_HEADER ofstdinc.h)
set(DCMTK_oflog_TEST_HEADER oflog.h)
set(DCMTK_dcmjpls_TEST_HEADER djlsutil.h)

set(DCMTK_INCLUDE_DIR_NAMES)

foreach(dir
    config
    dcmdata
    dcmimage
    dcmimgle
    dcmjpeg
    dcmjpls
    dcmnet
    dcmpstat
    dcmqrdb
    dcmsign
    dcmsr
    dcmtls
    ofstd
    oflog)
  if(EXTDCMTK_SOURCE_DIR)
    set(SOURCE_DIR_PATH
      ${EXTDCMTK_SOURCE_DIR}/${dir}/include/dcmtk/${dir})
  endif()
  find_path(DCMTK_${dir}_INCLUDE_DIR
    ${DCMTK_${dir}_TEST_HEADER}
    PATHS
    ${DCMTK_DIR}/${dir}/include
    ${DCMTK_DIR}/${dir}
    ${DCMTK_DIR}/include/dcmtk/${dir}
    ${DCMTK_DIR}/${dir}/include/dcmtk/${dir}
    ${DCMTK_DIR}/include/${dir}
    ${SOURCE_DIR_PATH}
    )
  mark_as_advanced(DCMTK_${dir}_INCLUDE_DIR)
  list(APPEND DCMTK_INCLUDE_DIR_NAMES DCMTK_${dir}_INCLUDE_DIR)

  if(DCMTK_${dir}_INCLUDE_DIR)
    # add the 'include' path so eg
    #include "dcmtk/dcmimgle/dcmimage.h"
    # works
    get_filename_component(_include ${DCMTK_${dir}_INCLUDE_DIR} PATH)
    get_filename_component(_include ${_include} PATH)
    list(APPEND
      DCMTK_INCLUDE_DIRS
      ${DCMTK_${dir}_INCLUDE_DIR}
      ${_include})
  endif()
endforeach()

list(APPEND DCMTK_INCLUDE_DIRS ${DCMTK_DIR}/include)

if(WIN32)
  list(APPEND DCMTK_LIBRARIES netapi32 wsock32)
endif()

if(DCMTK_ofstd_INCLUDE_DIR)
  get_filename_component(DCMTK_dcmtk_INCLUDE_DIR
    ${DCMTK_ofstd_INCLUDE_DIR}
    PATH
    CACHE)
  list(APPEND DCMTK_INCLUDE_DIRS ${DCMTK_dcmtk_INCLUDE_DIR})
  mark_as_advanced(DCMTK_dcmtk_INCLUDE_DIR)
endif()

# Compatibility: This variable is deprecated
set(DCMTK_INCLUDE_DIR ${DCMTK_INCLUDE_DIRS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(DCMTK
  REQUIRED_VARS ${DCMTK_INCLUDE_DIR_NAMES} DCMTK_LIBRARIES
  FAIL_MESSAGE "Please set DCMTK_DIR and re-run configure")

# Workaround bug in packaging of DCMTK 3.6.0 on Debian.
# See http://bugs.debian.org/cgi-bin/bugreport.cgi?bug=637687
if(DCMTK_FOUND AND UNIX AND NOT APPLE)
  include(${CMAKE_CURRENT_LIST_DIR}/CheckIncludeFiles.cmake)
  set(CMAKE_REQUIRED_FLAGS )
  set(CMAKE_REQUIRED_DEFINITIONS )
  set(CMAKE_REQUIRED_INCLUDES ${DCMTK_INCLUDE_DIRS})
  set(CMAKE_REQUIRED_LIBRARIES ${DCMTK_LIBRARIES})
  set(CMAKE_REQUIRED_QUIET ${DCMTK_FIND_QUIETLY})
  check_include_files("dcmtk/config/osconfig.h;dcmtk/ofstd/ofstream.h" DCMTK_HAVE_CONFIG_H_OPTIONAL LANGUAGE CXX)
  if(NOT DCMTK_HAVE_CONFIG_H_OPTIONAL)
    set(DCMTK_DEFINITIONS "HAVE_CONFIG_H")
  endif()
endif()
