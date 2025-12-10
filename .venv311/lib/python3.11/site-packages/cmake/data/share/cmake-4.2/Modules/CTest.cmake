# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
CTest
-----

Configure a project for testing with CTest/CDash

Include this module in the top CMakeLists.txt file of a project to
enable testing with CTest and dashboard submissions to CDash:

.. code-block:: cmake

  project(MyProject)
  ...
  include(CTest)

The module automatically creates the following variables:

:variable:`BUILD_TESTING`

  Option selecting whether ``include(CTest)`` calls :command:`enable_testing`.
  The option is ``ON`` by default when created by the module.

After including the module, use code like:

.. code-block:: cmake

  if(BUILD_TESTING)
    # ... CMake code to create tests ...
  endif()

to creating tests when testing is enabled.

To enable submissions to a CDash server, create a ``CTestConfig.cmake``
file at the top of the project with content such as:

.. code-block:: cmake

  set(CTEST_NIGHTLY_START_TIME "01:00:00 UTC")
  set(CTEST_SUBMIT_URL "http://my.cdash.org/submit.php?project=MyProject")

(the CDash server can provide the file to a project administrator who
configures ``MyProject``).  Settings in the config file are shared by
both this ``CTest`` module and the :manual:`ctest(1)` command-line
:ref:`Dashboard Client` mode (:option:`ctest -S`).

While building a project for submission to CDash, CTest scans the
build output for errors and warnings and reports them with surrounding
context from the build log.  This generic approach works for all build
tools, but does not give details about the command invocation that
produced a given problem.  One may get more detailed reports by setting
the :variable:`CTEST_USE_LAUNCHERS` variable:

.. code-block:: cmake

  set(CTEST_USE_LAUNCHERS 1)

in the ``CTestConfig.cmake`` file.
#]=======================================================================]

option(BUILD_TESTING "Build the testing tree." ON)

# function to turn generator name into a version string
# like vs9 or vs10
function(GET_VS_VERSION_STRING generator var)
  string(REGEX REPLACE "Visual Studio ([0-9][0-9]?)($|.*)" "\\1"
    NUMBER "${generator}")
    set(ver_string "vs${NUMBER}")
  set(${var} ${ver_string} PARENT_SCOPE)
endfunction()

include(CTestUseLaunchers)

if(BUILD_TESTING)
  # Setup some auxiliary macros
  macro(SET_IF_NOT_SET var val)
    if(NOT DEFINED "${var}")
      set("${var}" "${val}")
    endif()
  endmacro()

  macro(SET_IF_SET var val)
    if(NOT "${val}" STREQUAL "")
      set("${var}" "${val}")
    endif()
  endmacro()

  macro(SET_IF_SET_AND_NOT_SET var val)
    if(NOT "${val}" STREQUAL "")
      SET_IF_NOT_SET("${var}" "${val}")
    endif()
  endmacro()

  # Make sure testing is enabled
  enable_testing()

  if(EXISTS "${PROJECT_SOURCE_DIR}/CTestConfig.cmake")
    include("${PROJECT_SOURCE_DIR}/CTestConfig.cmake")
    SET_IF_SET_AND_NOT_SET(NIGHTLY_START_TIME "${CTEST_NIGHTLY_START_TIME}")
    SET_IF_SET_AND_NOT_SET(SUBMIT_URL "${CTEST_SUBMIT_URL}")
    SET_IF_SET_AND_NOT_SET(DROP_METHOD "${CTEST_DROP_METHOD}")
    SET_IF_SET_AND_NOT_SET(DROP_SITE "${CTEST_DROP_SITE}")
    SET_IF_SET_AND_NOT_SET(DROP_SITE_USER "${CTEST_DROP_SITE_USER}")
    SET_IF_SET_AND_NOT_SET(DROP_SITE_PASSWORD "${CTEST_DROP_SITE_PASSWORD}")
    SET_IF_SET_AND_NOT_SET(DROP_SITE_MODE "${CTEST_DROP_SITE_MODE}")
    SET_IF_SET_AND_NOT_SET(DROP_LOCATION "${CTEST_DROP_LOCATION}")
    SET_IF_SET_AND_NOT_SET(TRIGGER_SITE "${CTEST_TRIGGER_SITE}")
    SET_IF_SET_AND_NOT_SET(UPDATE_TYPE "${CTEST_UPDATE_TYPE}")
  endif()

  # the project can have a DartConfig.cmake file
  if(EXISTS "${PROJECT_SOURCE_DIR}/DartConfig.cmake")
    include("${PROJECT_SOURCE_DIR}/DartConfig.cmake")
  else()
    # Dashboard is opened for submissions for a 24 hour period starting at
    # the specified NIGHTLY_START_TIME. Time is specified in 24 hour format.
    SET_IF_NOT_SET (NIGHTLY_START_TIME "00:00:00 EDT")
    SET_IF_NOT_SET(DROP_METHOD "http")
    SET_IF_NOT_SET (COMPRESS_SUBMISSION ON)
  endif()
  SET_IF_NOT_SET (NIGHTLY_START_TIME "00:00:00 EDT")

  if(NOT SUBMIT_URL)
    set(SUBMIT_URL "${DROP_METHOD}://")
    if(DROP_SITE_USER)
      string(APPEND SUBMIT_URL "${DROP_SITE_USER}")
      if(DROP_SITE_PASSWORD)
        string(APPEND SUBMIT_URL ":${DROP_SITE_PASSWORD}")
      endif()
      string(APPEND SUBMIT_URL "@")
    endif()
    string(APPEND SUBMIT_URL "${DROP_SITE}${DROP_LOCATION}")
  endif()

  if(NOT UPDATE_TYPE)
    if(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/CVS")
      set(UPDATE_TYPE cvs)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.svn")
      set(UPDATE_TYPE svn)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.bzr")
      set(UPDATE_TYPE bzr)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.hg")
      set(UPDATE_TYPE hg)
    elseif(EXISTS "${CMAKE_CURRENT_SOURCE_DIR}/.git")
      set(UPDATE_TYPE git)
    endif()
  endif()

  string(TOLOWER "${UPDATE_TYPE}" _update_type)
  if("${_update_type}" STREQUAL "cvs")
    find_program(CVSCOMMAND cvs )
    set(CVS_UPDATE_OPTIONS "-d -A -P" CACHE STRING
      "Options passed to the cvs update command.")
    set(UPDATE_COMMAND "${CVSCOMMAND}")
    set(UPDATE_OPTIONS "${CVS_UPDATE_OPTIONS}")
  elseif("${_update_type}" STREQUAL "svn")
    find_program(SVNCOMMAND svn)
    set(UPDATE_COMMAND "${SVNCOMMAND}")
    set(UPDATE_OPTIONS "${SVN_UPDATE_OPTIONS}")
  elseif("${_update_type}" STREQUAL "bzr")
    find_program(BZRCOMMAND bzr)
    set(UPDATE_COMMAND "${BZRCOMMAND}")
    set(UPDATE_OPTIONS "${BZR_UPDATE_OPTIONS}")
  elseif("${_update_type}" STREQUAL "hg")
    find_program(HGCOMMAND hg)
    set(UPDATE_COMMAND "${HGCOMMAND}")
    set(UPDATE_OPTIONS "${HG_UPDATE_OPTIONS}")
  elseif("${_update_type}" STREQUAL "git")
    find_program(GITCOMMAND git)
    set(UPDATE_COMMAND "${GITCOMMAND}")
    set(UPDATE_OPTIONS "${GIT_UPDATE_OPTIONS}")
  elseif("${_update_type}" STREQUAL "p4")
    find_program(P4COMMAND p4)
    set(UPDATE_COMMAND "${P4COMMAND}")
    set(UPDATE_OPTIONS "${P4_UPDATE_OPTIONS}")
  endif()

  set(DART_TESTING_TIMEOUT 1500 CACHE STRING
    "Maximum time allowed before CTest will kill the test.")

  set(CTEST_SUBMIT_RETRY_DELAY 5 CACHE STRING
    "How long to wait between timed-out CTest submissions.")
  set(CTEST_SUBMIT_RETRY_COUNT 3 CACHE STRING
    "How many times to retry timed-out CTest submissions.")

  find_program(MEMORYCHECK_COMMAND
    NAMES purify valgrind boundscheck drmemory cuda-memcheck compute-sanitizer
    PATHS
    "[HKEY_LOCAL_MACHINE\\SOFTWARE\\Rational Software\\Purify\\Setup;InstallFolder]"
    DOC "Path to the memory checking command, used for memory error detection."
    )
  set(MEMORYCHECK_SUPPRESSIONS_FILE "" CACHE FILEPATH
    "File that contains suppressions for the memory checker")
  find_program(COVERAGE_COMMAND gcov DOC
    "Path to the coverage program that CTest uses for performing coverage inspection"
    )
  set(COVERAGE_EXTRA_FLAGS "-l" CACHE STRING
    "Extra command line flags to pass to the coverage tool")

  # set the site name
  if(COMMAND cmake_host_system_information)
    cmake_host_system_information(RESULT _ctest_hostname QUERY HOSTNAME)
    set(SITE "${_ctest_hostname}" CACHE STRING "Name of the computer/site where compile is being run")
    unset(_ctest_hostname)
  else()
    # This code path is needed for CMake itself during bootstrap.
    site_name(SITE)
  endif()
  # set the build name
  if(NOT BUILDNAME)
    set(DART_COMPILER "${CMAKE_CXX_COMPILER}")
    if(NOT DART_COMPILER)
      set(DART_COMPILER "${CMAKE_C_COMPILER}")
    endif()
    if(NOT DART_COMPILER)
      set(DART_COMPILER "unknown")
    endif()
    if(WIN32)
      set(DART_NAME_COMPONENT "NAME_WE")
    else()
      set(DART_NAME_COMPONENT "NAME")
    endif()
    if(NOT BUILD_NAME_SYSTEM_NAME)
      set(BUILD_NAME_SYSTEM_NAME "${CMAKE_SYSTEM_NAME}")
    endif()
    if(WIN32)
      set(BUILD_NAME_SYSTEM_NAME "Win32")
    endif()
    if(UNIX OR BORLAND)
      get_filename_component(DART_COMPILER_NAME
        "${DART_COMPILER}" ${DART_NAME_COMPONENT})
    else()
      get_filename_component(DART_COMPILER_NAME
        "${CMAKE_MAKE_PROGRAM}" ${DART_NAME_COMPONENT})
    endif()
    if(DART_COMPILER_NAME MATCHES "devenv")
      GET_VS_VERSION_STRING("${CMAKE_GENERATOR}" DART_COMPILER_NAME)
    endif()
    set(BUILDNAME "${BUILD_NAME_SYSTEM_NAME}-${DART_COMPILER_NAME}")
  endif()

  # the build command
  build_command(MAKECOMMAND_DEFAULT_VALUE
    CONFIGURATION "\${CTEST_CONFIGURATION_TYPE}")
  set(MAKECOMMAND ${MAKECOMMAND_DEFAULT_VALUE}
    CACHE STRING "Command to build the project")

  # the default build configuration the ctest build handler will use
  # if there is no -C arg given to ctest:
  set(DEFAULT_CTEST_CONFIGURATION_TYPE "$ENV{CMAKE_CONFIG_TYPE}")
  if(DEFAULT_CTEST_CONFIGURATION_TYPE STREQUAL "")
    set(DEFAULT_CTEST_CONFIGURATION_TYPE "Release")
  endif()

  mark_as_advanced(
    BZRCOMMAND
    COVERAGE_COMMAND
    COVERAGE_EXTRA_FLAGS
    CTEST_SUBMIT_RETRY_DELAY
    CTEST_SUBMIT_RETRY_COUNT
    CVSCOMMAND
    CVS_UPDATE_OPTIONS
    DART_TESTING_TIMEOUT
    GITCOMMAND
    P4COMMAND
    HGCOMMAND
    MAKECOMMAND
    MEMORYCHECK_COMMAND
    MEMORYCHECK_SUPPRESSIONS_FILE
    SITE
    SVNCOMMAND
    )
  if(NOT RUN_FROM_DART)
    set(RUN_FROM_CTEST_OR_DART 1)
    include(CTestTargets)
    set(RUN_FROM_CTEST_OR_DART)
  endif()
endif()
