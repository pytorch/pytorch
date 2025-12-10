# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
FindSquish
----------

Finds Squish, a cross-platform automated GUI testing framework for
applications built on various GUI technologies:

.. code-block:: cmake

  find_package(Squish [<version>] [...])

Squish supports testing of both native and cross-platform toolkits, such as
Qt, Java, and Tk.

Result Variables
^^^^^^^^^^^^^^^^

This module defines the following variables:

``Squish_FOUND``
  .. versionadded:: 3.3

  Boolean indicating whether (the requested version of) Squish was found.

``Squish_VERSION``
  .. versionadded:: 4.2

  The full version of the Squish found.

``SQUISH_INSTALL_DIR_FOUND``
  Boolean indicating whether the Squish installation directory was found.

``SQUISH_SERVER_EXECUTABLE_FOUND``
  Boolean indicating whether the Squish server executable was found.

``SQUISH_CLIENT_EXECUTABLE_FOUND``
  Boolean indicating whether the Squish client executable was found.

Cache Variables
^^^^^^^^^^^^^^^

The following cache variables may also be set:

``SQUISH_INSTALL_DIR``
  The Squish installation directory containing ``bin``, ``lib``, etc.

``SQUISH_SERVER_EXECUTABLE``
  The path to the ``squishserver`` executable.

``SQUISH_CLIENT_EXECUTABLE``
  The path to the ``squishrunner`` executable.

Commands
^^^^^^^^

This module provides the following commands, if Squish is found:

.. command:: squish_add_test

  Adds a Squish test to the project:

  .. code-block:: cmake

    squish_add_test(
      <name>
      AUT <target>
      SUITE <suite-name>
      TEST <squish-test-case-name>
      [PRE_COMMAND <command>]
      [POST_COMMAND <command>]
      [SETTINGSGROUP <group>]
    )

  This command is built on top of the :command:`add_test` command and adds a
  Squish test called ``<name>`` to the CMake project.  It supports Squish
  versions 4 and newer.

  During the CMake testing phase, the Squish server is started, the test is
  executed on the client, and the server is stopped once the test completes.  If
  any of these steps fail (including if the test itself fails), a fatal error is
  raised indicating the test did not pass.

  The arguments are:

  ``<name>``
    The name of the test.  This is passed as the first argument to the
    :command:`add_test` command.

  ``AUT <target>``
    The name of the CMake target to be used as the AUT (Application Under Test),
    i.e., the executable that will be tested.

  ``SUITE <suite-name>``
    Either the full path to the Squish test suite or just the suite name (i.e.,
    the last directory name of the suite).  In the latter case, the
    ``CMakeLists.txt`` invoking ``squish_add_test()`` must reside in the parent
    directory of the suite.

  ``TEST <squish-test-case-name>``
    The name of the Squish test, corresponding to the subdirectory of the test
    within the suite directory.

  ``PRE_COMMAND <command>``
    An optional command to execute before starting the Squish test.  Pass it as
    a string.  This may be a single command, or a :ref:`semicolon-separated list
    <CMake Language Lists>` of command and arguments.

  ``POST_COMMAND <command>``
    An optional command to execute after the Squish test has completed.  Pass it
    as a string.  This may be a single command, or a :ref:`semicolon-separated
    list <CMake Language Lists>` of command and arguments.

  ``SETTINGSGROUP <group>``
    .. deprecated:: 3.18
      This argument is now ignored.  It was previously used to specify a
      settings group name for executing the test instead of the default value
      ``CTest_<username>``.

  .. versionchanged:: 3.18
    In previous CMake versions, this command was named ``squish_v4_add_test()``.

.. command:: squish_v3_add_test

  Adds a Squish test to the project, when using Squish version 3.x:

  .. code-block:: cmake

    squish_v3_add_test(
      <test-name>
      <application-under-test>
      <squish-test-case-name>
      <environment-variables>
      <test-wrapper>
    )

  .. note::
    This command is for Squish version 3, which is not maintained anymore.  Use
    a newer Squish version, and ``squish_add_test()`` command.

  The arguments are:

  ``<name>``
    The name of the test.

  ``<application-under-test>``
    The path to the executable used as the AUT (Application Under Test), i.e.,
    the executable that will be tested.

  ``<squish-test-case-name>``
    The name of the Squish test, corresponding to the subdirectory of the test
    within the suite directory.

  ``<environment-variables>``
    A semicolon-separated list of environment variables and their values
    (VAR=VALUE).

  ``<test-wrapper>``
    A string of one or more (semicolon-separated list) test wrappers needed by
    the test case.

Deprecated Variables
^^^^^^^^^^^^^^^^^^^^

The following variables are provided for backward compatibility:

``SQUISH_FOUND``
  .. deprecated:: 4.2
    Use ``Squish_FOUND``, which has the same value.

  Boolean indicating whether (the requested version of) Squish was found.

``SQUISH_VERSION``
  .. deprecated:: 4.2
    Superseded by the ``Squish_VERSION``.

  The full version of the Squish found.

``SQUISH_VERSION_MAJOR``
  .. deprecated:: 4.2
    Superseded by the ``Squish_VERSION``.

  The major version of the Squish found.

``SQUISH_VERSION_MINOR``
  .. deprecated:: 4.2
    Superseded by the ``Squish_VERSION``.

  The minor version of the Squish found.

``SQUISH_VERSION_PATCH``
  .. deprecated:: 4.2
    Superseded by the ``Squish_VERSION``.

  The patch version of the Squish found.

Examples
^^^^^^^^

Finding Squish and specifying a minimum required version:

.. code-block:: cmake

  find_package(Squish 6.5)

Adding a Squish test:

.. code-block:: cmake

  enable_testing()

  find_package(Squish 6.5)
  if(Squish_FOUND)
    squish_add_test(
      projectTestName
      AUT projectApp
      SUITE ${CMAKE_CURRENT_SOURCE_DIR}/tests/projectSuite
      TEST someSquishTest
    )
  endif()

Example, how to use the ``squish_v3_add_test()`` command:

.. code-block:: cmake

  enable_testing()

  find_package(Squish 3.0)
  if(Squish_FOUND)
    squish_v3_add_test(
      projectTestName
      $<TARGET_FILE:projectApp>
      someSquishTest
      "FOO=1;BAR=2"
      testWrapper
    )
  endif()
#]=======================================================================]

set(SQUISH_INSTALL_DIR_STRING "Directory containing the bin, doc, and lib directories for Squish; this should be the root of the installation directory.")
set(SQUISH_SERVER_EXECUTABLE_STRING "The squishserver executable program.")
set(SQUISH_CLIENT_EXECUTABLE_STRING "The squishclient executable program.")

# Search only if the location is not already known.
if(NOT SQUISH_INSTALL_DIR)
  # Get the system search path as a list.
  file(TO_CMAKE_PATH "$ENV{PATH}" SQUISH_INSTALL_DIR_SEARCH2)

  # Construct a set of paths relative to the system search path.
  set(SQUISH_INSTALL_DIR_SEARCH "")
  foreach(dir ${SQUISH_INSTALL_DIR_SEARCH2})
    set(SQUISH_INSTALL_DIR_SEARCH ${SQUISH_INSTALL_DIR_SEARCH} "${dir}/../lib/fltk")
  endforeach()
  string(REPLACE "//" "/" SQUISH_INSTALL_DIR_SEARCH "${SQUISH_INSTALL_DIR_SEARCH}")

  # Look for an installation
  find_path(SQUISH_INSTALL_DIR
    NAMES bin/squishrunner bin/squishrunner.exe
    HINTS
    # Look for an environment variable SQUISH_INSTALL_DIR.
      ENV SQUISH_INSTALL_DIR

    # Look in places relative to the system executable search path.
    ${SQUISH_INSTALL_DIR_SEARCH}

    DOC "The ${SQUISH_INSTALL_DIR_STRING}"
    )
endif()

# search for the executables
if(SQUISH_INSTALL_DIR)
  set(SQUISH_INSTALL_DIR_FOUND 1)

  # find the client program
  if(NOT SQUISH_CLIENT_EXECUTABLE)
    find_program(SQUISH_CLIENT_EXECUTABLE ${SQUISH_INSTALL_DIR}/bin/squishrunner${CMAKE_EXECUTABLE_SUFFIX} DOC "The ${SQUISH_CLIENT_EXECUTABLE_STRING}")
  endif()

  # find the server program
  if(NOT SQUISH_SERVER_EXECUTABLE)
    find_program(SQUISH_SERVER_EXECUTABLE ${SQUISH_INSTALL_DIR}/bin/squishserver${CMAKE_EXECUTABLE_SUFFIX} DOC "The ${SQUISH_SERVER_EXECUTABLE_STRING}")
  endif()

else()
  set(SQUISH_INSTALL_DIR_FOUND 0)
endif()


unset(Squish_VERSION)
unset(SQUISH_VERSION)
unset(SQUISH_VERSION_MAJOR)
unset(SQUISH_VERSION_MINOR)
unset(SQUISH_VERSION_PATCH)

# record if executables are set
if(SQUISH_CLIENT_EXECUTABLE)
  set(SQUISH_CLIENT_EXECUTABLE_FOUND 1)
  execute_process(COMMAND "${SQUISH_CLIENT_EXECUTABLE}" --version
                  OUTPUT_VARIABLE _squishVersionOutput
                  ERROR_QUIET )
  if("${_squishVersionOutput}" MATCHES "([0-9]+)\\.([0-9]+)\\.([0-9]+)")
    set(SQUISH_VERSION_MAJOR "${CMAKE_MATCH_1}")
    set(SQUISH_VERSION_MINOR "${CMAKE_MATCH_2}")
    set(SQUISH_VERSION_PATCH "${CMAKE_MATCH_3}")
    set(Squish_VERSION "${SQUISH_VERSION_MAJOR}.${SQUISH_VERSION_MINOR}.${SQUISH_VERSION_PATCH}" )
    set(SQUISH_VERSION "${Squish_VERSION}")
  endif()
else()
  set(SQUISH_CLIENT_EXECUTABLE_FOUND 0)
endif()

if(SQUISH_SERVER_EXECUTABLE)
  set(SQUISH_SERVER_EXECUTABLE_FOUND 1)
else()
  set(SQUISH_SERVER_EXECUTABLE_FOUND 0)
endif()

# record if Squish was found
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Squish  REQUIRED_VARS  SQUISH_INSTALL_DIR SQUISH_CLIENT_EXECUTABLE SQUISH_SERVER_EXECUTABLE
                                          VERSION_VAR Squish_VERSION)


set(_SQUISH_MODULE_DIR "${CMAKE_CURRENT_LIST_DIR}")

macro(squish_v3_add_test testName testAUT testCase envVars testWrapper)
  if("${SQUISH_VERSION_MAJOR}" STRGREATER "3")
    message(STATUS "Using squish_v3_add_test(), but SQUISH_VERSION_MAJOR is ${SQUISH_VERSION_MAJOR}.\nThis may not work.")
  endif()

  # There's no target used for this command, so we don't need to do anything
  # here for CMP0178.
  add_test(${testName}
    ${CMAKE_COMMAND} -V -VV
    "-Dsquish_version:STRING=3"
    "-Dsquish_aut:STRING=${testAUT}"
    "-Dsquish_server_executable:STRING=${SQUISH_SERVER_EXECUTABLE}"
    "-Dsquish_client_executable:STRING=${SQUISH_CLIENT_EXECUTABLE}"
    "-Dsquish_libqtdir:STRING=${QT_LIBRARY_DIR}"
    "-Dsquish_test_case:STRING=${testCase}"
    "-Dsquish_env_vars:STRING=${envVars}"
    "-Dsquish_wrapper:STRING=${testWrapper}"
    "-Dsquish_module_dir:STRING=${_SQUISH_MODULE_DIR}"
    -P "${_SQUISH_MODULE_DIR}/SquishTestScript.cmake"
    )
  set_tests_properties(${testName}
    PROPERTIES FAIL_REGULAR_EXPRESSION "FAILED;ERROR;FATAL"
    )
endmacro()


function(squish_v4_add_test testName)
  if(NOT "${SQUISH_VERSION_MAJOR}" STRGREATER "3")
    message(STATUS "Using squish_add_test(), but SQUISH_VERSION_MAJOR is ${SQUISH_VERSION_MAJOR}.\nThis may not work.")
  endif()

  set(oneValueArgs AUT SUITE TEST SETTINGSGROUP PRE_COMMAND POST_COMMAND)

  cmake_parse_arguments(_SQUISH "" "${oneValueArgs}" "" ${ARGN} )

  if(_SQUISH_UNPARSED_ARGUMENTS)
    message(FATAL_ERROR "Unknown keywords given to SQUISH_ADD_TEST(): \"${_SQUISH_UNPARSED_ARGUMENTS}\"")
  endif()

  if(NOT _SQUISH_AUT)
    message(FATAL_ERROR "Required argument AUT not given for SQUISH_ADD_TEST()")
  endif()

  if(NOT _SQUISH_SUITE)
    message(FATAL_ERROR "Required argument SUITE not given for SQUISH_ADD_TEST()")
  endif()

  if(NOT _SQUISH_TEST)
    message(FATAL_ERROR "Required argument TEST not given for SQUISH_ADD_TEST()")
  endif()

  get_filename_component(absTestSuite "${_SQUISH_SUITE}" ABSOLUTE)
  if(NOT EXISTS "${absTestSuite}")
    message(FATAL_ERROR "Could not find squish test suite ${_SQUISH_SUITE} (checked ${absTestSuite})")
  endif()

  set(absTestCase "${absTestSuite}/${_SQUISH_TEST}")
  if(NOT EXISTS "${absTestCase}")
    message(FATAL_ERROR "Could not find squish testcase ${_SQUISH_TEST} (checked ${absTestCase})")
  endif()

  if(_SQUISH_SETTINGSGROUP)
    message("SETTINGSGROUP is deprecated and will be ignored.")
  endif()

  # There's no target used for this command, so we don't need to do anything
  # here for CMP0178.
  add_test(NAME ${testName}
    COMMAND ${CMAKE_COMMAND} -V -VV
    "-Dsquish_version:STRING=4"
    "-Dsquish_aut:STRING=$<TARGET_FILE_BASE_NAME:${_SQUISH_AUT}>"
    "-Dsquish_aut_dir:STRING=$<TARGET_FILE_DIR:${_SQUISH_AUT}>"
    "-Dsquish_server_executable:STRING=${SQUISH_SERVER_EXECUTABLE}"
    "-Dsquish_client_executable:STRING=${SQUISH_CLIENT_EXECUTABLE}"
    "-Dsquish_libqtdir:STRING=${QT_LIBRARY_DIR}"
    "-Dsquish_test_suite:STRING=${absTestSuite}"
    "-Dsquish_test_case:STRING=${_SQUISH_TEST}"
    "-Dsquish_env_vars:STRING=${envVars}"
    "-Dsquish_wrapper:STRING=${testWrapper}"
    "-Dsquish_module_dir:STRING=${_SQUISH_MODULE_DIR}"
    "-Dsquish_pre_command:STRING=${_SQUISH_PRE_COMMAND}"
    "-Dsquish_post_command:STRING=${_SQUISH_POST_COMMAND}"
    -P "${_SQUISH_MODULE_DIR}/SquishTestScript.cmake"
    )
  set_tests_properties(${testName}
    PROPERTIES FAIL_REGULAR_EXPRESSION "FAIL;FAILED;ERROR;FATAL"
    )
endfunction()

macro(squish_add_test)
  if("${SQUISH_VERSION_MAJOR}" STRGREATER "3")
    squish_v4_add_test(${ARGV})
  else()
    squish_v3_add_test(${ARGV})
  endif()
endmacro()
