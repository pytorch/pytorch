# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
SquishTestScript
----------------

.. note::

  This module is not intended to be included directly in a CMake project.  It is
  an internal CMake test script used to launch GUI tests with Squish.  For usage
  details, refer to the :command:`squish_add_test` command documentation in the
  ``FindSquish`` module.
#]=======================================================================]

# print out the variable that we are using
message(STATUS "squish_aut='${squish_aut}'")
message(STATUS "squish_aut_dir='${squish_aut_dir}'")

message(STATUS "squish_version='${squish_version}'")
message(STATUS "squish_server_executable='${squish_server_executable}'")
message(STATUS "squish_client_executable='${squish_client_executable}'")
message(STATUS "squish_libqtdir ='${squish_libqtdir}'")
message(STATUS "squish_test_suite='${squish_test_suite}'")
message(STATUS "squish_test_case='${squish_test_case}'")
message(STATUS "squish_wrapper='${squish_wrapper}'")
message(STATUS "squish_env_vars='${squish_env_vars}'")
message(STATUS "squish_module_dir='${squish_module_dir}'")
message(STATUS "squish_pre_command='${squish_pre_command}'")
message(STATUS "squish_post_command='${squish_post_command}'")

# parse environment variables
foreach(i ${squish_env_vars})
  message(STATUS "parsing env var key/value pair ${i}")
  string(REGEX MATCH "([^=]*)=(.*)" squish_env_name ${i})
  message(STATUS "key=${CMAKE_MATCH_1}")
  message(STATUS "value=${CMAKE_MATCH_2}")
  set ( ENV{${CMAKE_MATCH_1}} ${CMAKE_MATCH_2} )
endforeach()

if (QT4_INSTALLED)
  # record Qt lib directory
  set ( ENV{${SQUISH_LIBQTDIR}} ${squish_libqtdir} )
endif ()

if(squish_pre_command)
  message(STATUS "Executing pre command: ${squish_pre_command}")
  execute_process(COMMAND "${squish_pre_command}")
endif()

# run the test
if("${squish_version}" STREQUAL "4")
  if (WIN32)
    execute_process(COMMAND ${squish_module_dir}/Squish4RunTestCase.bat ${squish_server_executable} ${squish_client_executable} ${squish_test_suite} ${squish_test_case} ${squish_aut} ${squish_aut_dir}
                    RESULT_VARIABLE test_rv )
  elseif(UNIX)
    execute_process(COMMAND ${squish_module_dir}/Squish4RunTestCase.sh ${squish_server_executable} ${squish_client_executable} ${squish_test_suite} ${squish_test_case} ${squish_aut} ${squish_aut_dir}
                    RESULT_VARIABLE test_rv )
  endif ()

else()

  if (WIN32)
    execute_process(COMMAND ${squish_module_dir}/SquishRunTestCase.bat ${squish_server_executable} ${squish_client_executable} ${squish_test_case} ${squish_wrapper} ${squish_aut}
                    RESULT_VARIABLE test_rv )
  elseif(UNIX)
    execute_process(COMMAND ${squish_module_dir}/SquishRunTestCase.sh ${squish_server_executable} ${squish_client_executable} ${squish_test_case} ${squish_wrapper} ${squish_aut}
                    RESULT_VARIABLE test_rv )
  endif ()
endif()

if(squish_post_command)
  message(STATUS "Executing post command: ${squish_post_command}")
  execute_process(COMMAND "${squish_post_command}")
endif()

# check for an error with running the test
if(NOT "${test_rv}" STREQUAL "0")
  message(FATAL_ERROR "Error running Squish test")
endif()
