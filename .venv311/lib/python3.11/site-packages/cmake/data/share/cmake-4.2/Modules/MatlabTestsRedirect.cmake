# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file LICENSE.rst or https://cmake.org/licensing for details.

# This is an undocumented internal helper for the FindMatlab
# module ``matlab_add_unit_test`` command.

# Usage: cmake
#   -Dtest_timeout=180
#   -Doutput_directory=
#   -Dadditional_paths=""
#   -Dno_unittest_framework=""
#   -DMatlab_PROGRAM=matlab_exe_location
#   -DMatlab_ADDITIONAL_STARTUP_OPTIONS=""
#   -Dtest_name=name_of_the_test
#   -Dcustom_Matlab_test_command=""
#   -Dcmd_to_run_before_test=""
#   -Dunittest_file_to_run
#   -Dmaut_BATCH_OPTION="-batch"
#   -P FindMatlab_TestsRedirect.cmake

set(Matlab_UNIT_TESTS_CMD -nosplash -nodesktop -nodisplay ${Matlab_ADDITIONAL_STARTUP_OPTIONS})
if(WIN32 AND maut_BATCH_OPTION STREQUAL "-r")
  list(APPEND Matlab_UNIT_TESTS_CMD -wait)
endif()

if(NOT test_timeout)
  set(test_timeout 180)
endif()

# If timeout is -1, then do not put a timeout on the execute_process
if(test_timeout EQUAL -1)
  set(test_timeout "")
else()
  set(test_timeout TIMEOUT ${test_timeout})
endif()

if(NOT cmd_to_run_before_test)
  set(cmd_to_run_before_test)
endif()

get_filename_component(unittest_file_directory   "${unittest_file_to_run}" DIRECTORY)
get_filename_component(unittest_file_to_run_name "${unittest_file_to_run}" NAME_WE)

set(concat_string '${unittest_file_directory}')
foreach(s IN LISTS additional_paths)
  if(NOT "${s}" STREQUAL "")
    string(APPEND concat_string ", '${s}'")
  endif()
endforeach()

if(custom_Matlab_test_command)
  set(unittest_to_run "${custom_Matlab_test_command}")
else()
  set(unittest_to_run "runtests('${unittest_file_to_run_name}'), exit(max([ans(1,:).Failed]))")
endif()


if(no_unittest_framework)
  set(unittest_to_run "${unittest_file_to_run_name}")
endif()

set(command_to_run "try, ${unittest_to_run}, catch err, disp('An exception has been thrown during the execution'), disp(err), disp(err.stack), exit(1), end, exit(0)")
set(Matlab_SCRIPT_TO_RUN
    "addpath(${concat_string}); ${cmd_to_run_before_test}; ${command_to_run}"
   )
# if the working directory is not specified then default
# to the output_directory because the log file will go there
# if the working_directory is specified it will override the
# output_directory
if(NOT working_directory)
  set(working_directory "${output_directory}")
endif()

string(REPLACE "/" "_" log_file_name "${test_name}.log")
set(Matlab_LOG_FILE "${working_directory}/${log_file_name}")

set(devnull)
if(UNIX)
  set(devnull INPUT_FILE /dev/null)
elseif(WIN32)
  set(devnull INPUT_FILE NUL)
endif()

execute_process(
  # Do not use a full path to log file.  Depend on the fact that the log file
  # is always going to go in the working_directory.  This is because matlab
  # on unix is a shell script that does not handle spaces in the logfile path.
  COMMAND "${Matlab_PROGRAM}" ${Matlab_UNIT_TESTS_CMD} -logfile "${log_file_name}" "${maut_BATCH_OPTION}" "${Matlab_SCRIPT_TO_RUN}"
  RESULT_VARIABLE res
  ${test_timeout}
  OUTPUT_QUIET # we do not want the output twice
  WORKING_DIRECTORY "${working_directory}"
  ${devnull}
  )

if(NOT EXISTS ${Matlab_LOG_FILE})
  message( FATAL_ERROR "[MATLAB] ERROR: cannot find the log file ${Matlab_LOG_FILE}")
endif()

# print the output in any case.
file(READ ${Matlab_LOG_FILE} matlab_log_content)
message("Matlab test ${name_of_the_test} output:\n${matlab_log_content}") # if we put FATAL_ERROR here, the file is indented.


if(NOT (res EQUAL 0))
  message( FATAL_ERROR "[MATLAB] TEST FAILED Matlab returned ${res}" )
endif()
