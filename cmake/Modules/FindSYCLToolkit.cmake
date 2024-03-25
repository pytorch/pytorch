#[=======================================================================[.rst:
SYCLConfig
-------

Library to verify SYCL compatability of CMAKE_CXX_COMPILER
and passes relevant compiler flags.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``SYCLTOOLKIT_FOUND``
  True if the system has the SYCL library.
``SYCL_COMPILER``
  SYCL compiler executable.
``SYCL_INCLUDE_DIR``
  Include directories needed to use SYCL.
``SYCL_LIBRARY_DIR``
  Libaray directories needed to use SYCL.
``SYCL_FLAGS``
  SYCL specific flags for the compiler.
``SYCL_LANGUAGE_VERSION``
  The SYCL language spec version by Compiler.

``SYCL::SYCL_CXX``
  Interface target for using SYCL compiler.  The following properties are
  defined for the target: ``INTERFACE_COMPILE_OPTIONS``,
  ``INTERFACE_LINK_OPTIONS``, ``INTERFACE_INCLUDE_DIRECTORIES``, and
  ``INTERFACE_LINK_DIRECTORIES``

#]=======================================================================]

set(SYCLTOOLKIT_FOUND False)
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

set(SYCL_ROOT "")
if(DEFINED ENV{SYCL_ROOT})
  set(SYCL_ROOT $ENV{SYCL_ROOT})
elseif(DEFINED ENV{CMPLR_ROOT})
  set(SYCL_ROOT $ENV{CMPLR_ROOT})
endif()
if(NOT SYCL_ROOT)
  execute_process(
    COMMAND which icpx
    OUTPUT_VARIABLE SYCL_CMPLR_FULL_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT EXISTS "${SYCL_CMPLR_FULL_PATH}")
    message("Cannot find ENV{CMPLR_ROOT} or icpx, please setup SYCL compiler Tool kit enviroment before building!!")
    return()
  endif()

  get_filename_component(SYCL_BIN_DIR "${SYCL_CMPLR_FULL_PATH}" DIRECTORY)
  set(SYCL_ROOT ${SYCL_BIN_DIR}/..)
endif()

find_file(
  SYCL_COMPILER
  NAMES icpx
  HINTS ${SYCL_ROOT}/bin
  NO_DEFAULT_PATH
  )

string(COMPARE EQUAL "${SYCL_COMPILER}" "" nocmplr)
if(nocmplr)
  set(SYCLTOOLKIT_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL: CMAKE_CXX_COMPILER not set!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
endif()

find_file(
  SYCL_INCLUDE_DIR
  NAMES include
  HINTS ${SYCL_ROOT}
  NO_DEFAULT_PATH
  )

find_file(
  SYCL_INCLUDE_SYCL_DIR
  NAMES sycl
  HINTS ${SYCL_ROOT}/include
  NO_DEFAULT_PATH
  )

list(APPEND SYCL_INCLUDE_DIR ${SYCL_INCLUDE_SYCL_DIR})

find_file(
  SYCL_LIBRARY_DIR
  NAMES lib lib64
  HINTS ${SYCL_ROOT}
  NO_DEFAULT_PATH
  )

find_library(
  SYCL_LIBRARY
  NAMES sycl
  HINTS ${SYCL_LIBRARY_DIR}
  NO_DEFAULT_PATH
)

if((NOT SYCL_INCLUDE_DIR) OR (NOT SYCL_LIBRARY_DIR) OR (NOT SYCL_LIBRARY))
  set(SYCLTOOLKIT_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL sdk is incomplete!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
  return()
endif()

# Function to write a test case to verify SYCL features.

function(SYCL_CMPLR_TEST_WRITE src)

  set(cpp_macro_if "#if")
  set(cpp_macro_endif "#endif")

  set(SYCL_CMPLR_TEST_CONTENT "")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "#include <iostream>\nusing namespace std;\n")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "int main(){\n")

  # Feature tests goes here

  string(APPEND SYCL_CMPLR_TEST_CONTENT "${cpp_macro_if} defined(SYCL_LANGUAGE_VERSION)\n")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "cout << \"SYCL_LANGUAGE_VERSION=\"<<SYCL_LANGUAGE_VERSION<<endl;\n")
  string(APPEND SYCL_CMPLR_TEST_CONTENT "${cpp_macro_endif}\n")

  string(APPEND SYCL_CMPLR_TEST_CONTENT "return 0;}\n")

  file(WRITE ${src} "${SYCL_CMPLR_TEST_CONTENT}")

endfunction()

# Function to Build the feature check test case.

function(SYCL_CMPLR_TEST_BUILD error TEST_SRC_FILE TEST_EXE)

  set(SYCL_CXX_FLAGS_LIST "${SYCL_CXX_FLAGS}")
  separate_arguments(SYCL_CXX_FLAGS_LIST)

  execute_process(
    COMMAND "${SYCL_COMPILER}"
    ${SYCL_CXX_FLAGS_LIST}
    ${TEST_SRC_FILE}
    "-o"
    ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_CMPLR_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    OUTPUT_FILE ${SYCL_CMPLR_TEST_DIR}/Compile.log
    RESULT_VARIABLE result
    TIMEOUT 60
    )

  # Verify if test case build properly.
  if(result)
    message("SYCL: feature test compile failed!!")
    message("compile output is: ${output}")
  endif()

  set(${error} ${result} PARENT_SCOPE)

endfunction()

function(SYCL_CMPLR_TEST_RUN error TEST_EXE)

  execute_process(
    COMMAND ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_CMPLR_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    RESULT_VARIABLE result
    TIMEOUT 60
    )

  if(test_result)
    set(SYCLTOOLKIT_FOUND False)
    set(SYCL_REASON_FAILURE "SYCL: feature test execution failed!!")
  endif()

  set(test_result "${result}" PARENT_SCOPE)
  set(test_output "${output}" PARENT_SCOPE)

  set(${error} ${result} PARENT_SCOPE)

endfunction()

function(SYCL_CMPLR_TEST_EXTRACT test_output)

  string(REGEX REPLACE "\n" ";" test_output_list "${test_output}")

  set(SYCL_LANGUAGE_VERSION "")
  foreach(strl ${test_output_list})
     if(${strl} MATCHES "^SYCL_LANGUAGE_VERSION=([A-Za-z0-9_]+)$")
       string(REGEX REPLACE "^SYCL_LANGUAGE_VERSION=" "" extracted_sycl_lang "${strl}")
       set(SYCL_LANGUAGE_VERSION ${extracted_sycl_lang})
     endif()
  endforeach()

  set(SYCL_LANGUAGE_VERSION "${SYCL_LANGUAGE_VERSION}" PARENT_SCOPE)
endfunction()

set(SYCL_FLAGS "")
set(SYCL_LINK_FLAGS "")
list(APPEND SYCL_FLAGS "-fsycl")
list(APPEND SYCL_LINK_FLAGS "-fsycl")

set(SYCL_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${SYCL_FLAGS}")

# Create a clean working directory.
set(SYCL_CMPLR_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/TESTSYCLCMPLR")
file(REMOVE_RECURSE ${SYCL_CMPLR_TEST_DIR})
file(MAKE_DIRECTORY ${SYCL_CMPLR_TEST_DIR})

# Create the test source file
set(TEST_SRC_FILE "${SYCL_CMPLR_TEST_DIR}/sycl_features.cpp")
set(TEST_EXE "${TEST_SRC_FILE}.exe")
SYCL_CMPLR_TEST_WRITE(${TEST_SRC_FILE})

# Build the test and create test executable
SYCL_CMPLR_TEST_BUILD(error ${TEST_SRC_FILE} ${TEST_EXE})
if(error)
  return()
endif()

# Execute the test to extract information
SYCL_CMPLR_TEST_RUN(error ${TEST_EXE})
if(error)
  return()
endif()

# Extract test output for information
SYCL_CMPLR_TEST_EXTRACT(${test_output})

# As per specification, all the SYCL compatible compilers should
# define macro  SYCL_LANGUAGE_VERSION
string(COMPARE EQUAL "${SYCL_LANGUAGE_VERSION}" "" nosycllang)
if(nosycllang)
  set(SYCLTOOLKIT_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL: It appears that the ${SYCL_COMPILER} does not support SYCL")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
endif()

message(DEBUG "The SYCL compiler is ${SYCL_COMPILER}")
message(DEBUG "The SYCL Flags are ${SYCL_FLAGS}")
message(DEBUG "The SYCL Language Version is ${SYCL_LANGUAGE_VERSION}")

add_library(SYCL::SYCL_CXX INTERFACE IMPORTED)
set_property(TARGET SYCL::SYCL_CXX PROPERTY
  INTERFACE_COMPILE_OPTIONS ${SYCL_FLAGS})
set_property(TARGET SYCL::SYCL_CXX PROPERTY
  INTERFACE_LINK_OPTIONS ${SYCL_LINK_FLAGS})
set_property(TARGET SYCL::SYCL_CXX PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES ${SYCL_INCLUDE_DIR})
set_property(TARGET SYCL::SYCL_CXX PROPERTY
  INTERFACE_LINK_DIRECTORIES ${SYCL_LIBRARY_DIR})

find_package_handle_standard_args(
  SYCLToolkit
  FOUND_VAR SYCLTOOLKIT_FOUND
  REQUIRED_VARS SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_LIBRARY SYCL_FLAGS
  VERSION_VAR SYCL_LANGUAGE_VERSION
  REASON_FAILURE_MESSAGE "${SYCL_REASON_FAILURE}")

# Include in Cache
set(SYCL_LANGUAGE_VERSION "${SYCL_LANGUAGE_VERSION}" CACHE STRING "SYCL Language version")
