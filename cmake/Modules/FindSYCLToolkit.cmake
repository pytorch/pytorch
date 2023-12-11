#
# Modifications, Copyright (C) 2022 Intel Corporation
#
# This software and the related documents are Intel copyrighted materials, and
# your use of them is governed by the express license under which they were
# provided to you ("License"). Unless the License provides otherwise, you may not
# use, modify, copy, publish, distribute, disclose or transmit this software or
# the related documents without Intel's prior written permission.
#
# This software and the related documents are provided as is, with no express
# or implied warranties, other than those that are expressly stated in the
# License.
#
# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

#[=======================================================================[.rst:
SYCLConfig
-------

Library to verify SYCL compatability of CMAKE_CXX_COMPILER
and passes relevant compiler flags.

Result Variables
^^^^^^^^^^^^^^^^

This will define the following variables:

``SYCL_FOUND``
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
  Target for using Intel SYCL compiler (DPC++).  The following properties are
  defined for the target: ``INTERFACE_COMPILE_OPTIONS``,
  ``INTERFACE_LINK_OPTIONS``, ``INTERFACE_INCLUDE_DIRECTORIES``, and
  ``INTERFACE_LINK_DIRECTORIES``

#]=======================================================================]

include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

find_package(PkgConfig QUIET)
if(PKG_CONFIG_FOUND)
  # TODO add dependency package module checks, if any
endif()

set(SYCL_ROOT $ENV{CMPLR_ROOT})
if(NOT SYCL_ROOT)
  execute_process(COMMAND whereis icpx OUTPUT_VARIABLE {SYCL_ROOT)
endif()

if (NOT SYCL_ROOT)
  message(FATAL_ERROR "Please setup Intel SYCL compiler Tool kit enviroment before building!!")
endif()

set(SYCL_COMPILER ${SYCL_ROOT}/bin/icpx)
set(SYCL_INCLUDE_DIR ${SYCL_ROOT}/include ${SYCL_ROOT}/include/sycl)
set(SYCL_LIBRARY_DIR ${SYCL_ROOT}/lib)

string(COMPARE EQUAL "${SYCL_COMPILER}" "" nocmplr)
if(nocmplr)
  set(SYCL_FOUND False)
  set(SYCL_REASON_FAILURE "SYCL: CMAKE_CXX_COMPILER not set!!")
  set(SYCL_NOT_FOUND_MESSAGE "${SYCL_REASON_FAILURE}")
endif()

# Function to write a test case to verify SYCL features.

function(SYCL_FEATURE_TEST_WRITE src)

  set(pp_if "#if")
  set(pp_endif "#endif")

  set(SYCL_TEST_CONTENT "")
  string(APPEND SYCL_TEST_CONTENT "#include <iostream>\nusing namespace std;\n")
  string(APPEND SYCL_TEST_CONTENT "int main(){\n")

  # Feature tests goes here

  string(APPEND SYCL_TEST_CONTENT "${pp_if} defined(SYCL_LANGUAGE_VERSION)\n")
  string(APPEND SYCL_TEST_CONTENT "cout << \"SYCL_LANGUAGE_VERSION=\"<<SYCL_LANGUAGE_VERSION<<endl;\n")
  string(APPEND SYCL_TEST_CONTENT "${pp_endif}\n")

  string(APPEND SYCL_TEST_CONTENT "return 0;}\n")

  file(WRITE ${src} "${SYCL_TEST_CONTENT}")

endfunction()

# Function to Build the feature check test case.

function(SYCL_FEATURE_TEST_BUILD TEST_SRC_FILE TEST_EXE)

  # Convert CXX Flag string to list
  set(SYCL_CXX_FLAGS_LIST "${SYCL_CXX_FLAGS}")
  separate_arguments(SYCL_CXX_FLAGS_LIST)

  # Spawn a process to build the test case.
  execute_process(
    COMMAND "${SYCL_COMPILER}"
    ${SYCL_CXX_FLAGS_LIST}
    ${TEST_SRC_FILE}
    "-o"
    ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    OUTPUT_FILE ${SYCL_TEST_DIR}/Compile.log
    RESULT_VARIABLE result
    TIMEOUT 60
    )

  # Verify if test case build properly.
  if(result)
    message("SYCL feature test compile failed!")
    message("compile output is: ${output}")
  endif()

  # TODO: what to do if it doesn't build

endfunction()

# Function to run the test case to generate feature info.

function(SYCL_FEATURE_TEST_RUN TEST_EXE)

  # Spawn a process to run the test case.

  execute_process(
    COMMAND ${TEST_EXE}
    WORKING_DIRECTORY ${SYCL_TEST_DIR}
    OUTPUT_VARIABLE output ERROR_VARIABLE output
    RESULT_VARIABLE result
    TIMEOUT 60
    )

  # Verify the test execution output.
  if(test_result)
    set(SYCL_FOUND False)
    set(SYCL_REASON_FAILURE "SYCL: feature test execution failed!!")
  endif()
  # TODO: what iff the result is false.. error or ignore?

  set( test_result "${result}" PARENT_SCOPE)
  set( test_output "${output}" PARENT_SCOPE)

endfunction()


# Function to extract the information from test execution.
function(SYCL_FEATURE_TEST_EXTRACT test_output)

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
set(SYCL_TEST_DIR "${CMAKE_CURRENT_BINARY_DIR}/CMakeFiles/TESTSYCL")
file(REMOVE_RECURSE ${SYCL_TEST_DIR})
file(MAKE_DIRECTORY ${SYCL_TEST_DIR})

# Create the test source file
set(TEST_SRC_FILE "${SYCL_TEST_DIR}/sycl_features.cpp")
set(TEST_EXE "${TEST_SRC_FILE}.exe")
SYCL_FEATURE_TEST_WRITE(${TEST_SRC_FILE})

# Build the test and create test executable
SYCL_FEATURE_TEST_BUILD(${TEST_SRC_FILE} ${TEST_EXE})

# Execute the test to extract information
SYCL_FEATURE_TEST_RUN(${TEST_EXE})

# Extract test output for information
SYCL_FEATURE_TEST_EXTRACT(${test_output})

# As per specification, all the SYCL compatible compilers should
# define macro  SYCL_LANGUAGE_VERSION
string(COMPARE EQUAL "${SYCL_LANGUAGE_VERSION}" "" nosycllang)
if(nosycllang)
  set(SYCL_FOUND False)
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
  SYCL
  FOUND_VAR SYCL_FOUND
  REQUIRED_VARS SYCL_INCLUDE_DIR SYCL_LIBRARY_DIR SYCL_FLAGS
  VERSION_VAR SYCL_LANGUAGE_VERSION
  REASON_FAILURE_MESSAGE "${SYCL_REASON_FAILURE}")

# Include in Cache
set(SYCL_LANGUAGE_VERSION "${SYCL_LANGUAGE_VERSION}" CACHE STRING "SYCL Language version")
