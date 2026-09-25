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

#]=======================================================================]

set(SYCLTOOLKIT_FOUND False)
include(${CMAKE_ROOT}/Modules/FindPackageHandleStandardArgs.cmake)

set(SYCL_ROOT "")
if(DEFINED ENV{SYCL_ROOT})
  set(SYCL_ROOT $ENV{SYCL_ROOT})
elseif(DEFINED ENV{CMPLR_ROOT})
  set(SYCL_ROOT $ENV{CMPLR_ROOT})
endif()

if(WIN32)
  set(SYCL_EXECUTABLE_NAME icx)
else()
  set(SYCL_EXECUTABLE_NAME icpx)
endif()

if(NOT SYCL_ROOT)
  execute_process(
    COMMAND which ${SYCL_EXECUTABLE_NAME}
    OUTPUT_VARIABLE SYCL_CMPLR_FULL_PATH
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  if(NOT EXISTS "${SYCL_CMPLR_FULL_PATH}")
    message(WARNING "Cannot find ENV{CMPLR_ROOT} or icpx, please setup SYCL compiler Tool kit enviroment before building!!")
    return()
  endif()

  get_filename_component(SYCL_BIN_DIR "${SYCL_CMPLR_FULL_PATH}" DIRECTORY)
  set(SYCL_ROOT ${SYCL_BIN_DIR}/..)
endif()

find_program(
  SYCL_COMPILER
  NAMES ${SYCL_EXECUTABLE_NAME}
  PATHS "${SYCL_ROOT}"
  PATH_SUFFIXES bin bin64
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

message(DEBUG "The SYCL compiler is ${SYCL_COMPILER}")

set(SYCLTOOLKIT_FOUND True)
