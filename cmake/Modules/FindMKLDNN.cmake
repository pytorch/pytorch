# - Try to find MKLDNN
#
# The following variables are optionally searched for defaults
#  MKL_FOUND             : set to true if a library implementing the CBLAS interface is found
#
# The following are set after configuration is done:
#  MKLDNN_FOUND          : set to true if mkl-dnn is found.
#  MKLDNN_INCLUDE_DIR    : path to mkl-dnn include dir.
#  MKLDNN_LIBRARIES      : list of libraries for mkl-dnn

IF (NOT MKLDNN_FOUND)

SET(MKLDNN_LIBRARIES)
SET(MKLDNN_INCLUDE_DIR)

IF(MSVC)
  MESSAGE(STATUS "MKL-DNN needs omp 3+ which is not supported in MSVC so far")
  RETURN()
ENDIF(MSVC)

SET(IDEEP_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep")
SET(MKLDNN_ROOT "${IDEEP_ROOT}/mkl-dnn")

FIND_PACKAGE(BLAS)
FIND_PATH(IDEEP_INCLUDE_DIR ideep.hpp PATHS ${IDEEP_ROOT} PATH_SUFFIXES include)
FIND_PATH(MKLDNN_INCLUDE_DIR mkldnn.hpp mkldnn.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include)
IF (NOT MKLDNN_INCLUDE_DIR)
  EXECUTE_PROCESS(COMMAND git${CMAKE_EXECUTABLE_SUFFIX} submodule update --init mkl-dnn WORKING_DIRECTORY ${IDEEP_ROOT})
  FIND_PATH(MKLDNN_INCLUDE_DIR mkldnn.hpp mkldnn.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include)
ENDIF(NOT MKLDNN_INCLUDE_DIR)

IF (NOT IDEEP_INCLUDE_DIR OR NOT MKLDNN_INCLUDE_DIR)
  MESSAGE(STATUS "MKLDNN source files not found!")
  RETURN()
ENDIF(NOT IDEEP_INCLUDE_DIR OR NOT MKLDNN_INCLUDE_DIR)
LIST(APPEND MKLDNN_INCLUDE_DIR ${IDEEP_INCLUDE_DIR})

IF(MKL_FOUND)
  # Append to mkldnn dependencies
  LIST(APPEND MKLDNN_LIBRARIES ${MKL_LIBRARIES})
  LIST(APPEND MKLDNN_INCLUDE_DIR ${MKL_INCLUDE_DIR})
ELSE(MKL_FOUND)
  # If we cannot find MKL, we will use the Intel MKL Small library
  # comes with ${MKLDNN_ROOT}/external
  IF(NOT IS_DIRECTORY ${MKLDNN_ROOT}/external)
    IF(UNIX)
      EXECUTE_PROCESS(COMMAND "${MKLDNN_ROOT}/scripts/prepare_mkl.sh" RESULT_VARIABLE __result)
    ELSE(UNIX)
      EXECUTE_PROCESS(COMMAND "${MKLDNN_ROOT}/scripts/prepare_mkl.bat" RESULT_VARIABLE __result)
    ENDIF(UNIX)
  ENDIF(NOT IS_DIRECTORY ${MKLDNN_ROOT}/external)

  FILE(GLOB_RECURSE MKLML_INNER_INCLUDE_DIR ${MKLDNN_ROOT}/external/*/mkl.h)
  IF(NOT MKLML_INNER_INCLUDE_DIR)
    MESSAGE(STATUS "MKL-DNN not found. Compiling without MKL-DNN support")
    RETURN()
  ENDIF(NOT MKLML_INNER_INCLUDE_DIR)
  # if user has multiple version under external/ then guess last
  # one alphabetically is "latest" and warn
  LIST(LENGTH MKLML_INNER_INCLUDE_DIR MKLINCLEN)
  IF(MKLINCLEN GREATER 1)
    LIST(SORT MKLML_INNER_INCLUDE_DIR)
    LIST(REVERSE MKLML_INNER_INCLUDE_DIR)
    LIST(GET MKLML_INNER_INCLUDE_DIR 0 MKLINCLST)
    SET(MKLML_INNER_INCLUDE_DIR "${MKLINCLST}")
  ENDIF(MKLINCLEN GREATER 1)
  GET_FILENAME_COMPONENT(MKLML_INNER_INCLUDE_DIR ${MKLML_INNER_INCLUDE_DIR} DIRECTORY)
  LIST(APPEND MKLDNN_INCLUDE_DIR ${MKLML_INNER_INCLUDE_DIR})

  IF(APPLE)
    SET(__mklml_inner_libs mklml iomp5)
  ELSE(APPLE)
    SET(__mklml_inner_libs mklml_intel iomp5)
  ENDIF(APPLE)
  FOREACH(__mklml_inner_lib ${__mklml_inner_libs})
    STRING(TOUPPER ${__mklml_inner_lib} __mklml_inner_lib_upper)
    FIND_LIBRARY(${__mklml_inner_lib_upper}_LIBRARY
          NAMES ${__mklml_inner_lib}
          PATHS  "${MKLML_INNER_INCLUDE_DIR}/../lib"
          DOC "The path to Intel(R) MKLML ${__mklml_inner_lib} library")
    MARK_AS_ADVANCED(${__mklml_inner_lib_upper}_LIBRARY)
    IF(NOT ${__mklml_inner_lib_upper}_LIBRARY)
      MESSAGE(STATUS "MKL-DNN not found. Compiling without MKL-DNN support")
      RETURN()
    ENDIF(NOT ${__mklml_inner_lib_upper}_LIBRARY)
    LIST(APPEND MKLDNN_LIBRARIES ${${__mklml_inner_lib_upper}_LIBRARY})
  ENDFOREACH(__mklml_inner_lib)
ENDIF(MKL_FOUND)

IF(MKL_FOUND)
  SET(MKL_cmake_included TRUE)
  SET(MKLDNN_THREADING "OMP:COMP" CACHE STRING "")
ENDIF(MKL_FOUND)
SET(WITH_TEST FALSE CACHE BOOL "" FORCE)
SET(WITH_EXAMPLE FALSE CACHE BOOL "" FORCE)
SET(MKLDNN_LIBRARY_TYPE STATIC CACHE STRING "" FORCE)
ADD_SUBDIRECTORY(${MKLDNN_ROOT})
IF(NOT TARGET mkldnn)
  MESSAGE("Failed to include MKL-DNN target")
  RETURN()
ENDIF(NOT TARGET mkldnn)
IF(MKL_FOUND)
  TARGET_COMPILE_DEFINITIONS(mkldnn PRIVATE -DUSE_MKL)
ENDIF(MKL_FOUND)
IF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
  TARGET_COMPILE_OPTIONS(mkldnn PRIVATE -Wno-maybe-uninitialized)
  TARGET_COMPILE_OPTIONS(mkldnn PRIVATE -Wno-strict-overflow)
  TARGET_COMPILE_OPTIONS(mkldnn PRIVATE -Wno-error=strict-overflow)
ENDIF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
LIST(APPEND MKLDNN_LIBRARIES mkldnn)

SET(MKLDNN_FOUND TRUE)
MESSAGE(STATUS "Found MKL-DNN: TRUE")

ENDIF(NOT MKLDNN_FOUND)
