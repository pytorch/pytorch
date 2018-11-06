# - Try to find MKLDNN
#
# The following variables are optionally searched for defaults
#  MKL_FOUND             : set to true if a library implementing the CBLAS interface is found
#  USE_MKLDNN
#
# The following are set after configuration is done:
#  MKLDNN_FOUND          : set to true if mkl-dnn is found.
#  MKLDNN_INCLUDE_DIR    : path to mkl-dnn include dir.
#  MKLDNN_LIBRARIES      : list of libraries for mkl-dnn

IF (NOT MKLDNN_FOUND)

SET(MKLDNN_LIBRARIES)
SET(MKLDNN_INCLUDE_DIR)

IF (NOT USE_MKLDNN)
  RETURN()
ENDIF(NOT USE_MKLDNN)

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
  LIST(APPEND MKLDNN_LIBRARIES ${MKL_LIBRARIES})
  LIST(APPEND MKLDNN_INCLUDE_DIR ${MKL_INCLUDE_DIR})
  # The OMP-related variables of MKL-DNN have to be overwritten here,
  # if MKL is used, and the OMP version is defined by MKL.
  # MKL_LIBRARIES_xxxx_LIBRARY is defined by MKL.
  # INTEL_MKL_DIR gives the MKL root path.
  IF (INTEL_MKL_DIR)
    SET(MKLROOT ${INTEL_MKL_DIR})
    IF(WIN32)
      SET(MKLIOMP5DLL ${MKL_LIBRARIES_libiomp5md_LIBRARY} CACHE STRING "Overwrite MKL-DNN omp dependency" FORCE)
    ELSE(WIN32)
      IF (MKL_LIBRARIES_gomp_LIBRARY)
        SET(MKLOMPLIB ${MKL_LIBRARIES_gomp_LIBRARY})
      ELSE(MKL_LIBRARIES_gomp_LIBRARY)
        SET(MKLOMPLIB ${MKL_LIBRARIES_iomp5_LIBRARY})
      ENDIF(MKL_LIBRARIES_gomp_LIBRARY)
      SET(MKLIOMP5LIB ${MKLOMPLIB} CACHE STRING "Overwrite MKL-DNN omp dependency" FORCE)
    ENDIF(WIN32)
  ELSE(INTEL_MKL_DIR)
    MESSAGE(STATUS "Warning: MKL is found, but INTEL_MKL_DIR is not set!")
  ENDIF(INTEL_MKL_DIR)

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
  IF(MKLML_INNER_INCLUDE_DIR)
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
      LIST(APPEND MKLDNN_LIBRARIES ${${__mklml_inner_lib_upper}_LIBRARY})
    ENDFOREACH(__mklml_inner_lib)
  ENDIF(MKLML_INNER_INCLUDE_DIR)
ENDIF(MKL_FOUND)

LIST(APPEND __mkldnn_looked_for MKLDNN_LIBRARIES)
LIST(APPEND __mkldnn_looked_for MKLDNN_INCLUDE_DIR)
INCLUDE(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKLDNN DEFAULT_MSG ${__mkldnn_looked_for})

IF(MKLDNN_FOUND)
  IF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
    ADD_COMPILE_OPTIONS(-Wno-maybe-uninitialized)
  ENDIF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
  SET(WITH_TEST FALSE CACHE BOOL "build with mkl-dnn test" FORCE)
  SET(WITH_EXAMPLE FALSE CACHE BOOL "build with mkl-dnn examples" FORCE)
  ADD_SUBDIRECTORY(${MKLDNN_ROOT})
  SET(MKLDNN_LIB "${CMAKE_SHARED_LIBRARY_PREFIX}mkldnn${CMAKE_SHARED_LIBRARY_SUFFIX}")
  IF(WIN32)
    LIST(APPEND MKLDNN_LIBRARIES "${PROJECT_BINARY_DIR}/bin/${MKLDNN_LIB}")
  ELSE(WIN32)
    LIST(APPEND MKLDNN_LIBRARIES "${PROJECT_BINARY_DIR}/lib/${MKLDNN_LIB}")
  ENDIF(WIN32)
ELSE(MKLDNN_FOUND)
  MESSAGE(STATUS "MKLDNN source files not found!")
ENDIF(MKLDNN_FOUND)

UNSET(__mklml_inner_libs)
UNSET(__mkldnn_looked_for)

ENDIF(NOT MKLDNN_FOUND)
