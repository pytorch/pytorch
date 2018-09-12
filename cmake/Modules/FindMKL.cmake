# - Find INTEL MKL library
#
# This module finds the Intel Mkl libraries.
#
#   USE_IDEEP                         : use IDEEP interface
#
# This module sets the following variables:
#  MKL_FOUND - set to true if a library implementing the CBLAS interface is found
#  MKL_VERSION - best guess
#  MKL_INCLUDE_DIR - path to include dir.
#  MKL_LIBRARIES - list of libraries for base mkl
#  MKL_LAPACK_LIBRARIES - list of libraries to add for lapack
#  MKL_SCALAPACK_LIBRARIES - list of libraries to add for scalapack
#  MKL_SOLVER_LIBRARIES - list of libraries to add for the solvers
#  MKL_CDFT_LIBRARIES - list of libraries to add for the solvers
#  CAFFE2_USE_MKL - set to ON if MKL_FOUND is true
#  CAFFE2_USE_IDEEP - set to ON if use IDEEP interfaces

# Do nothing if MKL_FOUND was set before!
IF (NOT MKL_FOUND)

SET(MKL_VERSION)
SET(MKL_INCLUDE_DIR)
SET(MKL_LIBRARIES)
SET(MKL_LAPACK_LIBRARIES)
SET(MKL_SCALAPACK_LIBRARIES)
SET(MKL_SOLVER_LIBRARIES)
SET(MKL_CDFT_LIBRARIES)
SET(CAFFE2_USE_MKL OFF)
SET(CAFFE2_USE_IDEEP OFF)

# Includes
INCLUDE(CheckTypeSize)
INCLUDE(CheckFunctionExists)

# Intel Compiler Suite
SET(INTEL_COMPILER_DIR "/opt/intel" CACHE STRING
  "Root directory of the Intel Compiler Suite (contains ipp, mkl, etc.)")
SET(INTEL_MKL_DIR "/opt/intel/mkl" CACHE STRING
  "Root directory of the Intel MKL (standalone)")
SET(INTEL_MKL_SEQUENTIAL OFF CACHE BOOL
  "Force using the sequential (non threaded) libraries")

# Checks
CHECK_TYPE_SIZE("void*" SIZE_OF_VOIDP)
IF ("${SIZE_OF_VOIDP}" EQUAL 8)
  SET(mklvers "intel64")
  SET(iccvers "intel64")
  SET(mkl64s "_lp64")
ELSE ("${SIZE_OF_VOIDP}" EQUAL 8)
  SET(mklvers "32")
  SET(iccvers "ia32")
  SET(mkl64s)
ENDIF ("${SIZE_OF_VOIDP}" EQUAL 8)
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(mklthreads "mkl_gnu_thread" "mkl_intel_thread")
  SET(mklifaces  "intel" "gf")
  SET(mklrtls "gomp" "iomp5")
ELSE(CMAKE_COMPILER_IS_GNUCC)
  SET(mklthreads "mkl_intel_thread")
  SET(mklifaces  "intel")
  SET(mklrtls "iomp5" "guide")
  IF (MSVC)
    SET(mklrtls "libiomp5md")
  ENDIF (MSVC)
ENDIF (CMAKE_COMPILER_IS_GNUCC)

# Kernel libraries dynamically loaded
SET(mklkerlibs "mc" "mc3" "nc" "p4n" "p4m" "p4m3" "p4p" "def")
SET(mklseq)


# Paths
SET(saved_CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH})
SET(saved_CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH})
IF(WIN32)
  # Set default MKLRoot for Windows
  IF($ENV{MKLProductDir})
    SET(INTEL_COMPILER_DIR $ENV{MKLProductDir})
  ELSE()
    SET(INTEL_COMPILER_DIR
     "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows")
  ENDIF()
  # Change mklvers and iccvers when we are using MSVC instead of ICC
  IF(MSVC AND NOT CMAKE_CXX_COMPILER_ID STREQUAL "Intel")
    SET(mklvers "${mklvers}_win")
    SET(iccvers "${iccvers}_win")
  ENDIF()
ENDIF(WIN32)
IF (EXISTS ${INTEL_COMPILER_DIR})
  # TODO: diagnostic if dir does not exist
  SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
    "${INTEL_COMPILER_DIR}/lib/${iccvers}")
  IF(MSVC)
    SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
      "${INTEL_COMPILER_DIR}/compiler/lib/${iccvers}")
  ENDIF()
  IF (NOT EXISTS ${INTEL_MKL_DIR})
    SET(INTEL_MKL_DIR "${INTEL_COMPILER_DIR}/mkl")
  ENDIF()
ENDIF()
IF (EXISTS ${INTEL_MKL_DIR})
  # TODO: diagnostic if dir does not exist
  SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}
    "${INTEL_MKL_DIR}/include")
  SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
    "${INTEL_MKL_DIR}/lib/${mklvers}")
  IF (MSVC)
    SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
      "${INTEL_MKL_DIR}/lib/${iccvers}")
  ENDIF()
ENDIF()

# Try linking multiple libs
MACRO(CHECK_ALL_LIBRARIES LIBRARIES _name _list _flags)
  # This macro checks for the existence of the combination of libraries given by _list.
  # If the combination is found, this macro checks whether we can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to FALSE.
  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.
  SET(_prefix "${LIBRARIES}")
  # start checking
  SET(_libraries_work TRUE)
  SET(${LIBRARIES})
  SET(_combined_name)
  SET(_paths)
  set(__list)
  foreach(_elem ${_list})
    if(__list)
      set(__list "${__list} - ${_elem}")
    else(__list)
      set(__list "${_elem}")
    endif(__list)
  endforeach(_elem)
  message(STATUS "Checking for [${__list}]")
  FOREACH(_library ${_list})
    SET(_combined_name ${_combined_name}_${_library})
    IF(_libraries_work)
      IF(${_library} STREQUAL "gomp")
          FIND_PACKAGE(OpenMP)
          IF(OPENMP_FOUND)
            SET(${_prefix}_${_library}_LIBRARY ${OpenMP_C_FLAGS})
          ENDIF(OPENMP_FOUND)
      ELSE(${_library} STREQUAL "gomp")
          FIND_LIBRARY(${_prefix}_${_library}_LIBRARY NAMES ${_library})
      ENDIF(${_library} STREQUAL "gomp")
      MARK_AS_ADVANCED(${_prefix}_${_library}_LIBRARY)
      SET(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      SET(_libraries_work ${${_prefix}_${_library}_LIBRARY})
      IF(${_prefix}_${_library}_LIBRARY)
        MESSAGE(STATUS "  Library ${_library}: ${${_prefix}_${_library}_LIBRARY}")
      ELSE(${_prefix}_${_library}_LIBRARY)
        MESSAGE(STATUS "  Library ${_library}: not found")
      ENDIF(${_prefix}_${_library}_LIBRARY)
    ENDIF(_libraries_work)
  ENDFOREACH(_library ${_list})
  # Test this combination of libraries.
  IF(_libraries_work)
    SET(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}})
    SET(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${CMAKE_REQUIRED_LIBRARIES}")
    CHECK_FUNCTION_EXISTS(${_name} ${_prefix}${_combined_name}_WORKS)
    SET(CMAKE_REQUIRED_LIBRARIES)
    MARK_AS_ADVANCED(${_prefix}${_combined_name}_WORKS)
    SET(_libraries_work ${${_prefix}${_combined_name}_WORKS})
  ENDIF(_libraries_work)
  # Fin
  IF(_libraries_work)
  ELSE (_libraries_work)
    SET(${LIBRARIES})
    MARK_AS_ADVANCED(${LIBRARIES})
  ENDIF(_libraries_work)
ENDMACRO(CHECK_ALL_LIBRARIES)

if(WIN32)
  set(mkl_m "")
  set(mkl_pthread "")
else(WIN32)
  set(mkl_m "m")
  set(mkl_pthread "pthread")
endif(WIN32)

if(UNIX AND NOT APPLE)
  set(mkl_dl "${CMAKE_DL_LIBS}")
else(UNIX AND NOT APPLE)
  set(mkl_dl "")
endif(UNIX AND NOT APPLE)

# Check for version 10/11
IF (NOT MKL_LIBRARIES)
  SET(MKL_VERSION 1011)
ENDIF (NOT MKL_LIBRARIES)
FOREACH(mklrtl ${mklrtls} "")
  FOREACH(mkliface ${mklifaces})
    FOREACH(mkl64 ${mkl64s} "")
      FOREACH(mklthread ${mklthreads})
        IF (NOT MKL_LIBRARIES AND NOT INTEL_MKL_SEQUENTIAL)
          CHECK_ALL_LIBRARIES(MKL_LIBRARIES cblas_sgemm
            "mkl_${mkliface}${mkl64};${mklthread};mkl_core;${mklrtl};${mkl_pthread};${mkl_m};${mkl_dl}" "")
        ENDIF (NOT MKL_LIBRARIES AND NOT INTEL_MKL_SEQUENTIAL)
      ENDFOREACH(mklthread)
    ENDFOREACH(mkl64)
  ENDFOREACH(mkliface)
ENDFOREACH(mklrtl)
FOREACH(mklrtl ${mklrtls} "")
  FOREACH(mkliface ${mklifaces})
    FOREACH(mkl64 ${mkl64s} "")
      IF (NOT MKL_LIBRARIES)
        CHECK_ALL_LIBRARIES(MKL_LIBRARIES cblas_sgemm
          "mkl_${mkliface}${mkl64};mkl_sequential;mkl_core;${mkl_m};${mkl_dl}" "")
        IF (MKL_LIBRARIES)
          SET(mklseq "_sequential")
        ENDIF (MKL_LIBRARIES)
      ENDIF (NOT MKL_LIBRARIES)
    ENDFOREACH(mkl64)
  ENDFOREACH(mkliface)
ENDFOREACH(mklrtl)
FOREACH(mklrtl ${mklrtls} "")
  FOREACH(mkliface ${mklifaces})
    FOREACH(mkl64 ${mkl64s} "")
      FOREACH(mklthread ${mklthreads})
        IF (NOT MKL_LIBRARIES)
          CHECK_ALL_LIBRARIES(MKL_LIBRARIES cblas_sgemm
            "mkl_${mkliface}${mkl64};${mklthread};mkl_core;${mklrtl};pthread;${mkl_m};${mkl_dl}" "")
        ENDIF (NOT MKL_LIBRARIES)
      ENDFOREACH(mklthread)
    ENDFOREACH(mkl64)
  ENDFOREACH(mkliface)
ENDFOREACH(mklrtl)

# Check for older versions
IF (NOT MKL_LIBRARIES)
  SET(MKL_VERSION 900)
  CHECK_ALL_LIBRARIES(MKL_LIBRARIES cblas_sgemm
    "mkl;guide;pthread;m" "")
ENDIF (NOT MKL_LIBRARIES)

# Include files
IF (MKL_LIBRARIES)
  FIND_PATH(MKL_INCLUDE_DIR "mkl_cblas.h")
  MARK_AS_ADVANCED(MKL_INCLUDE_DIR)
ENDIF (MKL_LIBRARIES)

# Other libraries
IF (MKL_LIBRARIES)
  FOREACH(mkl64 ${mkl64s} "_core" "")
    FOREACH(mkls ${mklseq} "")
      IF (NOT MKL_LAPACK_LIBRARIES)
        FIND_LIBRARY(MKL_LAPACK_LIBRARIES NAMES "mkl_lapack${mkl64}${mkls}")
        MARK_AS_ADVANCED(MKL_LAPACK_LIBRARIES)
      ENDIF (NOT MKL_LAPACK_LIBRARIES)
      IF (NOT MKL_SCALAPACK_LIBRARIES)
        FIND_LIBRARY(MKL_SCALAPACK_LIBRARIES NAMES "mkl_scalapack${mkl64}${mkls}")
        MARK_AS_ADVANCED(MKL_SCALAPACK_LIBRARIES)
      ENDIF (NOT MKL_SCALAPACK_LIBRARIES)
      IF (NOT MKL_SOLVER_LIBRARIES)
        FIND_LIBRARY(MKL_SOLVER_LIBRARIES NAMES "mkl_solver${mkl64}${mkls}")
        MARK_AS_ADVANCED(MKL_SOLVER_LIBRARIES)
      ENDIF (NOT MKL_SOLVER_LIBRARIES)
      IF (NOT MKL_CDFT_LIBRARIES)
        FIND_LIBRARY(MKL_CDFT_LIBRARIES NAMES "mkl_cdft${mkl64}${mkls}")
        MARK_AS_ADVANCED(MKL_CDFT_LIBRARIES)
      ENDIF (NOT MKL_CDFT_LIBRARIES)
    ENDFOREACH(mkls)
  ENDFOREACH(mkl64)
ENDIF (MKL_LIBRARIES)

# LibIRC: intel compiler always links this;
# gcc does not; but mkl kernels sometimes need it.
IF (MKL_LIBRARIES)
  IF (CMAKE_COMPILER_IS_GNUCC)
    FIND_LIBRARY(MKL_KERNEL_libirc "irc")
  ELSEIF (CMAKE_C_COMPILER_ID AND NOT CMAKE_C_COMPILER_ID STREQUAL "Intel")
    FIND_LIBRARY(MKL_KERNEL_libirc "irc")
  ENDIF (CMAKE_COMPILER_IS_GNUCC)
  MARK_AS_ADVANCED(MKL_KERNEL_libirc)
  IF (MKL_KERNEL_libirc)
    SET(MKL_LIBRARIES ${MKL_LIBRARIES} ${MKL_KERNEL_libirc})
  ENDIF (MKL_KERNEL_libirc)
ENDIF (MKL_LIBRARIES)

# Final
SET(CMAKE_LIBRARY_PATH ${saved_CMAKE_LIBRARY_PATH})
SET(CMAKE_INCLUDE_PATH ${saved_CMAKE_INCLUDE_PATH})
IF (MKL_LIBRARIES AND MKL_INCLUDE_DIR)
  SET(MKL_FOUND TRUE)
  SET(CAFFE2_USE_MKL ON)
  SET(MKL_cmake_included true)
  ENDIF()
ELSE (MKL_LIBRARIES AND MKL_INCLUDE_DIR)
  SET(MKL_FOUND FALSE)
  SET(MKL_VERSION)
ENDIF (MKL_LIBRARIES AND MKL_INCLUDE_DIR)

# Standard termination
IF(NOT MKL_FOUND AND MKL_FIND_REQUIRED)
  MESSAGE(FATAL_ERROR "MKL library not found. Please specify library location")
ENDIF(NOT MKL_FOUND AND MKL_FIND_REQUIRED)
IF(NOT MKL_FIND_QUIETLY)
  IF(MKL_FOUND)
    MESSAGE(STATUS "MKL library found")
  ELSE(MKL_FOUND)
    MESSAGE(STATUS "MKL library not found")
  ENDIF(MKL_FOUND)
ENDIF(NOT MKL_FIND_QUIETLY)

# Do nothing if MKL_FOUND was set before!
ENDIF (NOT MKL_FOUND)

IF (USE_IDEEP AND NOT IDEEP_FOUND)

SET(IDEEP_LIBRARIES)
SET(IDEEP_MKLDNN_INCLUDE_DIR)
SET(CAFFE2_USE_IDEEP OFF)

IF(MSVC)
  MESSAGE(STATUS "MKL-DNN needs omp 3+ which is not supported in MSVC so far")
  RETURN()
ENDIF(MSVC)

SET(IDEEP_ROOT "${PROJECT_SOURCE_DIR}/third_party/ideep")
SET(MKLDNN_ROOT "${IDEEP_ROOT}/mkl-dnn")

FIND_PATH(IDEEP_INCLUDE_DIR ideep.hpp PATHS ${IDEEP_ROOT} PATH_SUFFIXES include)
FIND_PATH(IDEEP_MKLDNN_INCLUDE_DIR mkldnn.hpp mkldnn.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include)
IF (NOT IDEEP_MKLDNN_INCLUDE_DIR)
  EXECUTE_PROCESS(COMMAND git submodule update --init mkl-dnn WORKING_DIRECTORY ${IDEEP_ROOT})
  FIND_PATH(IDEEP_MKLDNN_INCLUDE_DIR mkldnn.hpp mkldnn.h PATHS ${MKLDNN_ROOT} PATH_SUFFIXES include)
ENDIF(NOT IDEEP_MKLDNN_INCLUDE_DIR)

IF (NOT IDEEP_INCLUDE_DIR OR NOT IDEEP_MKLDNN_INCLUDE_DIR)
  MESSAGE(STATUS "MKLDNN source files not found!")

ELSE(NOT IDEEP_INCLUDE_DIR OR NOT IDEEP_MKLDNN_INCLUDE_DIR)
  LIST(APPEND IDEEP_MKLDNN_INCLUDE_DIR ${IDEEP_INCLUDE_DIR})

  IF(MKL_FOUND)
    LIST(APPEND IDEEP_LIBRARIES ${MKL_LIBRARIES})
    LIST(APPEND IDEEP_MKLDNN_INCLUDE_DIR ${MKL_INCLUDE_DIR})

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
      LIST(APPEND IDEEP_MKLDNN_INCLUDE_DIR ${MKLML_INNER_INCLUDE_DIR})

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
        LIST(APPEND IDEEP_LIBRARIES ${${__mklml_inner_lib_upper}_LIBRARY})
      ENDFOREACH(__mklml_inner_lib)
    ENDIF(MKLML_INNER_INCLUDE_DIR)
  ENDIF(MKL_FOUND)

  LIST(APPEND __ideep_looked_for IDEEP_LIBRARIES)
  LIST(APPEND __ideep_looked_for IDEEP_MKLDNN_INCLUDE_DIR)
  INCLUDE(FindPackageHandleStandardArgs)
  find_package_handle_standard_args(IDEEP DEFAULT_MSG ${__ideep_looked_for})

ENDIF(NOT IDEEP_INCLUDE_DIR OR NOT IDEEP_MKLDNN_INCLUDE_DIR)

if(IDEEP_FOUND)
  SET(MKL_FOUND TRUE)
  SET(CAFFE2_USE_IDEEP ON)
  IF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
    ADD_COMPILE_OPTIONS(-Wno-maybe-uninitialized)
  ENDIF(NOT APPLE AND CMAKE_COMPILER_IS_GNUCC)
  SET(WITH_TEST FALSE CACHE BOOL "build with mkl-dnn test" FORCE)
  SET(WITH_EXAMPLE FALSE CACHE BOOL "build with mkl-dnn examples" FORCE)
  ADD_SUBDIRECTORY(${MKLDNN_ROOT})
  SET(MKLDNN_LIB "${CMAKE_SHARED_LIBRARY_PREFIX}mkldnn${CMAKE_SHARED_LIBRARY_SUFFIX}")
  IF(MINGW OR MSVC)
    LIST(APPEND IDEEP_LIBRARIES "${PROJECT_BINARY_DIR}/bin/${MKLDNN_LIB}")
  ELSE(MINGW OR MSVC)
    LIST(APPEND IDEEP_LIBRARIES "${PROJECT_BINARY_DIR}/lib/${MKLDNN_LIB}")
  ENDIF(MINGW OR MSVC)
  #LIST(APPEND IDEEP_LIBRARIES mkldnn)
  SET(MKL_LIBRARIES ${IDEEP_LIBRARIES})
  SET(MKL_INCLUDE_DIR ${IDEEP_MKLDNN_INCLUDE_DIR})
ELSE(IDEEP_FOUND)
  MESSAGE(STATUS "MKLDNN source files not found!")
ENDIF(IDEEP_FOUND)

caffe_clear_vars(__mkldnn_looked_for __mklml_inner_libs)

ENDIF(USE_IDEEP AND NOT IDEEP_FOUND)
