# - Find INTEL MKL library
#
# This module sets the following variables:
#  MKL_FOUND - set to true if a library implementing the CBLAS interface is found
#  MKL_VERSION - best guess of the found mkl version
#  MKL_INCLUDE_DIR - path to include dir.
#  MKL_LIBRARIES - list of libraries for base mkl
#  MKL_OPENMP_TYPE - OpenMP flavor that the found mkl uses: GNU or Intel
#  MKL_OPENMP_LIBRARY - path to the OpenMP library the found mkl uses
#  MKL_LAPACK_LIBRARIES - list of libraries to add for lapack
#  MKL_SCALAPACK_LIBRARIES - list of libraries to add for scalapack
#  MKL_SOLVER_LIBRARIES - list of libraries to add for the solvers
#  MKL_CDFT_LIBRARIES - list of libraries to add for the solvers

# Do nothing if MKL_FOUND was set before!
IF (NOT MKL_FOUND)

SET(MKL_VERSION)
SET(MKL_INCLUDE_DIR)
SET(MKL_LIBRARIES)
SET(MKL_OPENMP_TYPE)
SET(MKL_OPENMP_LIBRARY)
SET(MKL_LAPACK_LIBRARIES)
SET(MKL_SCALAPACK_LIBRARIES)
SET(MKL_SOLVER_LIBRARIES)
SET(MKL_CDFT_LIBRARIES)

# Includes
INCLUDE(CheckTypeSize)
INCLUDE(CheckFunctionExists)

# Set default value of INTEL_COMPILER_DIR and INTEL_MKL_DIR
IF (WIN32)
  IF(DEFINED ENV{MKLProductDir})
    SET(DEFAULT_INTEL_COMPILER_DIR $ENV{MKLProductDir})
  ELSE()
    SET(DEFAULT_INTEL_COMPILER_DIR
     "C:/Program Files (x86)/IntelSWTools/compilers_and_libraries/windows")
  ENDIF()
  SET(DEFAULT_INTEL_MKL_DIR "${INTEL_COMPILER_DIR}/mkl")
ELSE (WIN32)
  SET(DEFAULT_INTEL_COMPILER_DIR "/opt/intel")
  SET(DEFAULT_INTEL_MKL_DIR "/opt/intel/mkl")
  if (EXISTS "/opt/intel/oneapi")
    SET(DEFAULT_INTEL_COMPILER_DIR "/opt/intel/oneapi")
    if (EXISTS "/opt/intel/oneapi/mkl/latest")
      SET(DEFAULT_INTEL_MKL_DIR "/opt/intel/oneapi/mkl/latest")
    endif()
  endif()
ENDIF (WIN32)

# Intel Compiler Suite
SET(INTEL_COMPILER_DIR "${DEFAULT_INTEL_COMPILER_DIR}" CACHE STRING
  "Root directory of the Intel Compiler Suite (contains ipp, mkl, etc.)")
SET(INTEL_MKL_DIR "${DEFAULT_INTEL_MKL_DIR}" CACHE STRING
  "Root directory of the Intel MKL (standalone)")
SET(INTEL_OMP_DIR "${DEFAULT_INTEL_MKL_DIR}" CACHE STRING
  "Root directory of the Intel OpenMP (standalone)")
SET(MKL_THREADING "OMP" CACHE STRING "MKL flavor: SEQ, TBB or OMP (default)")

IF (NOT "${MKL_THREADING}" STREQUAL "SEQ" AND
    NOT "${MKL_THREADING}" STREQUAL "TBB" AND
    NOT "${MKL_THREADING}" STREQUAL "OMP")
  MESSAGE(FATAL_ERROR "Invalid MKL_THREADING (${MKL_THREADING}), should be one of: SEQ, TBB, OMP")
ENDIF()

IF ("${MKL_THREADING}" STREQUAL "TBB" AND NOT USE_TBB)
  MESSAGE(FATAL_ERROR "MKL_THREADING is TBB but USE_TBB is turned off")
ENDIF()

MESSAGE(STATUS "MKL_THREADING = ${MKL_THREADING}")

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
  IF ("${MKL_THREADING}" STREQUAL "TBB")
    SET(mklthreads "mkl_tbb_thread")
    SET(mklrtls "tbb")
  ELSE()
    SET(mklthreads "mkl_gnu_thread" "mkl_intel_thread")
    SET(mklrtls "gomp" "iomp5")
  ENDIF()
  SET(mklifaces  "intel" "gf")
ELSE(CMAKE_COMPILER_IS_GNUCC)
  IF ("${MKL_THREADING}" STREQUAL "TBB")
    SET(mklthreads "mkl_tbb_thread")
    SET(mklrtls "tbb")
  ELSE()
    SET(mklthreads "mkl_intel_thread")
    SET(mklrtls "iomp5" "guide")
    IF (MSVC)
      SET(mklrtls "libiomp5md")
    ENDIF (MSVC)
  ENDIF()
  SET(mklifaces  "intel")
ENDIF (CMAKE_COMPILER_IS_GNUCC)

# Kernel libraries dynamically loaded
SET(mklkerlibs "mc" "mc3" "nc" "p4n" "p4m" "p4m3" "p4p" "def")
SET(mklseq)

# Paths
SET(saved_CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH})
SET(saved_CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH})
IF(WIN32)
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
  IF (APPLE)
    SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
      "${INTEL_COMPILER_DIR}/lib")
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
    IF ("${SIZE_OF_VOIDP}" EQUAL 8)
      SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
        "${INTEL_MKL_DIR}/win-x64")
    ENDIF ()
  ENDIF()
  IF (APPLE)
    SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
      "${INTEL_MKL_DIR}/lib")
  ENDIF()
ENDIF()

IF (EXISTS ${INTEL_OMP_DIR})
  # TODO: diagnostic if dir does not exist
  SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}
    "${INTEL_OMP_DIR}/include")
  SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
    "${INTEL_OMP_DIR}/lib/${mklvers}")
  IF (MSVC)
    SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
      "${INTEL_OMP_DIR}/lib/${iccvers}")
    IF ("${SIZE_OF_VOIDP}" EQUAL 8)
      SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
        "${INTEL_OMP_DIR}/win-x64")
    ENDIF ()
  ENDIF()
  IF (APPLE)
    SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
      "${INTEL_OMP_DIR}/lib")
  ENDIF()
ENDIF()

MACRO(GET_MKL_LIB_NAMES LIBRARIES INTERFACE MKL64)
  cmake_parse_arguments("" "" "THREAD" "" ${ARGN})
  SET(${LIBRARIES} mkl_${INTERFACE}${MKL64} mkl_core)
  IF(_THREAD)
    LIST(INSERT ${LIBRARIES} 1 ${_THREAD})
    IF(UNIX AND ${USE_STATIC_MKL})
      # The thread library defines symbols required by the other MKL libraries so also add it last
      LIST(APPEND ${LIBRARIES} ${_THREAD})
    ENDIF()
  ENDIF()
  IF(${USE_STATIC_MKL})
    IF(UNIX)
      list(TRANSFORM ${LIBRARIES} PREPEND "lib")
      list(TRANSFORM ${LIBRARIES} APPEND ".a")
    ELSE()
      message(WARNING "Ignoring USE_STATIC_MKL")
    ENDIF()
  ENDIF()
ENDMACRO()

# Try linking multiple libs
MACRO(CHECK_ALL_LIBRARIES LIBRARIES OPENMP_TYPE OPENMP_LIBRARY _name _list _flags)
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
  SET(${OPENMP_TYPE})
  SET(${OPENMP_LIBRARY})
  SET(_combined_name)
  SET(_openmp_type)
  SET(_openmp_library)
  SET(_paths)
  IF (NOT MKL_FIND_QUIETLY)
    set(_str_list)
    foreach(_elem ${_list})
      if(_str_list)
        set(_str_list "${_str_list} - ${_elem}")
      else()
        set(_str_list "${_elem}")
      endif()
    endforeach(_elem)
    message(STATUS "Checking for [${_str_list}]")
  ENDIF ()
  SET(_found_tbb FALSE)
  FOREACH(_library ${_list})
    SET(_combined_name ${_combined_name}_${_library})
    UNSET(${_prefix}_${_library}_LIBRARY)
    IF(_libraries_work)
      IF(${_library} MATCHES "omp")
        IF(_openmp_type)
          MESSAGE(FATAL_ERROR "More than one OpenMP libraries appear in the MKL test: ${_list}")
        ELSEIF(${_library} MATCHES "gomp")
          SET(_openmp_type "GNU")
          # Use FindOpenMP to find gomp
          FIND_PACKAGE(OpenMP QUIET)
          IF(OPENMP_FOUND)
            # Test that none of the found library names contains "iomp" (Intel
            # OpenMP). This doesn't necessarily mean that we have gomp... but it
            # is probably good enough since on gcc we should already have
            # OpenMP_CXX_FLAGS="-fopenmp" and OpenMP_CXX_LIB_NAMES="".
            SET(_found_gomp true)
            FOREACH(_lib_name ${OpenMP_CXX_LIB_NAMES})
              IF (_found_gomp AND "${_lib_name}" MATCHES "iomp")
                SET(_found_gomp false)
              ENDIF()
            ENDFOREACH()
            IF(_found_gomp)
              SET(${_prefix}_${_library}_LIBRARY ${OpenMP_CXX_FLAGS})
              SET(_openmp_library "${${_prefix}_${_library}_LIBRARY}")
            ENDIF()
          ENDIF(OPENMP_FOUND)
        ELSEIF(${_library} MATCHES "iomp")
          SET(_openmp_type "Intel")
          FIND_LIBRARY(${_prefix}_${_library}_LIBRARY NAMES ${_library})
          SET(_openmp_library "${${_prefix}_${_library}_LIBRARY}")
        ELSE()
          MESSAGE(FATAL_ERROR "Unknown OpenMP flavor: ${_library}")
        ENDIF()
      ELSEIF(${_library} STREQUAL "tbb")
        # Separately handling compiled TBB
        SET(_found_tbb TRUE)
      ELSE()
        SET(lib_names ${_library})
        FIND_LIBRARY(${_prefix}_${_library}_LIBRARY NAMES ${lib_names})
      ENDIF()
      MARK_AS_ADVANCED(${_prefix}_${_library}_LIBRARY)
      IF(NOT (${_library} STREQUAL "tbb"))
        SET(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
        SET(_libraries_work ${${_prefix}_${_library}_LIBRARY})
        IF (NOT MKL_FIND_QUIETLY)
          IF(${_prefix}_${_library}_LIBRARY)
            MESSAGE(STATUS "  Library ${_library}: ${${_prefix}_${_library}_LIBRARY}")
          ELSE(${_prefix}_${_library}_LIBRARY)
            MESSAGE(STATUS "  Library ${_library}: not found")
          ENDIF(${_prefix}_${_library}_LIBRARY)
        ENDIF ()
      ENDIF()
    ENDIF(_libraries_work)
  ENDFOREACH(_library ${_list})
  # Test this combination of libraries.
  IF(_libraries_work)
    IF (NOT _found_tbb)
      SET(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}})
      SET(CMAKE_REQUIRED_LIBRARIES "${CMAKE_REQUIRED_LIBRARIES};${CMAKE_REQUIRED_LIBRARIES}")
      CHECK_FUNCTION_EXISTS(${_name} ${_prefix}${_combined_name}_WORKS)
      SET(CMAKE_REQUIRED_LIBRARIES)
      MARK_AS_ADVANCED(${_prefix}${_combined_name}_WORKS)
      SET(_libraries_work ${${_prefix}${_combined_name}_WORKS})
    ENDIF()
  ENDIF(_libraries_work)
  # Fin
  IF(_libraries_work)
    SET(${OPENMP_TYPE} ${_openmp_type})
    MARK_AS_ADVANCED(${OPENMP_TYPE})
    SET(${OPENMP_LIBRARY} ${_openmp_library})
    MARK_AS_ADVANCED(${OPENMP_LIBRARY})
  ELSE (_libraries_work)
    SET(${LIBRARIES})
    MARK_AS_ADVANCED(${LIBRARIES})
  ENDIF(_libraries_work)
ENDMACRO(CHECK_ALL_LIBRARIES)

IF(WIN32)
  SET(mkl_m "")
  SET(mkl_pthread "")
ELSE(WIN32)
  SET(mkl_m "m")
  SET(mkl_pthread "pthread")
ENDIF(WIN32)

IF(UNIX AND NOT APPLE)
  SET(mkl_dl "${CMAKE_DL_LIBS}")
ELSE(UNIX AND NOT APPLE)
  SET(mkl_dl "")
ENDIF(UNIX AND NOT APPLE)

# Check for version 10/11
IF (NOT MKL_LIBRARIES)
  SET(MKL_VERSION 1011)
ENDIF (NOT MKL_LIBRARIES)

# First: search for parallelized ones with intel thread lib
IF (NOT "${MKL_THREADING}" STREQUAL "SEQ")
  FOREACH(mklrtl ${mklrtls} "")
    FOREACH(mkliface ${mklifaces})
      FOREACH(mkl64 ${mkl64s} "")
        FOREACH(mklthread ${mklthreads})
          IF (NOT MKL_LIBRARIES)
            GET_MKL_LIB_NAMES(mkl_lib_names "${mkliface}" "${mkl64}" THREAD "${mklthread}")
            CHECK_ALL_LIBRARIES(MKL_LIBRARIES MKL_OPENMP_TYPE MKL_OPENMP_LIBRARY cblas_sgemm
              "${mkl_lib_names};${mklrtl};${mkl_pthread};${mkl_m};${mkl_dl}" "")
          ENDIF (NOT MKL_LIBRARIES)
        ENDFOREACH(mklthread)
      ENDFOREACH(mkl64)
    ENDFOREACH(mkliface)
  ENDFOREACH(mklrtl)
ENDIF (NOT "${MKL_THREADING}" STREQUAL "SEQ")

# Second: search for sequential ones
FOREACH(mkliface ${mklifaces})
  FOREACH(mkl64 ${mkl64s} "")
    IF (NOT MKL_LIBRARIES)
      GET_MKL_LIB_NAMES(mkl_lib_names "${mkliface}" "${mkl64}" THREAD "mkl_sequential")
      CHECK_ALL_LIBRARIES(MKL_LIBRARIES MKL_OPENMP_TYPE MKL_OPENMP_LIBRARY cblas_sgemm
        "${mkl_lib_names};${mkl_m};${mkl_dl}" "")
      IF (MKL_LIBRARIES)
        SET(mklseq "_sequential")
      ENDIF (MKL_LIBRARIES)
    ENDIF (NOT MKL_LIBRARIES)
  ENDFOREACH(mkl64)
ENDFOREACH(mkliface)

# First: search for parallelized ones with native pthread lib
FOREACH(mklrtl ${mklrtls} "")
  FOREACH(mkliface ${mklifaces})
    FOREACH(mkl64 ${mkl64s} "")
      IF (NOT MKL_LIBRARIES)
        GET_MKL_LIB_NAMES(mkl_lib_names "${mkliface}" "${mkl64}" THREAD "${mklthread}")
        CHECK_ALL_LIBRARIES(MKL_LIBRARIES MKL_OPENMP_TYPE MKL_OPENMP_LIBRARY cblas_sgemm
          "${mkl_lib_names};${mklrtl};pthread;${mkl_m};${mkl_dl}" "")
      ENDIF (NOT MKL_LIBRARIES)
    ENDFOREACH(mkl64)
  ENDFOREACH(mkliface)
ENDFOREACH(mklrtl)

IF (MKL_LIBRARIES)
  SET(CMAKE_REQUIRED_LIBRARIES ${MKL_LIBRARIES})
  check_function_exists("cblas_gemm_bf16bf16f32" MKL_HAS_SBGEMM)
  set(CMAKE_REQUIRED_LIBRARIES)
  IF(MKL_HAS_SBGEMM)
    add_compile_options(-DMKL_HAS_SBGEMM)
  ENDIF(MKL_HAS_SBGEMM)
ENDIF (MKL_LIBRARIES)

# Check for older versions
IF (NOT MKL_LIBRARIES)
  SET(MKL_VERSION 900)
  if (USE_STATIC_MKL)
      message(WARNING "Ignoring USE_STATIC_MKL")
  endif()
  CHECK_ALL_LIBRARIES(MKL_LIBRARIES MKL_OPENMP_TYPE MKL_OPENMP_LIBRARY cblas_sgemm
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

# Final
SET(CMAKE_LIBRARY_PATH ${saved_CMAKE_LIBRARY_PATH})
SET(CMAKE_INCLUDE_PATH ${saved_CMAKE_INCLUDE_PATH})
IF (MKL_LIBRARIES AND MKL_INCLUDE_DIR)
  SET(MKL_FOUND TRUE)
ELSE (MKL_LIBRARIES AND MKL_INCLUDE_DIR)
  if (MKL_LIBRARIES AND NOT MKL_INCLUDE_DIR)
    MESSAGE(WARNING "MKL libraries files are found, but MKL header files are \
      not. You can get them by `conda install mkl-include` if using conda (if \
      it is missing, run `conda upgrade -n root conda` first), and \
      `pip install mkl-devel` if using pip. If build fails with header files \
      available in the system, please make sure that CMake will search the \
      directory containing them, e.g., by setting CMAKE_INCLUDE_PATH.")
  endif()
  SET(MKL_FOUND FALSE)
  SET(MKL_VERSION)  # clear MKL_VERSION
ENDIF (MKL_LIBRARIES AND MKL_INCLUDE_DIR)

# Standard termination
IF(NOT MKL_FOUND AND MKL_FIND_REQUIRED)
  MESSAGE(FATAL_ERROR "MKL library not found. Please specify library location \
    by appending the root directory of the MKL installation to the environment variable CMAKE_PREFIX_PATH.")
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
