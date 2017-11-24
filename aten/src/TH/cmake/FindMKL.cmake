# - Find INTEL MKL library
#
# This module finds the Intel Mkl libraries.
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


# Do nothing if MKL_FOUND was set before!
IF (NOT MKL_FOUND)

SET(MKL_VERSION)
SET(MKL_INCLUDE_DIR)
SET(MKL_LIBRARIES)
SET(MKL_LAPACK_LIBRARIES)
SET(MKL_SCALAPACK_LIBRARIES)
SET(MKL_SOLVER_LIBRARIES)
SET(MKL_CDFT_LIBRARIES)

# Includes
INCLUDE(CheckTypeSize)
INCLUDE(CheckFunctionExists)

# Intel Compiler Suite
SET(INTEL_COMPILER_DIR CACHE STRING
  "Root directory of the Intel Compiler Suite (contains ipp, mkl, etc.)")
SET(INTEL_MKL_DIR CACHE STRING
  "Root directory of the Intel MKL (standalone)")
SET(INTEL_MKL_SEQUENTIAL OFF CACHE BOOL
  "Force using the sequential (non threaded) libraries")

# Checks
CHECK_TYPE_SIZE("void*" SIZE_OF_VOIDP)
IF ("${SIZE_OF_VOIDP}" EQUAL 8)
  SET(mklvers "em64t")
  SET(iccvers "intel64")
  SET(mkl64s "_lp64")
ELSE ("${SIZE_OF_VOIDP}" EQUAL 8)
  SET(mklvers "32")
  SET(iccvers "ia32")
  SET(mkl64s)
ENDIF ("${SIZE_OF_VOIDP}" EQUAL 8)
IF(CMAKE_COMPILER_IS_GNUCC)
  SET(mklthreads "mkl_gnu_thread" "mkl_intel_thread")
  SET(mklifaces  "gf" "intel")
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
IF (INTEL_COMPILER_DIR)
  # TODO: diagnostic if dir does not exist
  SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
    "${INTEL_COMPILER_DIR}/lib/${iccvers}")
  IF (NOT INTEL_MKL_DIR)
    SET(INTEL_MKL_DIR "${INTEL_COMPILER_DIR}/mkl")
  ENDIF (NOT INTEL_MKL_DIR)
ENDIF (INTEL_COMPILER_DIR)
IF (INTEL_MKL_DIR)
  # TODO: diagnostic if dir does not exist
  SET(CMAKE_INCLUDE_PATH ${CMAKE_INCLUDE_PATH}
    "${INTEL_MKL_DIR}/include")
  SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
    "${INTEL_MKL_DIR}/lib/${mklvers}")
  IF (MSVC)
    SET(CMAKE_LIBRARY_PATH ${CMAKE_LIBRARY_PATH}
      "${INTEL_MKL_DIR}/lib/${iccvers}")
  ENDIF (MSVC)
ENDIF (INTEL_MKL_DIR)

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
else(WIN32)
  set(mkl_m "m")
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
            "mkl_${mkliface}${mkl64};${mklthread};mkl_core;${mklrtl};pthread;${mkl_m};${mkl_dl}" "")
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
IF (MKL_LIBRARIES)
  SET(MKL_FOUND TRUE)
ELSE (MKL_LIBRARIES)
  SET(MKL_FOUND FALSE)
  SET(MKL_VERSION)
ENDIF (MKL_LIBRARIES)

# Standard termination
IF(NOT MKL_FOUND AND MKL_FIND_REQUIRED)
  MESSAGE(FATAL_ERROR "MKL library not found. Please specify library  location")
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


