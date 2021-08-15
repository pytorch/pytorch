# - Find LAPACK library
# This module finds an installed fortran library that implements the LAPACK
# linear-algebra interface (see http://www.netlib.org/lapack/).
#
# The approach follows that taken for the autoconf macro file, acx_lapack.m4
# (distributed at http://ac-archive.sourceforge.net/ac-archive/acx_lapack.html).
#
# This module sets the following variables:
#  LAPACK_FOUND - set to true if a library implementing the LAPACK interface is found
#  LAPACK_LIBRARIES - list of libraries (using full path name) for LAPACK

# Note: I do not think it is a good idea to mixup different BLAS/LAPACK versions
# Hence, this script wants to find a Lapack library matching your Blas library

# Do nothing if LAPACK was found before
IF(NOT LAPACK_FOUND)

SET(LAPACK_LIBRARIES)
SET(LAPACK_INFO)

IF(LAPACK_FIND_QUIETLY OR NOT LAPACK_FIND_REQUIRED)
  FIND_PACKAGE(BLAS)
ELSE(LAPACK_FIND_QUIETLY OR NOT LAPACK_FIND_REQUIRED)
  FIND_PACKAGE(BLAS REQUIRED)
ENDIF(LAPACK_FIND_QUIETLY OR NOT LAPACK_FIND_REQUIRED)

# Old search lapack script
include(CheckFortranFunctionExists)

macro(Check_Lapack_Libraries LIBRARIES _prefix _name _flags _list _blas)
  # This macro checks for the existence of the combination of fortran libraries
  # given by _list.  If the combination is found, this macro checks (using the
  # Check_Fortran_Function_Exists macro) whether can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to FALSE.
  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.
  set(_libraries_work TRUE)
  set(${LIBRARIES})
  set(_combined_name)
  foreach(_library ${_list})
    set(_combined_name ${_combined_name}_${_library})
    if(_libraries_work)
      if (WIN32)
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library} PATHS ENV LIB PATHS ENV PATH)
      else (WIN32)
        if(APPLE)
          find_library(${_prefix}_${_library}_LIBRARY
            NAMES ${_library}
            PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 /usr/lib/aarch64-linux-gnu
            ENV DYLD_LIBRARY_PATH)
        else(APPLE)
          find_library(${_prefix}_${_library}_LIBRARY
            NAMES ${_library}
            PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 /usr/lib/aarch64-linux-gnu
            ENV LD_LIBRARY_PATH)
        endif(APPLE)
      endif(WIN32)
      mark_as_advanced(${_prefix}_${_library}_LIBRARY)
      set(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      set(_libraries_work ${${_prefix}_${_library}_LIBRARY})
    endif(_libraries_work)
  endforeach(_library ${_list})
  if(_libraries_work)
    # Test this combination of libraries.
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}} ${_blas})
    if (CMAKE_Fortran_COMPILER_WORKS)
      check_fortran_function_exists(${_name} ${_prefix}${_combined_name}_WORKS)
    else (CMAKE_Fortran_COMPILER_WORKS)
      check_function_exists("${_name}_" ${_prefix}${_combined_name}_WORKS)
    endif (CMAKE_Fortran_COMPILER_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES)
    mark_as_advanced(${_prefix}${_combined_name}_WORKS)
    set(_libraries_work ${${_prefix}${_combined_name}_WORKS})
  endif(_libraries_work)
  if(NOT _libraries_work)
    set(${LIBRARIES} FALSE)
  endif(NOT _libraries_work)
endmacro(Check_Lapack_Libraries)


if(BLAS_FOUND)

  # Intel MKL
  IF((NOT LAPACK_INFO) AND (BLAS_INFO STREQUAL "mkl"))
    IF(MKL_LAPACK_LIBRARIES)
      SET(LAPACK_LIBRARIES ${MKL_LAPACK_LIBRARIES} ${MKL_LIBRARIES})
    ELSE(MKL_LAPACK_LIBRARIES)
      SET(LAPACK_LIBRARIES ${MKL_LIBRARIES})
    ENDIF(MKL_LAPACK_LIBRARIES)
    SET(LAPACK_INCLUDE_DIR ${MKL_INCLUDE_DIR})
    SET(LAPACK_INFO "mkl")
  ENDIF()

  # Accelerate
  IF((NOT LAPACK_INFO) AND (BLAS_INFO STREQUAL "accelerate"))
    SET(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    check_function_exists("cheev_" ACCELERATE_LAPACK_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES)
    if(ACCELERATE_LAPACK_WORKS)
      SET(LAPACK_INFO "accelerate")
    else()
      message(STATUS "Strangely, this Accelerate library does not support Lapack?!")
    endif()
  endif()

  # vecLib
  IF((NOT LAPACK_INFO) AND (BLAS_INFO STREQUAL "veclib"))
    SET(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    check_function_exists("cheev_" VECLIB_LAPACK_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES)
    if(VECLIB_LAPACK_WORKS)
      SET(LAPACK_INFO "veclib")
    else()
      message(STATUS "Strangely, this vecLib library does not support Lapack?!")
    endif()
  endif()

  # OpenBlas
  IF((NOT LAPACK_INFO) AND (BLAS_INFO STREQUAL "open"))
    SET(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    check_function_exists("cheev_" OPEN_LAPACK_WORKS)
    if(OPEN_LAPACK_WORKS)
      check_function_exists("cgesdd_" LAPACK_CGESDD_WORKS)
      if(NOT LAPACK_CGESDD_WORKS)
        find_library(GFORTRAN_LIBRARY
          NAMES libgfortran.a gfortran
          PATHS ${CMAKE_C_IMPLICIT_LINK_DIRECTORIES})
       list(APPEND CMAKE_REQUIRED_LIBRARIES "${GFORTRAN_LIBRARY}")
       unset(LAPACK_CGESDD_WORKS CACHE)
       check_function_exists("cgesdd_" LAPACK_CGESDD_WORKS)
       if(LAPACK_CGESDD_WORKS)
         list(APPEND LAPACK_LIBRARIES "${GFORTRAN_LIBRARY}")
       else()
         message(WARNING "OpenBlas has been compiled with Lapack support, but cgesdd can not be used")
         set(OPEN_LAPACK_WORKS NO)
       endif()
      endif()
    endif()

    set(CMAKE_REQUIRED_LIBRARIES)
    if(OPEN_LAPACK_WORKS)
      SET(LAPACK_INFO "open")
    else()
      message(STATUS "It seems OpenBlas has not been compiled with Lapack support")
    endif()
  endif()

  # GotoBlas
  IF((NOT LAPACK_INFO) AND (BLAS_INFO STREQUAL "goto"))
    SET(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    check_function_exists("cheev_" GOTO_LAPACK_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES)
    if(GOTO_LAPACK_WORKS)
      SET(LAPACK_INFO "goto")
    else()
      message(STATUS "It seems GotoBlas has not been compiled with Lapack support")
    endif()
  endif()

  # FLAME
  IF((NOT LAPACK_INFO) AND (BLAS_INFO STREQUAL "FLAME"))
    check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "flame"
      "${BLAS_LIBRARIES}"
      )
    if(LAPACK_LIBRARIES)
      SET(LAPACK_INFO "FLAME")
    endif(LAPACK_LIBRARIES)
  endif()

  # ACML
  IF((NOT LAPACK_INFO) AND (BLAS_INFO STREQUAL "acml"))
    SET(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
    check_function_exists("cheev_" ACML_LAPACK_WORKS)
    set(CMAKE_REQUIRED_LIBRARIES)
    if(ACML_LAPACK_WORKS)
      SET(LAPACK_INFO "acml")
    else()
      message(STATUS "Strangely, this ACML library does not support Lapack?!")
    endif()
  endif()

  # Generic LAPACK library?
  IF((NOT LAPACK_INFO) AND (BLAS_INFO STREQUAL "generic"))
    check_lapack_libraries(
      LAPACK_LIBRARIES
      LAPACK
      cheev
      ""
      "lapack"
      "${BLAS_LIBRARIES}"
      )
    if(LAPACK_LIBRARIES)
      SET(LAPACK_INFO "generic")
    endif(LAPACK_LIBRARIES)
  endif()

else(BLAS_FOUND)
  message(STATUS "LAPACK requires BLAS")
endif(BLAS_FOUND)

if(LAPACK_INFO)
  set(LAPACK_FOUND TRUE)
else(LAPACK_INFO)
  set(LAPACK_FOUND FALSE)
endif(LAPACK_INFO)

IF (NOT LAPACK_FOUND AND LAPACK_FIND_REQUIRED)
  message(FATAL_ERROR "Cannot find a library with LAPACK API. Please specify library location.")
ENDIF (NOT LAPACK_FOUND AND LAPACK_FIND_REQUIRED)
IF(NOT LAPACK_FIND_QUIETLY)
  IF(LAPACK_FOUND)
    MESSAGE(STATUS "Found a library with LAPACK API (${LAPACK_INFO}).")
  ELSE(LAPACK_FOUND)
    MESSAGE(STATUS "Cannot find a library with LAPACK API. Not using LAPACK.")
  ENDIF(LAPACK_FOUND)
ENDIF(NOT LAPACK_FIND_QUIETLY)

# Do nothing if LAPACK was found before
ENDIF(NOT LAPACK_FOUND)
