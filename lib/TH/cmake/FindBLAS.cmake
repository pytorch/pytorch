# - Find BLAS library
# This module finds an installed fortran library that implements the BLAS 
# linear-algebra interface (see http://www.netlib.org/blas/).  
# The list of libraries searched for is taken
# from the autoconf macro file, acx_blas.m4 (distributed at
# http://ac-archive.sourceforge.net/ac-archive/acx_blas.html).
#
# This module sets the following variables:
#  BLAS_FOUND - set to true if a library implementing the BLAS interface is found.
#  BLAS_INFO - name of the detected BLAS library.
#  BLAS_F2C - set to true if following the f2c return convention
#  BLAS_LIBRARIES - list of libraries to link against to use BLAS
#  BLAS_INCLUDE_DIR - include directory

SET(BLAS_LIBRARIES)
SET(BLAS_INCLUDE_DIR)
SET(BLAS_INFO)
SET(BLAS_F2C)

# CBLAS in Intel mkl
FIND_PACKAGE(MKL)
IF (MKL_FOUND AND NOT BLAS_LIBRARIES)
  SET(BLAS_INFO imkl)
  SET(BLAS_LIBRARIES ${MKL_LIBRARIES})
  SET(BLAS_INCLUDE_DIR ${MKL_INCLUDE_DIR})
  SET(BLAS_VERSION ${MKL_VERSION})
ENDIF (MKL_FOUND AND NOT BLAS_LIBRARIES)

# Old FindBlas
INCLUDE(CheckCSourceRuns)
INCLUDE(CheckFortranFunctionExists)
SET(_verbose TRUE)

MACRO(Check_Fortran_Libraries LIBRARIES _prefix _name _flags _list)
  # This macro checks for the existence of the combination of fortran libraries
  # given by _list.  If the combination is found, this macro checks (using the 
  # Check_Fortran_Function_Exists macro) whether can link against that library
  # combination using the name of a routine given by _name using the linker
  # flags given by _flags.  If the combination of libraries is found and passes
  # the link test, LIBRARIES is set to the list of complete library paths that
  # have been found.  Otherwise, LIBRARIES is set to NOTFOUND.
  # N.B. _prefix is the prefix applied to the names of all cached variables that
  # are generated internally and marked advanced by this macro.
  
  set(__list)
  foreach(_elem ${_list})
    if(__list)
      set(__list "${__list} - ${_elem}")
    else(__list)
      set(__list "${_elem}")
    endif(__list)
  endforeach(_elem)
  if(_verbose)
    message(STATUS "Checking for [${__list}]")
  endif(_verbose)

  set(_libraries_work TRUE)
  set(${LIBRARIES})
  set(_combined_name)
  foreach(_library ${_list})
    set(_combined_name ${_combined_name}_${_library})
    if(_libraries_work)
      if ( WIN32 )
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS ENV LIB 
          PATHS ENV PATH )
      endif ( WIN32 )
      if ( APPLE ) 
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 
          ENV DYLD_LIBRARY_PATH )
      else ( APPLE )
        find_library(${_prefix}_${_library}_LIBRARY
          NAMES ${_library}
          PATHS /usr/local/lib /usr/lib /usr/local/lib64 /usr/lib64 
          ENV LD_LIBRARY_PATH )
      endif( APPLE )
      mark_as_advanced(${_prefix}_${_library}_LIBRARY)
      set(${LIBRARIES} ${${LIBRARIES}} ${${_prefix}_${_library}_LIBRARY})
      set(_libraries_work ${${_prefix}_${_library}_LIBRARY})
    endif(_libraries_work)
  endforeach(_library ${_list})
  if(_libraries_work)
    # Test this combination of libraries.
    set(CMAKE_REQUIRED_LIBRARIES ${_flags} ${${LIBRARIES}})
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
    set(${LIBRARIES} NOTFOUND)
  endif(NOT _libraries_work)
endmacro(Check_Fortran_Libraries)


# Apple BLAS library?
if(NOT BLAS_LIBRARIES)
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "Accelerate")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "accelerate")
  endif (BLAS_LIBRARIES)
endif(NOT BLAS_LIBRARIES)
if ( NOT BLAS_LIBRARIES )
  check_fortran_libraries(
    BLAS_LIBRARIES
    BLAS
    sgemm
    ""
    "vecLib")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "veclib")
  endif (BLAS_LIBRARIES)
endif ( NOT BLAS_LIBRARIES )

# BLAS in ATLAS library? (http://math-atlas.sourceforge.net/)
if(NOT BLAS_LIBRARIES)
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "cblas;f77blas;atlas")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "atlas")
  endif (BLAS_LIBRARIES)
endif(NOT BLAS_LIBRARIES)

# Generic BLAS library?
if(NOT BLAS_LIBRARIES)
  check_fortran_libraries(
  BLAS_LIBRARIES
  BLAS
  sgemm
  ""
  "blas")
  if (BLAS_LIBRARIES)
    set(BLAS_INFO "generic")
  endif (BLAS_LIBRARIES)
endif(NOT BLAS_LIBRARIES)

# Determine if blas was compiled with the f2c conventions
IF (BLAS_LIBRARIES)
  SET(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  CHECK_C_SOURCE_RUNS("
#include <stdio.h>
float x[4] = { 1, 2, 3, 4 };
float y[4] = { .1, .01, .001, .0001 };
int four = 4;
int one = 1;
extern double sdot_();
int main() {
  int i;
  double r = sdot_(&four, x, &one, y, &one);
  exit((float)r != (float).1234);
}" BLAS_F2C_DOUBLE_WORKS )
  CHECK_C_SOURCE_RUNS("
#include <stdio.h>
float x[4] = { 1, 2, 3, 4 };
float y[4] = { .1, .01, .001, .0001 };
int four = 4;
int one = 1;
extern float sdot_();
int main() {
  int i;
  double r = sdot_(&four, x, &one, y, &one);
  exit((float)r != (float).1234);
}" BLAS_F2C_FLOAT_WORKS )
  IF (BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
    IF (_verbose)
      MESSAGE(STATUS "This BLAS uses the F2C return conventions")
    ENDIF(_verbose)
    SET(BLAS_F2C TRUE)
  ELSE (BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
    SET(BLAS_F2C FALSE)
  ENDIF (BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
ENDIF(BLAS_LIBRARIES)

# epilogue

if(BLAS_LIBRARIES)
  set(BLAS_FOUND TRUE)
else(BLAS_LIBRARIES)
  set(BLAS_FOUND FALSE)
endif(BLAS_LIBRARIES)

IF (NOT BLAS_FOUND AND BLAS_FIND_REQUIRED)
  message(FATAL_ERROR "Cannot find a library with BLAS API. Please specify library location.")
ENDIF (NOT BLAS_FOUND AND BLAS_FIND_REQUIRED)
IF(NOT BLAS_FIND_QUIETLY)
  IF(BLAS_FOUND)
    MESSAGE(STATUS "Found a library with BLAS API (${BLAS_INFO}).")
  ELSE(BLAS_FOUND)
    MESSAGE(STATUS "Cannot find a library with BLAS API. Not using BLAS.")
  ENDIF(BLAS_FOUND)
ENDIF(NOT BLAS_FIND_QUIETLY)





