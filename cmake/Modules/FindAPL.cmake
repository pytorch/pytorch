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
IF (NOT APL_FOUND)

SET(APL_VERSION)
SET(APL_INCLUDE_DIR)
SET(APL_LIBRARIES)
SET(APL_OPENMP_TYPE)
SET(APL_OPENMP_LIBRARY)
SET(APL_LAPACK_LIBRARIES)
SET(APL_SCALAPACK_LIBRARIES)
SET(APL_SOLVER_LIBRARIES)
SET(APL_CDFT_LIBRARIES)
FIND_LIBRARY(APL_LIBRARIES armpl_ilp64 FortranRuntime
  PATH_SUFFIXES lib)
  FIND_LIBRARY(FORTRAN_RUNTIME_LIB FortranRuntime
  PATH_SUFFIXES lib)
if(FORTRAN_RUNTIME_LIB)
  list(APPEND APL_LIBRARIES ${FORTRAN_RUNTIME_LIB})
else()
  message(FATAL_ERROR "FortranRuntime library not found")
endif()
message("ionut3" ${APL_LIBRARIES})
#message(FATAL_ERROR "Terminating CMake due to some condition")

IF (APL_LIBRARIES)
  FIND_PATH(APL_INCLUDE_DIR blas.h
  PATH_SUFFIXES include_lp64)
  SET(APL_FOUND TRUE)
ENDIF (APL_LIBRARIES)
ENDIF (NOT APL_FOUND)
