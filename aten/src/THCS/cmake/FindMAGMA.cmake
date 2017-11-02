# - Find MAGMA library
# This module finds an installed MAGMA library, a matrix algebra library
# similar to LAPACK for GPU and multicore systems
# (see http://icl.cs.utk.edu/magma/).
#
# This module sets the following variables:
#  MAGMA_FOUND - set to true if the MAGMA library is found.
#  MAGMA_LIBRARIES - list of libraries to link against to use MAGMA
#  MAGMA_INCLUDE_DIR - include directory

IF(NOT MAGMA_FOUND)

include(FindPackageHandleStandardArgs)

SET(MAGMA_LIBRARIES)
SET(MAGMA_INCLUDE_DIR)

FIND_LIBRARY(MAGMA_LIBRARIES magma /usr/local/magma/lib)
FIND_PATH(MAGMA_INCLUDE_DIR magma.h /usr/local/magma/include)

IF (MAGMA_LIBRARIES)
  SET(MAGMA_FOUND TRUE)
ELSE (MAGMA_LIBRARIES)
  SET(MAGMA_FOUND FALSE)
ENDIF (MAGMA_LIBRARIES)

ENDIF(NOT MAGMA_FOUND)
