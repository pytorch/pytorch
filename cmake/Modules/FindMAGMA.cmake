# - Find MAGMA library
# This module finds an installed MAGMA library, a matrix algebra library
# similar to LAPACK for GPU and multicore systems
# (see http://icl.cs.utk.edu/magma/).
#
# This module will look for MAGMA library under /usr/local/magma by
# default. To use a different installed version of the library set
# environment variable MAGMA_HOME before running cmake (e.g.
# MAGMA_HOME=${HOME}/lib/magma instead of default /usr/local/magma)
#
# This module sets the following variables:
#  MAGMA_FOUND - set to true if the MAGMA library is found.
#  MAGMA_LIBRARIES - list of libraries to link against to use MAGMA
#  MAGMA_INCLUDE_DIR - include directory

if(MAGMA_FOUND)
  return()
endif()

include(FindPackageHandleStandardArgs)

SET(MAGMA_LIBRARIES)
SET(MAGMA_INCLUDE_DIR)

FIND_LIBRARY(MAGMA_LIBRARIES magma
  HINTS $ENV{MAGMA_HOME} /usr/local/magma
  PATH_SUFFIXES lib)

FIND_PATH(MAGMA_INCLUDE_DIR magma.h
  HINTS $ENV{MAGMA_HOME} /usr/local/magma
  PATH_SUFFIXES include)

IF (MAGMA_LIBRARIES)
  SET(MAGMA_FOUND TRUE)
ELSE (MAGMA_LIBRARIES)
  SET(MAGMA_FOUND FALSE)
ENDIF (MAGMA_LIBRARIES)

add_library(torch::magma INTERFACE IMPORTED)
set_property(TARGET torch::magma
             PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${MAGMA_INCLUDE_DIR}")
set_property(TARGET torch::magma
             PROPERTY INTERFACE_LINK_LIBRARIES "${MAGMA_LIBRARIES}")

# Check for Magma V2
include(CheckPrototypeDefinition)
check_prototype_definition(magma_get_sgeqrf_nb
  "magma_int_t magma_get_sgeqrf_nb( magma_int_t m, magma_int_t n );"
  "0"
  "magma.h"
  MAGMA_V2)
if(MAGMA_V2)
  set_property(TARGET torch::magma
               PROPERTY INTERFACE_COMPILE_DEFINITIONS "MAGMA_V2")
endif(MAGMA_V2)
