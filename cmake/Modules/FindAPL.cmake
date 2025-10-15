# - Find APL (Arm Performance Libraries)
#
# This module sets the following variables:
#   APL_INCLUDE_SEARCH_PATHS - list of paths to search for APL include files
#   APL_LIB_SEARCH_PATHS - list of paths to search for APL libraries
#   APL_FOUND - set to true if APL is found
#   APL_INCLUDE_DIR - path to include dir.
#   APL_LIB_DIR - path to include dir.
#   APL_LIBRARIES - list of libraries for base APL

SET(APL_INCLUDE_SEARCH_PATHS $ENV{ARMPL_DIR}/include)
SET(APL_LIB_SEARCH_PATHS $ENV{ARMPL_DIR}/lib)
SET(APL_BIN_SEARCH_PATHS $ENV{ARMPL_DIR}/bin)

SET(APL_FOUND ON)

# Check include file
FIND_PATH(APL_INCLUDE_DIR NAMES armpl.h PATHS ${APL_INCLUDE_SEARCH_PATHS})
IF(NOT APL_INCLUDE_DIR)
    SET(APL_FOUND OFF)
    MESSAGE(STATUS "Could not verify APL include directory. Turning APL_FOUND off")
ENDIF()

# Check lib file
FIND_PATH(APL_LIB_DIR NAMES armpl_lp64.dll.lib libarmpl_lp64.a PATHS ${APL_LIB_SEARCH_PATHS})
IF(NOT APL_LIB_DIR)
    SET(APL_FOUND OFF)
    MESSAGE(STATUS "Could not verify APL lib directory. Turning APL_FOUND off")
ENDIF()

# Check bin file
FIND_PATH(APL_BIN_DIR NAMES armpl_lp64.dll armpl-info PATHS ${APL_BIN_SEARCH_PATHS})
IF(NOT APL_BIN_DIR)
    SET(APL_FOUND OFF)
    MESSAGE(STATUS "Could not verify APL bin directory. Turning APL_FOUND off")
ENDIF()

IF (APL_FOUND)
  IF(WIN32)
    set(APL_LIBRARIES
      "${APL_LIB_DIR}/armpl_lp64.dll.lib"
    )
    set(APL_DLLS
      "${CMAKE_INSTALL_PREFIX}/lib/armpl_lp64.dll"
    )
    add_custom_command(
      OUTPUT ${APL_DLLS}
      COMMAND ${CMAKE_COMMAND} -E make_directory "${CMAKE_INSTALL_PREFIX}/lib"
      COMMAND ${CMAKE_COMMAND} -E copy_if_different "${APL_BIN_DIR}/armpl_lp64.dll" "${CMAKE_INSTALL_PREFIX}/lib/armpl_lp64.dll"
    )
    add_custom_target(copy_apl_dlls ALL DEPENDS ${APL_DLLS})
  ELSEIF(UNIX)
    set(APL_LIBRARIES
      "${APL_LIB_DIR}/libarmpl_lp64.a"
    )
  ENDIF()
  MESSAGE(STATUS "Found APL header: ${APL_INCLUDE_DIR}")
  MESSAGE(STATUS "Found APL library: ${APL_LIB_DIR}")
  message(STATUS "APL_LIBRARIES: ${APL_LIBRARIES}")
  SET(CMAKE_REQUIRED_LIBRARIES ${APL_LIBRARIES})
  include(CheckCSourceRuns)
  CHECK_C_SOURCE_RUNS("
#include <stdlib.h>
#include <stdio.h>
float x[4] = { 1, 2, 3, 4 };
float y[4] = { .1, .01, .001, .0001 };
extern float cblas_sdot();
int main() {
  int i;
  double r = cblas_sdot(4, x, 1, y, 1);
  exit((float)r != (float).1234);
}" BLAS_USE_CBLAS_DOT )
  MESSAGE(STATUS "BLAS_USE_CBLAS_DOT: ${BLAS_USE_CBLAS_DOT}")
ENDIF (APL_FOUND)