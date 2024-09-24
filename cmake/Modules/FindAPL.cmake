SET(APL_INCLUDE_SEARCH_PATHS
$ENV{ARMPL_DIR}/include
)

SET(APL_LIB_SEARCH_PATHS
$ENV{ARMPL_DIR}/lib
 )

FIND_PATH(APL_INCLUDE_DIR NAMES armpl.h PATHS ${APL_INCLUDE_SEARCH_PATHS})

IF(WIN32)
  FIND_LIBRARY(APL_LIB NAMES armpl_lp64.dll.lib PATHS ${APL_LIB_SEARCH_PATHS})
  FIND_LIBRARY(APL_MATH NAMES amath.dll.lib PATHS ${APL_LIB_SEARCH_PATHS})
ELSEIF(UNIX)
  FIND_LIBRARY(APL_LIB NAMES armpl_lp64 libarmpl.so libarmpl.a PATHS ${APL_LIB_SEARCH_PATHS})
  FIND_LIBRARY(APL_MATH NAMES amath libamath.so libamath.a PATHS ${APL_LIB_SEARCH_PATHS})
ENDIF()

SET(APL_FOUND ON)

#    Check include files
IF(NOT APL_INCLUDE_DIR)
    SET(APL_FOUND OFF)
    MESSAGE(STATUS "Could not find APL include directory. Turning APL_FOUND off")
ENDIF()

#    Check libraries
IF(NOT APL_LIB)
    SET(APL_FOUND OFF)
    MESSAGE(STATUS "Could not find APL lib. Turning APL_FOUND off")
ENDIF()

IF(NOT APL_MATH)
    SET(APL_FOUND OFF)
    MESSAGE(STATUS "Could not find APL_MATH lib. Turning APL_FOUND off")
ENDIF()

IF (APL_FOUND)
    MESSAGE(STATUS "Found APL header: ${APL_INCLUDE_DIR}")
    MESSAGE(STATUS "Found APL library: ${APL_LIB}")
    MESSAGE(STATUS "Found APL MATH: ${APL_MATH}")
    SET(APL_LIBRARIES ${APL_LIB} ${APL_MATH})
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