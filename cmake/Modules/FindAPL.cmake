SET(APL_INCLUDE_SEARCH_PATHS
$ENV{ARMPL_DIR}/include_lp64
)

SET(APL_LIB_SEARCH_PATHS
$ENV{ARMPL_DIR}/lib
 )

FIND_PATH(APL_INCLUDE_DIR NAMES armpl.h PATHS ${APL_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(APL_LIB NAMES armpl_lp64 PATHS ${APL_LIB_SEARCH_PATHS})
FIND_LIBRARY(APL_FortranRuntime NAMES FortranRuntime PATHS ${APL_LIB_SEARCH_PATHS})
FIND_LIBRARY(APL_FortranDecimal NAMES FortranDecimal PATHS ${APL_LIB_SEARCH_PATHS})

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

#    Check FortranRuntime
IF(NOT APL_FortranRuntime)
    SET(APL_LIB OFF)
    MESSAGE(STATUS "Could not find FortranRuntime lib. Turning APL_FOUND off")
ENDIF()

#    Check FortranDecimal
IF(NOT APL_FortranDecimal)
    SET(APL_LIB OFF)
    MESSAGE(STATUS "Could not find FortranDecimal lib. Turning APL_FOUND off")
ENDIF()

IF (APL_FOUND)
    MESSAGE(STATUS "Found APL header: ${APL_INCLUDE_DIR}")
    MESSAGE(STATUS "Found APL library: ${APL_LIB}")
    MESSAGE(STATUS "Found FortranRuntime dependency: ${APL_FortranRuntime}")
    MESSAGE(STATUS "Found FortranDecimal dependency: ${APL_FortranDecimal}")
    SET(APL_LIBRARIES ${APL_LIB} ${APL_FortranRuntime} ${APL_FortranDecimal})
    SET(CMAKE_REQUIRED_LIBRARIES ${APL_LIBRARIES})
  CHECK_C_SOURCE_RUNS("
#include <stdlib.h>
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
#include <stdlib.h>
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
MESSAGE(STATUS "BLAS_F2C_FLOAT_WORKS: ${BLAS_F2C_FLOAT_WORKS}")
MESSAGE(STATUS "BLAS_F2C_DOUBLE_WORKS: ${BLAS_F2C_DOUBLE_WORKS}")  
  IF (BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
    MESSAGE(STATUS "This BLAS uses the F2C return conventions")
    SET(BLAS_F2C TRUE)
  ELSE (BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
    SET(BLAS_F2C FALSE)
  ENDIF(BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
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
