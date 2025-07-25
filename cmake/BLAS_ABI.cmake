# Push host architecture when cross-compiling otherwise check would fail when
# cross-compiling for arm64 on x86_64
cmake_push_check_state(RESET)
if(CMAKE_SYSTEM_NAME STREQUAL "Darwin" AND CMAKE_OSX_ARCHITECTURES MATCHES
                                           "^(x86_64|arm64)$")
  list(APPEND CMAKE_REQUIRED_FLAGS "-arch ${CMAKE_HOST_SYSTEM_PROCESSOR}")
endif()

# Set values through env variables if cross compiling
if(CMAKE_CROSSCOMPILING)
  if("$ENV{PYTORCH_BLAS_F2C}" STREQUAL "ON")
    set(BLAS_F2C TRUE)
  else()
    set(BLAS_F2C FALSE)
  endif()

  if("$ENV{PYTORCH_BLAS_USE_CBLAS_DOT}" STREQUAL "ON")
    set(BLAS_USE_CBLAS_DOT TRUE)
  else()
    set(BLAS_USE_CBLAS_DOT FALSE)
  endif()
else()
  set(CMAKE_REQUIRED_LIBRARIES ${BLAS_LIBRARIES})
  check_c_source_runs(
    "
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
}"
    BLAS_F2C_DOUBLE_WORKS)
  check_c_source_runs(
    "
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
}"
    BLAS_F2C_FLOAT_WORKS)

  if(BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
    message(STATUS "This BLAS uses the F2C return conventions")
    set(BLAS_F2C TRUE)
  else(BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
    set(BLAS_F2C FALSE)
  endif(BLAS_F2C_DOUBLE_WORKS AND NOT BLAS_F2C_FLOAT_WORKS)
  check_c_source_runs(
    "
#include <stdlib.h>
#include <stdio.h>
float x[4] = { 1, 2, 3, 4 };
float y[4] = { .1, .01, .001, .0001 };
extern float cblas_sdot();
int main() {
  int i;
  double r = cblas_sdot(4, x, 1, y, 1);
  exit((float)r != (float).1234);
}"
    BLAS_USE_CBLAS_DOT)
  if(BLAS_USE_CBLAS_DOT)
    set(BLAS_USE_CBLAS_DOT TRUE)
  else(BLAS_USE_CBLAS_DOT)
    set(BLAS_USE_CBLAS_DOT FALSE)
  endif(BLAS_USE_CBLAS_DOT)
  set(CMAKE_REQUIRED_LIBRARIES)
endif(CMAKE_CROSSCOMPILING)
message(STATUS "BLAS_USE_CBLAS_DOT: ${BLAS_USE_CBLAS_DOT}")
message(STATUS "BLAS_F2C: ${BLAS_F2C}")
cmake_pop_check_state()
