

SET(Open_BLAS_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/openblas
  /usr/include/openblas-base
  /usr/local/include
  /usr/local/include/openblas
  /usr/local/include/openblas-base
  /usr/local/opt/openblas/include
  /opt/OpenBLAS/include
  $ENV{OpenBLAS_HOME}
  $ENV{OpenBLAS_HOME}/include
  $ENV{OpenBLAS_HOME}/include/openblas
)

SET(Open_BLAS_LIB_SEARCH_PATHS
        /lib/
        /lib/openblas-base
        /lib64/
        /usr/lib
        /usr/lib/openblas-base
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /usr/local/opt/openblas/lib
        /opt/OpenBLAS/lib
        $ENV{OpenBLAS}
        $ENV{OpenBLAS}/lib
        $ENV{OpenBLAS_HOME}
        $ENV{OpenBLAS_HOME}/lib
 )

FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(OpenBLAS_LIB NAMES openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS})

SET(OpenBLAS_FOUND ON)

#    Check include files
IF(NOT OpenBLAS_INCLUDE_DIR)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS include. Turning OpenBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT OpenBLAS_LIB)
    SET(OpenBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find OpenBLAS lib. Turning OpenBLAS_FOUND off")
ENDIF()

IF (OpenBLAS_FOUND)
  IF (NOT OpenBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found OpenBLAS libraries: ${OpenBLAS_LIB}")
    MESSAGE(STATUS "Found OpenBLAS include: ${OpenBLAS_INCLUDE_DIR}")
  ENDIF (NOT OpenBLAS_FIND_QUIETLY)
ELSE (OpenBLAS_FOUND)
  IF (OpenBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find OpenBLAS")
  ENDIF (OpenBLAS_FIND_REQUIRED)
ENDIF (OpenBLAS_FOUND)

IF(OpenBLAS_LIB)
 # Run ldd on the OpenBLAS library
execute_process(
  COMMAND ldd "${OpenBLAS_LIB}"
  OUTPUT_VARIABLE LDD_OUTPUT
  ERROR_VARIABLE LDD_ERROR
  RESULT_VARIABLE LDD_RESULT
  OUTPUT_STRIP_TRAILING_WHITESPACE
)

if(NOT LDD_RESULT EQUAL 0)
  message(WARNING "ldd failed on ${OpenBLAS_LIB}: ${LDD_ERROR}")
endif()

# Check if the output contains "libgomp"
string(FIND "${LDD_OUTPUT}" "libgomp" LIBGOMP_FOUND_INDEX)
if(LIBGOMP_FOUND_INDEX GREATER -1)
  message(STATUS "OpenBLAS is directly linked against libgomp")
  set(OPENBLAS_USES_LIBGOMP TRUE CACHE BOOL "OpenBLAS uses libgomp")
else()
  message(STATUS "OpenBLAS is not directly linked against libgomp")
  set(OPENBLAS_USES_LIBGOMP FALSE CACHE BOOL "OpenBLAS uses libgomp")
endif()

ENDIF(OpenBLAS_LIB)

MARK_AS_ADVANCED(
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIB
    OpenBLAS
)
