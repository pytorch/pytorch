IF(DEFINED ENV{OpenBLAS_HOME})
  FIND_PATH(
    OpenBLAS_INCLUDE_DIR
    NAMES cblas.h
    PATHS
      $ENV{OpenBLAS_HOME}
      $ENV{OpenBLAS_HOME}/include
      $ENV{OpenBLAS_HOME}/include/openblas
    NO_DEFAULT_PATH
  )
  FIND_LIBRARY(
    OpenBLAS_LIB
    NAMES openblas
    PATHS
      $ENV{OpenBLAS_HOME}
      $ENV{OpenBLAS_HOME}/lib
    NO_DEFAULT_PATH
  )
ELSEIF(DEFINED ENV{OpenBLAS} AND NOT OpenBLAS_LIB)
  FIND_LIBRARY(
    OpenBLAS_LIB
    NAMES openblas
    PATHS
      $ENV{OpenBLAS}
      $ENV{OpenBLAS}/lib
    NO_DEFAULT_PATH
  )
ENDIF()
 
IF(NOT OpenBLAS_INCLUDE_DIR AND NOT OpenBLAS_LIB)
  IF(DEFINED ENV{OpenBLAS_HOME} OR DEFINED ENV{OpenBLAS} AND NOT OpenBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Could not find OpenBLAS at user-specified location, searching system libraries")
  ENDIF()

  SET(Open_BLAS_INCLUDE_SEARCH_PATHS
    /usr/include
    /usr/include/openblas
    /usr/include/openblas-base
    /usr/local/include
    /usr/local/include/openblas
    /usr/local/include/openblas-base
    /usr/local/opt/openblas/include
    /opt/OpenBLAS/include
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
  )

  FIND_PATH(OpenBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Open_BLAS_INCLUDE_SEARCH_PATHS})
  FIND_LIBRARY(OpenBLAS_LIB NAMES openblas PATHS ${Open_BLAS_LIB_SEARCH_PATHS})
ENDIF()

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

MARK_AS_ADVANCED(
    OpenBLAS_INCLUDE_DIR
    OpenBLAS_LIB
    OpenBLAS
)
