

SET(Flexi_BLAS_INCLUDE_SEARCH_PATHS
  /usr/include
  /usr/include/flexiblas
  /usr/include/flexiblas-base
  /usr/local/include
  /usr/local/include/flexiblas
  /usr/local/include/flexiblas-base
  /usr/local/opt/flexiblas/include
  /opt/Flexiblas/include
  $ENV{FlexiBLAS_HOME}
  $ENV{FlexiBLAS_HOME}/include
)

SET(Flexi_BLAS_LIB_SEARCH_PATHS
        /lib/
        /lib/flexiblas-base
        /lib64/
        /usr/lib
        /usr/lib/flexiblas-base
        /usr/lib64
        /usr/local/lib
        /usr/local/lib64
        /usr/local/opt/flexiblas/lib
        /opt/FlexiBLAS/lib
        $ENV{FlexiBLAS}
        $ENV{FlexiBLAS}/lib
        $ENV{FlexiBLAS_HOME}
        $ENV{FlexiBLAS_HOME}/lib
 )

FIND_PATH(FlexiBLAS_INCLUDE_DIR NAMES cblas.h PATHS ${Flexi_BLAS_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(FlexiBLAS_LIB NAMES flexiblas PATHS ${Flexi_BLAS_LIB_SEARCH_PATHS})

SET(FlexiBLAS_FOUND ON)

#    Check include files
IF(NOT FlexiBLAS_INCLUDE_DIR)
    SET(FlexiBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find FlexiBLAS include. Turning FlexiBLAS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT FlexiBLAS_LIB)
    SET(FlexiBLAS_FOUND OFF)
    MESSAGE(STATUS "Could not find FlexiBLAS lib. Turning FlexiBLAS_FOUND off")
ENDIF()

IF (FlexiBLAS_FOUND)
  IF (NOT FlexiBLAS_FIND_QUIETLY)
    MESSAGE(STATUS "Found FlexiBLAS libraries: ${FlexiBLAS_LIB}")
    MESSAGE(STATUS "Found FlexiBLAS include: ${FlexiBLAS_INCLUDE_DIR}")
  ENDIF (NOT FlexiBLAS_FIND_QUIETLY)
ELSE (FlexiBLAS_FOUND)
  IF (FlexiBLAS_FIND_REQUIRED)
    MESSAGE(FATAL_ERROR "Could not find FlexiBLAS")
  ENDIF (FlexiBLAS_FIND_REQUIRED)
ENDIF (FlexiBLAS_FOUND)

MARK_AS_ADVANCED(
    FlexiBLAS_INCLUDE_DIR
    FlexiBLAS_LIB
    FlexiBLAS
)
