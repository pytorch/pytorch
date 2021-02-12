SET(BLIS_INCLUDE_SEARCH_PATHS
 /usr/include
 /usr/include/blis
 /usr/local/include
 /usr/local/include/blis
 /usr/local/opt/blis/include
 /opt/blis/include
 $ENV{BLIS_HOME}
 $ENV{BLIS_HOME}/include
)

SET(BLIS_LIB_SEARCH_PATHS
 /lib
 /lib/blis
 /lib64
 /lib64/blis
 /usr/lib
 /usr/lib/blis
 /usr/lib64
 /usr/lib64/blis
 /usr/local/lib
 /usr/local/lib64
 /usr/local/opt/blis/lib
 /opt/blis/lib
 $ENV{BLIS}
 $ENV{BLIS}/lib
 $ENV{BLIS_HOME}
 $ENV{BLIS_HOME}/lib
)

FIND_PATH(BLIS_INCLUDE_DIR NAMES cblas.h blis.h PATHS ${BLIS_INCLUDE_SEARCH_PATHS})
FIND_LIBRARY(BLIS_LIB NAMES blis PATHS ${BLIS_LIB_SEARCH_PATHS})

SET(BLIS_FOUND ON)

#    Check include files
IF(NOT BLIS_INCLUDE_DIR)
        SET(BLIS_FOUND OFF)
        MESSAGE(STATUS "Could not find BLIS include. Turning BLIS_FOUND off")
ENDIF()

#    Check libraries
IF(NOT BLIS_LIB)
        SET(BLIS_FOUND OFF)
        MESSAGE(STATUS "Could not find BLIS lib. Turning BLIS_FOUND off")
ENDIF()

IF (BLIS_FOUND)
        IF (NOT BLIS_FIND_QUIETLY)
                MESSAGE(STATUS "Found BLIS libraries: ${BLIS_LIB}")
                MESSAGE(STATUS "Found BLIS include: ${BLIS_INCLUDE_DIR}")
        ENDIF (NOT BLIS_FIND_QUIETLY)
ELSE (BLIS_FOUND)
        MESSAGE(FATAL_ERROR "Could not find BLIS")
ENDIF (BLIS_FOUND)

MARK_AS_ADVANCED(
        BLIS_INCLUDE_DIR
        BLIS_LIB
        blis
)

