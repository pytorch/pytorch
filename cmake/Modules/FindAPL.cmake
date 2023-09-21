# Find APL libraries and dependencies for Windows and MSVC

SET(APL_LIB_SEARCH_PATHS $ENV{ARMPL_DIR}/bin)

FIND_LIBRARY(APL_LIB NAMES armpl_ilp64 PATHS ${APL_LIB_SEARCH_PATHS})
FIND_LIBRARY(APL_FortranRuntime NAMES FortranRuntime PATHS ${APL_LIB_SEARCH_PATHS})
FIND_LIBRARY(APL_FortranDecimal NAMES FortranDecimal PATHS ${APL_LIB_SEARCH_PATHS})

SET(APL_FOUND ON)

#    Check dll
IF(NOT APL_LIB)
    SET(APL_LIB OFF)
    MESSAGE(STATUS "Could not find APL .dll. Turning APL_FOUND off")
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
    MESSAGE(STATUS "Found APL library: ${APL_LIB}")
    MESSAGE(STATUS "Found FortranRuntime dependency: ${APL_FortranRuntime}")
    MESSAGE(STATUS "Found FortranDecimal dependency: ${APL_FortranDecimal}")
    SET(APL_LIBRARIES ${APL_LIB} ${APL_FortranRuntime} ${APL_FortranDecimal})
ENDIF (APL_FOUND)

