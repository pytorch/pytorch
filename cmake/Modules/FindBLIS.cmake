# - Find BLIS library
#
# This module sets the following variables:
#  BLIS_FOUND - set to true if a library implementing CBLAS interface is found.
#  BLIS_INCLUDE_DIR - path to include dir.
#  BLIS_LIB - list of libraries for BLIS.
#
# CPU only Dockerfile to build with AMD BLIS is available at the location
# pytorch/docker/pytorch/cpu-blis/Dockerfile
##

IF (NOT BLIS_FOUND)

add_custom_target(libamdblis ALL
    DEPENDS
        ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libblis-mt.a
)

add_custom_command(
    OUTPUT
       ${CMAKE_CURRENT_SOURCE_DIR}/build/lib/libblis-mt.a
   WORKING_DIRECTORY
       ${CMAKE_CURRENT_SOURCE_DIR}/third_party/blis
   COMMAND
       make clean && make distclean && CC=gcc  ./configure --prefix=${CMAKE_CURRENT_SOURCE_DIR}/build/blis_gcc_build  --enable-threading=openmp --enable-cblas zen3 && make -j install
   COMMAND
       cd ${CMAKE_CURRENT_SOURCE_DIR}/build
   COMMAND
       cp blis_gcc_build/lib/libblis-mt.a ${CMAKE_CURRENT_SOURCE_DIR}/build/lib
   COMMAND
       cp -r blis_gcc_build/include/blis/* blis_gcc_build/include
)

SET(BLIS_INCLUDE_DIR
  ${CMAKE_CURRENT_SOURCE_DIR}/build/include
)

SET(BLIS_LIB_SEARCH_PATHS
  ${CMAKE_CURRENT_SOURCE_DIR}/build/lib
)

LIST(APPEND BLIS_LIBRARIES ${BLIS_LIB_SEARCH_PATHS}/libblis-mt.a)

INCLUDE(FindPackageHandleStandardArgs)
FIND_PACKAGE_HANDLE_STANDARD_ARGS(BLIS DEFAULT_MSG BLIS_INCLUDE_DIR BLIS_LIBRARIES)

MARK_AS_ADVANCED(
        BLIS_INCLUDE_DIR
        BLIS_LIBRARIES
        blis-mt
)


SET(BLIS_FOUND ON)
IF(BLIS_FOUND)
        IF(NOT BLIS_FIND_QUIETLY)
                MESSAGE(STATUS "Found BLIS libraries: ${BLIS_LIBRARIES}")
                MESSAGE(STATUS "Found BLIS include: ${BLIS_INCLUDE_DIR}")
        ENDIF()
ELSE(BLIS_FOUND)
        MESSAGE(FATAL_ERROR "Could not find BLIS")
ENDIF(BLIS_FOUND)
ENDIF (NOT BLIS_FOUND)
