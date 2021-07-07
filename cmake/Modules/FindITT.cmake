# - Try to find torch-ccl
#
# The following are set after configuration is done:
#  ITT_FOUND          : set to true if ITT is found.
#  ITT_INCLUDE_DIR    : path to ITT include dir.
#  ITT_LIBRARIES      : list of libraries for ITT

IF (NOT ITT_FOUND)
SET(ITT_FOUND OFF)

SET(ITT_INCLUDE_DIR)
SET(ITT_LIBRARIES)

IF(NOT CMAKE_BUILD_TYPE)
  SET(CMAKE_BUILD_TYPE "Release" CACHE STRING
      "Choose the type of build, options are: Debug Release"
      FORCE)
ENDIF(NOT CMAKE_BUILD_TYPE)

SET(ITT_ROOT "${PROJECT_SOURCE_DIR}/third_party/ittapi")
FIND_PATH(ITT_INCLUDE_DIR ittnotify.h PATHS ${ITT_ROOT} PATH_SUFFIXES include)
IF (ITT_INCLUDE_DIR)
  ADD_SUBDIRECTORY(${ITT_ROOT})
  SET(ITT_LIBRARIES ittnotify)
  SET(ITT_FOUND ON)
ENDIF (ITT_INCLUDE_DIR)

ENDIF(NOT ITT_FOUND)
