# - Try to find MIOpen
#
# The following variables are optionally searched for defaults
#  MIOPEN_ROOT_DIR:            Base directory where all MIOpen components are found
#
# The following are set after configuration is done:
#  MIOPEN_FOUND
#  MIOPEN_INCLUDE_DIRS
#  MIOPEN_LIBRARIES
#  MIOPEN_LIBRARY_DIRS
#
# Borrowed from https://github.com/caffe2/caffe2/blob/master/cmake/Modules/FindCuDNN.cmake

include(FindPackageHandleStandardArgs)

set(MIOPEN_ROOT_DIR "" CACHE PATH "Folder contains MIOpen")

if($ENV{MIOPEN_INCLUDE_DIR})
  SET(MIOPEN_INCLUDE_DIR $ENV{MIOPEN_INCLUDE_DIR})
else($ENV{MIOPEN_INCLUDE_DIR})
  find_path(MIOPEN_INCLUDE_DIR miopen.h
    HINTS ${MIOPEN_ROOT_DIR}
    PATH_SUFFIXES include include/miopen)
endif($ENV{MIOPEN_INCLUDE_DIR})

if($ENV{MIOPEN_LIBRARY})
  SET(MIOPEN_LIBRARY $ENV{MIOPEN_LIBRARY})
else($ENV{MIOPEN_LIBRARY})
  find_library(MIOPEN_LIBRARY MIOpen
    HINTS ${MIOPEN_LIB_DIR} ${MIOPEN_ROOT_DIR}
    PATH_SUFFIXES lib)
endif($ENV{MIOPEN_LIBRARY})

find_package_handle_standard_args(
    MIOPEN DEFAULT_MSG MIOPEN_INCLUDE_DIR MIOPEN_LIBRARY)

if(MIOPEN_FOUND)
	# get MIOpen version
  file(READ ${MIOPEN_INCLUDE_DIR}/version.h MIOPEN_HEADER_CONTENTS)
	string(REGEX MATCH "define MIOPEN_VERSION_MAJOR * +([0-9]+)"
				 MIOPEN_VERSION_MAJOR "${MIOPEN_HEADER_CONTENTS}")
	string(REGEX REPLACE "define MIOPEN_VERSION_MAJOR * +([0-9]+)" "\\1"
				 MIOPEN_VERSION_MAJOR "${MIOPEN_VERSION_MAJOR}")
	string(REGEX MATCH "define MIOPEN_VERSION_MINOR * +([0-9]+)"
				 MIOPEN_VERSION_MINOR "${MIOPEN_HEADER_CONTENTS}")
	string(REGEX REPLACE "define MIOPEN_VERSION_MINOR * +([0-9]+)" "\\1"
				 MIOPEN_VERSION_MINOR "${MIOPEN_VERSION_MINOR}")
	string(REGEX MATCH "define MIOPEN_VERSION_PATCH * +([0-9]+)"
				 MIOPEN_VERSION_PATCH "${MIOPEN_HEADER_CONTENTS}")
	string(REGEX REPLACE "define MIOPEN_VERSION_PATCH * +([0-9]+)" "\\1"
				 MIOPEN_VERSION_PATCH "${MIOPEN_VERSION_PATCH}")
  # Assemble MIOpen version
  if(NOT MIOPEN_VERSION_MAJOR)
    set(MIOPEN_VERSION "?")
  else()
    set(MIOPEN_VERSION "${MIOPEN_VERSION_MAJOR}.${MIOPEN_VERSION_MINOR}.${MIOPEN_VERSION_PATCH}")
  endif()

  set(MIOPEN_INCLUDE_DIRS ${MIOPEN_INCLUDE_DIR})
  set(MIOPEN_LIBRARIES ${MIOPEN_LIBRARY})
  message(STATUS "Found MIOpen: v${MIOPEN_VERSION}  (include: ${MIOPEN_INCLUDE_DIR}, library: ${MIOPEN_LIBRARY})")
  mark_as_advanced(MIOPEN_ROOT_DIR MIOPEN_LIBRARY MIOPEN_INCLUDE_DIR)
endif()
