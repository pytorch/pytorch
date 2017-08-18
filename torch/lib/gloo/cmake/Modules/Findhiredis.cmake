# Find the hiredis libraries
#
# The following variables are optionally searched for defaults
#  HIREDIS_ROOT_DIR: Base directory where all hiredis components are found
#
# The following are set after configuration is done:
#  HIREDIS_FOUND
#  hiredis_INCLUDE_DIR
#  hiredis_LIBRARIES

find_path(hiredis_INCLUDE_DIR
  NAMES hiredis.h
  HINTS ${HIREDIS_ROOT_DIR} ${HIREDIS_ROOT_DIR}/include
  PATH_SUFFIXES hiredis)

find_library(hiredis_LIBRARIES
  NAMES hiredis
  HINTS ${HIREDIS_ROOT_DIR} ${HIREDIS_ROOT_DIR}/lib
  PATH_SUFFIXES hiredis)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(hiredis DEFAULT_MSG hiredis_INCLUDE_DIR hiredis_LIBRARIES)

if(HIREDIS_FOUND)
  set(include_message "include: ${hiredis_INCLUDE_DIR}")
  set(library_message "library: ${hiredis_LIBRARIES}")
  message(STATUS "Found hiredis (${include_message}, ${library_message})")
  mark_as_advanced(hiredis_INCLUDE_DIR hiredis_LIBRARIES)
endif()
