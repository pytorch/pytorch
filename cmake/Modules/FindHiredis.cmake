# Find the Hiredis libraries
#
# The following variables are optionally searched for defaults
#  HIREDIS_ROOT_DIR:    Base directory where all Hiredis components are found
#
# The following are set after configuration is done:
#  HIREDIS_FOUND
#  Hiredis_INCLUDE_DIR
#  Hiredis_LIBRARIES

find_path(Hiredis_INCLUDE_DIR NAMES hiredis/hiredis.h
                             PATHS ${HIREDIS_ROOT_DIR} ${HIREDIS_ROOT_DIR}/include)

find_library(Hiredis_LIBRARIES NAMES hiredis
                              PATHS ${HIREDIS_ROOT_DIR} ${HIREDIS_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Hiredis DEFAULT_MSG Hiredis_INCLUDE_DIR Hiredis_LIBRARIES)

if(HIREDIS_FOUND)
  message(STATUS "Found Hiredis  (include: ${Hiredis_INCLUDE_DIR}, library: ${Hiredis_LIBRARIES})")
  mark_as_advanced(Hiredis_INCLUDE_DIR Hiredis_LIBRARIES)
endif()
