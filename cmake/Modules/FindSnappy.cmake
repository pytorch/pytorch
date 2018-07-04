# Find the Snappy libraries
#
# The following variables are optionally searched for defaults
#  SNAPPY_ROOT_DIR:    Base directory where all Snappy components are found
#
# The following are set after configuration is done:
#  SNAPPY_FOUND
#  Snappy_INCLUDE_DIR
#  Snappy_LIBRARIES

find_path(Snappy_INCLUDE_DIR NAMES snappy.h
                             PATHS ${SNAPPY_ROOT_DIR} ${SNAPPY_ROOT_DIR}/include)

find_library(Snappy_LIBRARIES NAMES snappy
                              PATHS ${SNAPPY_ROOT_DIR} ${SNAPPY_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(Snappy DEFAULT_MSG Snappy_INCLUDE_DIR Snappy_LIBRARIES)

if(SNAPPY_FOUND)
  message(STATUS "Found Snappy  (include: ${Snappy_INCLUDE_DIR}, library: ${Snappy_LIBRARIES})")
  mark_as_advanced(Snappy_INCLUDE_DIR Snappy_LIBRARIES)
  caffe_parse_header(${Snappy_INCLUDE_DIR}/snappy-stubs-public.h
                     SNAPPY_VERION_LINES SNAPPY_MAJOR SNAPPY_MINOR SNAPPY_PATCHLEVEL)
  set(Snappy_VERSION "${SNAPPY_MAJOR}.${SNAPPY_MINOR}.${SNAPPY_PATCHLEVEL}")
endif()

