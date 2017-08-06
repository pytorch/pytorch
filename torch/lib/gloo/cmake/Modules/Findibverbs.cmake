# Find the ibverbs libraries
#
# The following variables are optionally searched for defaults
#  IBVERBS_ROOT_DIR: Base directory where all Ibverbs components are found
#
# The following are set after configuration is done:
#  IBVERBS_FOUND
#  ibverbs_INCLUDE_DIR
#  ibverbs_LIBRARIES

find_path(ibverbs_INCLUDE_DIR
  NAMES infiniband/verbs.h
  HINTS ${IBVERBS_ROOT_DIR} ${IBVERBS_ROOT_DIR}/include)

find_library(ibverbs_LIBRARIES
  NAMES ibverbs
  HINTS ${IBVERBS_ROOT_DIR} ${IBVERBS_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ibverbs DEFAULT_MSG ibverbs_INCLUDE_DIR ibverbs_LIBRARIES)

if(IBVERBS_FOUND)
  set(include_message "include: ${ibverbs_INCLUDE_DIR}")
  set(library_message "library: ${ibverbs_LIBRARIES}")
  message(STATUS "Found ibverbs (${include_message}, ${library_message})")
  mark_as_advanced(ibverbs_INCLUDE_DIR ibverbs_LIBRARIES)
endif()
