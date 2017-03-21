# Find the nccl libraries
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR: Base directory where all nccl components are found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  nccl_INCLUDE_DIR
#  nccl_LIBRARIES

find_path(nccl_INCLUDE_DIR
  NAMES nccl.h
  HINTS ${NCCL_ROOT_DIR} ${NCCL_ROOT_DIR}/include)

find_library(nccl_LIBRARIES
  NAMES nccl
  HINTS ${NCCL_ROOT_DIR} ${NCCL_ROOT_DIR}/lib ${NCCL_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(nccl DEFAULT_MSG nccl_INCLUDE_DIR nccl_LIBRARIES)

if(NCCL_FOUND)
  set(include_message "include: ${nccl_INCLUDE_DIR}")
  set(library_message "library: ${nccl_LIBRARIES}")
  message(STATUS "Found nccl (${include_message}, ${library_message})")
  mark_as_advanced(nccl_INCLUDE_DIR nccl_LIBRARIES)
endif()
