# - Try to find NCCL
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR:            Base directory where all NCCL components are found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES

set(NCCL_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA NCCL")

find_path(NCCL_INCLUDE_DIR
  NAMES nccl.h
  HINTS
  ${NCCL_ROOT_DIR}
  ${NCCL_ROOT_DIR}/include)

find_library(NCCL_LIBRARY
  NAMES nccl
  HINTS
  ${NCCL_ROOT_DIR}
  ${NCCL_ROOT_DIR}/lib
  ${NCCL_ROOT_DIR}/lib/x86_64-linux-gnu
  ${NCCL_ROOT_DIR}/lib64)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

if(NCCL_FOUND)
  set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
  set(NCCL_LIBRARIES ${NCCL_LIBRARY})
  message(STATUS "Found NCCL (include: ${NCCL_INCLUDE_DIRS}, library: ${NCCL_LIBRARIES})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_INCLUDE_DIRS NCCL_LIBRARIES)
endif()
