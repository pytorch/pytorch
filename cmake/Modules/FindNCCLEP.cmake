# Find NCCL EP.
#
# Optional search hints:
#  NCCL_EP_ROOT: Base directory containing NCCL EP, or an nccl-extensions tree
#  NCCL_EP_INCLUDE_DIR: Directory containing nccl_ep.h
#  NCCL_EP_LIB_DIR: Directory containing libnccl_ep
#
# Result variables:
#  NCCL_EP_FOUND
#  NCCL_EP_INCLUDE_DIRS
#  NCCL_EP_JIT_INCLUDE_DIR
#  NCCL_EP_LIBRARIES

set(NCCL_EP_INCLUDE_DIR $ENV{NCCL_EP_INCLUDE_DIR} CACHE PATH "Folder containing NCCL EP headers")
set(NCCL_EP_LIB_DIR $ENV{NCCL_EP_LIB_DIR} CACHE PATH "Folder containing NCCL EP libraries")
set(NCCL_EP_ROOT $ENV{NCCL_EP_ROOT} CACHE PATH "NCCL EP install prefix or nccl-extensions tree")

find_path(NCCL_EP_INCLUDE_DIRS
  NAMES nccl_ep.h
  HINTS ${NCCL_EP_INCLUDE_DIR} ${NCCL_EP_ROOT}
  PATH_SUFFIXES include build/include)

find_path(NCCL_EP_JIT_INCLUDE_DIR
  NAMES nccl_ep/device/ht_ep.cuh
  HINTS ${NCCL_EP_INCLUDE_DIRS}
  NO_DEFAULT_PATH)

find_library(NCCL_EP_LIBRARIES
  NAMES nccl_ep
  HINTS ${NCCL_EP_LIB_DIR} ${NCCL_EP_ROOT}
  PATH_SUFFIXES lib build/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  NCCLEP DEFAULT_MSG NCCL_EP_INCLUDE_DIRS NCCL_EP_JIT_INCLUDE_DIR NCCL_EP_LIBRARIES)

if(NCCLEP_FOUND)
  set(NCCL_EP_FOUND TRUE)
  message(STATUS "Found NCCL EP (include: ${NCCL_EP_INCLUDE_DIRS}, library: ${NCCL_EP_LIBRARIES})")
endif()

mark_as_advanced(NCCL_EP_INCLUDE_DIR NCCL_EP_LIB_DIR NCCL_EP_INCLUDE_DIRS NCCL_EP_JIT_INCLUDE_DIR NCCL_EP_LIBRARIES)
