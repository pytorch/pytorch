# Find the nccl-extensions NCCL EP library (libnccl_ep)
#
# The following variables are optionally searched for defaults
#  NCCL_EP_ROOT: Base directory where all NCCL EP components are found
#  NCCL_EP_INCLUDE_DIR: Directory where the NCCL EP header is found
#  NCCL_EP_LIB_DIR: Directory where the NCCL EP library is found
#
# The following are set after configuration is done:
#  NCCLEP_FOUND
#  NCCL_EP_INCLUDE_DIRS
#  NCCL_EP_LIBRARIES
#  NCCL_EP_JIT_INCLUDE_DIR: directory holding the nccl_ep/ runtime-JIT header
#   tree (nccl_ep/device/*.cuh, nccl_ep/common.hpp, nccl_ep/ep_enums.h). Its
#   parent is baked into torch._nccl_ep as NCCL_EP_HOME; the JIT resolves
#   kernel sources from $NCCL_EP_HOME/include/nccl_ep.

set(NCCL_EP_INCLUDE_DIR $ENV{NCCL_EP_INCLUDE_DIR} CACHE PATH "Folder contains NVIDIA NCCL EP headers")
set(NCCL_EP_LIB_DIR $ENV{NCCL_EP_LIB_DIR} CACHE PATH "Folder contains NVIDIA NCCL EP libraries")

list(APPEND NCCL_EP_ROOT $ENV{NCCL_EP_ROOT})
# Compatible layer for CMake <3.12. NCCL_EP_ROOT will be accounted in for searching paths and libraries for CMake >=3.12.
list(APPEND CMAKE_PREFIX_PATH ${NCCL_EP_ROOT})

find_path(NCCL_EP_INCLUDE_DIRS
  NAMES nccl_ep.h
  HINTS ${NCCL_EP_INCLUDE_DIR})

find_library(NCCL_EP_LIBRARIES
  NAMES nccl_ep
  HINTS ${NCCL_EP_LIB_DIR})

# The runtime JIT recompiles device kernels from the header tree installed
# next to nccl_ep.h, so require it for a usable install.
find_path(NCCL_EP_JIT_INCLUDE_DIR
  NAMES nccl_ep/device/ht_ep.cuh
  HINTS ${NCCL_EP_INCLUDE_DIR} ${NCCL_EP_INCLUDE_DIRS})

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(NCCLEP DEFAULT_MSG NCCL_EP_INCLUDE_DIRS NCCL_EP_LIBRARIES NCCL_EP_JIT_INCLUDE_DIR)

if(NCCLEP_FOUND)
  message(STATUS "Found NCCL EP (include: ${NCCL_EP_INCLUDE_DIRS}, JIT include: ${NCCL_EP_JIT_INCLUDE_DIR}, library: ${NCCL_EP_LIBRARIES})")
  mark_as_advanced(NCCL_EP_ROOT NCCL_EP_INCLUDE_DIRS NCCL_EP_LIBRARIES NCCL_EP_JIT_INCLUDE_DIR)
endif()
