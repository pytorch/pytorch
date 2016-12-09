# - Try to find NCCL
#
# The following variables are optionally searched for defaults
#  NCCL_ROOT_DIR:            Base directory where all NCCL components are found
#
# The following are set after configuration is done:
#  NCCL_FOUND
#  NCCL_INCLUDE_DIRS
#  NCCL_LIBRARIES
#  NCCL_LIBRARYRARY_DIRS

include(FindPackageHandleStandardArgs)

# TODO(slayton): Do this properly
# set(NCCL_ROOT_DIR "/home/slayton/git/nccl/build" CACHE PATH "Folder contains NVIDIA NCCL")
set(NCCL_ROOT_DIR "" CACHE PATH "Folder contains NVIDIA NCCL")

find_path(NCCL_INCLUDE_DIR nccl.h
    PATHS ${NCCL_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(NCCL_LIBRARY nccl
    PATHS ${NCCL_ROOT_DIR}
    PATH_SUFFIXES lib lib64)

find_package_handle_standard_args(NCCL DEFAULT_MSG NCCL_INCLUDE_DIR NCCL_LIBRARY)

if(NCCL_FOUND)
  set(NCCL_INCLUDE_DIRS ${NCCL_INCLUDE_DIR})
  set(NCCL_LIBRARIES ${NCCL_LIBRARY})
  message(STATUS "Found NCCL    (include: ${NCCL_INCLUDE_DIR}, library: ${NCCL_LIBRARY})")
  mark_as_advanced(NCCL_ROOT_DIR NCCL_LIBRARY_RELEASE NCCL_LIBRARY_DEBUG
                                 NCCL_LIBRARY NCCL_INCLUDE_DIR)
endif()
