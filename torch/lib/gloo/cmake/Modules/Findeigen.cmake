# Find the eigen library
#
# The following variables are optionally searched for defaults
#  EIGEN_ROOT_DIR: Base directory where all eigen components are found
#
# The following are set after configuration is done:
#  EIGEN_FOUND
#  eigen_INCLUDE_DIR

find_path(eigen_INCLUDE_DIR
  NAMES Eigen/Core
  HINTS ${EIGEN_ROOT_DIR} ${EIGEN_ROOT_DIR}/include
  PATH_SUFFIXES eigen3)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(eigen DEFAULT_MSG eigen_INCLUDE_DIR)

if(EIGEN_FOUND)
  message(STATUS "Found eigen (include: ${eigen_INCLUDE_DIR})")
  mark_as_advanced(eigen_INCLUDE_DIR eigen_LIBRARIES)
endif()
