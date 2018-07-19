# - Try to find MKLDNN
#
# The following variables are optionally searched for defaults
#  MKLDNN_ROOT_DIR:            Base directory where all MKLDNN components are found
#
# The following are set after configuration is done:
#  MKLDNN_FOUND
#  MKLDNN_INCLUDE_DIRS
#  MKLDNN_LIBRARIES
#  MKLDNN_LIBRARY_DIRS

include(FindPackageHandleStandardArgs)

set(MKLDNN_ROOT_DIR "" CACHE PATH "Folder contains Intel MKLDNN")

find_path(MKLDNN_INCLUDE_DIR mkldnn.h
    HINTS ${MKLDNN_ROOT_DIR}
    PATH_SUFFIXES include)

find_library(MKLDNN_LIBRARY mkldnn
    HINTS ${MKLDNN_LIB_DIR} ${MKLDNN_ROOT_DIR}
    PATH_SUFFIXES lib lib64)

find_package_handle_standard_args(
    MKLDNN DEFAULT_MSG MKLDNN_INCLUDE_DIR MKLDNN_LIBRARY)

if(MKLDNN_FOUND)
  set(MKLDNN_INCLUDE_DIRS ${MKLDNN_INCLUDE_DIR})
  set(MKLDNN_LIBRARIES ${MKLDNN_LIBRARY})
  message(STATUS "Found MKLDNN      (include: ${MKLDNN_INCLUDE_DIR}, library: ${MKLDNN_LIBRARY})")
  mark_as_advanced(MKLDNN_ROOT_DIR MKLDNN_LIBRARY MKLDNN_INCLUDE_DIR)
endif()
