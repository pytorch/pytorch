# Find the Numa libraries
#
# The following variables are optionally searched for defaults
#  NUMA_ROOT_DIR:    Base directory where all Numa components are found
#
# The following are set after configuration is done:
#  NUMA_FOUND
#  Numa_INCLUDE_DIR
#  Numa_LIBRARIES

find_path(
    Numa_INCLUDE_DIR NAMES numa.h
    PATHS ${NUMA_ROOT_DIR} ${NUMA_ROOT_DIR}/include)

find_library(
    Numa_LIBRARIES NAMES numa
    PATHS ${NUMA_ROOT_DIR} ${NUMA_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
    Numa DEFAULT_MSG Numa_INCLUDE_DIR Numa_LIBRARIES)

if(NUMA_FOUND)
  message(
      STATUS
      "Found Numa  (include: ${Numa_INCLUDE_DIR}, library: ${Numa_LIBRARIES})")
  mark_as_advanced(Numa_INCLUDE_DIR Numa_LIBRARIES)
endif()

