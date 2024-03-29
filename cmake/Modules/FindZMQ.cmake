# Find the ZMQ libraries
#
# The following variables are optionally searched for defaults
#  ZMQ_ROOT_DIR:    Base directory where all ZMQ components are found
#
# The following are set after configuration is done:
#  ZMQ_FOUND
#  ZMQ_INCLUDE_DIR
#  ZMQ_LIBRARIES
#  ZMQ_VERSION_MAJOR

find_path(ZMQ_INCLUDE_DIR NAMES zmq.h
                             PATHS ${ZMQ_ROOT_DIR} ${ZMQ_ROOT_DIR}/include)

find_library(ZMQ_LIBRARIES NAMES zmq
                              PATHS ${ZMQ_ROOT_DIR} ${ZMQ_ROOT_DIR}/lib)

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(ZMQ DEFAULT_MSG ZMQ_INCLUDE_DIR ZMQ_LIBRARIES)

if(ZMQ_FOUND)
  message(STATUS "Found ZMQ  (include: ${ZMQ_INCLUDE_DIR}, library: ${ZMQ_LIBRARIES})")
  mark_as_advanced(ZMQ_INCLUDE_DIR ZMQ_LIBRARIES)

  caffe_parse_header(${ZMQ_INCLUDE_DIR}/zmq.h ZMQ_VERSION_LINES ZMQ_VERSION_MAJOR)
  if(${ZMQ_VERSION_MAJOR} VERSION_LESS "3")
    message(WARNING "Caffe2 requires zmq version 3 or above, but found " ${ZMQ_VERSION_MAJOR} ". Disabling zmq for now.")
    set(ZMQ_FOUND)
  else()

  endif()
endif()
