# glog depends on gflags
include("cmake/External/gflags.cmake")

if (NOT __GLOG_INCLUDED)
  set(__GLOG_INCLUDED TRUE)

  # try the system-wide glog first
  find_package(Glog)
  if (GLOG_FOUND)
    # Great, we will use glog.
    message(STATUS "Found system glog install.")
  else()
    message(WARNING "glog is not found. Caffe2 will build without glog support but it is strongly recommended that you install glog.")
  endif()

endif()

