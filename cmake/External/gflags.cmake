if (NOT __GFLAGS_INCLUDED) # guard against multiple includes
  set(__GFLAGS_INCLUDED TRUE)

  # try the system-wide glog first
  find_package(GFlags)
  if (GFLAGS_FOUND)
    # Great, we will use gflags.
    message(STATUS "Found system gflags install.")
  else()
    message(WARNING "gflags is not found. Caffe2 will build without gflags support but it is strongly recommended that you install gflags.")
  endif()

endif()
