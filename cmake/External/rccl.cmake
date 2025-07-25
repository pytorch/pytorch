if(NOT __NCCL_INCLUDED)
  set(__NCCL_INCLUDED TRUE)

  if(USE_SYSTEM_NCCL)
    # NCCL_ROOT, NCCL_LIB_DIR, NCCL_INCLUDE_DIR will be accounted in the
    # following line.
    find_package(rccl REQUIRED)
    if(rccl_FOUND)
      message(STATUS "RCCL Found!")
      add_library(__caffe2_nccl INTERFACE)
      target_link_libraries(__caffe2_nccl INTERFACE roc::rccl)
    else()
      message(STATUS "RCCL NOT Found!")
    endif()
  else()
    message(STATUS "USE_SYSTEM_NCCL=OFF is not supported yet when using RCCL")
  endif()
endif()
