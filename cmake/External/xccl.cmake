if(NOT __XCCL_INCLUDED)
  set(__XCCL_INCLUDED TRUE)

  if(USE_XCCL)
    # XCCL_ROOT, XCCL_LIBRARY_DIR, XCCL_INCLUDE_DIR are handled by FindXCCL.cmake.
    find_package(XCCL REQUIRED)
    if(XCCL_FOUND)
      add_library(__caffe2_xccl INTERFACE)
      target_link_libraries(__caffe2_xccl INTERFACE ${XCCL_LIBRARY})
      target_include_directories(__caffe2_xccl INTERFACE ${XCCL_INCLUDE_DIR})
    endif()
  endif()
endif()
