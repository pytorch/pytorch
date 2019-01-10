if (NOT __NCCL_INCLUDED)
  set(__NCCL_INCLUDED TRUE)

  # try the system-wide nccl first
  find_package(NCCL)
  if (NCCL_FOUND)
      add_library(__caffe2_nccl INTERFACE)
      target_link_libraries(__caffe2_nccl INTERFACE ${NCCL_LIBRARIES})
      target_include_directories(__caffe2_nccl INTERFACE ${NCCL_INCLUDE_DIRS})
  else()
    # build directory
    set(nccl_PREFIX ${PROJECT_SOURCE_DIR}/third_party/nccl)

    # we build nccl statically, but want to link it into the caffe shared library
    # this requires position-independent code
    if (UNIX)
      set(NCCL_EXTRA_COMPILER_FLAGS "-fPIC")
    endif()

    set(NCCL_CXX_FLAGS ${CMAKE_CXX_FLAGS} ${NCCL_EXTRA_COMPILER_FLAGS})
    set(NCCL_C_FLAGS ${CMAKE_C_FLAGS} ${NCCL_EXTRA_COMPILER_FLAGS})

    ExternalProject_Add(nccl_external
      SOURCE_DIR ${nccl_PREFIX}
      BUILD_IN_SOURCE 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND
        make
        "CXX=${CMAKE_CXX_COMPILER}"
        "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}"
        "NVCC=${CUDA_NVCC_EXECUTABLE}"
      BUILD_BYPRODUCTS "${nccl_PREFIX}/build/lib/libnccl_static.a"
      INSTALL_COMMAND ""
      )

    set(NCCL_FOUND TRUE)
    add_library(__caffe2_nccl INTERFACE)
    # The following old-style variables are set so that other libs, such as Gloo,
    # can still use it.
    set(NCCL_INCLUDE_DIRS ${nccl_PREFIX}/build/include)
    set(NCCL_LIBRARIES ${nccl_PREFIX}/build/lib/libnccl_static.a)
    add_dependencies(__caffe2_nccl nccl_external)
    target_link_libraries(__caffe2_nccl INTERFACE ${NCCL_LIBRARIES})
    target_include_directories(__caffe2_nccl INTERFACE ${NCCL_INCLUDE_DIRS})
  endif()

endif()
