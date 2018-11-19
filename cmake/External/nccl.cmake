if (NOT __NCCL_INCLUDED)
  set(__NCCL_INCLUDED TRUE)

  # try the system-wide nccl first
  find_package(NCCL)
  if (NCCL_FOUND)
      add_library(__caffe2_nccl INTERFACE)
      target_link_libraries(__caffe2_nccl INTERFACE ${NCCL_LIBRARIES})
      target_include_directories(__caffe2_nccl INTERFACE ${NCCL_INCLUDE_DIRS})
  else()
    if (TORCH_CUDA_ARCH_LIST)
      torch_cuda_get_nvcc_gencode_flag(NVCC_GENCODE)
      string(REPLACE "-gencode;" "-gencode=" NVCC_GENCODE "${NVCC_GENCODE}")
      # this second replacement is needed when there are multiple archs
      string(REPLACE ";-gencode" " -gencode" NVCC_GENCODE "${NVCC_GENCODE}")
    endif()

    ExternalProject_Add(nccl_external
      SOURCE_DIR ${PROJECT_SOURCE_DIR}/third_party/nccl/nccl
      BUILD_IN_SOURCE 1
      CONFIGURE_COMMAND ""
      BUILD_COMMAND
        env
        "CCACHE_DISABLE=1"
        make
        "CXX=${CMAKE_CXX_COMPILER}"
        "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}"
        "NVCC=${CUDA_NVCC_EXECUTABLE}"
        "NVCC_GENCODE=${NVCC_GENCODE}"
        "BUILDDIR=${CMAKE_CURRENT_BINARY_DIR}/nccl"
        "VERBOSE=0"
        "-j"
      BUILD_BYPRODUCTS "${CMAKE_CURRENT_BINARY_DIR}/nccl/lib/libnccl_static.a"
      INSTALL_COMMAND ""
      )

    set(NCCL_FOUND TRUE)
    add_library(__caffe2_nccl INTERFACE)
    # The following old-style variables are set so that other libs, such as Gloo,
    # can still use it.
    set(NCCL_INCLUDE_DIRS ${CMAKE_CURRENT_BINARY_DIR}/nccl/include)
    set(NCCL_LIBRARIES ${CMAKE_CURRENT_BINARY_DIR}/nccl/lib/libnccl_static.a)
    add_dependencies(__caffe2_nccl nccl_external)
    target_link_libraries(__caffe2_nccl INTERFACE ${NCCL_LIBRARIES})
    target_include_directories(__caffe2_nccl INTERFACE ${NCCL_INCLUDE_DIRS})
  endif()

endif()
