if (NOT __NCCL_INCLUDED)
  set(__NCCL_INCLUDED TRUE)

  # try the system-wide nccl first
  find_package(NCCL)
  if (NCCL_FOUND)
      set(NCCL_EXTERNAL FALSE)
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
          make "CXX=${CMAKE_CXX_COMPILER}" "CUDA_HOME=${CUDA_TOOLKIT_ROOT_DIR}"
      INSTALL_COMMAND ""
      )

    set(NCCL_FOUND TRUE)
    set(NCCL_INCLUDE_DIRS ${nccl_PREFIX}/build/include)
    set(NCCL_LIBRARIES ${nccl_PREFIX}/build/lib/libnccl_static.a)
    set(NCCL_LIBRARY_DIRS ${nccl_PREFIX}/build/lib)
    set(NCCL_EXTERNAL TRUE)

    list(APPEND Caffe2_EXTERNAL_DEPENDENCIES nccl_external)
  endif()

endif()
