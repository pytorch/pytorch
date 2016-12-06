if (NOT __NCCL_INCLUDED)
  set(__NCCL_INCLUDED TRUE)

  # try the system-wide nccl first
  find_package(NCCL)
  if (NCCL_FOUND)
      set(NCCL_EXTERNAL FALSE)
  else()
    # fetch and build glog from github

    # build directory
    set(nccl_PREFIX ${CMAKE_SOURCE_DIR}/third_party/nccl)
    # install directory
    set(nccl_INSTALL ${CMAKE_BINARY_DIR}/external/nccl-install)

    # we build glog statically, but want to link it into the caffe shared library
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
      BUILD_COMMAND make
      INSTALL_COMMAND ""
      )

    set(NCCL_FOUND TRUE)
    set(NCCL_INCLUDE_DIRS ${nccl_INSTALL}/include)
    set(NCCL_LIBRARIES ${nccl_INSTALL}/lib/libnccl.a)
    set(NCCL_LIBRARY_DIRS ${nccl_INSTALL}/lib)
    set(NCCL_EXTERNAL TRUE)

    list(APPEND external_project_dependencies nccl_external)
  endif()

endif()

