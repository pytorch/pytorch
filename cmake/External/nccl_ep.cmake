if(NOT __NCCL_EP_INCLUDED)
  set(__NCCL_EP_INCLUDED TRUE)

  # NCCL is built (Makefile) into this tree by cmake/External/nccl.cmake.
  set(__NCCL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl")

  # Build contrib/nccl_ep as its own CMake project against that prebuilt NCCL
  # (headers + libnccl in __NCCL_BUILD_DIR). NCCL_EP_BUILDDIR == NCCL_HOME so
  # nccl_ep's artifacts (libnccl_ep.a, headers) land in the same tree, leaving
  # the core NCCL build untouched.
  # Build nccl_ep for the same CUDA archs as the rest of PyTorch
  # (TORCH_CUDA_ARCH_LIST), dropping anything below sm_90 since nccl_ep requires
  # Hopper+. PyTorch disables CMAKE_CUDA_ARCHITECTURES in favor of
  # TORCH_CUDA_ARCH_LIST, so convert "9.0;10.0;..." to "90;100;...". If nothing
  # qualifies (list unset, or a symbolic value like "Common"/"All"), pass no
  # flag and let nccl_ep's CMakeLists pick its own CUDA-version default.
  if((NOT DEFINED TORCH_CUDA_ARCH_LIST) AND (DEFINED ENV{TORCH_CUDA_ARCH_LIST}))
    set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
  endif()
  set(__NCCL_EP_ARCHS "")
  string(REPLACE " " ";" __nccl_ep_arch_list "${TORCH_CUDA_ARCH_LIST}")
  foreach(__arch IN LISTS __nccl_ep_arch_list)
    if(__arch MATCHES "^([0-9]+)\\.([0-9]+)")
      set(__compact "${CMAKE_MATCH_1}${CMAKE_MATCH_2}")
      if(__compact GREATER_EQUAL 90)
        list(APPEND __NCCL_EP_ARCHS "${__compact}")
      endif()
    endif()
  endforeach()
  set(__NCCL_EP_ARCH_ARG "")
  if(__NCCL_EP_ARCHS)
    list(REMOVE_DUPLICATES __NCCL_EP_ARCHS)
    set(__NCCL_EP_ARCH_ARG "-DCMAKE_CUDA_ARCHITECTURES=${__NCCL_EP_ARCHS}")
  endif()

  message(STATUS "Configuring NCCL EP as third-party dependency (__caffe2_nccl_ep)")
  ExternalProject_Add(nccl_ep_external
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/nccl_ep_build
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nccl_ep
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
      ${__NCCL_EP_ARCH_ARG}
      -DNCCL_HOME=${__NCCL_BUILD_DIR}
      -DNCCL_EP_BUILDDIR=${__NCCL_BUILD_DIR}
      -DNCCL_EP_SOURCE_DIR=${PROJECT_SOURCE_DIR}/third_party/nccl/contrib/nccl_ep
    BUILD_BYPRODUCTS "${__NCCL_BUILD_DIR}/lib/libnccl_ep.a"
    INSTALL_COMMAND ""
    # NCCL (Makefile) must finish first: nccl_ep links -lnccl and includes its headers.
    DEPENDS nccl_external
  )

  set(NCCL_EP_LIBRARIES "${__NCCL_BUILD_DIR}/lib/libnccl_ep.a")
  set(NCCL_EP_INCLUDE_DIRS "${__NCCL_BUILD_DIR}/include")

  add_library(__caffe2_nccl_ep INTERFACE)
  add_dependencies(__caffe2_nccl_ep nccl_ep_external)
  target_link_libraries(__caffe2_nccl_ep INTERFACE ${NCCL_EP_LIBRARIES})
  target_include_directories(__caffe2_nccl_ep INTERFACE ${NCCL_EP_INCLUDE_DIRS})
  # libnccl_ep.a's JIT calls CUDA Driver APIs; pull in libcuda.so.
  if(TARGET CUDA::cuda_driver)
    target_link_libraries(__caffe2_nccl_ep INTERFACE CUDA::cuda_driver)
  endif()
endif()
