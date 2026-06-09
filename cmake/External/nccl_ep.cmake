if(NOT __NCCL_EP_INCLUDED)
  set(__NCCL_EP_INCLUDED TRUE)

  # NCCL is built (Makefile) into this tree by cmake/External/nccl.cmake.
  set(__NCCL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl")

  # Build contrib/nccl_ep as its own CMake project against that prebuilt NCCL
  # (headers + libnccl in __NCCL_BUILD_DIR). NCCL_EP_BUILDDIR == NCCL_HOME so
  # nccl_ep's artifacts (libnccl_ep.a, headers) land in the same tree, leaving
  # the core NCCL build untouched.
  message(STATUS "Configuring NCCL EP as third-party dependency (__caffe2_nccl_ep)")
  ExternalProject_Add(nccl_ep_external
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/nccl_ep_build
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/nccl_ep
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
      -DCMAKE_CUDA_ARCHITECTURES=90
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

  # nccl_ep JIT-compiles its CUDA kernels at runtime from .cuh/.h sources on
  # disk. NCCL is statically linked here (USE_NCCL_EP implies NOT
  # USE_SYSTEM_NCCL), so the NCCL pip wheel is absent at runtime and in any case
  # ships neither the device-header tree nor a version-matched copy. Bundle the
  # whole version-matched include tree (nccl_ep/ + nccl_device/ + nccl.h +
  # ep_enums.h, ~1.2MB) into the wheel under torch/include/nccl_ep_jit; the
  # existing include/**/*.{h,hpp,cuh} package_data globs pick it up, and
  # token_switch.py points NCCL_EP_JIT_SOURCE_DIR / _BUILD_INCLUDE_DIR here.
  install(DIRECTORY "${NCCL_EP_INCLUDE_DIRS}/"
          DESTINATION "include/nccl_ep_jit"
          FILES_MATCHING
            PATTERN "*.h"
            PATTERN "*.hpp"
            PATTERN "*.cuh")
endif()
