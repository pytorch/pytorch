if(NOT __NCCL_EP_INCLUDED)
  set(__NCCL_EP_INCLUDED TRUE)

  message(STATUS "Configuring NCCL EP as third-party dependency (__caffe2_nccl_ep)")
  add_library(__caffe2_nccl_ep INTERFACE)
  add_dependencies(__caffe2_nccl_ep nccl_external)
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
