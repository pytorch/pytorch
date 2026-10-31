if(NOT __NCCL_EP_INCLUDED)
  set(__NCCL_EP_INCLUDED TRUE)

  # Stage public headers for the bindings; nccl-extensions supplies the runtime.
  set(__NCCL_EP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/nccl-extensions")
  if(NOT EXISTS "${__NCCL_EP_SOURCE_DIR}")
    # Keep the bundled notices in third_party/nccl_ep in sync with this pin.
    file(READ "${PROJECT_SOURCE_DIR}/third_party/nccl_ep/pin.txt" __NCCL_EP_PIN)
    string(STRIP "${__NCCL_EP_PIN}" __NCCL_EP_PIN)
    include(FetchContent)
    FetchContent_Declare(
      nccl_extensions
      GIT_REPOSITORY https://github.com/NVIDIA/nccl-extensions.git
      GIT_TAG "${__NCCL_EP_PIN}"
      GIT_SUBMODULES ""
      SOURCE_DIR "${__NCCL_EP_SOURCE_DIR}"
    )
    FetchContent_Populate(nccl_extensions)
  endif()

  set(NCCL_EP_INCLUDE_DIRS "${CMAKE_CURRENT_BINARY_DIR}/nccl_ep/include")
  configure_file("${__NCCL_EP_SOURCE_DIR}/nccl_ep/include/nccl_ep.h"
    "${NCCL_EP_INCLUDE_DIRS}/nccl_ep.h" COPYONLY)
  configure_file("${__NCCL_EP_SOURCE_DIR}/nccl_ep/include/ep_enums.h"
    "${NCCL_EP_INCLUDE_DIRS}/nccl_ep/ep_enums.h" COPYONLY)

  add_library(__caffe2_nccl_ep INTERFACE)
  target_include_directories(__caffe2_nccl_ep INTERFACE ${NCCL_EP_INCLUDE_DIRS})
endif()
