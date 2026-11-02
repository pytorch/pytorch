if(NOT __NCCL_EP_INCLUDED)
  set(__NCCL_EP_INCLUDED TRUE)

  # Both modes compile the bindings against the pinned public headers.
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

  if(NOT USE_SYSTEM_NCCL_EP)
    # Match PyTorch's requested architectures, keeping only Hopper and newer.
    if((NOT DEFINED TORCH_CUDA_ARCH_LIST) AND (DEFINED ENV{TORCH_CUDA_ARCH_LIST}))
      set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
    endif()
    string(REPLACE " " ";" __nccl_ep_archs "${TORCH_CUDA_ARCH_LIST}")
    list(TRANSFORM __nccl_ep_archs REPLACE "^([0-9]+)\\.([0-9]+).*$" "\\1\\2")
    list(FILTER __nccl_ep_archs INCLUDE REGEX "^[0-9]+$")
    set(__NCCL_EP_ARCHS "")
    foreach(__arch IN LISTS __nccl_ep_archs)
      if(__arch GREATER_EQUAL 90)
        list(APPEND __NCCL_EP_ARCHS "${__arch}")
      endif()
    endforeach()
    set(__NCCL_EP_ARCH_ARG "")
    if(__NCCL_EP_ARCHS)
      list(REMOVE_DUPLICATES __NCCL_EP_ARCHS)
      list(JOIN __NCCL_EP_ARCHS "|" __NCCL_EP_ARCHS)
      set(__NCCL_EP_ARCH_ARG "-DCMAKE_CUDA_ARCHITECTURES=${__NCCL_EP_ARCHS}")
    endif()

    set(__NCCL_EP_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl_ep")
    set(__NCCL_EP_OUTPUT_DIR "${__NCCL_EP_BUILD_DIR}/artifacts")
    set(NCCL_EP_LIBRARIES "${__NCCL_EP_OUTPUT_DIR}/lib/libnccl_ep.a")
    set(__NCCL_EP_DEPENDS "")
    if(NOT USE_SYSTEM_NCCL)
      set(__NCCL_EP_DEPENDS nccl_external)
    endif()

    ExternalProject_Add(nccl_ep_external
      SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/nccl_ep_build
      BINARY_DIR ${__NCCL_EP_BUILD_DIR}
      LIST_SEPARATOR |
      CMAKE_ARGS
        -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
        -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
        -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
        # Match PyTorch's shared CUDA runtime for minor-version compatibility.
        -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
        ${__NCCL_EP_ARCH_ARG}
        -DNCCL_INCLUDE_DIR=${NCCL_INCLUDE_DIRS}
        -DNCCL_EP_BUILDDIR=${__NCCL_EP_OUTPUT_DIR}
        -DNCCL_EP_SOURCE_DIR=${__NCCL_EP_SOURCE_DIR}/nccl_ep
      BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG>
        --target nccl_ep_static
      BUILD_BYPRODUCTS "${NCCL_EP_LIBRARIES}"
      BUILD_ALWAYS TRUE
      INSTALL_COMMAND ""
      DEPENDS ${__NCCL_EP_DEPENDS}
    )

    add_dependencies(__caffe2_nccl_ep nccl_ep_external)
    target_link_libraries(__caffe2_nccl_ep INTERFACE
      ${NCCL_EP_LIBRARIES} CUDA::cuda_driver)
  endif()
endif()
