if(NOT __NCCL_EP_INCLUDED)
  set(__NCCL_EP_INCLUDED TRUE)

  # EP sources are needed with both source and system NCCL.
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

  # Build nccl_ep for the same CUDA archs as the rest of PyTorch, dropping
  # anything below sm_90 since nccl_ep requires Hopper+. PyTorch disables
  # CMAKE_CUDA_ARCHITECTURES in favor of TORCH_CUDA_ARCH_LIST, so derive the arch
  # list from there. If nothing qualifies (list unset, or a symbolic value like
  # "Common"/"All"), pass no flag and let nccl_ep's CMakeLists pick its own
  # CUDA-version default.
  if((NOT DEFINED TORCH_CUDA_ARCH_LIST) AND (DEFINED ENV{TORCH_CUDA_ARCH_LIST}))
    # Usually only set in the environment (e.g. by setup.py), not as a cache var.
    set(TORCH_CUDA_ARCH_LIST $ENV{TORCH_CUDA_ARCH_LIST})
  endif()
  # TORCH_CUDA_ARCH_LIST may be space- or ;-separated; normalize to a CMake list.
  string(REPLACE " " ";" __nccl_ep_archs "${TORCH_CUDA_ARCH_LIST}")
  # Reduce each entry to its compact sm number, dropping any decoration:
  # "9.0" / "9.0a" / "10.0+PTX" -> "90" / "90" / "100".
  list(TRANSFORM __nccl_ep_archs REPLACE "^([0-9]+)\\.([0-9]+).*$" "\\1\\2")
  # Drop anything that wasn't a numeric arch spec (e.g. "Common", "All", "").
  list(FILTER __nccl_ep_archs INCLUDE REGEX "^[0-9]+$")
  set(__NCCL_EP_ARCHS "")
  foreach(__arch IN LISTS __nccl_ep_archs)
    if(__arch GREATER_EQUAL 90)  # nccl_ep is Hopper+ only
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
  set(__NCCL_EP_DEPENDS "")
  if(NOT USE_SYSTEM_NCCL)
    set(__NCCL_EP_DEPENDS nccl_external)
  endif()

  message(STATUS "Building static NCCL EP against NCCL headers at ${NCCL_INCLUDE_DIRS}")
  ExternalProject_Add(nccl_ep_external
    SOURCE_DIR ${CMAKE_CURRENT_LIST_DIR}/nccl_ep_build
    BINARY_DIR ${__NCCL_EP_BUILD_DIR}
    LIST_SEPARATOR |
    CMAKE_ARGS
      -DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}
      -DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}
      -DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}
      # Link the CUDA runtime dynamically (CMake defaults to static). The static
      # cudart does not support CUDA minor-version (enhanced) compatibility, so a
      # static-cudart libnccl_ep cannot run on a minor-older driver; the shared
      # libcudart does, matching how libtorch_cuda links it.
      -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
      ${__NCCL_EP_ARCH_ARG}
      -DNCCL_INCLUDE_DIR=${NCCL_INCLUDE_DIRS}
      -DNCCL_EP_BUILDDIR=${__NCCL_EP_OUTPUT_DIR}
      -DNCCL_EP_SOURCE_DIR=${__NCCL_EP_SOURCE_DIR}/nccl_ep
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG>
      --target nccl_ep_artifacts
    BUILD_BYPRODUCTS "${__NCCL_EP_OUTPUT_DIR}/lib/libnccl_ep.a"
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND ""
    DEPENDS ${__NCCL_EP_DEPENDS}
  )

  set(NCCL_EP_LIBRARIES "${__NCCL_EP_OUTPUT_DIR}/lib/libnccl_ep.a")
  set(NCCL_EP_INCLUDE_DIRS "${__NCCL_EP_OUTPUT_DIR}/include")

  add_library(__caffe2_nccl_ep INTERFACE)
  add_dependencies(__caffe2_nccl_ep nccl_ep_external)
  target_link_libraries(__caffe2_nccl_ep INTERFACE ${NCCL_EP_LIBRARIES} ${CMAKE_DL_LIBS})
  target_include_directories(__caffe2_nccl_ep INTERFACE ${NCCL_EP_INCLUDE_DIRS})
  # libnccl_ep's JIT calls CUDA Driver APIs; pull in libcuda.so.
  if(TARGET CUDA::cuda_driver)
    target_link_libraries(__caffe2_nccl_ep INTERFACE CUDA::cuda_driver)
  endif()

  install(DIRECTORY "${NCCL_EP_INCLUDE_DIRS}/"
    DESTINATION share/nccl_ep/include PATTERN "config.h" EXCLUDE)
  install(DIRECTORY "${__NCCL_EP_OUTPUT_DIR}/nccl/"
    DESTINATION share/nccl_ep/nccl)
endif()
