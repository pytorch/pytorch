if(NOT __NCCL_EP_INCLUDED)
  set(__NCCL_EP_INCLUDED TRUE)

  # NCCL is built (Makefile) into this tree by cmake/External/nccl.cmake.
  set(__NCCL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl")

  # Build contrib/nccl_ep as its own CMake project against that prebuilt NCCL
  # (headers + libnccl in __NCCL_BUILD_DIR). NCCL_EP_BUILDDIR == NCCL_HOME so
  # nccl_ep's artifacts (libnccl_ep.a, headers) land in the same tree, leaving
  # the core NCCL build untouched.
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
      # Link the CUDA runtime dynamically (CMake defaults to static). The static
      # cudart does not support CUDA minor-version (enhanced) compatibility, so a
      # static-cudart libnccl_ep cannot run on a minor-older driver; the shared
      # libcudart does, matching how libtorch_cuda links it.
      -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
      ${__NCCL_EP_ARCH_ARG}
      -DNCCL_HOME=${__NCCL_BUILD_DIR}
      -DNCCL_EP_BUILDDIR=${__NCCL_BUILD_DIR}
      -DNCCL_EP_SOURCE_DIR=${PROJECT_SOURCE_DIR}/third_party/nccl/contrib/nccl_ep
    # Build the static lib; the _nccl_ep extension statically links it, so its
    # ncclEp* symbols are embedded in the extension and its nccl* references
    # resolve against torch_cuda's own (statically-linked) NCCL. No runtime
    # libnccl_ep.so and no nccl4py dependency.
    BUILD_BYPRODUCTS "${__NCCL_BUILD_DIR}/lib/libnccl_ep.a"
    INSTALL_COMMAND ""
    # NCCL (Makefile) must finish first: nccl_ep links -lnccl and includes its headers.
    DEPENDS nccl_external
  )

  set(NCCL_EP_LIBRARIES "${__NCCL_BUILD_DIR}/lib/libnccl_ep.a")
  set(NCCL_EP_INCLUDE_DIRS "${__NCCL_BUILD_DIR}/include")
  # The runtime JIT resolves NCCL_EP_HOME -> include/nccl_ep and NCCL_HOME ->
  # include/nccl.h; the submodule build installs both under this one dir, which
  # the extension bakes in as the default (see torch/CMakeLists.txt).
  set(NCCL_EP_JIT_HOME "${__NCCL_BUILD_DIR}" CACHE INTERNAL "nccl-ep JIT header home")

  add_library(__caffe2_nccl_ep INTERFACE)
  add_dependencies(__caffe2_nccl_ep nccl_ep_external)
  target_link_libraries(__caffe2_nccl_ep INTERFACE ${NCCL_EP_LIBRARIES})
  target_include_directories(__caffe2_nccl_ep INTERFACE ${NCCL_EP_INCLUDE_DIRS})
  # libnccl_ep's JIT calls CUDA Driver APIs; pull in libcuda.so.
  if(TARGET CUDA::cuda_driver)
    target_link_libraries(__caffe2_nccl_ep INTERFACE CUDA::cuda_driver)
  endif()
endif()
