if(NOT __NCCL_EP_INCLUDED)
  set(__NCCL_EP_INCLUDED TRUE)

  # NCCL is built (Makefile) into this tree by cmake/External/nccl.cmake.
  set(__NCCL_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl")

  # NCCL EP sources come from NVIDIA/nccl-extensions at the pinned tag. Keep the
  # pin in sync with the nccl-extensions wheel version _token_switch.py asks
  # for: a USE_SYSTEM_NCCL=ON build compiles against these headers but loads
  # that wheel's libnccl_ep at runtime.
  set(__NCCL_EP_SOURCE_ROOT "${PROJECT_SOURCE_DIR}/third_party/nccl-extensions")
  # Test a file, not the dir, so an interrupted clone is fetched again.
  if(NOT EXISTS "${__NCCL_EP_SOURCE_ROOT}/nccl_ep/CMakeLists.txt")
    set(__NCCL_EP_PIN_FILE
        "${PROJECT_SOURCE_DIR}/.ci/docker/ci_commit_pins/nccl-extensions.txt")
    if(EXISTS "${__NCCL_EP_PIN_FILE}")
      file(READ "${__NCCL_EP_PIN_FILE}" __NCCL_EP_TAG)
      string(STRIP "${__NCCL_EP_TAG}" __NCCL_EP_TAG)
      message(STATUS "Checking out nccl-extensions tag: ${__NCCL_EP_TAG}")
      include(FetchContent)
      FetchContent_Declare(
        nccl_extensions
        GIT_REPOSITORY https://github.com/NVIDIA/nccl-extensions
        GIT_TAG        "${__NCCL_EP_TAG}"
        GIT_SHALLOW    TRUE
        # No submodules: the repo's third_party/nccl and third_party/googletest
        # are unused by the nccl_ep build and GIT_SHALLOW does not apply to them.
        GIT_SUBMODULES ""
        SOURCE_DIR     "${__NCCL_EP_SOURCE_ROOT}"
      )
      FetchContent_Populate(nccl_extensions)
    else()
      message(FATAL_ERROR
        "nccl-extensions sources not found at ${__NCCL_EP_SOURCE_ROOT} and "
        "no pin file ${__NCCL_EP_PIN_FILE} exists. Provide them there or set "
        "USE_NCCL_EP=OFF.")
    endif()
  endif()

  # Build nccl_ep as its own CMake project against that prebuilt NCCL
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

  # Branch on how torch links NCCL (USE_NCCL_EP supports both):
  #  - USE_SYSTEM_NCCL=OFF (source/dev): torch statically embeds the submodule
  #    NCCL, so build libnccl_ep.a against it and have the extension statically
  #    link it -- self-contained, no nccl-extensions wheel, no runtime
  #    libnccl_ep.so.
  #  - USE_SYSTEM_NCCL=ON (wheel/CI): torch dynamically links the system NCCL, so
  #    build libnccl_ep.so against that same NCCL; the extension NEEDED-links it
  #    and resolves it (and its JIT headers) from the nccl-extensions wheel at
  #    runtime.
  if(USE_SYSTEM_NCCL)
    if(DEFINED ENV{NCCL_INCLUDE_DIR})
      get_filename_component(__NCCL_EP_NCCL_HOME "$ENV{NCCL_INCLUDE_DIR}" DIRECTORY)
    else()
      list(GET NCCL_INCLUDE_DIRS 0 __nccl_inc)
      get_filename_component(__NCCL_EP_NCCL_HOME "${__nccl_inc}" DIRECTORY)
    endif()
    set(__NCCL_EP_LIB "libnccl_ep.so")
    set(__NCCL_EP_DEPENDS "")
  else()
    set(__NCCL_EP_NCCL_HOME "${__NCCL_BUILD_DIR}")
    set(__NCCL_EP_LIB "libnccl_ep.a")
    set(__NCCL_EP_DEPENDS nccl_external)
  endif()

  message(STATUS "Configuring NCCL EP (__caffe2_nccl_ep): ${__NCCL_EP_LIB} against NCCL at ${__NCCL_EP_NCCL_HOME}")
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
      -DNCCL_HOME=${__NCCL_EP_NCCL_HOME}
      -DNCCL_EP_BUILDDIR=${__NCCL_BUILD_DIR}
      -DNCCL_EP_SOURCE_DIR=${__NCCL_EP_SOURCE_ROOT}/nccl_ep
    BUILD_BYPRODUCTS "${__NCCL_BUILD_DIR}/lib/${__NCCL_EP_LIB}"
    INSTALL_COMMAND ""
    # In the static path NCCL (Makefile) must finish first; in the system path
    # NCCL is already present so there is no in-tree dependency.
    DEPENDS ${__NCCL_EP_DEPENDS}
  )

  set(NCCL_EP_LIBRARIES "${__NCCL_BUILD_DIR}/lib/${__NCCL_EP_LIB}")
  set(NCCL_EP_INCLUDE_DIRS "${__NCCL_BUILD_DIR}/include")

  if(NOT USE_SYSTEM_NCCL)
    # Static path only: bake the in-tree JIT header dir so the extension
    # self-configures NCCL_EP_HOME / NCCL_HOME (both resolve under this one dir:
    # include/nccl_ep and include/nccl.h). The dynamic path instead gets these
    # from the nccl-extensions wheel at runtime, so leave NCCL_EP_JIT_HOME unset
    # there.
    set(NCCL_EP_JIT_HOME "${__NCCL_BUILD_DIR}" CACHE INTERNAL "nccl-ep JIT header home")
  endif()

  add_library(__caffe2_nccl_ep INTERFACE)
  add_dependencies(__caffe2_nccl_ep nccl_ep_external)
  target_link_libraries(__caffe2_nccl_ep INTERFACE ${NCCL_EP_LIBRARIES})
  target_include_directories(__caffe2_nccl_ep INTERFACE ${NCCL_EP_INCLUDE_DIRS})
  # libnccl_ep's JIT calls CUDA Driver APIs; pull in libcuda.so.
  if(TARGET CUDA::cuda_driver)
    target_link_libraries(__caffe2_nccl_ep INTERFACE CUDA::cuda_driver)
  endif()
endif()
