if(NOT __NCCL_EP_INCLUDED)
  set(__NCCL_EP_INCLUDED TRUE)

  if(NOT NCCL_EP_SOURCE_DIR)
    set(NCCL_EP_SOURCE_DIR "${PROJECT_SOURCE_DIR}/third_party/nccl-extensions/nccl_ep")
  endif()
  get_filename_component(NCCL_EP_SOURCE_DIR "${NCCL_EP_SOURCE_DIR}" ABSOLUTE
    BASE_DIR "${PROJECT_SOURCE_DIR}")
  if(NOT EXISTS "${NCCL_EP_SOURCE_DIR}/CMakeLists.txt" OR
     NOT EXISTS "${NCCL_EP_SOURCE_DIR}/include/nccl_ep.h")
    message(FATAL_ERROR
      "NCCL EP sources are missing at ${NCCL_EP_SOURCE_DIR}. "
      "Run git submodule update --init --recursive third_party/nccl-extensions "
      "or set NCCL_EP_SOURCE_DIR to the nccl_ep directory of another checkout.")
  endif()

  # Reuse PyTorch's architecture expansion, including named GPUs and +PTX.
  torch_cuda_get_nvcc_gencode_flag(__nccl_ep_gencode)
  set(__NCCL_EP_NVCC_FLAGS "")
  foreach(__flag IN LISTS __nccl_ep_gencode)
    if(__flag MATCHES "code=(sm|compute)_([0-9]+)([af]?)$")
      if(CMAKE_MATCH_2 GREATER_EQUAL 90)
        list(APPEND __NCCL_EP_NVCC_FLAGS "-gencode=${__flag}")
      endif()
    endif()
  endforeach()
  if(NOT __NCCL_EP_NVCC_FLAGS)
    message(FATAL_ERROR "NCCL EP requires a TORCH_CUDA_ARCH_LIST with SM90 or newer")
  endif()
  list(REMOVE_DUPLICATES __NCCL_EP_NVCC_FLAGS)
  # Supported CMake versions cannot represent every architecture NVCC accepts.
  list(JOIN __NCCL_EP_NVCC_FLAGS " " __nccl_ep_nvcc_arg)

  set(__NCCL_EP_BUILD_DIR "${CMAKE_CURRENT_BINARY_DIR}/nccl_ep")
  set(__NCCL_EP_OUTPUT_DIR "${__NCCL_EP_BUILD_DIR}/artifacts")
  set(__NCCL_EP_DEPENDS "")
  if(NOT USE_SYSTEM_NCCL)
    set(__NCCL_EP_DEPENDS nccl_external)
  endif()

  if(USE_SYSTEM_NCCL AND NOT USE_STATIC_NCCL)
    set(__NCCL_EP_TARGET nccl_ep_shared)
    set(__NCCL_EP_LIB libnccl_ep.so)
    file(STRINGS "${NCCL_EP_SOURCE_DIR}/include/nccl_ep.h" __nccl_ep_major
      REGEX "^#define NCCL_EP_MAJOR [0-9]+$")
    string(REGEX REPLACE ".* ([0-9]+)$" "\\1" __nccl_ep_major "${__nccl_ep_major}")
    if(NOT __nccl_ep_major MATCHES "^[0-9]+$")
      message(FATAL_ERROR "Cannot determine the NCCL EP shared library version")
    endif()
    set(NCCL_EP_SONAME "libnccl_ep.so.${__nccl_ep_major}")
  else()
    set(__NCCL_EP_TARGET nccl_ep_static)
    set(__NCCL_EP_LIB libnccl_ep.a)
  endif()

  message(STATUS "Building ${__NCCL_EP_TARGET} from ${NCCL_EP_SOURCE_DIR} against ${NCCL_LIBRARIES}")
  ExternalProject_Add(nccl_ep_external
    SOURCE_DIR "${CMAKE_CURRENT_LIST_DIR}/nccl_ep_build"
    BINARY_DIR "${__NCCL_EP_BUILD_DIR}"
    CMAKE_ARGS
      "-DCMAKE_BUILD_TYPE=${CMAKE_BUILD_TYPE}"
      "-DCMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER}"
      "-DCMAKE_CUDA_COMPILER=${CMAKE_CUDA_COMPILER}"
      "-DCMAKE_CUDA_HOST_COMPILER=${CMAKE_CUDA_HOST_COMPILER}"
      "-DCUDAToolkit_ROOT=${CUDA_TOOLKIT_ROOT_DIR}"
      -DCMAKE_CUDA_RUNTIME_LIBRARY=Shared
      -DCMAKE_CUDA_ARCHITECTURES=OFF
      "-DCMAKE_CUDA_FLAGS=${__nccl_ep_nvcc_arg}"
      "-DNCCL_INCLUDE_DIR=${NCCL_INCLUDE_DIRS}"
      "-DNCCL_LIBRARY=${NCCL_LIBRARIES}"
      "-DNCCL_EP_BUILDDIR=${__NCCL_EP_OUTPUT_DIR}"
      "-DNCCL_EP_SOURCE_DIR=${NCCL_EP_SOURCE_DIR}"
      "-DNCCL_EP_TARGET=${__NCCL_EP_TARGET}"
    BUILD_COMMAND "${CMAKE_COMMAND}" --build <BINARY_DIR> --config $<CONFIG>
      --target nccl_ep_artifacts
    BUILD_BYPRODUCTS "${__NCCL_EP_OUTPUT_DIR}/lib/${__NCCL_EP_LIB}"
    BUILD_ALWAYS TRUE
    INSTALL_COMMAND ""
    DEPENDS ${__NCCL_EP_DEPENDS}
  )

  set(NCCL_EP_LIBRARIES "${__NCCL_EP_OUTPUT_DIR}/lib/${__NCCL_EP_LIB}")
  set(NCCL_EP_INCLUDE_DIRS "${__NCCL_EP_OUTPUT_DIR}/include")

  add_library(__caffe2_nccl_ep INTERFACE)
  add_dependencies(__caffe2_nccl_ep nccl_ep_external)
  target_link_libraries(__caffe2_nccl_ep INTERFACE ${NCCL_EP_LIBRARIES})
  target_include_directories(__caffe2_nccl_ep INTERFACE ${NCCL_EP_INCLUDE_DIRS})
  if(TARGET CUDA::cuda_driver)
    target_link_libraries(__caffe2_nccl_ep INTERFACE CUDA::cuda_driver)
  endif()

  install(DIRECTORY "${NCCL_EP_INCLUDE_DIRS}/" DESTINATION share/nccl_ep/include)
  install(FILES "${NCCL_EP_SOURCE_DIR}/LICENSE.txt" DESTINATION share/nccl_ep)
  if(NOT USE_SYSTEM_NCCL)
    install(FILES "${PROJECT_SOURCE_DIR}/third_party/nccl/LICENSE.txt"
      DESTINATION share/nccl_ep RENAME NCCL-LICENSE.txt)
  endif()
  if(USE_SYSTEM_NCCL AND NOT USE_STATIC_NCCL)
    # Install a real file with the SONAME, independent of wheel symlink handling.
    install(FILES "${__NCCL_EP_OUTPUT_DIR}/lib/${NCCL_EP_SONAME}.package"
      DESTINATION lib RENAME "${NCCL_EP_SONAME}")
  endif()
endif()
