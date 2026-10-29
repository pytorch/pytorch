# Build EP against PyTorch's selected NCCL and stage its runtime JIT headers.
cmake_minimum_required(VERSION 3.18)
project(nccl_ep LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

string(REPLACE "." ";" __cuda_ver_parts "${CUDAToolkit_VERSION}")
list(GET __cuda_ver_parts 0 CUDA_MAJOR)
list(GET __cuda_ver_parts 1 CUDA_MINOR)

foreach(header nccl.h nccl_device.h)
  if(NOT EXISTS "${NCCL_INCLUDE_DIR}/${header}")
    message(FATAL_ERROR "NCCL EP requires NCCL device headers: missing ${NCCL_INCLUDE_DIR}/${header}")
  endif()
endforeach()

get_filename_component(nccl_home "${NCCL_INCLUDE_DIR}" DIRECTORY)
set(NCCL_HOME "${nccl_home}" CACHE PATH "PyTorch NCCL prefix" FORCE)
add_subdirectory("${NCCL_EP_SOURCE_DIR}" nccl_ep EXCLUDE_FROM_ALL)

# Resolve JIT headers relative to this EP copy without changing the environment.
set_property(SOURCE "${NCCL_EP_SOURCE_DIR}/device/jit/jit_runtime.cc"
  TARGET_DIRECTORY nccl_ep_static
  APPEND PROPERTY COMPILE_DEFINITIONS env_value=torch_nccl_ep_jit_env_value)
target_sources(nccl_ep_static PRIVATE "${CMAKE_CURRENT_SOURCE_DIR}/jit_paths.cpp")

# Core NCCL symbols resolve through torch_cuda, including for system NCCL.
foreach(prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
  get_target_property(libs nccl_ep_static ${prop})
  list(REMOVE_ITEM libs -lnccl)
  set_property(TARGET nccl_ep_static PROPERTY ${prop} "${libs}")
endforeach()
set_target_properties(nccl_ep_static PROPERTIES
  LINK_DIRECTORIES "" INTERFACE_LINK_DIRECTORIES "")
target_include_directories(nccl_ep_static BEFORE PUBLIC
  "${NCCL_EP_BUILDDIR}/include" "${NCCL_EP_BUILDDIR}/include/nccl_ep"
  "${NCCL_INCLUDE_DIR}" "${NCCL_INCLUDE_DIR}/nccl_device")

# Stage the matching core NCCL headers alongside EP's headers.
add_custom_target(nccl_ep_artifacts
  COMMAND "${CMAKE_COMMAND}" -E make_directory "${NCCL_EP_BUILDDIR}/nccl/include"
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${NCCL_INCLUDE_DIR}/nccl.h" "${NCCL_INCLUDE_DIR}/nccl_device.h"
    "${NCCL_EP_BUILDDIR}/nccl/include/"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
    "${NCCL_INCLUDE_DIR}/nccl_device" "${NCCL_EP_BUILDDIR}/nccl/include/nccl_device"
  DEPENDS nccl_ep_static
  VERBATIM)
