# Build native NCCL EP against the NCCL selected by PyTorch.
cmake_minimum_required(VERSION 3.18)
project(nccl_ep LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

string(REPLACE "." ";" __cuda_ver_parts "${CUDAToolkit_VERSION}")
list(GET __cuda_ver_parts 0 CUDA_MAJOR)
list(GET __cuda_ver_parts 1 CUDA_MINOR)

foreach(header nccl.h nccl_device.h nccl_device/impl/core__types.h)
  if(NOT EXISTS "${NCCL_INCLUDE_DIR}/${header}")
    message(FATAL_ERROR "NCCL EP requires NCCL device headers: missing ${NCCL_INCLUDE_DIR}/${header}")
  endif()
endforeach()

get_filename_component(nccl_home "${NCCL_INCLUDE_DIR}" DIRECTORY)
set(NCCL_HOME "${nccl_home}" CACHE PATH "PyTorch NCCL prefix" FORCE)
add_subdirectory("${NCCL_EP_SOURCE_DIR}" nccl_ep EXCLUDE_FROM_ALL)

# Upstream assumes NCCL_HOME/{include,lib} and links -lnccl. Honor PyTorch's
# actual include/library paths; static EP must resolve NCCL through torch_cuda.
foreach(prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
  get_target_property(libs ${NCCL_EP_TARGET} ${prop})
  list(REMOVE_ITEM libs -lnccl)
  set_property(TARGET ${NCCL_EP_TARGET} PROPERTY ${prop} "${libs}")
endforeach()
set_target_properties(${NCCL_EP_TARGET} PROPERTIES
  LINK_DIRECTORIES "" INTERFACE_LINK_DIRECTORIES "")
target_include_directories(${NCCL_EP_TARGET} BEFORE PUBLIC
  "${NCCL_INCLUDE_DIR}" "${NCCL_INCLUDE_DIR}/nccl_device")
if(NCCL_EP_TARGET STREQUAL "nccl_ep_shared")
  target_link_libraries(nccl_ep_shared PRIVATE "${NCCL_LIBRARY}")
  set_target_properties(nccl_ep_shared PROPERTIES INSTALL_RPATH "$ORIGIN")
endif()

add_custom_target(nccl_ep_artifacts
  COMMAND "${CMAKE_COMMAND}" -E copy_if_different
    "${NCCL_INCLUDE_DIR}/nccl.h" "${NCCL_INCLUDE_DIR}/nccl_device.h"
    "${NCCL_EP_BUILDDIR}/include/"
  COMMAND "${CMAKE_COMMAND}" -E copy_directory
    "${NCCL_INCLUDE_DIR}/nccl_device" "${NCCL_EP_BUILDDIR}/include/nccl_device"
  DEPENDS ${NCCL_EP_TARGET}
  VERBATIM)
if(NCCL_EP_TARGET STREQUAL "nccl_ep_shared")
  add_custom_command(TARGET nccl_ep_artifacts POST_BUILD
    COMMAND "${CMAKE_COMMAND}" -E copy_if_different
      "$<TARGET_FILE:nccl_ep_shared>"
      "${NCCL_EP_BUILDDIR}/lib/$<TARGET_SONAME_FILE_NAME:nccl_ep_shared>.package"
    VERBATIM)
endif()
