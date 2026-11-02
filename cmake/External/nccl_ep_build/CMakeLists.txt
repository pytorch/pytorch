# Build EP against PyTorch's selected NCCL and stage its runtime JIT headers.
cmake_minimum_required(VERSION 3.18)
project(nccl_ep LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)
set(CUDA_MAJOR ${CUDAToolkit_VERSION_MAJOR})
set(CUDA_MINOR ${CUDAToolkit_VERSION_MINOR})

get_filename_component(nccl_home "${NCCL_INCLUDE_DIR}" DIRECTORY)
set(NCCL_HOME "${nccl_home}" CACHE PATH "PyTorch NCCL prefix" FORCE)
add_subdirectory("${NCCL_EP_SOURCE_DIR}" nccl_ep EXCLUDE_FROM_ALL)

# Core NCCL symbols resolve through torch_cuda, including for system NCCL.
foreach(prop LINK_LIBRARIES INTERFACE_LINK_LIBRARIES)
  get_target_property(libs nccl_ep_static ${prop})
  list(REMOVE_ITEM libs -lnccl)
  set_property(TARGET nccl_ep_static PROPERTY ${prop} "${libs}")
endforeach()
set_target_properties(nccl_ep_static PROPERTIES
  LINK_DIRECTORIES "" INTERFACE_LINK_DIRECTORIES "")

# Keep matching core and EP headers together for the JIT include path and cache
# fingerprint. Copy only core headers, excluding stale EP headers in NCCL_HOME.
file(GLOB_RECURSE nccl_device_headers CONFIGURE_DEPENDS
  RELATIVE "${NCCL_INCLUDE_DIR}" "${NCCL_INCLUDE_DIR}/nccl_device/*")
foreach(header nccl.h nccl_device.h ${nccl_device_headers})
  configure_file("${NCCL_INCLUDE_DIR}/${header}"
    "${NCCL_EP_BUILDDIR}/include/${header}" COPYONLY)
endforeach()
