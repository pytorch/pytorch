# Driver project that builds nccl-extensions/nccl_ep against a prebuilt NCCL
# (the Makefile-built or installed tree in NCCL_HOME). nccl_ep is written to be
# add_subdirectory'd, so it has no project()/languages of its own and expects
# CUDA_MAJOR/CUDA_MINOR (mirroring NCCL's root CMakeLists); this driver supplies
# them. Invoked by the nccl_ep_external ExternalProject in
# cmake/External/nccl_ep.cmake, which passes NCCL_HOME / NCCL_EP_BUILDDIR /
# NCCL_EP_SOURCE_DIR via -D.
cmake_minimum_required(VERSION 3.18)
project(nccl_ep LANGUAGES CXX CUDA)

find_package(CUDAToolkit REQUIRED)

string(REPLACE "." ";" __cuda_ver_parts "${CUDAToolkit_VERSION}")
list(GET __cuda_ver_parts 0 CUDA_MAJOR)
list(GET __cuda_ver_parts 1 CUDA_MINOR)

add_subdirectory("${NCCL_EP_SOURCE_DIR}" nccl_ep)

# The static EP target's CUDA device link does not need a separate libnccl.so.
get_target_property(libs nccl_ep_static LINK_LIBRARIES)
list(REMOVE_ITEM libs -lnccl)
set_property(TARGET nccl_ep_static PROPERTY LINK_LIBRARIES "${libs}")

# Keep matching core and EP headers together for the JIT include path and cache
# fingerprint. Copy only core headers, excluding stale EP headers in NCCL_HOME.
file(GLOB_RECURSE nccl_device_headers CONFIGURE_DEPENDS
  RELATIVE "${NCCL_HOME}/include" "${NCCL_HOME}/include/nccl_device/*")
foreach(header nccl.h nccl_device.h ${nccl_device_headers})
  configure_file("${NCCL_HOME}/include/${header}"
    "${NCCL_EP_BUILDDIR}/include/${header}" COPYONLY)
endforeach()
