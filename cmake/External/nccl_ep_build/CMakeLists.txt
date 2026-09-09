# Driver project that builds NCCL's contrib/nccl_ep against a prebuilt NCCL
# (the Makefile-built tree in NCCL_HOME). contrib/nccl_ep is written to be
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
