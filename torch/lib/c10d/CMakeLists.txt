cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

# Find modules.
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/public
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules_CUDA_fix)

if(USE_CUDA)
  add_definitions(-DUSE_CUDA=1)
elseif(USE_ROCM)
  add_definitions(-DUSE_ROCM=1)
  add_definitions(-D__HIP_PLATFORM_HCC__=1)
else()
  message(STATUS "Building c10d without CUDA/ROCm support")
endif()

#if(USE_TBB)
#include_directories(${TBB_ROOT_DIR}/include)
#endif()

if(USE_GLOO)
  option(USE_C10D_GLOO "USE C10D GLOO" ON)
endif()

if(USE_NCCL)
  option(USE_C10D_NCCL "USE C10D NCCL" ON)
endif()

if(USE_MPI)
  find_package(MPI)
  if(MPI_FOUND)
    message(STATUS "MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
    message(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")
    message(STATUS "MPIEXEC: ${MPIEXEC}")
    option(USE_C10D_MPI "USE C10D MPI" ON)
  else()
    message(STATUS "Not able to find MPI, will compile c10d without MPI support")
  endif()
endif()

function(copy_header file)
  configure_file(${file} ${CMAKE_BINARY_DIR}/include/c10d/${file} COPYONLY)
endfunction()

set(C10D_SRCS
  FileStore.cpp
  ProcessGroup.cpp
  Store.cpp
  PrefixStore.cpp
  Utils.cpp
  )

if(NOT WIN32)
  list(APPEND C10D_SRCS HashStore.cpp ProcessGroupRoundRobin.cpp TCPStore.cpp)
endif()

set(C10D_LIBS)

if(USE_C10D_NCCL)
  list(APPEND C10D_SRCS cuda/ProcessGroupNCCL.cpp cuda/NCCLUtils.cpp)
  list(APPEND C10D_LIBS __caffe2_nccl)
endif()

if(USE_C10D_MPI)
  list(APPEND C10D_SRCS ProcessGroupMPI.cpp)
  list(APPEND C10D_LIBS ${MPI_LIBRARIES})
endif()

if(USE_C10D_GLOO)
  list(APPEND C10D_SRCS ProcessGroupGloo.cpp GlooDeviceFactory.cpp)
  list(APPEND C10D_LIBS gloo)
  if(USE_CUDA)
    list(APPEND C10D_LIBS gloo_cuda)
  endif()
endif()

#add_library(c10d STATIC ${C10D_SRCS})
#set_property(TARGET torch PROPERTY POSITION_INDEPENDENT_CODE ON)
#set_property(TARGET torch PROPERTY CXX_STANDARD 14)

#add_dependencies(c10d torch)

#if(USE_C10D_GLOO)
#  add_dependencies(torch gloo)
#  if(USE_CUDA)
#    add_dependencies(torch gloo_cuda)
#  endif()
#endif()

#target_include_directories(torch PUBLIC
#  ${CMAKE_BINARY_DIR}/aten/src # provides "ATen/TypeExtendedInterface.h" to ATen.h
#  ${CMAKE_BINARY_DIR}/caffe2/aten/src # provides <TH/THGeneral.h> to THC.h
#  )
#
## For <c10d/...>
#target_include_directories(torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
## For torch/csrc/utils/hash.h
#target_include_directories(torch PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

#if(USE_C10D_NCCL)
#  target_compile_definitions(torch INTERFACE USE_C10D_NCCL)
#endif()
#
#if(USE_C10D_MPI)
#  target_compile_definitions(torch INTERFACE USE_C10D_MPI)
#endif()
#
#if(USE_C10D_GLOO)
#  target_compile_definitions(torch INTERFACE USE_C10D_GLOO)
#endif()

copy_header(FileStore.hpp)
copy_header(PrefixStore.hpp)
copy_header(ProcessGroup.hpp)
copy_header(Store.hpp)
copy_header(Types.hpp)
copy_header(Utils.hpp)
if(USE_GLOO)
  copy_header(ProcessGroupGloo.hpp)
  copy_header(GlooDeviceFactory.hpp)
endif()
if(NOT WIN32)
  copy_header(HashStore.hpp)
  copy_header(TCPStore.hpp)
endif()

if(USE_C10D_NCCL)
  copy_header(cuda/ProcessGroupNCCL.hpp)
  copy_header(cuda/NCCLUtils.hpp)
endif()

if(USE_C10D_MPI)
  target_include_directories(torch PUBLIC ${MPI_INCLUDE_PATH})
  copy_header(ProcessGroupMPI.hpp)
endif()

#target_link_libraries(torch PUBLIC ${C10D_LIBS})

#install(TARGETS c10d DESTINATION lib)

option(BUILD_EXAMPLES "Build examples" OFF)
if(BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

option(BUILD_TEST "Build tests" ON)
#if(BUILD_TEST)
#  enable_testing()
#  add_subdirectory(test)
#endif()

# Install all header files that were prepared in the build directory
install(DIRECTORY ${CMAKE_BINARY_DIR}/include/ DESTINATION include)

# C10D_CPU_SRCS and C10D_CPU_HDRS includes all cpp source and header files needed for CPU-only build.
#file(GLOB C10D_CPU_SRCS "*.cpp")
file(GLOB C10D_CPU_HDRS "*.hpp")

set(C10D_CPU_SRCS)
list(APPEND C10D_BASE_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/FileStore.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ProcessGroup.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Store.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PrefixStore.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
)

if(NOT WIN32)
  list(APPEND C10D_BASE_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/HashStore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ProcessGroupRoundRobin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TCPStore.cpp
  )
endif()

if(USE_C10D_MPI)
  list(APPEND C10D_BASE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/ProcessGroupMPI.cpp)
endif()

if(USE_C10D_GLOO)
  list(APPEND C10D_BASE_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/ProcessGroupGloo.cpp ${CMAKE_CURRENT_SOURCE_DIR}/GlooDeviceFactory.cpp)
endif()

set(C10D_CPU_SRCS ${C10D_BASE_SRCS})
set(C10D_CUDA_SRCS ${C10D_BASE_SRCS})

if(USE_C10D_NCCL)
  list(APPEND C10D_CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/cuda/ProcessGroupNCCL.cpp ${CMAKE_CURRENT_SOURCE_DIR}/cuda/NCCLUtils.cpp)
endif()

# C10D_CUDA_SRCS and C10D_CUDA_HDRS includes all cpp source and header files needed for CUDA build.
file(GLOB C10D_CUDA_SRCS "cuda/*.cpp")
file(GLOB C10D_CUDA_HDRS "cuda/*.hpp")

# C10D_PUBLIC_HDRS contains all header files we expose in torch/include.
set(C10D_PUBLIC_HDRS)
list(APPEND C10D_PUBLIC_HDRS ${C10D_CPU_HDRS})
list(APPEND C10D_PUBLIC_HDRS ${C10D_CUDA_HDRS})
# ProcessGroupRoundRobin.hpp isn't officially supported, thus don't make it
# part of public headers.
list(REMOVE_ITEM C10D_PUBLIC_HDRS "ProcessGroupRoundRobin.hpp")

# All include directories for building C10D source files
set(C10D_INCLUDE)

list(APPEND C10D_INCLUDE ${CMAKE_BINARY_DIR}/aten/src) # provides "ATen/TypeExtendedInterface.h" to ATen.h
list(APPEND C10D_INCLUDE ${CMAKE_BINARY_DIR}/caffe2/aten/src) # provides <TH/THGeneral.h> to THC.h
list(APPEND C10D_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/..) # For <c10d/...>
list(APPEND C10D_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/../../..) # For torch/csrc/utils/hash.h
if(USE_C10D_MPI)
  list(APPEND C10D_INCLUDE ${MPI_INCLUDE_PATH})
endif()

if(USE_TBB)
  list(APPEND C10D_INCLUDE ${TBB_ROOT_DIR}/include)
endif()


# Export c10d cpu and cuda source file lists to parent scopes
set(C10D_CPU_SRCS ${C10D_CPU_SRCS} PARENT_SCOPE)
set(C10D_CPU_HDRS ${C10D_CPU_HDRS} PARENT_SCOPE)
set(C10D_CUDA_SRCS ${C10D_CUDA_SRCS} PARENT_SCOPE)
set(C10D_CUDA_HDRS ${C10D_CUDA_HDRS} PARENT_SCOPE)
set(C10D_PUBLIC_HDRS ${C10D_PUBLIC_HDRS} PARENT_SCOPE) 
set(C10D_INCLUDE ${C10D_INCLUDE} PARENT_SCOPE) 
set(C10D_LIBS ${C10D_LIBS} PARENT_SCOPE) 

message(STATUS "C10D_CPU_SRCS: ${C10D_CPU_SRCS}")