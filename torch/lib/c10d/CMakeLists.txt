cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

#############################
#   Gather Build Options    #
#############################
if(USE_GLOO)
  if(USE_CUDA)
    option(USE_C10D_GLOO_CUDA "USE C10D GLOO with CUDA support" ON)
  else()
    option(USE_C10D_GLOO_CPU "USE C10D GLOO with CPU-only" ON)
  endif()
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

#############################
#   Expose Public Headers   #
#############################
function(copy_header file)
  configure_file(${file} ${CMAKE_BINARY_DIR}/include/c10d/${file} COPYONLY)
endfunction()

copy_header(FileStore.hpp)
copy_header(PrefixStore.hpp)
copy_header(ProcessGroup.hpp)
copy_header(Store.hpp)
copy_header(comm.hpp)
copy_header(Types.hpp)
copy_header(Utils.hpp)
copy_header(reducer.hpp)
copy_header(default_comm_hooks.hpp)

if(USE_GLOO)
  copy_header(ProcessGroupGloo.hpp)
  copy_header(GlooDeviceFactory.hpp)
endif()

if(NOT WIN32)
  copy_header(HashStore.hpp)
  copy_header(TCPStore.hpp)
endif()

if(USE_C10D_NCCL)
  copy_header(ProcessGroupNCCL.hpp)
  copy_header(NCCLUtils.hpp)
endif()

if(USE_C10D_MPI)
  copy_header(ProcessGroupMPI.hpp)
endif()

# Install all header files that were prepared in the build directory
install(DIRECTORY ${CMAKE_BINARY_DIR}/include/ DESTINATION include)

#############################
#   C10D_{CPU,CUDA}_SRCS    #
#############################

# Sources to be added to libtorch_cpu and libtorch_cuda respectively.
set(C10D_CPU_SRCS)
set(C10D_CUDA_SRCS)

list(APPEND C10D_CPU_SRCS
  ${CMAKE_CURRENT_SOURCE_DIR}/FileStore.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/ProcessGroup.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Store.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/PrefixStore.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/Utils.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/reducer.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/comm.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/default_comm_hooks.cpp
)

if(NOT WIN32)
  list(APPEND C10D_CPU_SRCS
    ${CMAKE_CURRENT_SOURCE_DIR}/HashStore.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/ProcessGroupRoundRobin.cpp
    ${CMAKE_CURRENT_SOURCE_DIR}/TCPStore.cpp
  )
endif()

if(USE_C10D_MPI)
  list(APPEND C10D_CPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/ProcessGroupMPI.cpp)
endif()

if(USE_C10D_GLOO_CPU)
  list(APPEND
    C10D_CPU_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/ProcessGroupGloo.cpp ${CMAKE_CURRENT_SOURCE_DIR}/GlooDeviceFactory.cpp)
else()
  if(USE_C10D_GLOO_CUDA)
    list(APPEND C10D_CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/ProcessGroupGloo.cpp ${CMAKE_CURRENT_SOURCE_DIR}/GlooDeviceFactory.cpp)
  endif()
endif()

if(USE_C10D_NCCL)
  list(APPEND C10D_CUDA_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/ProcessGroupNCCL.cpp ${CMAKE_CURRENT_SOURCE_DIR}/NCCLUtils.cpp)
endif()

#############################
#       C10D_INCLUDE        #
#############################

# All include directories for building C10D source files
set(C10D_INCLUDE)

list(APPEND C10D_INCLUDE ${CMAKE_CURRENT_SOURCE_DIR}/..) # For <c10d/...>
if(USE_C10D_MPI)
  list(APPEND C10D_INCLUDE ${MPI_INCLUDE_PATH})
endif()

if(USE_TBB)
  list(APPEND C10D_INCLUDE ${TBB_ROOT_DIR}/include)
endif()

#############################
#        C10D_*_LIBS        #
#############################

# C10D_{CPU, CUDA}_LIBS includes required third-party library dependencies
set(C10D_CPU_LIBS)
set(C10D_CUDA_LIBS)
if(USE_C10D_MPI)
  list(APPEND C10D_CPU_LIBS ${MPI_LIBRARIES})
  list(APPEND C10D_CUDA_LIBS ${MPI_LIBRARIES})
endif()

if(USE_C10D_GLOO_CPU)
  list(APPEND C10D_CPU_LIBS gloo)
endif()

if(USE_C10D_GLOO_CUDA)
  list(APPEND C10D_CUDA_LIBS gloo_cuda)
endif()

#############################
#   Pass Back Build Data    #
#############################

# Export c10d cpu and cuda source file lists to parent scopes
set(C10D_CPU_SRCS ${C10D_CPU_SRCS} PARENT_SCOPE)
set(C10D_CUDA_SRCS ${C10D_CUDA_SRCS} PARENT_SCOPE)
set(C10D_INCLUDE ${C10D_INCLUDE} PARENT_SCOPE)
set(C10D_CPU_LIBS ${C10D_CPU_LIBS} PARENT_SCOPE)
set(C10D_CUDA_LIBS ${C10D_CUDA_LIBS} PARENT_SCOPE)

#############################
#     Examples and Tests    #
#############################

option(BUILD_EXAMPLES "Build examples" OFF)
if(BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()