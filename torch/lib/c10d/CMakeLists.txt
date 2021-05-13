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

if(USE_TBB)
  include_directories(${TBB_ROOT_DIR}/include)
endif()

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
  ParamCommsUtils.cpp
  PrefixStore.cpp
  ProcessGroup.cpp
  sequence_num.cpp
  Store.cpp
  TCPStore.cpp
  Utils.cpp
  )

if(NOT WIN32)
  list(APPEND C10D_SRCS HashStore.cpp ProcessGroupRoundRobin.cpp)
endif()

set(C10D_LIBS torch)

if(USE_C10D_NCCL)
  list(APPEND C10D_SRCS ProcessGroupNCCL.cpp NCCLUtils.cpp)
  list(APPEND C10D_LIBS __caffe2_nccl)
endif()

if(USE_C10D_MPI)
  list(APPEND C10D_SRCS ProcessGroupMPI.cpp)
  if(CMAKE_CXX_COMPILER_ID MATCHES "Clang" OR CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
    set_source_files_properties(ProcessGroupMPI.cpp PROPERTIES COMPILE_FLAGS -Wno-deprecated-declarations)
  endif()
  list(APPEND C10D_LIBS ${MPI_LIBRARIES})
endif()

if(USE_C10D_GLOO)
  list(APPEND C10D_SRCS ProcessGroupGloo.cpp GlooDeviceFactory.cpp ProcessGroupWrapper.cpp)
  list(APPEND C10D_LIBS gloo)
  if(USE_CUDA)
    list(APPEND C10D_LIBS gloo_cuda)
  endif()
endif()

add_library(c10d STATIC ${C10D_SRCS})
set_property(TARGET c10d PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET c10d PROPERTY CXX_STANDARD 14)

if(NOT MSVC)
  target_compile_options(c10d PUBLIC
    -Wall
    -Wextra
    -Wno-unused-parameter
    -Wno-missing-field-initializers
    -Wno-write-strings
    -Wno-unknown-pragmas
    )
endif()

add_dependencies(c10d torch)

if(USE_C10D_GLOO)
  add_dependencies(c10d gloo)
  if(USE_CUDA)
    add_dependencies(c10d gloo_cuda)
  endif()
endif()

target_include_directories(c10d PUBLIC
  ${CMAKE_BINARY_DIR}/aten/src # provides "ATen/TypeExtendedInterface.h" to ATen.h
  ${CMAKE_BINARY_DIR}/caffe2/aten/src # provides <TH/THGeneral.h> to THC.h
  )

# For <c10d/...>
target_include_directories(c10d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)

if(USE_C10D_NCCL)
  target_compile_definitions(c10d INTERFACE USE_C10D_NCCL)
endif()

if(USE_C10D_MPI)
  target_compile_definitions(c10d INTERFACE USE_C10D_MPI)
endif()

if(USE_C10D_GLOO)
  target_compile_definitions(c10d INTERFACE USE_C10D_GLOO)
endif()

copy_header(FileStore.hpp)
copy_header(ParamCommsUtils.hpp)
copy_header(PrefixStore.hpp)
copy_header(ProcessGroup.hpp)
copy_header(Store.hpp)
copy_header(TCPStore.hpp)
copy_header(Types.hpp)
copy_header(Utils.hpp)
copy_header(sequence_num.hpp)
if(USE_GLOO)
  copy_header(ProcessGroupGloo.hpp)
  copy_header(GlooDeviceFactory.hpp)
  copy_header(ProcessGroupWrapper.hpp)
endif()
if(NOT WIN32)
  copy_header(HashStore.hpp)
  copy_header(UnixSockUtils.hpp)
else()
  copy_header(WinSockUtils.hpp)
endif()

if(USE_C10D_NCCL)
  copy_header(ProcessGroupNCCL.hpp)
  copy_header(NCCLUtils.hpp)
endif()

if(USE_C10D_MPI)
  target_include_directories(c10d PUBLIC ${MPI_INCLUDE_PATH})
  copy_header(ProcessGroupMPI.hpp)
endif()

target_link_libraries(c10d PUBLIC ${C10D_LIBS})

install(TARGETS c10d DESTINATION lib)

option(BUILD_EXAMPLES "Build examples" OFF)
if(BUILD_EXAMPLES)
  add_subdirectory(example)
endif()

option(BUILD_TEST "Build tests" ON)
if(BUILD_TEST)
  enable_testing()
  add_subdirectory(test)
endif()

# Install all header files that were prepared in the build directory
install(DIRECTORY ${CMAKE_BINARY_DIR}/include/ DESTINATION include)
