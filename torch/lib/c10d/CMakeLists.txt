cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

# Find modules.
list(APPEND CMAKE_MODULE_PATH
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/public
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules_CUDA_fix)

if(USE_CUDA)
  find_package(CUDA)
  if(CUDA_FOUND)
    set(C10D_USE_CUDA true)
    message(STATUS "Building C10D with CUDA support")
    add_definitions(-DUSE_CUDA=1)
  else()
    set(C10D_USE_CUDA false)
    message(STATUS "CUDA not found, building C10D without CUDA support")
  endif()
else()
  set(C10D_USE_CUDA false)
  message(STATUS "Building C10D without CUDA support")
endif()

find_package(MPI)
if(MPI_FOUND)
  message(STATUS "MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
  message(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")
  message(STATUS "MPIEXEC: ${MPIEXEC}")
else()
  message(STATUS "Not able to find MPI, will compile c10d without MPI support")
endif()

function(copy_header file)
  configure_file(${file} ${CMAKE_BINARY_DIR}/include/c10d/${file} COPYONLY)
endfunction()

if(USE_NCCL)
  option(USE_C10D_NCCL "USE C10D NCCL" ON)
endif()

if(MPI_FOUND)
  option(USE_C10D_MPI "USE C10D MPI" ON)
endif()

set(C10D_SRCS
  FileStore.cpp
  ProcessGroup.cpp
  Store.cpp
  PrefixStore.cpp
  TCPStore.cpp
  Utils.cpp
  ProcessGroupGloo.cpp
  )

if(C10D_USE_CUDA)
  set(C10D_LIBS
    caffe2_gpu
    )
else()
  set(C10D_LIBS
    caffe2
    )
endif()


if(USE_C10D_NCCL)
  list(APPEND C10D_SRCS ProcessGroupNCCL.cpp)
  list(APPEND C10D_LIBS __caffe2_nccl)
endif()

if(USE_C10D_MPI)
  list(APPEND C10D_SRCS ProcessGroupMPI.cpp)
  list(APPEND C10D_LIBS ${MPI_LIBRARIES})
endif()

add_library(c10d STATIC ${C10D_SRCS})
set_property(TARGET c10d PROPERTY POSITION_INDEPENDENT_CODE ON)
set_property(TARGET c10d PROPERTY CXX_STANDARD 11)
target_compile_options(c10d PUBLIC
  -Wall
  -Wextra
  -Wno-unused-parameter
  -Wno-missing-field-initializers
  -Wno-write-strings
  -Wno-unknown-pragmas
  )


add_dependencies(c10d caffe2)

if (USE_GLOO)
  add_dependencies(c10d gloo)
  target_link_libraries(c10d PUBLIC gloo)
  if (USE_CUDA)
    target_link_libraries(c10d PUBLIC gloo_cuda)
  endif()
endif()

target_include_directories(c10d PUBLIC
  ${CMAKE_BINARY_DIR}/aten/src # provides "ATen/TypeExtendedInterface.h" to ATen.h
  ${CMAKE_BINARY_DIR}/caffe2/aten/src # provides <TH/THGeneral.h> to THC.h
  )

# For <c10d/...>
target_include_directories(c10d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
# For torch/csrc/utils/hash.h
target_include_directories(c10d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

if(USE_C10D_NCCL)
  target_compile_definitions(c10d INTERFACE USE_C10D_NCCL)
endif()

if(USE_C10D_MPI)
  target_compile_definitions(c10d INTERFACE USE_C10D_MPI)
endif()

copy_header(FileStore.hpp)
copy_header(PrefixStore.hpp)
copy_header(ProcessGroup.hpp)
copy_header(Store.hpp)
copy_header(TCPStore.hpp)
copy_header(Types.hpp)
copy_header(Utils.hpp)
copy_header(ProcessGroupGloo.hpp)

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
