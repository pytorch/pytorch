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
configure_file(cmake/Def.hpp.in ${CMAKE_BINARY_DIR}/include/c10d/Def.hpp @ONLY)

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


if(USE_NCCL)
  list(APPEND C10D_SRCS ProcessGroupNCCL.cpp)
  list(APPEND C10D_LIBS __caffe2_nccl)
endif()

if(MPI_FOUND)
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
  ${CMAKE_SOURCE_DIR}/aten/src/TH # provides "THAllocator.h" to THCGeneral.h
  ${CMAKE_SOURCE_DIR}/aten/src/THC # provides "THC.h" to ProcessGroupGloo.cpp
  ${CMAKE_BINARY_DIR}/aten/src # provides "ATen/TypeExtendedInterface.h" to ATen.h
  ${CMAKE_BINARY_DIR}/caffe2/aten/src # provides <TH/THGeneral.h> to THC.h
  ${CMAKE_BINARY_DIR}/caffe2/aten/src/THC # provides "THCGeneral.h" to THC.h
  )

# For <c10d/...>
target_include_directories(c10d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
# For torch/csrc/utils/hash.h and torch/csrc/utils/functional.h
target_include_directories(c10d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)

copy_header(FileStore.hpp)
copy_header(PrefixStore.hpp)
copy_header(ProcessGroup.hpp)
copy_header(Store.hpp)
copy_header(TCPStore.hpp)
copy_header(Types.hpp)
copy_header(Utils.hpp)
copy_header(ProcessGroupGloo.hpp)

if(USE_NCCL)
  copy_header(ProcessGroupNCCL.hpp)
  copy_header(NCCLUtils.hpp)
endif()

if(MPI_FOUND)
  target_include_directories(c10d PUBLIC ${MPI_INCLUDE_PATH})
  copy_header(ProcessGroupMPI.hpp)
  file(WRITE ${CMAKE_BINARY_DIR}/include/c10d/mpi_include_path "")
  foreach(MPI_INC ${MPI_INCLUDE_PATH})
    file(APPEND ${CMAKE_BINARY_DIR}/include/c10d/mpi_include_path ${MPI_INC}\n)
  endforeach()
endif()

target_link_libraries(c10d PUBLIC ${C10D_LIBS})
target_include_directories(c10d PRIVATE ${CMAKE_BINARY_DIR}/include)

install(TARGETS c10d
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)

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
