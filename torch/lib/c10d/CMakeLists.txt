cmake_minimum_required(VERSION 3.2 FATAL_ERROR)

# Find modules.
list(APPEND CMAKE_MODULE_PATH
  /usr/lib/x86_64-linux-gnu/
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/public
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules
  ${CMAKE_CURRENT_SOURCE_DIR}/../../../cmake/Modules_CUDA_fix)

# Relies on CMAKE_INSTALL_PREFIX to be set to ../tmp_install.
# It then finds $PREFIX/share/cmake/ATen/ATenConfig.cmake,
# which defines ATEN_INCLUDE_DIR and ATEN_LIBRARIES.
find_package(Caffe2 REQUIRED)
if(NOT Caffe2_FOUND)
  message(FATAL_ERROR "Caffe2 not found")
endif()

find_package(Gloo REQUIRED)
if(Gloo_FOUND)
  message(STATUS "Gloo_LIBRARY: ${Gloo_LIBRARY}")
  message(STATUS "Gloo_NATIVE_LIBRARY: ${Gloo_NATIVE_LIBRARY}")
  message(STATUS "Gloo_INCLUDE_DIR: ${Gloo_INCLUDE_DIR}")
else()
  message(FATAL_ERROR "Gloo not found")
endif()

find_package(MPI)
if(MPI_FOUND)
  message(STATUS "MPI_INCLUDE_PATH: ${MPI_INCLUDE_PATH}")
  message(STATUS "MPI_LIBRARIES: ${MPI_LIBRARIES}")
  message(STATUS "MPIEXEC: ${MPIEXEC}")
else()
  message(STATUS "Not able to find MPI, will compile c10d without MPI support")
endif()

find_package(NCCL)
if(NCCL_FOUND)
  message(STATUS "NCCL_LIBRARIES: ${NCCL_LIBRARIES}")
  message(STATUS "NCCL_INCLUDE_DIRS: ${NCCL_INCLUDE_DIRS}")
  IF(NCCL_MAJOR_VERSION AND NOT (NCCL_MAJOR_VERSION LESS 2))
    message(STATUS "NCCL Version 2 or higher found, will "
                   "compile with NCCL distributed backend")
    SET(DISTRIBUTED_NCCL_FOUND TRUE)
  else()
    message(STATUS "Found NCCL, but the NCCL version is either not 2+ or not "
                   "determinable, will not compile with NCCL distributed "
                   "backend")
  endif()
else()
  message(STATUS "Not able to find NCCL, will not "
                 "compile with NCCL distributed backend")
endif()

find_package(CUDA REQUIRED)
if(NOT CUDA_FOUND)
  message(FATAL_ERROR "CUDA not found")
endif()

function(copy_header file)
  configure_file(${file} ${CMAKE_BINARY_DIR}/include/c10d/${file} COPYONLY)
endfunction()

if(DISTRIBUTED_NCCL_FOUND)
  option(USE_C10D_NCCL "USE C10D NCCL" ON)
endif()
if(MPI_FOUND)
  option(USE_C10D_MPI "USE C10D MPI" ON)
endif()
configure_file(cmake/Def.hpp.in ${CMAKE_BINARY_DIR}/include/c10d/Def.hpp @ONLY)

set(C10D_SRCS
  CUDAUtils.cpp
  FileStore.cpp
  ProcessGroup.cpp
  Store.cpp
  TCPStore.cpp
  Utils.cpp
  ProcessGroupGloo.cpp
  )

set(C10D_LIBS
  caffe2_gpu
  ${Gloo_LIBRARY}
  ${Gloo_NATIVE_LIBRARY}
  )

if(DISTRIBUTED_NCCL_FOUND)
  list(APPEND C10D_SRCS ProcessGroupNCCL.cpp)
  list(APPEND C10D_LIBS ${NCCL_LIBRARIES})
endif()

if(MPI_FOUND)
  list(APPEND C10D_SRCS ProcessGroupMPI.cpp)
  list(APPEND C10D_LIBS ${MPI_LIBRARIES})
endif()

add_library(c10d ${C10D_SRCS})
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

# c10d links to Caffe2/ATen, but the targets don't add TH/THC to the include path
target_include_directories(c10d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../tmp_install/include/TH)
target_include_directories(c10d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../tmp_install/include/THC)
# For <c10d/...>
target_include_directories(c10d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/..)
# For torch/csrc/utils/hash.h and torch/csrc/utils/functional.h
target_include_directories(c10d PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/../../..)
# For <gloo/...>
target_include_directories(c10d PUBLIC ${GLOO_INCLUDE_DIR})

copy_header(CUDAUtils.hpp)
copy_header(FileStore.hpp)
copy_header(ProcessGroup.hpp)
copy_header(Store.hpp)
copy_header(TCPStore.hpp)
copy_header(Types.hpp)
copy_header(Utils.hpp)
copy_header(ProcessGroupGloo.hpp)

if(DISTRIBUTED_NCCL_FOUND)
  target_include_directories(c10d PUBLIC ${NCCL_INCLUDE_DIRS})
  copy_header(ProcessGroupNCCL.hpp)
  copy_header(NCCLUtils.hpp)
endif()

if(MPI_FOUND)
  target_include_directories(c10d PUBLIC ${MPI_INCLUDE_PATH})
  copy_header(ProcessGroupMPI.hpp)
endif()

target_link_libraries(c10d PUBLIC ${C10D_LIBS})
target_include_directories(c10d PRIVATE ${CMAKE_BINARY_DIR}/include)

install(TARGETS c10d ARCHIVE DESTINATION lib)

option(BUILD_EXAMPLES "Build examples" ON)
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
