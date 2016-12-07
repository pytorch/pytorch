# Excludes
set(Caffe2_DIR_EXCLUDES
        "/binaries"
        "/contrib"
        "/distributed"
        "/mpi"
        "/experiments"
        "/python"
)
set(Caffe2_FILE_EXCLUDES
        "/core/init_omp.cc"
        "/operators/fully_connected_op_sparse.cc"
        "/test/caffe2_gtest_main.cc"
        "/core/init_omp.cc"
        "/db/db_test.cc"
        "/db/leveldb.cc"
        "/db/lmdb.cc"
        "/db/rocksdb.cc"
        "/db/zmqdb.cc"
)

set(Caffe2_CPU_FILE_EXCLUDES
        "/contrib/nervana/nervana_fc_op_gpu_test.cc"
        "/core/blob_gpu_test.cc"
        "/core/context_gpu_test.cc"
        "/mpi/mpi_gpu_test.cc"
        "/operators/conv_op_cache_cudnn_test.cc"
        "/operators/elementwise_op_gpu_test.cc"
        "/operators/fully_connected_op_gpu_test.cc"
        "/operators/operator_fallback_gpu_test.cc"
        "/operators/utility_ops_gpu_test.cc"
)

# Expand all the exludes.
prepend(Caffe2_DIR_EXCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_DIR_EXCLUDES})
prepend(Caffe2_FILE_EXCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_FILE_EXCLUDES})
prepend(Caffe2_CPU_FILE_EXCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_CPU_FILE_EXCLUDES})

# Get all source directories.
# TODO(bwasti): GLOB is not considered best practice.
file(GLOB Caffe2_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/*)

# Exclude all excluded directories.
exclude(Caffe2_DIRS "${Caffe2_DIRS}" ${Caffe2_DIR_EXCLUDES})

# Get all source files from remaining directories.

# GPU files
foreach(dir ${Caffe2_DIRS})
        file(GLOB tmp ${dir}/*_cudnn.cc)
        set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
endforeach()
foreach(dir ${Caffe2_DIRS})
        file(GLOB tmp ${dir}/*_gpu.cc)
        set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
endforeach()
exclude(Caffe2_GPU_SRCS "${Caffe2_GPU_SRCS}" ${Caffe2_FILE_EXCLUDES})

# CPU files.
foreach(dir ${Caffe2_DIRS})
        file(GLOB tmp ${dir}/*.cc)
        set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})
endforeach()
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_FILE_EXCLUDES})
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_CPU_FILE_EXCLUDES})
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_GPU_SRCS})

# CPU test files.
set(Caffe2_CPU_TEST_SRCS
        "/test/caffe2_gtest_main.cc"
)
prepend(Caffe2_CPU_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_CPU_TEST_SRCS})

# Compile protobufs.
add_subdirectory(proto)
include_directories(BEFORE ${CMAKE_BINARY_DIR})

# Compile exposed libraries.
LIST(APPEND CMAKE_CXX_FLAGS "--std=c++11 -fPIC")
add_library(Caffe2_CPU ${Caffe2_CPU_SRCS})
target_link_libraries(Caffe2_CPU Caffe2_PROTO)

# CUDA library
# TODO(slayton): Move this somewhere sane
find_package(CUDA)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler -std=c++11)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,compute=sm_52)
CUDA_ADD_LIBRARY(Caffe2_GPU ${Caffe2_GPU_SRCS})
target_link_libraries(Caffe2_GPU Caffe2_PROTO glog gflags )

# Compile test binaries.
add_executable(Caffe2_CPU_TEST ${Caffe2_CPU_TEST_SRCS})
# TODO: Make BLAS backend interchangeable
target_link_libraries(Caffe2_CPU_TEST Caffe2_CPU gtest glog atlas cblas ${Caffe2_LINKER_LIBS})
target_compile_features(Caffe2_CPU_TEST PRIVATE cxx_range_for)

# ---[ Python
add_subdirectory(python)

# ---[ All binaries
file(GLOB Caffe2_BINARY_SRCS ${CMAKE_CURRENT_SOURCE_DIR}/binaries/*.cc)

# ZMQ src
set(Caffe2_ZMQ_BINARY_SRC
    "zmq_feeder.cc"
)
# Prune out ZMQ if no support
if(NOT USE_ZMQ)
  message(STATUS "Excluding ZMQ binary")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_ZMQ_BINARY_SRC}")
  exclude(Caffe2_BINARY_SRCS "${Caffe2_BINARY_SRCS}" "${tmp}")
endif()

# MPI-based binaries
set(Caffe2_MPI_BINARY_SRC
    "fb_run_plan_mpi.cc"
    "run_plan_mpi.cc"
)
if(NOT USE_MPI)
  message(STATUS "Excluding MPI binaries")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_MPI_BINARY_SRC}")
  exclude(Caffe2_BINARY_SRCS "${Caffe2_BINARY_SRCS}" "${tmp}")
endif()

foreach(binary_src ${Caffe2_BINARY_SRCS})
  get_filename_component(bin_name ${binary_src} NAME_WE)
  # message(STATUS ${bin_name})
  # message(STATUS "${CMAKE_CURRENT_SOURCE_DIR}/"${binary_src})
  add_executable(${bin_name} "${CMAKE_CURRENT_SOURCE_DIR}/"${binary_src})
  target_link_libraries(${bin_name} Caffe2_CPU Caffe2_GPU Caffe_PROTO Caffe2_PROTO glog atlas cblas ${Caffe2_LINKER_LIBS})
endforeach()


