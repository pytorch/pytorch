# Excludes
set(Caffe2_DIR_EXCLUDES
        "/binaries"
        "/contrib"
        "/distributed"
        "/mpi"
        "/image"
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
add_library(Caffe2_CPU SHARED ${Caffe2_CPU_SRCS})
target_link_libraries(Caffe2_CPU Caffe2_PROTO)

# CUDA library
# TODO(slayton): Move this somewhere sane
find_package(CUDA)
LIST(APPEND CUDA_NVCC_FLAGS -Xcompiler -std=c++11)
LIST(APPEND CUDA_NVCC_FLAGS -gencode arch=compute_52,compute=sm_52)
CUDA_ADD_LIBRARY(Caffe2_GPU SHARED ${Caffe2_GPU_SRCS})
target_link_libraries(Caffe2_GPU Caffe2_PROTO glog gflags)

# Compile test binaries.
add_executable(Caffe2_CPU_TEST ${Caffe2_CPU_TEST_SRCS})

target_link_libraries(Caffe2_CPU_TEST Caffe2_CPU gtest glog atlas cblas)
target_compile_features(Caffe2_CPU_TEST PRIVATE cxx_range_for)

# ---[ Python
add_subdirectory(python)
