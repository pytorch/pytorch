# ---[ Excludes
set(Caffe2_DIR_EXCLUDES
        "/binaries"      # handled below
        "/contrib"       # To be handled
        "/distributed"   # Exclude for now
        "/mpi"           # Exclude for now
        "/experiments"   # Exclude for now
        "/python"        # handled below
        "/image"         # Exclude for now
)

set(Caffe2_FILE_EXCLUDES
        "/operators/fully_connected_op_sparse.cc"
        "/test/caffe2_gtest_main.cc"
        "/db/db_test.cc"
        "/db/rocksdb.cc"
        "/db/zmqdb.cc"
        "/utils/mkl_utils_test.cc"
        "/mpi/mpi_test.cc"
)

# ---[ Specific builds
set(Caffe2_LEVELDB_SRCS
        "/db/leveldb.cc"
)
set(Caffe2_LMDB_SRCS
        "/db/lmdb.cc"
)

set(Caffe2_CPU_TEST_SRCS
    "/test/caffe2_gtest_main.cc"
    "/core/parallel_net_test.cc"
    "/core/timer_test.cc"
    "/core/workspace_test.cc"
    "/core/registry_test.cc"
    "/core/context_test.cc"
    "/core/init_test.cc"
    "/core/operator_test.cc"
    "/core/typeid_test.cc"
    "/core/predictor_test.cc"
    "/core/operator_schema_test.cc"
    "/core/logging_test.cc"
    "/core/blob_test.cc"
    "/core/net_test.cc"
    "/db/db_test.cc"
    "/operators/utility_ops_test.cc"
    "/operators/elementwise_op_test.cc"
    "/operators/fully_connected_op_test.cc"
    "/operators/text_file_reader_utils_test.cc"
    "/utils/proto_utils_test.cc"
    "/utils/math_test.cc"
    "/utils/simple_queue_test.cc"
    "/utils/mkl_utils_test.cc"
    "/utils/fixed_divisor_test.cc"
)

set(Caffe2_GPU_TEST_SRCS
    "/contrib/nervana/nervana_fc_op_gpu_test.cc"
    "/core/blob_gpu_test.cc"
    "/mpi/mpi_gpu_test.cc"
    "/core/context_gpu_test.cc"
    "/operators/conv_op_cache_cudnn_test.cc"
    "/operators/operator_fallback_gpu_test.cc"
    "/operators/utility_ops_gpu_test.cc"
    "/operators/fully_connected_op_gpu_test.cc"
    "/operators/elementwise_op_gpu_test.cc"
)

# Expand all the exludes.
prepend(Caffe2_DIR_EXCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_DIR_EXCLUDES})
prepend(Caffe2_FILE_EXCLUDES ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_FILE_EXCLUDES})
prepend(Caffe2_CPU_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_CPU_TEST_SRCS})
prepend(Caffe2_GPU_TEST_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_GPU_TEST_SRCS})
prepend(Caffe2_LEVELDB_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_LEVELDB_SRCS})
prepend(Caffe2_LMDB_SRCS ${CMAKE_CURRENT_SOURCE_DIR} ${Caffe2_LMDB_SRCS})

# ---[ List of libraries to link with
set(Caffe2_LINK_LIBS Caffe_PROTO)

# Get all source directories.
# TODO(bwasti): GLOB is not considered best practice.
file(GLOB Caffe2_DIRS ${CMAKE_CURRENT_SOURCE_DIR}/*)

# Exclude all excluded directories.
exclude(Caffe2_DIRS "${Caffe2_DIRS}" ${Caffe2_DIR_EXCLUDES})

# Get all source files from remaining directories.

# ---[ GPU files
# ------[ cuDNN
foreach(dir ${Caffe2_DIRS})
        file(GLOB tmp ${dir}/*_cudnn.cc)
        set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
endforeach()
# ------[ general GPU
foreach(dir ${Caffe2_DIRS})
        file(GLOB tmp ${dir}/*_gpu.cc)
        set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
endforeach()
# ------[ CUDA sources
foreach(dir ${Caffe2_DIRS})
        file(GLOB tmp ${dir}/*.cu)
        set(Caffe2_GPU_SRCS ${Caffe2_GPU_SRCS} ${tmp})
endforeach()
exclude(Caffe2_GPU_SRCS "${Caffe2_GPU_SRCS}" ${Caffe2_FILE_EXCLUDES})
exclude(Caffe2_GPU_SRCS "${Caffe2_GPU_SRCS}" ${Caffe2_GPU_TEST_SRCS})

# CPU files.
foreach(dir ${Caffe2_DIRS})
        file(GLOB tmp ${dir}/*.cc)
        set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} ${tmp})
endforeach()
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_FILE_EXCLUDES})
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_GPU_SRCS})
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_CPU_TEST_SRCS})
exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_GPU_TEST_SRCS})
if (NOT USE_LEVELDB)
        exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_LEVELDB_SRCS})
endif()
if (NOT USE_LMDB)
        exclude(Caffe2_CPU_SRCS "${Caffe2_CPU_SRCS}" ${Caffe2_LMDB_SRCS})
endif()

# Compile protobufs.
add_subdirectory(proto)
include_directories(BEFORE ${CMAKE_BINARY_DIR})
list(APPEND Caffe2_LINK_LIBS Caffe2_PROTO)

# Compile exposed libraries.
LIST(APPEND CMAKE_CXX_FLAGS "-std=c++11 -fPIC")
add_library(Caffe2_CPU ${Caffe2_CPU_SRCS})
target_link_libraries(Caffe2_CPU ${Caffe2_LINK_LIBS} glog gflags ${Caffe2_LINKER_LIBS})
install(TARGETS Caffe2_CPU DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
list(APPEND Caffe2_LINK_LIBS Caffe2_CPU)

# ---[ CUDA library
if(HAVE_CUDA)
  CUDA_ADD_LIBRARY(Caffe2_GPU ${Caffe2_GPU_SRCS})
  target_link_libraries(Caffe2_GPU ${Caffe2_LINK_LIBS} glog gflags ${Caffe2_LINKER_LIBS})
  install(TARGETS Caffe2_GPU DESTINATION ${CMAKE_INSTALL_PREFIX}/lib)
  link_directories(${CMAKE_INSTALL_PREFIX}/lib)
  list(APPEND Caffe2_LINK_LIBS Caffe2_GPU)
  list(APPEND Caffe2_LINK Caffe2_GPU)
endif()

# Compile test binaries.
foreach(test_src ${Caffe2_CPU_TEST_SRCS})
  get_filename_component(test_name ${test_src} NAME_WE)
  add_executable(${test_name} "${test_src}")
  target_link_libraries(${test_name} ${Caffe2_LINK} gtest_main)
  target_compile_features(${test_name} PRIVATE cxx_range_for)
  install(TARGETS ${test_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/test)
endforeach()

# ---[ Python
add_subdirectory(python)

# ---[ Binaries
set(Caffe2_CPU_BINARY_SRCS
    "convert_caffe_image_db.cc"
    "convert_db.cc"
    "db_throughput.cc"
    "fb_run_plan_mpi.cc"
    "make_cifar_db.cc"
    "make_image_db.cc"
    "make_mnist_db.cc"
    "predictor_verifier.cc"
    "print_registered_core_operators.cc"
    "run_plan.cc"
    "run_plan_mpi.cc"
    "speed_benchmark.cc"
    "split_db.cc"
    "zmq_feeder.cc"
)

set(Caffe2_GPU_BINARY_SRCS
    "core_overhead_benchmark.cc"
    "inspect_gpus.cc"
    "print_core_object_sizes.cc"
)

prepend(Caffe2_CPU_BINARY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_CPU_BINARY_SRCS}")
prepend(Caffe2_GPU_BINARY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_GPU_BINARY_SRCS}")

# ZMQ src
set(Caffe2_ZMQ_BINARY_SRCS
    "zmq_feeder.cc"
)
# Prune out ZMQ if no support
if(NOT USE_ZMQ)
  message(STATUS "Excluding ZMQ binary")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_ZMQ_BINARY_SRCS}")
  exclude(Caffe2_CPU_BINARY_SRCS "${Caffe2_CPU_BINARY_SRCS}" "${tmp}")
  exclude(Caffe2_GPU_BINARY_SRCS "${Caffe2_GPU_BINARY_SRCS}" "${tmp}")
endif()

# MPI-based binaries
set(Caffe2_MPI_BINARY_SRCS
    "fb_run_plan_mpi.cc"
    "run_plan_mpi.cc"
)
if(NOT USE_MPI)
  message(STATUS "Excluding MPI binaries")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_MPI_BINARY_SRCS}")
  exclude(Caffe2_CPU_BINARY_SRCS "${Caffe2_CPU_BINARY_SRCS}" "${tmp}")
  exclude(Caffe2_GPU_BINARY_SRCS "${Caffe2_GPU_BINARY_SRCS}" "${tmp}")
endif()

# OpenCV-based binaries
set(Caffe2_OPENCV_BINARY_SRCS
    "convert_encoded_to_raw_leveldb.cc"
    "make_image_db.cc"
)
if(NOT USE_OPENCV)
  message(STATUS "Excluding OPENCV binaries")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/binaries/" "${Caffe2_OPENCV_BINARY_SRCS}")
  exclude(Caffe2_CPU_BINARY_SRCS "${Caffe2_CPU_BINARY_SRCS}" "${tmp}")
  exclude(Caffe2_GPU_BINARY_SRCS "${Caffe2_GPU_BINARY_SRCS}" "${tmp}")
endif()

foreach(binary_src ${Caffe2_CPU_BINARY_SRCS})
  get_filename_component(bin_name ${binary_src} NAME_WE)
  add_executable(${bin_name} ${binary_src})
  target_link_libraries(${bin_name} ${Caffe2_LINK} glog benchmark ${Caffe2_LINKER_LIBS})
  install(TARGETS ${bin_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/binaries)
endforeach()

if(HAVE_CUDA)
  foreach(binary_src ${Caffe2_GPU_BINARY_SRCS})
    get_filename_component(bin_name ${binary_src} NAME_WE)
    add_executable(${bin_name} ${binary_src})
    target_link_libraries(${bin_name} ${Caffe2_LINK} glog cnmem benchmark ${Caffe2_LINKER_LIBS})
    install(TARGETS ${bin_name} DESTINATION ${CMAKE_INSTALL_PREFIX}/binaries)
  endforeach()
endif()
