# ---[ Binaries
set(Caffe2_CPU_BINARY_SRCS
    "convert_caffe_image_db.cc"
    "convert_db.cc"
    "db_throughput.cc"
    "make_cifar_db.cc"
    "make_mnist_db.cc"
    "predictor_verifier.cc"
    "print_registered_core_operators.cc"
    "run_plan.cc"
    "speed_benchmark.cc"
    "split_db.cc"
)

set(Caffe2_GPU_BINARY_SRCS
    # "core_overhead_benchmark.cc" # todo: will need to figure out its dependency because it relies on benchmark
    "inspect_gpus.cc"
    "print_core_object_sizes.cc"
)

prepend(Caffe2_CPU_BINARY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/" "${Caffe2_CPU_BINARY_SRCS}")
prepend(Caffe2_GPU_BINARY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/" "${Caffe2_GPU_BINARY_SRCS}")

# ZMQ src
set(Caffe2_ZMQ_BINARY_SRCS
    "zmq_feeder.cc"
)
# Prune out ZMQ if no support
if(USE_ZMQ AND ZMQ_FOUND)
  message(STATUS "Excluding ZMQ binary")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/" "${Caffe2_ZMQ_BINARY_SRCS}")
  list(APPEND Caffe2_CPU_BINARY_SRCS "${tmp}")
endif()

# MPI-based binaries
set(Caffe2_MPI_BINARY_SRCS
    "fb_run_plan_mpi.cc"
    "run_plan_mpi.cc"
)
if(USE_MPI AND MPI_CXX_FOUND)
  message(STATUS "Including MPI binaries")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/" "${Caffe2_MPI_BINARY_SRCS}")
  list(APPEND Caffe2_CPU_BINARY_SRCS "${tmp}")
endif()

# OpenCV-based binaries
set(Caffe2_OPENCV_BINARY_SRCS
    # Temporarily commented out because this depends on both opencv and
    # leveldb, so we should test both USE_LEVELDB and USE_OPENCV
    # "convert_encoded_to_raw_leveldb.cc"
    "make_image_db.cc"
)
if(USE_OPENCV AND OpenCV_FOUND)
  message(STATUS "Including OPENCV binaries")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/" "${Caffe2_OPENCV_BINARY_SRCS}")
  list(APPEND Caffe2_CPU_BINARY_SRCS "${tmp}")
endif()

set(Caffe2_CPU_BINARY_SRCS ${Caffe2_CPU_BINARY_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_BINARY_SRCS ${Caffe2_GPU_BINARY_SRCS} PARENT_SCOPE)
