
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

prepend(Caffe2_CPU_BINARY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/" "${Caffe2_CPU_BINARY_SRCS}")
prepend(Caffe2_GPU_BINARY_SRCS "${CMAKE_CURRENT_SOURCE_DIR}/" "${Caffe2_GPU_BINARY_SRCS}")

# ZMQ src
set(Caffe2_ZMQ_BINARY_SRCS
    "zmq_feeder.cc"
)
# Prune out ZMQ if no support
if(NOT USE_ZMQ)
  message(STATUS "Excluding ZMQ binary")
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/" "${Caffe2_ZMQ_BINARY_SRCS}")
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
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/" "${Caffe2_MPI_BINARY_SRCS}")
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
  prepend(tmp "${CMAKE_CURRENT_SOURCE_DIR}/" "${Caffe2_OPENCV_BINARY_SRCS}")
  exclude(Caffe2_CPU_BINARY_SRCS "${Caffe2_CPU_BINARY_SRCS}" "${tmp}")
  exclude(Caffe2_GPU_BINARY_SRCS "${Caffe2_GPU_BINARY_SRCS}" "${tmp}")
endif()

set(Caffe2_CPU_BINARY_SRCS ${Caffe2_CPU_BINARY_SRCS} PARENT_SCOPE)
set(Caffe2_GPU_BINARY_SRCS ${Caffe2_GPU_BINARY_SRCS} PARENT_SCOPE)
