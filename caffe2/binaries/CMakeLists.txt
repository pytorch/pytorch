caffe2_binary_target("convert_caffe_image_db.cc")
caffe2_binary_target("convert_db.cc")
caffe2_binary_target("db_throughput.cc")
caffe2_binary_target("make_cifar_db.cc")
caffe2_binary_target("make_mnist_db.cc")
caffe2_binary_target("predictor_verifier.cc")
caffe2_binary_target("print_registered_core_operators.cc")
caffe2_binary_target("run_plan.cc")
caffe2_binary_target("speed_benchmark.cc")
caffe2_binary_target("split_db.cc")

if (USE_CUDA)
  caffe2_binary_target("inspect_gpus.cc")
  caffe2_binary_target("print_core_object_sizes.cc")
endif()

if (USE_ZMQ)
  caffe2_binary_target("zmq_feeder.cc")
endif()

if(USE_MPI)
  caffe2_binary_target("run_plan_mpi.cc")
endif()

if (USE_OPENCV AND USE_LEVELDB)
  caffe2_binary_target("convert_encoded_to_raw_leveldb.cc")
endif()

if (USE_OPENCV)
  caffe2_binary_target("make_image_db.cc")
endif()

# ---[ tutorials
caffe2_binary_target("tutorial_blob.cc")
