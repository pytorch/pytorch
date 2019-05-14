caffe2_binary_target("convert_caffe_image_db.cc")
caffe2_binary_target("convert_db.cc")
caffe2_binary_target("make_cifar_db.cc")
caffe2_binary_target("make_mnist_db.cc")
if (NOT ANDROID)
  caffe2_binary_target("parallel_info.cc")
  target_include_directories(parallel_info PUBLIC
    ${CMAKE_BINARY_DIR}/aten/src) # provides "ATen/TypeExtendedInterface.h" to ATen.h
endif()
caffe2_binary_target("predictor_verifier.cc")
caffe2_binary_target("print_registered_core_operators.cc")
caffe2_binary_target("run_plan.cc")
caffe2_binary_target("speed_benchmark.cc")
caffe2_binary_target("split_db.cc")

caffe2_binary_target("db_throughput.cc")


if (USE_CUDA)
  caffe2_binary_target("inspect_gpu.cc")
  target_link_libraries(inspect_gpu ${CUDA_LIBRARIES})
  caffe2_binary_target("print_core_object_sizes_gpu.cc")

  if (BUILD_TEST)
    # Core overhead benchmark
    caffe2_binary_target("core_overhead_benchmark_gpu.cc")
    target_link_libraries(core_overhead_benchmark_gpu benchmark ${CUDA_curand_LIBRARY})
  endif()
endif()

if (USE_ROCM)
  caffe2_hip_binary_target("hip/inspect_gpu.cc")
  caffe2_hip_binary_target("hip/print_core_object_sizes_gpu.cc")

  if (BUILD_TEST)
    # Core overhead benchmark
    caffe2_hip_binary_target("hip/core_overhead_benchmark_gpu.cc")
    target_link_libraries(core_overhead_benchmark_gpu benchmark)
  endif()
endif()

if (USE_ZMQ)
  caffe2_binary_target("zmq_feeder.cc")
  target_link_libraries(zmq_feeder ${ZMQ_LIBRARIES})
endif()

if(USE_MPI)
  caffe2_binary_target("run_plan_mpi.cc")
  target_link_libraries(run_plan_mpi ${MPI_CXX_LIBRARIES})
endif()

if (USE_OPENCV AND USE_LEVELDB)
  caffe2_binary_target("convert_encoded_to_raw_leveldb.cc")
  target_link_libraries(
      convert_encoded_to_raw_leveldb
      ${OpenCV_LIBS} ${LevelDB_LIBRARIES} ${Snappy_LIBRARIES})
endif()

if (USE_OPENCV)
  caffe2_binary_target("make_image_db.cc")
  target_link_libraries(make_image_db ${OpenCV_LIBS})
  caffe2_binary_target("convert_image_to_tensor.cc")
  target_link_libraries(convert_image_to_tensor ${OpenCV_LIBS})
endif()

if (USE_OBSERVERS)
  caffe2_binary_target(caffe2_benchmark "caffe2_benchmark.cc" "benchmark_helper.cc")
endif()

if (USE_OBSERVERS AND USE_OPENCV)
  caffe2_binary_target("convert_and_benchmark.cc")
  target_link_libraries(convert_and_benchmark ${OpenCV_LIBS})
endif()

# ---[ tutorials
caffe2_binary_target("tutorial_blob.cc")
