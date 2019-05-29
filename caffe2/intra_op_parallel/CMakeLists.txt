list(APPEND Caffe2_CPU_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/intra_op_parallel.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/net_tbb_task_graph.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/numa_all_reduce_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/parallel_adagrad_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/parallel_batch_matmul_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/parallel_fully_connected_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/parallel_lengths_reducer_ops.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/partition.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/tbb_fully_connected_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/tbb_numa_all_reduce_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/intra_op_parallel.cc")

if (CAFFE2_COMPILER_SUPPORTS_AVX2_EXTENSIONS)
  set(caffe2_intra_op_parallel_avx_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/numa_all_reduce_op_avx.cc")
  add_library(caffe2_intra_op_parallel_avx OBJECT ${caffe2_intra_op_parallel_avx_SRCS})
  set_property(SOURCE ${caffe2_intra_op_parallel_avx_SRCS}
    APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx ")
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS}
    $<TARGET_OBJECTS:caffe2_intra_op_parallel_avx>)
else()
  list(APPEND Caffe2_CPU_SRCS
    "${CMAKE_CURRENT_SOURCE_DIR}/numa_all_reduce_op_avx.cc")
endif()

# ---[ Send the lists to the parent scope.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
