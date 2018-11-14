# ---[ AVX2 Ops
set(caffe2_dnnlowp_avx2_ops_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/conv_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/elementwise_sum_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/fully_connected_fake_lowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/group_norm_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/relu_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/transpose.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/dnnlowp.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/norm_minimization_avx2.cc")

# ---[ CPU files only
list(APPEND Caffe2_CPU_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/activation_distribution_observer.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/batch_matmul_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/caffe2_dnnlowp_utils.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/channel_shuffle_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/concat_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/conv_dnnlowp_acc16_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/conv_relu_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/dequantize_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/dnnlowp_partition.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/elementwise_add_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/elementwise_linear_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/elementwise_mul_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/elementwise_sum_relu_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/fbgemm_pack_matrix_cache.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/fully_connected_dnnlowp_acc16_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/fully_connected_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/fully_connected_rowwise_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/lstm_unit_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/pool_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/quantize_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/sigmoid_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/tanh_dnnlowp_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/utility_dnnlowp_ops.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/dynamic_histogram.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/kl_minimization.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/norm_minimization.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/p99.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/sigmoid.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/tanh.cc")

#Common sources

# ---[ CPU test files
# TODO: fc_fake_lowp_test.cc needs avx flags
# sigmoid_test.cc doesn build; error: undefined Sigmoid and Compute
list(APPEND Caffe2_CPU_TEST_SRCS
  #"${CMAKE_CURRENT_SOURCE_DIR}/dynamic_histogram_test.cc"
  #"${CMAKE_CURRENT_SOURCE_DIR}/l2_minimization_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/requantization_test.cc")
  #"${CMAKE_CURRENT_SOURCE_DIR}/sigmoid_test.cc")
  #"${CMAKE_CURRENT_SOURCE_DIR}/tanh_test.cc")

if (NOT MSVC)
  add_library(caffe2_dnnlowp_avx2_ops OBJECT ${caffe2_dnnlowp_avx2_ops_SRCS})
  add_dependencies(caffe2_dnnlowp_avx2_ops fbgemm Caffe2_PROTO c10)
  target_include_directories(caffe2_dnnlowp_avx2_ops BEFORE
    PRIVATE $<BUILD_INTERFACE:${FBGEMM_SOURCE_DIR}/include>)
  set_property(SOURCE ${caffe2_dnnlowp_avx2_ops_SRCS}
    APPEND_STRING PROPERTY COMPILE_FLAGS " -mavx2 -mfma -mf16c -mxsave ")
  set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS}
    $<TARGET_OBJECTS:caffe2_dnnlowp_avx2_ops>)
endif()


# ---[ Send the lists to the parent scope.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
