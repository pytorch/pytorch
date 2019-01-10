# ---[ CPU files only
list(APPEND Caffe2_CPU_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/init_qnnpack.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_add_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_average_pool_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_channel_shuffle_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_concat_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_conv_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_conv_transpose_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_dequantize_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_fc_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_flatten_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_given_tensor_fill_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_leaky_relu_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_max_pool_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_quantize_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_relu_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_reshape_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_resize_nearest_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_roi_align_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_slice_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_sigmoid_op.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_softmax_op.cc")

# ---[ CPU test files
list(APPEND Caffe2_CPU_TEST_SRCS
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_test.cc"
  "${CMAKE_CURRENT_SOURCE_DIR}/int8_roi_align_op_test.cc")

# ---[ Send the lists to the parent scope.
set(Caffe2_CPU_SRCS ${Caffe2_CPU_SRCS} PARENT_SCOPE)
set(Caffe2_CPU_TEST_SRCS ${Caffe2_CPU_TEST_SRCS} PARENT_SCOPE)
