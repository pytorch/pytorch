# @package onnx
# Module caffe2.python.onnx.tests.onnx_backend_test






import os

import unittest
import onnx.backend.test

import caffe2.python.onnx.backend as c2

from caffe2.python import core
core.SetEnginePref({}, {})

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(c2, __name__)

backend_test.exclude(r'(test_hardsigmoid'  # Does not support Hardsigmoid.
                     '|test_hardmax'  # Does not support Hardmax.
                     '|test_.*FLOAT16.*'  # Does not support Cast on Float16.
                     '|test_depthtospace.*'  # Does not support DepthToSpace.
                     '|test_reduce_l1.*'  # Does not support ReduceL1.
                     '|test_reduce_l2.*'  # Does not support ReduceL2.
                     '|test_reduce_log_sum.*'  # Does not support ReduceLogSum.
                     '|test_reduce_prod.*'  # Does not support ReduceProd.
                     '|test_reduce_sum_square.*'  # Does not support ReduceSumSquare
                     '|test_det.*'  # Does not support Det
                     '|test_range.*'  # Does not support Range
                     '|test_tile.*'  # Tile's Caffe2 implementation needs some tweak
                     '|test_lstm.*'  # Seems LSTM case has some problem
                     '|test_simple_rnn.*'  # Seems simple RNN case has some problem
                     '|test_gru.*'  # Seems GRU case has some problem
                     '|test_prelu.*'  # PRelu is not compliant with ONNX yet
                     '|test_operator_repeat.*'  # Tile is not compliant with ONNX yet
                     '|test_.*pool_.*same.*'  # Does not support pool same.
                     '|test_.*pool_.*ceil.*'  # Does not support pool same.
                     '|test_maxpool_with_argmax.*'  # MaxPool outputs indices in different format.
                     '|test_maxpool.*dilation.*'  # MaxPool doesn't support dilation yet.
                     '|test_maxpool.*uint8.*'  # MaxPool doesn't support uint8 yet.
                     '|test_convtranspose.*'  # ConvTranspose needs some more complicated translation
                     '|test_mvn.*'  # MeanVarianceNormalization is experimental and not supported.
                     '|test_dynamic_slice.*'  # MeanVarianceNormalization is experimental and not supported.
                     '|test_eyelike.*'  # Needs implementation
                     '|test_maxunpool.*'  # Needs implementation
                     '|test_acosh.*'  # Needs implementation
                     '|test_asinh.*'  # Needs implementation
                     '|test_atanh.*'  # Needs implementation
                     '|test_onehot.*'  # Needs implementation
                     '|test_scan.*'  # Needs implementation
                     '|test_isnan.*'  # Needs implementation
                     '|test_scatter.*'  # Should be similar to ScatterAssign
                     '|test_constantofshape_int.*'  # Needs implementation
                     '|test_shrink.*'  # Needs implementation
                     '|test_strnorm.*'  # Needs implementation
                     '|test_nonzero.*'  # Needs implementation
                     '|test_tfidfvectorizer.*'  # Needs implementation
                     '|test_top_k.*'  # opset 10 is not supported yet
                     '|test_resize.*'  # opset 10 is not supported yet
                     '|test_slice.*'  # opset 10 is not supported yet
                     '|test_.*qlinear.*'  # Skip quantized op test
                     '|test_.*quantize.*'  # Skip quantized op test
                     '|test_.*matmulinteger.*'  # Skip quantized op test
                     '|test_.*convinteger.*'  # Skip quantized op test
                     '|test_isinf.*'  # Needs implementation
                     '|test_mod.*'  # Needs implementation
                     '|test_nonmaxsuppression.*'  # Needs implementation
                     '|test_reversesequence.*'  # Needs implementation
                     '|test_roialign.*'  # Needs implementation
                     '|test_bitshift.*'  # Needs implementation
                     '|test_round.*'  # Needs implementation
                     '|test_cumsum.*'  # Needs implementation
                     '|test_clip.*'  # opset 11 is not supported yet
                     '|test_gather_elements.*'  # opset 11 is not supported yet
                     '|test_scatter.*'  # opset 11 is not supported yet
                     '|test_unique.*'  # opset 11 is not supported yet
                     '|test_gathernd.*'  # opset 11 is not supported yet
                     '|test_dropout_random.*'  # opset 12 is not supported
                     '|test_dropout_default.*'  # opset 12 is not supported
                     '|test_einsum.*'  # opset 12 is not supported
                     '|test_.*training.*'  # training is not supported
                     '|test_.*_loss.*'  # training is not supported
                     '|test_split_zero_size.*'  # unsupported case
                     '|test_constantofshape_int_shape_zero.*'  # unsupported case
                     '|test_constant_pad.*'  # 1d pad is not supported
                     '|test_edge_pad.*'  # 1d pad is not supported
                     '|test_reflect_pad.*'  # 1d pad is not supported
                     '|test_gemm_default_no_bias.*'  # no bias is not supported
                     '|test_gemm_default_scalar_bias.*'  # incorrect type
                     '|test_sequence_.*'  # type sequence is not supported yet
                     '|test_.*negative_ax.*'  # negative axis is not supported yet
                     '|test_.*negative_ind.*'  # negative axis is not supported yet
                     '|test_argmax_.*select_last_index.*'  # unsupported case
                     '|test_argmin_.*select_last_index_.*'  # unsupported case
                     '|test_celu.*'  # unsupported case
                     '|test_gathernd.*'  # unsupported case
                     '|test_greater_equal.*'  # unsupported case
                     '|test_less_equal.*'  # unsupported case
                     '|test_max_.*'  # unsupported case
                     '|test_min_.*'  # unsupported case
                     '|test_.*momentum_.*'  # unsupported case
                     '|test_sce.*'  # unsupported case
                     '|test_nllloss.*'  # unsupported case
                     '|test_unfoldtodepth.*'  # unsupported case
                     '|test_.*gradient.*'  # no support for gradient op in c2-onnx
                     '|test_.*adagrad.*'  # no support for gradient op in c2-onnx
                     '|test_.*loss.*'  # no support for loss op in c2-onnx
                     '|test_.*adam.*'  # no support for adam op
                     ')')

# Quick patch to unbreak master CI, is working on the debugging.
backend_test.exclude('(test_cast_.*'
                     '|test_compress_.*'
                     '|test_Conv1d_.*cuda'
                     '|test_Conv3d_groups_cuda'
                     '|test_rnn_seq_length'
                     '|test_operator_add.*_cuda'
                     '|test_operator_lstm_cuda'
                     '|test_operator_rnn.*_cuda'
                     '|test_lrn_default_cuda)')

# Temporarily skip some ONNX backend tests with broadcasting.
backend_test.exclude('(test_pow_bcast'
                     '|test_pow_types.*'
                     ')')

# Temporarily skip some ONNX backend tests due to updates in opset 13.
backend_test.exclude('(test_if_.*'  # added support for sequence type inputs
                     '|test_if_seq_.*'  # added support for sequence type inputs
                     '|test_logsoftmax_.*'  # axis attr default value changed from 1 to -1
                     '|test_loop11_.*'  # seg fault issue
                     '|test_loop13_seq_.*'  # no support for sequence inputs for scan input
                     '|test_reduce_sum_.*'  # axes is now an input (not attr), added noop_with_empty_axes
                     '|test_softmax_.*'  # axis attr default value changed from 1 to -1
                     '|test_split_variable_parts_.*'  # axes is now an input (not attr)
                     '|test_squeeze_.*'  # axes is now an input (not attr)
                     '|test_unsqueeze_.*'  # axes is now an input (not attr)
                     '|test_MaxPool1d_stride_padding_dilation_.*'
                     '|test_MaxPool2d_stride_padding_dilation_.*'
                     ')')

# Temporarily skip some ONNX backend tests due to updates in opset 14.
backend_test.exclude('(test_add_uint8_.*'  # uint8 dtype added
                     '|test_div_uint8_.*'  # uint8 dtype added
                     '|test_hardswish_.*'  # new operator added
                     '|test_mul_uint8_.*'  # uint8 dtype added
                     '|test_sub_uint8_.*'  # uint8 dtype added
                     '|test_tril_.*'  # new operator added
                     '|test_triu_.*'  # new operator added
                     '|test_identity_sequence_.*'  # new operator added
                     '|test_reshape_allowzero_reordered_.*'
                     '|test_conv_with_autopad_same_.*'
                     ')')

# Unsupported ops in opset 15
backend_test.exclude('(test_bernoulli_*'
                     '|test_castlike_*'
                     '|test_optional_*'
                     '|test_shape_end_*'
                     '|test_shape_start_*'
                     ')')

# Skip vgg to speed up CI
if 'JENKINS_URL' in os.environ:
    backend_test.exclude(r'(test_vgg19|test_vgg)')

# import all test cases at global scope to make them visible to python.unittest
globals().update(backend_test
                 .enable_report()
                 .test_cases)

if __name__ == '__main__':
    unittest.main()
