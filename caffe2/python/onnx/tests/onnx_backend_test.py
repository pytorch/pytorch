# @package onnx
# Module caffe2.python.onnx.tests.onnx_backend_test

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import os

import unittest
import onnx.backend.test

import caffe2.python.onnx.backend as c2

from caffe2.python import core, workspace
core.SetEnginePref({}, {})

# This is a pytest magic variable to load extra plugins
pytest_plugins = 'onnx.backend.test.report',

backend_test = onnx.backend.test.BackendTest(c2, __name__)

backend_test.exclude(r'(test_hardsigmoid'  # Does not support Hardsigmoid.
                     '|test_hardmax'  # Does not support Hardmax.
                     '|test_cast.*FLOAT16.*'  # Does not support Cast on Float16.
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
                     '|test_maxpool.*dilation.*'  # MaxPool doesn't support dilation yet
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
                     '|test_sequence_.*'  # type sequence is not supported yet
                     '|test_.*negative_ax.*'  # negative axis is not supported yet
                     '|test_.*negative_ind.*'  # negative axis is not supported yet
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
