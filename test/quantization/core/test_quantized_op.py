# Owner(s): ["oncall: quantization"]

from builtins import round

import copy
import itertools
import numpy as np
import unittest
import operator
import random

import torch
from torch import _VF
import torch.jit
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair

from hypothesis import settings, HealthCheck
from hypothesis import assume, given, note
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()

from torch.testing._internal.common_utils import TestCase
from torch.testing._internal.common_utils import IS_PPC, TEST_WITH_UBSAN, IS_MACOS, BUILD_WITH_CAFFE2
from torch.testing._internal.common_quantization import skipIfNoFBGEMM, skipIfNoQNNPACK, skipIfNoONEDNN
from torch.testing._internal.common_quantized import _quantize, _dequantize, _calculate_dynamic_qparams, \
    override_quantized_engine, supported_qengines, override_qengines, _snr
from torch.testing._internal.common_quantized import (
    qengine_is_qnnpack,
    qengine_is_onednn,
)
from torch.ao.quantization import PerChannelMinMaxObserver
from torch.testing._internal.common_cuda import TEST_CUDNN, TEST_CUDA
import torch.backends.xnnpack

from typing import Optional

np_dtype = {
    torch.quint8 : np.uint8,
    torch.qint8 : np.int8,
    torch.qint32 : np.int32
}

# Make sure we won't have overflows from vpmaddubsw instruction used in FBGEMM.
# On the current Intel x86 architecture, we need to utilize vpmaddubsw instruction
# for the 8-bit int multiplication. This instruction vertically multiplies each
# unsigned 8-bit integer from a with the corresponding signed 8-bit integer from
# b, producing intermediate signed 16-bit integers. This function modifies the
# weights to eliminate the overflow on the signed 16-bit integers.
def avoid_vpmaddubsw_overflow_linear(
    batch_size, input_channels, output_channels, X, X_min, X_max, W, W_min, W_max
):
    for i, j in np.ndindex((batch_size, output_channels)):
        for k in range(0, input_channels // 2 * 2, 2):
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            if x0 * w0 + x1 * w1 < -(1 << 15):
                w1_adjusted = (-(1 << 15) - float(x0) * w0) / x1
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min
            elif x0 * w0 + x1 * w1 > (1 << 15) - 1:
                w1_adjusted = ((1 << 15) - 1 - float(x0) * w0) / x1
                W[j, k + 1] = int(w1_adjusted) + 128 + W_min

    # Go through the same loop again to double check we don't have any overflow
    for i, j in np.ndindex((batch_size, output_channels)):
        for k in range(0, input_channels // 2 * 2, 2):
            x0 = X[i, k] - X_min
            x1 = X[i, k + 1] - X_min
            w0 = W[j, k] - 128 - W_min
            w1 = W[j, k + 1] - 128 - W_min
            assert -(1 << 15) <= x0 * w0 + x1 * w1 < (1 << 15)


# Reference quantized Linear operator
def qlinear_ref(X_q, X_scale, X_zp, W_q, W_scale, W_zp, b_q, Y_scale, Y_zp, dtype=np.uint8):
    X_q = np.reshape(X_q, (-1, X_q.shape[X_q.ndim - 1]))
    row_offsets_ref = X_q.sum(axis=1).astype(np.int32).reshape((-1, 1))
    col_offsets_ref = W_q.sum(axis=1).astype(np.int32).reshape((1, -1))
    assert X_q.ndim == 2
    batch_size, input_channels = X_q.shape
    Prod_XqWq_ref = (
        np.matmul(X_q.astype(np.int32), W_q.astype(np.int32).T)
        - W_zp * row_offsets_ref
        - X_zp * col_offsets_ref
        + input_channels * X_zp * W_zp
    )
    if b_q is not None:
        Prod_XqWq_ref += b_q
    Y_q_ref = _quantize(Prod_XqWq_ref, Y_scale / (X_scale * W_scale), Y_zp, dtype=dtype)
    return Y_q_ref

"""Computes the output shape given pooling parameters."""
def pool_output_shape(input_size, kernel_size, padding, stride,
                      dilation, ceiling_mode=False):
    if stride is None:
        stride = kernel_size
    output_size = (
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1
         + (stride - 1 if ceiling_mode else 0)) // stride + 1)
    if (ceiling_mode and
            ((output_size - 1) * stride >= input_size + padding)):
        output_size -= 1
    return output_size

"""
Util for creating a random tensor and quantization params when Hypothesis
is undesirable.
"""
def _get_random_tensor_and_q_params(shapes, rand_scale, torch_type):
    X = (torch.rand(*shapes, dtype=torch.float) - 0.5) * rand_scale
    # Calculate reasonable quantization params
    min_val = torch.min(X)
    max_val = torch.max(X)
    if torch_type == torch.qint32:
        X_zero_point = int(torch.randint(-1 * (2 ** 31), 2 ** 31 - 1, (1,)))
        num_bins = 2 ** 32
        X_scale = float(max_val - min_val) / num_bins
    elif torch_type == torch.qint8:
        X_zero_point = int(torch.randint(-128, 127, (1,)))
        num_bins = 2 ** 8
        X_scale = float(max_val - min_val) / num_bins
    else:  # torch.quint8
        X_zero_point = 127
        num_bins = 2 ** 8
        X_scale = float(max_val - min_val) / num_bins
    if X_scale == 0:
        X_scale = 1e-10
    return X, X_scale, X_zero_point

class TestQuantizedOps(TestCase):

    """Helper function to test quantized activation functions."""
    def _test_activation_function(self, X, fn_name, test_configs):
        r"""
            When writing a unit test for the activation function,
            instead of specifying the test routines only applicable to the activation function itself,
            you utilize the _test_activation_function that provides general testing.
            To utilize the helper function, a test config must be provided.
            A test config is a list that contains metadata about the quantized activation
            functions that will be tested and how the tests need to be set up; it allows simpler and
            more concise unit tests to be written by specifying the configurations needed
            and calling the provided helper function _test_activation_function.
            Inside the list, each config (as a dictionary) represents a suite of tests that assert the
            correctness of various quantization functions.
            You can check out the test_qrelu, test_qrelu6, test_qsigmoid, and test_qhardsigmoid for
            how their test configs are specified.
            Here's a list of the fields that can be included in a test config:
            quantized_fn: a list of the quantized functions to be tested
            reference_fn: the original reference function to be called on the
            the dequantized X
            extra_kwargs: the additional keyword arguments
            for each test entry in ops_under_test, it must have at least the fields
            for quantized_fn and reference_fn.
            output_range: the output range the operator will map to. By default, if it is
            no specified, the range will not be controlled and depend on Xmin and Xmax.
            change_zero_point: a boolean flag indicating if the zero point parameter should
            be determined based on torch_type during quantization (see sigmoid/hardsigmoid for
            examples). By default, if it is not specified, change_zero_point is assumed to be
            False and zero point will just take on the default value from X.
            `output_is_observed`: if specified and is True, we'll append extra
             output_scale/output_zero_point keyword argument when calling quantized op
        """
        # Retrives the default parameters from X.
        X, (scale, zero_point, torch_type) = X
        if not isinstance(X, torch.Tensor):
            X = torch.from_numpy(X)
        if (X.device.type == 'cuda') and (torch.backends.quantized.engine == 'qnnpack'):
            return
        # Quantizes the reference to account for max error.
        # q_min and q_max only depend on the initial torch_type.
        q_min, q_max = torch.iinfo(torch_type).min, torch.iinfo(torch_type).max

        for op_group in test_configs:
            ref_op = op_group['reference_fn']
            for q_op in op_group['quantized_fn']:

                for memory_format in (torch.channels_last, torch.contiguous_format):
                    if memory_format == torch.channels_last and len(X.shape) != 4:
                        continue
                    X = X.to(memory_format=memory_format)

                    # Retrieves the inplace keyword arguments
                    # some functions require inplace=True to test in-place.
                    # copy.copy is needed because these are modified in place
                    extra_kwargs = \
                        copy.copy(op_group.get('extra_kwargs', {}))
                    output_is_observed = \
                        copy.copy(op_group.get('output_is_observed', False))

                    # Quantizes and dequantizes to account for max error.
                    qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                                   dtype=torch_type)
                    dqX = qX.dequantize()
                    dqY_hat = ref_op(dqX.clone(), **extra_kwargs)

                    # Adjusts output_scale if needed.
                    # The output_scale determines the quantization scale for functions that
                    # have a constrained output range. e.x. sigmoid ranges from 0 to 1.
                    output_scale = scale
                    if 'output_range' in op_group:
                        (f_min, f_max) = op_group['output_range']
                        output_scale = (f_max - f_min) / (q_max - q_min + 1.0)

                    # Adjusts output_zero_point if needed (see explanation for the
                    # change_zero_point parameter above).
                    # output_zero_point determines the additional offset that will be
                    # added to a scaled value during quantization.
                    if op_group.get('change_zero_point', False):
                        output_zero_point = 0 if torch_type == torch.qint32 else q_min
                    else:
                        output_zero_point = zero_point

                    # Quantizes the dequantized version of Y_hat.
                    qY_hat = torch.quantize_per_tensor(dqY_hat, scale=output_scale,
                                                       zero_point=output_zero_point,
                                                       dtype=torch_type)

                    if output_is_observed:
                        extra_kwargs.update({'output_scale': output_scale, 'output_zero_point': output_zero_point})

                    # Finds qY using in-place or non-in-place quantized operators.
                    qY = q_op(qX, **extra_kwargs)

                    self.assertEqual(qY, qY_hat, msg='{} - {} failed: ({} vs. {})'.format(
                        fn_name, q_op, qY, qY_hat
                    ))

    """Tests the correctness of the quantized::relu op."""
    @override_qengines
    def test_qrelu(self):
        relu_test_configs = [
            {
                'quantized_fn': [
                    torch.relu,
                    torch.relu_,
                    torch.nn.functional.relu,
                    torch.nn.functional.relu,
                ],
                'reference_fn': torch.nn.functional.relu
            },
            {
                'quantized_fn': [
                    torch.nn.functional.relu,
                    torch.nn.functional.relu,
                ],
                'reference_fn': torch.nn.functional.relu,
                'extra_kwargs': {
                    'inplace': True
                }
            }
        ]
        devices = ["cpu", "cuda"] if TEST_CUDA else ["cpu"]
        for device in devices:
            shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
            dtypes = (torch.quint8, torch.qint8)
            scales = (0.05, 0.1)
            zero_points = (0, 5)
            test_cases = itertools.product(shapes, dtypes, scales, zero_points)
            for shape, dtype, scale, zero_point in test_cases:
                X = torch.randn(*shape, device=device)
                X = (X, (scale, zero_point, dtype))
                self._test_activation_function(X, 'relu', relu_test_configs)

    """Tests the correctness of the quantized::relu6 op."""
    def test_qrelu6(self):
        relu6_test_configs = [
            {
                'quantized_fn': [
                    torch.ops.quantized.relu6,
                    torch.ao.nn.quantized.ReLU6(inplace=False),
                    torch.ao.nn.quantized.ReLU6(inplace=True)
                ],
                'reference_fn': torch.nn.functional.relu6
            }
        ]
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        scales = (0.05, 0.1)
        zero_points = (0, 5)
        test_cases = itertools.product(shapes, dtypes, scales, zero_points)
        for shape, dtype, scale, zero_point in test_cases:
            X = torch.randn(*shape) * 10
            X = (X, (scale, zero_point, dtype))
            self._test_activation_function(X, 'relu6', relu6_test_configs)

    """Tests the correctness of the quantized::sigmoid op."""
    @override_qengines
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_sigmoid_non_observed(self, X):
        sigmoid_test_configs = [
            {
                'quantized_fn': [
                    torch.sigmoid
                ],
                'reference_fn': torch.sigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True
            }
        ]
        self._test_activation_function(X, 'sigmoid', sigmoid_test_configs)

    """Tests the correctness of the quantized::sigmoid op."""
    # TODO: enable after observed output is supported in qnnpack
    # @override_qengines
    @skipIfNoFBGEMM
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_sigmoid(self, X):
        sigmoid_test_configs = [
            {
                'quantized_fn': [
                    torch.ops.quantized.sigmoid
                ],
                'reference_fn': torch.sigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True,
                'output_is_observed': True,
            }
        ]
        self._test_activation_function(X, 'sigmoid', sigmoid_test_configs)

    """Tests the correctness of the quantized::hardsigmoid op."""
    @override_qengines
    def test_qhardsigmoid(self):
        hardsigmoid_test_configs = [
            {
                'quantized_fn': [
                    torch.ao.nn.quantized.functional.hardsigmoid,
                    torch.nn.quantized.functional.hardsigmoid,
                ],
                'reference_fn': torch.nn.functional.hardsigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True,
            },
            {
                'quantized_fn': [
                    torch.ao.nn.quantized.functional.hardsigmoid,
                    torch.nn.quantized.functional.hardsigmoid,
                ],
                'reference_fn': torch.nn.functional.hardsigmoid,
                'output_range': (0.0, 1.0),
                'change_zero_point': True,
                'extra_kwargs': {
                    'inplace': True,
                },
            },
        ]
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        test_cases = itertools.product(shapes, dtypes)
        for shape, dtype in test_cases:
            X = (np.random.rand(*shape).astype(np.float32), (1.0, 0, dtype))
            self._test_activation_function(X, 'hardsigmoid', hardsigmoid_test_configs)

    @override_qengines
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_leaky_relu_observed_output(self, X):
        leaky_relu_test_configs = [
            {
                'quantized_fn': [
                    torch.ops.quantized.leaky_relu
                ],
                'reference_fn': torch.nn.functional.leaky_relu,
                'extra_kwargs': {
                    'negative_slope': 0.1,
                    'inplace': False,
                },
                'output_is_observed': True,
            }
        ]
        self._test_activation_function(X, 'leaky_relu', leaky_relu_test_configs)

    """Tests the correctness of the quantized::relu op."""
    def test_leaky_relu(self):
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        memory_formats = (torch.channels_last, torch.contiguous_format)
        test_cases = itertools.product(shapes, dtypes, memory_formats)
        for shape, dtype, memory_format in test_cases:
            if memory_format == torch.channels_last and len(shape) != 4:
                continue
            X, scale, zero_point, torch_type, alpha = \
                torch.randn(*shape), 0.1, 0, dtype, 0.01
            X = X.to(memory_format=memory_format)

            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            dqX = qX.dequantize()

            # torch.nn.functional
            op = torch.nn.functional.leaky_relu
            dqY = op(dqX, negative_slope=alpha)
            qY = torch.quantize_per_tensor(dqY, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            qY_hat = op(qX, negative_slope=alpha)
            self.assertEqual(qY.dequantize(), qY_hat.dequantize(),
                             msg="F.leaky_relu failed ({} vs {})".format(qY, qY_hat))

    """Tests the correctness of the quantized::elu op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams()),
           alpha=st.floats(0.01, 10.0, allow_nan=False, allow_infinity=False))
    def test_qelu(self, X, alpha):
        X, (scale, zero_point, torch_type) = X
        output_scale = 0.5
        output_zero_point = 1

        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # calculate ELU(dqX) and quantize
        dqX = qX.dequantize()
        dqY_hat = dqX.clone()
        dqY_hat = torch.nn.functional.elu(dqX, alpha)
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=output_scale, zero_point=output_zero_point,
                                           dtype=torch_type)

        qY = torch.ao.nn.quantized.functional.elu(qX, output_scale, output_zero_point, alpha=alpha)
        self.assertEqual(qY, qY_hat,
                         msg="F.elu failed ({} vs {})".format(qY, qY_hat))


    """Tests the correctness of the quantized::celu op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       elements=hu.floats(-1e2, 1e2, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(scale_max=9.999999747378752e-06)),
           alpha=st.floats(0.01, 100.0, allow_nan=False, allow_infinity=False))
    def test_qcelu(self, X, alpha):
        X, (scale, zero_point, torch_type) = X
        output_scale = 0.5
        output_zero_point = 1

        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # calculate CELU(dqX) and quantize
        dqX = qX.dequantize()
        dqY_hat = torch.nn.functional.celu(dqX, alpha)
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=output_scale, zero_point=output_zero_point,
                                           dtype=torch_type)

        # test regular
        qY = torch.ops.quantized.celu(qX, output_scale, output_zero_point, alpha=alpha)
        self.assertEqual(qY, qY_hat,
                         msg="F.celu failed ({} vs {})".format(qY, qY_hat))

    """Tests the correctness of the quantized::gelu op."""
    def test_qgelu(self):
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        memory_formats = (torch.channels_last, torch.contiguous_format)
        approximation = ['none', 'tanh']
        test_cases = itertools.product(shapes, dtypes, memory_formats, approximation)
        devices = ["cpu", "cuda"] if TEST_CUDA else ["cpu"]
        for shape, dtype, memory_format, approximate in test_cases:
            if memory_format == torch.channels_last and len(shape) != 4:
                continue

            X, scale, zero_point, torch_type = \
                torch.randn(*shape), 0.1, 0, dtype
            X = X.to(memory_format=memory_format)
            for device in devices:
                X = X.to(device=device)
                qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                               dtype=torch_type)
                dqX = qX.dequantize()

                op = torch.nn.functional.gelu
                dqY = op(dqX, approximate=approximate)
                qY = torch.quantize_per_tensor(dqY, scale=scale, zero_point=zero_point,
                                               dtype=torch_type)
                qY_hat = op(qX)
                self.assertEqual(qY.dequantize(), qY_hat.dequantize(),
                                 msg="F.gelu failed ({} vs {})".format(qY, qY_hat))

    """Tests the correctness of the quantized::prelu op."""
    def test_qprelu(self):
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        num_params = (0, 1)  # 0: num_parameter = num_channels
        dtypes = (torch.quint8, torch.qint8)
        memory_formats = (torch.channels_last, torch.contiguous_format)
        test_cases = itertools.product(shapes, num_params, dtypes, memory_formats)
        for shape, num_param, dtype, memory_format in test_cases:
            if memory_format == torch.channels_last and len(shape) != 4:
                continue
            X, scale, zero_point, torch_type = \
                torch.randn(*shape), 0.1, 0, dtype
            X = X.to(memory_format=memory_format)
            num_parameter = 1 if num_param == 1 or len(shape) == 1 else shape[1]
            W = torch.randn(num_parameter)
            W, w_scale, w_zero_point = \
                torch.randn(num_parameter), 0.2, 0

            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            dqX = qX.dequantize()
            qW = torch.quantize_per_tensor(W, scale=w_scale, zero_point=w_zero_point,
                                           dtype=torch_type)
            dqW = qW.dequantize()

            op = torch.nn.functional.prelu
            qop = torch.ops.quantized.prelu
            dqY = op(dqX, dqW)
            qY = torch.quantize_per_tensor(dqY, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            qY_hat = qop(qX, qW, scale, zero_point)
            self.assertEqual(qY.dequantize(), qY_hat.dequantize(),
                             msg="F.prelu failed ({} vs {})".format(qY, qY_hat))

    """Tests the correctness of the quantized::qlayer_norm op."""
    @skipIfNoFBGEMM
    def test_qlayer_norm(self):
        # hypothesis is flaky for this test, create test cases manually
        side_lens = (1, 8, 11)
        torch_types = (torch.qint8, torch.quint8)
        y_scales = (0.1, 4.23)
        y_zero_points = (0, 1)
        channels_last_list = (True, False)
        affine_list = (True, False)
        combined = [side_lens, torch_types, y_scales, y_zero_points,
                    channels_last_list, affine_list]
        test_cases = itertools.product(*combined)

        with override_quantized_engine("fbgemm"):
            for test_case in test_cases:

                side_len, torch_type, Y_scale, Y_zero_point, channels_last, \
                    affine = test_case
                shapes = [side_len] * 4

                # In the FP kernel, mean and variance are calculated in floating point.
                # In the quantized kernel, they are calculated in integer arithmetic.
                # Because of this, the numerics do not always match exactly which is
                # expected and acceptable. We do two things to allow this failure
                # in this test:
                # 1. do not use Hypothesis to generate the input tensor.  Hypothesis
                #    favors homogeneous inputs in its search strategies which isn't
                #    representative of the inputs we care about, and tends to maximize
                #    this particular numerics difference.
                # 2. allow a small % of off by Y_scale errors.  Even when the
                #    variance of the input is high, there can be off by one errors
                #    in the result if the input value happens to fall exactly on
                #    the bin boundary of the output scale.
                #
                # If we want the numerics to match we could switch to calculating
                # mean+var in floating point in the future, at the cost of speed.
                X, X_scale, X_zero_point = \
                    _get_random_tensor_and_q_params(shapes, 1.0, torch_type)

                qX = torch.quantize_per_tensor(X, scale=X_scale,
                                               zero_point=X_zero_point,
                                               dtype=torch_type)
                if channels_last:
                    qX = qX.contiguous(memory_format=torch.channels_last)
                dqX = qX.dequantize()

                # Enforce non-homogeneous inputs
                enough_unique_vals_in_each_layer = sum(
                    1 if (
                        dqX[i].shape[0] < 5 or
                        float(torch.unique(dqX[i]).shape[0]) / dqX[i].shape[0] > 0.01
                    ) else 0
                    for i in range(dqX.shape[0])
                ) == dqX.shape[0]
                assume(enough_unique_vals_in_each_layer)

                # Initialize the weights non-randomly for reproducibility, to avoid
                # flaky tests
                if affine:
                    weight = torch.ones(*qX.size()[1:], dtype=torch.float) * 0.5
                    bias = torch.ones(*qX.size()[1:], dtype=torch.float) * 1
                else:
                    weight = None
                    bias = None
                epsilon = 1e-5

                qY = torch.ops.quantized.layer_norm(
                    qX, qX.size()[1:], weight=weight, bias=bias, eps=epsilon,
                    output_scale=Y_scale, output_zero_point=Y_zero_point)

                Y_hat = F.layer_norm(
                    dqX, dqX.size()[1:], weight=weight, bias=bias, eps=epsilon)
                qY_hat = torch.quantize_per_tensor(
                    Y_hat, scale=Y_scale, zero_point=Y_zero_point, dtype=torch_type)

                # Due to the numerics difference mentioned above between calculating
                # the variance in float vs int, the results can still be slightly
                # different.
                dqY = qY.dequantize()
                dqY_hat = qY_hat.dequantize()
                diff = dqY - dqY_hat

                # off-by-one errors are magnitude of Y_scale
                num_diff = torch.sum(diff > Y_scale * 1.0001)
                pct_diff = float(num_diff) / (diff.numel() + 1e-5)
                num_diff_off_by_one = torch.sum((diff > 0) * (diff <= Y_scale))
                pct_diff_off_by_one = float(num_diff_off_by_one) / (diff.numel() + 1e-5)

                self.assertTrue(pct_diff < 1e-6)
                self.assertTrue(pct_diff_off_by_one < 0.01)


    """Tests the correctness of the quantized::qnnpack_tanh op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    def test_qtanh(self, X):
        # Note: QNNPACK is tested separately in TestQNNPackOps
        X, (scale, zero_point, torch_type) = X

        X = torch.from_numpy(X)
        Y = torch.tanh(X)

        qX = torch.quantize_per_tensor(X, scale=scale,
                                       zero_point=zero_point,
                                       dtype=torch_type)

        # Quantize the reference to account for max error.
        # Note that the output scale has +1, because we use scale of 2.0/2^BITS
        # in the implementations.
        f_min, f_max = -1.0, 1.0
        q_min, q_max = torch.iinfo(torch_type).min, torch.iinfo(torch_type).max
        output_scale = (f_max - f_min) / (q_max - q_min + 1.0)
        output_zero_point = int(round((q_max + q_min) / 2.0))
        qY = torch.quantize_per_tensor(Y, scale=output_scale,
                                       zero_point=output_zero_point,
                                       dtype=torch_type)
        qY_hat = torch.tanh(qX)
        self.assertEqual(qY, qY_hat,
                         msg="TanH failed: {} vs. {}".format(qY, qY_hat))

    """Tests the correctness of the quantized::threshold op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams()),
           threshold=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
           value=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False))
    def test_qthreshold(self, X, threshold, value):
        X, (scale, zero_point, torch_type) = X
        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # calculate threshold(dqX) and quantize
        dqX = qX.dequantize()
        dqY_hat = dqX.clone()
        dqY_hat = torch.nn.functional.threshold(dqY_hat, threshold, value)
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

        ops_under_test = {
            'native': torch.threshold,
            'nn.functional': torch.nn.functional.threshold,
            'nn.quantized.functional': torch.nn.quantized.functional.threshold,
            'ao.nn.quantized.functional': torch.ao.nn.quantized.functional.threshold,
        }

        for name, op in ops_under_test.items():
            qY = op(qX, threshold, value)
            self.assertEqual(qY, qY_hat, msg="{} qthreshold failed".format(name))

    """Tests the correctness of the quantized::clamp op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 8, 1, 8, max_numel=10**5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False),
                       qparams=hu.qparams()),
           min_val=hu.floats(-1e6, 1e6, allow_nan=False),
           max_val=hu.floats(-1e6, 1e6, allow_nan=False))
    def test_qclamp(self, X, min_val, max_val):
        X, (scale, zero_point, torch_type) = X

        assume(min_val <= max_val)
        Y_clamp = torch.clamp(torch.from_numpy(X), min=min_val, max=max_val)
        qY_clamp = torch.quantize_per_tensor(Y_clamp, scale=scale,
                                             zero_point=zero_point, dtype=torch_type)

        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)
        ops_under_test = {
            'ops.quantized': torch.ops.quantized.clamp,
        }

        for name, op in ops_under_test.items():
            qY_clamp_hat = op(qX, min=min_val, max=max_val)
            self.assertEqual(qY_clamp, qY_clamp_hat, msg="{} qclamp failed".format(name))

        if torch.backends.quantized.engine == 'fbgemm':
            with override_quantized_engine('fbgemm'):
                Y_min_clamp = torch.clamp(X, min=min_val)
                Y_max_clamp = torch.clamp(X, max=max_val)

                qY_min_clamp = torch.quantize_per_tensor(Y_min_clamp, scale=scale,
                                                         zero_point=zero_point, dtype=torch_type)
                qY_max_clamp = torch.quantize_per_tensor(Y_max_clamp, scale=scale,
                                                         zero_point=zero_point, dtype=torch_type)


                for name, op in ops_under_test.items():
                    qY_min_clamp_hat = op(qX, min=min_val)
                    self.assertEqual(qY_min_clamp, qY_min_clamp_hat, msg="{} qclamp failed".format(name))
                    qY_max_clamp_hat = op(qX, max=max_val)
                    self.assertEqual(qY_max_clamp, qY_max_clamp_hat, msg="{} qclamp failed".format(name))

    """Tests the correctness of the quantized::hardtanh op."""
    @skipIfNoFBGEMM
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 8, 1, 8, max_numel=10**5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams()),
           min_val=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
           max_val=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
    def test_hardtanh(self, X, min_val, max_val):
        with override_quantized_engine('fbgemm'):
            X, (scale, zero_point, torch_type) = X

            assume(min_val <= max_val)
            Y = X.copy()
            Y[Y < min_val] = min_val
            Y[Y > max_val] = max_val
            qY = torch.quantize_per_tensor(torch.from_numpy(Y), scale=scale,
                                           zero_point=zero_point, dtype=torch_type)
            X = torch.from_numpy(X)
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

            ops_under_test = {
                'nn.quantized.functional.hardtanh':
                    torch.nn.quantized.functional.hardtanh,
                'ao.nn.quantized.functional.hardtanh':
                    torch.ao.nn.quantized.functional.hardtanh,
            }

            for name, op in ops_under_test.items():
                qY_hat = op(qX, min_val, max_val)
                self.assertEqual(qY, qY_hat, msg="{} hardtanh failed".format(name))

            ops_under_test_inplace = {
                'inplace nn.quantized.functional.hardtanh':
                    torch.nn.quantized.functional.hardtanh,
                'inplace ao.nn.quantized.functional.hardtanh':
                    torch.ao.nn.quantized.functional.hardtanh,
            }

            for name, op_ in ops_under_test_inplace.items():
                qY_hat = qX.clone()
                op_(qY_hat, min_val, max_val, inplace=True)
                self.assertEqual(qY, qY_hat, msg="{} hardtanh failed".format(name))

    """Tests the correctness of the quantized::hardswish op."""
    @override_qengines
    def test_hardswish(self):
        max_sides = (3, 4)
        side_lens = (1, 7)
        torch_types = (torch.quint8, torch.qint8)
        y_scales = (0.1, )
        y_zero_points = (1,)
        combined = [max_sides, side_lens, torch_types, y_scales, y_zero_points]
        test_cases = itertools.product(*combined)
        for test_case in test_cases:
            max_side, side_len, torch_type, Y_scale, Y_zero_point = test_case

            if torch.backends.quantized.engine == 'qnnpack' and torch_type != torch.quint8:
                continue

            shapes = [side_len] * max_side
            X, X_scale, X_zero_point = \
                _get_random_tensor_and_q_params(shapes, 2.0, torch_type)
            for memory_format in torch.channels_last, torch.contiguous_format:
                if memory_format == torch.channels_last and len(shapes) == 4:
                    X = X.to(memory_format=memory_format)
                qX = torch.quantize_per_tensor(X, scale=X_scale, zero_point=X_zero_point,
                                               dtype=torch_type)
                dqX = qX.dequantize()

                dqY_hat = F.hardswish(dqX)
                qY_hat = torch.quantize_per_tensor(dqY_hat, scale=Y_scale,
                                                   zero_point=Y_zero_point,
                                                   dtype=torch_type)

                qY = torch.ao.nn.quantized.functional.hardswish(
                    qX, scale=Y_scale, zero_point=Y_zero_point)
                self.assertEqual(
                    qY, qY_hat,
                    msg="Hardswish failed: {} vs {}, {}".format(qY, qY_hat, torch.backends.quantized.engine))

    """Tests the correctness of the binary op + scalar."""
    def _test_binary_op_scalar_relu(self, A, b, binary_op_name, binary_op, quantized_op, quantized_op_relu):
        import copy
        op_scalar = quantized_op
        op_scalar_relu = quantized_op_relu

        A, (scale, zero_point, dtype) = A
        A = A.astype(np.float32)
        qA = torch.quantize_per_tensor(torch.from_numpy(A), scale, zero_point, dtype)

        if binary_op_name == 'add':
            C = binary_op(qA.dequantize(), round(b / scale) * scale)
        else:
            C = binary_op(qA.dequantize(), b)
        C_relu = copy.deepcopy(C)
        C_relu[C_relu < 0] = 0

        C_hat = op_scalar(qA, b)
        C_ref = torch.quantize_per_tensor(C, C_hat.q_scale(), C_hat.q_zero_point(), dtype)
        C_relu_hat = op_scalar_relu(qA, b)
        C_relu_ref = torch.quantize_per_tensor(
            C_relu, C_relu_hat.q_scale(), C_relu_hat.q_zero_point(), dtype)

        self.assertEqual(C_ref.dequantize(), C_hat.dequantize(),
                         msg="{}_scalar results don't match: "
                         "{} vs {}".format(binary_op_name, C_ref.dequantize(), C_hat.dequantize()))
        self.assertEqual(C_relu_ref.dequantize(), C_relu_hat.dequantize(),
                         msg="{}_scalar_relu results don't match: "
                         "{} vs {}".format(binary_op_name, C_relu_ref.dequantize(), C_relu_hat.dequantize()))

    @unittest.skipIf(IS_MACOS, "skipping macos test")
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 4, 1, 5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False),
                       qparams=hu.qparams()),
           b=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
    def test_add_scalar_relu(self, A, b):
        self._test_binary_op_scalar_relu(A, b, "add", operator.add, torch.ops.quantized.add, torch.ops.quantized.add_relu)

    @unittest.skipIf(IS_MACOS, "skipping macos test")
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 4, 1, 5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False),
                       qparams=hu.qparams()),
           b=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
    def test_mul_scalar_relu(self, A, b):
        self._test_binary_op_scalar_relu(A, b, "mul", operator.mul, torch.ops.quantized.mul, torch.ops.quantized.mul_relu)

    """Tests the correctness of the add and add_relu op."""
    def test_qadd_relu_same_qparams(self):
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            add_relu = torch.ops.quantized.add_relu
            add = torch.ops.quantized.add
            add_out = torch.ops.quantized.add
            add_relu_out = torch.ops.quantized.add_relu

            # NB: This is a strange size so that we exercise both the vectorized
            # implementation (64-element chunks at at time) as well as the scalar
            # implementation
            A = torch.arange(-128, 130, dtype=torch.float)
            B = torch.arange(-128, 130, dtype=torch.float)
            scale = 2.0
            zero_point = 127
            qA = torch.quantize_per_tensor(A, scale=scale, zero_point=zero_point,
                                           dtype=dtype)
            qB = torch.quantize_per_tensor(B, scale=scale, zero_point=zero_point,
                                           dtype=dtype)

            # Add ReLU ground truth
            C = (qA.dequantize() + qB.dequantize()).numpy()
            qC = _quantize(C, scale, zero_point, dtype=np_dtype[dtype])
            qC_hat = add(qA, qB, scale=scale, zero_point=zero_point)
            np.testing.assert_equal(qC, qC_hat.int_repr(),
                                    "Quantized addition failed.")
            qC_out_hat = torch._empty_affine_quantized(qC.shape,
                                                       scale=scale,
                                                       zero_point=zero_point,
                                                       dtype=dtype)
            add_out(qA, qB, out=qC_out_hat)
            self.assertEqual(qC_hat, qC_out_hat, msg="Add.out failed")

            # Add + ReLU ground truth
            Crelu = C.copy()
            Crelu[C < 0] = 0
            qCrelu = _quantize(Crelu, scale, zero_point, dtype=np_dtype[dtype])
            qCrelu_hat = add_relu(qA, qB, scale=scale, zero_point=zero_point)
            np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                    "Quantized addition with ReLU failed.")
            qCrelu_out_hat = torch._empty_affine_quantized(qCrelu.shape,
                                                           scale=scale,
                                                           zero_point=zero_point,
                                                           dtype=dtype)
            add_relu_out(qA, qB, out=qCrelu_out_hat)
            self.assertEqual(qCrelu_hat, qCrelu_out_hat,
                             msg="AddReLU.out failed")

    """Tests the correctness of the cudnn add and add_relu op
    (Similar to test_qadd_relu_different_qparams, will probably merge in the future)"""
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    @unittest.skip("Local only - currently the test_qadd_relu_cudnn op is bulid "
                   "with USE_EXPERIMENTAL_CUDNN_V8_API, we can enable the test "
                   "after it is built by default")
    def test_qadd_relu_cudnn(self):
        dtype = torch.qint8
        add_relu = torch.ops.quantized.add_relu
        add = torch.ops.quantized.add

        A = torch.arange(-128, 130, dtype=torch.float).to(torch.device("cuda"))
        B = torch.arange(-128, 130, dtype=torch.float).to(torch.device("cuda"))
        scale_A = 2.5
        scale_B = 6.3
        scale_C = 12.9
        zero_point = 0
        qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point,
                                       dtype=dtype)
        qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point,
                                       dtype=dtype)
        # Add ground truth
        C = (qA.dequantize() + qB.dequantize()).to(device="cpu").numpy()
        qC = _quantize(C, scale_C, zero_point, dtype=np_dtype[dtype])
        qC_hat = add(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized addition failed.")

        # Add + ReLU ground truth
        Crelu = C.copy()
        Crelu[C < 0] = 0
        qCrelu = _quantize(Crelu, scale_C, zero_point, dtype=np_dtype[dtype])
        qCrelu_hat = add_relu(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                "Quantized addition with ReLU failed.")

    """Tests the correctness of the cudnn add and add_relu op for nhwc format"""
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    @unittest.skip("Local only - currently the test_qadd_relu_cudnn_nhwc op is bulid "
                   "with USE_EXPERIMENTAL_CUDNN_V8_API, we can enable the test "
                   "after it is built by default")
    def test_qadd_relu_cudnn_nhwc(self):
        dtype = torch.qint8
        add_relu = torch.ops.quantized.add_relu
        add = torch.ops.quantized.add

        A = torch.rand(16, 8, 4, 12).to(device="cuda")
        B = torch.rand(16, 8, 4, 12).to(device="cuda")
        scale_A = 2.5
        scale_B = 6.3
        scale_C = 12.9
        zero_point = 0
        qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point,
                                       dtype=dtype)
        qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point,
                                       dtype=dtype)
        # Add ground truth
        C = (qA.dequantize() + qB.dequantize()).to(device="cpu").numpy()
        qC = _quantize(C, scale_C, zero_point, dtype=np_dtype[dtype])
        qC_hat = add(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized addition failed.")

        # Add + ReLU ground truth
        Crelu = C.copy()
        Crelu[C < 0] = 0
        qCrelu = _quantize(Crelu, scale_C, zero_point, dtype=np_dtype[dtype])
        qCrelu_hat = add_relu(qA, qB, scale=scale_C, zero_point=zero_point).to(device="cpu")
        np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                "Quantized addition with ReLU failed.")

    """Tests the correctness of the add and add_relu op."""
    def test_qadd_relu_different_qparams(self):
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            add_relu = torch.ops.quantized.add_relu
            add = torch.ops.quantized.add
            add_out = torch.ops.quantized.add
            add_relu_out = torch.ops.quantized.add_relu

            # NB: This is a strange size so that we exercise both the vectorized
            # implementation (64-element chunks at at time) as well as the scalar
            # implementation
            A = torch.arange(-128, 130, dtype=torch.float)
            B = torch.arange(-128, 130, dtype=torch.float)
            scale_A = 3.0
            zero_point_A = 7
            scale_B = 5.0
            zero_point_B = 127

            scale_C = 0.5
            zero_point_C = 5

            qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point_A,
                                           dtype=dtype)
            qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point_B,
                                           dtype=dtype)

            # Add ground truth
            C = (qA.dequantize() + qB.dequantize()).numpy()
            qC = _quantize(C, scale_C, zero_point_C, dtype=np_dtype[dtype])
            qC_hat = add(qA, qB, scale=scale_C, zero_point=zero_point_C)
            np.testing.assert_equal(qC, qC_hat.int_repr(),
                                    "Quantized addition failed.")
            qC_out_hat = torch._empty_affine_quantized(qC.shape,
                                                       scale=scale_C,
                                                       zero_point=zero_point_C,
                                                       dtype=dtype)
            add_out(qA, qB, out=qC_out_hat)
            self.assertEqual(qC_hat, qC_out_hat, msg="Add.out failed")

            # Add + ReLU ground truth
            Crelu = C.copy()
            Crelu[C < 0] = 0
            qCrelu = _quantize(Crelu, scale_C, zero_point_C, dtype=np_dtype[dtype])
            qCrelu_hat = add_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)
            np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                    "Quantized addition with ReLU failed.")
            qCrelu_out_hat = torch._empty_affine_quantized(qCrelu.shape,
                                                           scale=scale_C,
                                                           zero_point=zero_point_C,
                                                           dtype=dtype)
            add_relu_out(qA, qB, out=qCrelu_out_hat)
            self.assertEqual(qCrelu_hat, qCrelu_out_hat,
                             msg="AddReLU.out failed")

    """Tests the correctness of the mul and mul_relu op."""
    def test_qmul_relu_same_qparams(self):
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            mul_relu = torch.ops.quantized.mul_relu
            mul = torch.ops.quantized.mul
            mul_out = torch.ops.quantized.mul
            mul_relu_out = torch.ops.quantized.mul_relu

            A = torch.arange(-100, 100, dtype=torch.float)
            B = torch.arange(-100, 100, dtype=torch.float)
            scale = 2
            zero_point = 127
            qA = torch.quantize_per_tensor(A, scale=scale, zero_point=zero_point,
                                           dtype=dtype)
            qB = torch.quantize_per_tensor(B, scale=scale, zero_point=zero_point,
                                           dtype=dtype)

            # mul ReLU ground truth
            C = (qA.dequantize() * qB.dequantize()).numpy()
            qC = _quantize(C, scale, zero_point, dtype=np_dtype[dtype])
            qC_hat = mul(qA, qB, scale=scale, zero_point=zero_point)
            np.testing.assert_equal(qC, qC_hat.int_repr(),
                                    "Quantized mulition failed.")
            qC_out_hat = torch._empty_affine_quantized(qC.shape,
                                                       scale=scale,
                                                       zero_point=zero_point,
                                                       dtype=dtype)
            mul_out(qA, qB, out=qC_out_hat)
            self.assertEqual(qC_hat, qC_out_hat, msg="mul.out failed")

            # mul + ReLU ground truth
            Crelu = C.copy()
            Crelu[C < 0] = 0
            qCrelu = _quantize(Crelu, scale, zero_point, dtype=np_dtype[dtype])
            qCrelu_hat = mul_relu(qA, qB, scale=scale, zero_point=zero_point)
            np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                    "Quantized mulition with ReLU failed.")
            qCrelu_out_hat = torch._empty_affine_quantized(qCrelu.shape,
                                                           scale=scale,
                                                           zero_point=zero_point,
                                                           dtype=dtype)
            mul_relu_out(qA, qB, out=qCrelu_out_hat)
            self.assertEqual(qCrelu_hat, qCrelu_out_hat,
                             msg="mulReLU.out failed")

            # Scalar multiplication
            for b in B:
                C_ref = qA.dequantize().numpy() * b.item()
                qC_hat = torch.ops.quantized.mul(qA, b.item())

                self.assertEqual(C_ref, qC_hat.dequantize())

            # Scalar multiplication + relu
            for b in B:
                C_ref = qA.dequantize().numpy() * b.item()
                C_ref[C_ref < 0] = 0
                qC_hat = torch.ops.quantized.mul_relu(qA, b.item())

                self.assertEqual(C_ref, qC_hat.dequantize())

    """Tests the correctness of the mul and mul_relu op."""
    def test_qmul_relu_different_qparams(self):
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            mul_relu = torch.ops.quantized.mul_relu
            mul = torch.ops.quantized.mul
            mul_out = torch.ops.quantized.mul
            mul_relu_out = torch.ops.quantized.mul_relu

            A = torch.arange(-100, 100, dtype=torch.float)
            B = torch.arange(-100, 100, dtype=torch.float)
            scale_A = 3.0
            zero_point_A = 7
            scale_B = 5.0
            zero_point_B = 127

            scale_C = 0.5
            zero_point_C = 5

            qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point_A,
                                           dtype=dtype)
            qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point_B,
                                           dtype=dtype)

            # mul ground truth
            C = (qA.dequantize() * qB.dequantize()).numpy()
            qC = _quantize(C, scale_C, zero_point_C, dtype=np_dtype[dtype])
            qC_hat = mul(qA, qB, scale=scale_C, zero_point=zero_point_C)
            np.testing.assert_equal(qC, qC_hat.int_repr(),
                                    "Quantized multiplication failed.")
            qC_out_hat = torch._empty_affine_quantized(qC.shape,
                                                       scale=scale_C,
                                                       zero_point=zero_point_C,
                                                       dtype=dtype)
            mul_out(qA, qB, out=qC_out_hat)
            self.assertEqual(qC_hat, qC_out_hat, msg="mul.out failed")

            # mul + ReLU ground truth
            Crelu = C.copy()
            Crelu[C < 0] = 0
            qCrelu = _quantize(Crelu, scale_C, zero_point_C, dtype=np_dtype[dtype])
            qCrelu_hat = mul_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)
            np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                    "Quantized multiplication with ReLU failed.")
            qCrelu_out_hat = torch._empty_affine_quantized(qCrelu.shape,
                                                           scale=scale_C,
                                                           zero_point=zero_point_C,
                                                           dtype=dtype)
            mul_relu_out(qA, qB, out=qCrelu_out_hat)
            self.assertEqual(qCrelu_hat, qCrelu_out_hat,
                             msg="mulReLU.out failed")

    """Tests the correctness of the matmul op."""
    @given(num_dims=st.integers(2, 5),
           outer_dims=st.lists(st.integers(2, 6), min_size=3, max_size=3),
           m=st.integers(2, 6),
           k=st.integers(2, 6),
           n=st.integers(2, 6),
           dtypes=st.sampled_from(((torch.qint8, np.int8),
                                   (torch.quint8, np.uint8))))
    def test_qmatmul(self, num_dims, outer_dims, m, k, n, dtypes):
        (torch_dtype, np_dtype) = dtypes

        size_a = outer_dims[:num_dims - 2] + [m, k]
        size_b = outer_dims[:num_dims - 2] + [k, n]
        A = torch.randn(size=size_a, dtype=torch.float32) * 3
        B = torch.randn(size=size_b, dtype=torch.float32) * 3

        scale_A = 3.1
        zero_point_A = 7
        scale_B = 5.3
        zero_point_B = 127

        scale_C = 1.3
        zero_point_C = 5

        qA = torch.quantize_per_tensor(A,
                                       scale=scale_A,
                                       zero_point=zero_point_A,
                                       dtype=torch_dtype)
        qB = torch.quantize_per_tensor(B,
                                       scale=scale_B,
                                       zero_point=zero_point_B,
                                       dtype=torch_dtype)

        # matmul ground truth
        C = torch.matmul(qA.dequantize(), qB.dequantize()).numpy()
        qC = _quantize(C, scale_C, zero_point_C, dtype=(np_dtype))
        qC_hat = torch.ops.quantized.matmul(qA,
                                            qB,
                                            scale=scale_C,
                                            zero_point=zero_point_C)
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized multiplication failed.")

        # Using per channel quantization fails
        axis = 0
        scales_A = torch.rand(size=(A.shape[axis],))
        zero_points_A = torch.randint(low=0, high=5, size=(A.shape[axis],))
        scales_B = torch.rand(size=(B.shape[axis],))
        zero_points_B = torch.randint(low=0, high=5, size=(B.shape[axis],))

        qA = torch.quantize_per_channel(A,
                                        scales=scales_A,
                                        zero_points=zero_points_A,
                                        axis=axis,
                                        dtype=torch.qint8)
        qB = torch.quantize_per_channel(B,
                                        scales=scales_B,
                                        zero_points=zero_points_B,
                                        axis=axis,
                                        dtype=torch.qint8)
        np.testing.assert_raises_regex(RuntimeError,
                                       ".*per-tensor.*",
                                       torch.ops.quantized.matmul,
                                       qA,
                                       qB,
                                       scale_C,
                                       zero_point_C)


    """Tests the correctness of the quantized softmax op."""
    @given(dims=st.lists(st.integers(2, 5), min_size=5, max_size=5))
    def test_qsoftmax(self, dims):
        for (num_dims, dim, memory_format) in [
            (2, 1, torch.contiguous_format),  # 2d softmax over last dim
            (4, 3, torch.contiguous_format),  # >2 dims, softmax along last dim
            (5, 2, torch.contiguous_format),  # >2 dims, softmax along not last dim (requires permute)
            (4, 3, torch.channels_last),      # >2 dims, softmax along last dim, but not contiguous
            (4, 1, torch.channels_last),      # Channels Last, doesn't require permute
            (5, 1, torch.channels_last_3d),   # Channels Last 3D, doesn't require permute
        ]:
            size = dims[:num_dims]
            torch_dtype = torch.quint8
            np_dtype = np.uint8

            scale_X = 1.3
            zero_point_X = 5
            X = torch.rand(size=size, dtype=torch.float32) * 8 + zero_point_X
            X = X.to(memory_format=memory_format)

            scale_Y = 1 / 256
            zero_point_Y = 0

            qX = torch.quantize_per_tensor(X,
                                           scale=scale_X,
                                           zero_point=zero_point_X,
                                           dtype=torch_dtype)


            # softmax ground truth
            Y = torch.softmax(qX.dequantize(), dim=dim).numpy()
            qY = _quantize(Y, scale_Y, zero_point_Y, dtype=np_dtype)
            qY_hat = torch.ops.quantized.softmax(qX,
                                                 dim=dim,
                                                 output_scale=scale_Y,
                                                 output_zero_point=zero_point_Y)

            np.testing.assert_equal(qY, qY_hat.int_repr(),
                                    "Quantized softmax failed.")

    """Tests the correctness of the quantized softmax op using qnnpack."""
    @skipIfNoQNNPACK
    def test_qsoftmax_qnnpack(self):
        with override_quantized_engine('qnnpack'):
            self.test_qsoftmax()

    """Tests the correctness of the mul and mul_relu op."""
    def test_qmul_broadcast(self):
        mul_relu = torch.ops.quantized.mul_relu
        mul = torch.ops.quantized.mul
        mul_out = torch.ops.quantized.mul
        mul_relu_out = torch.ops.quantized.mul_relu

        # A = torch.arange(-25, 25, dtype=torch.float)
        # B = torch.arange(-25, 25, dtype=torch.float)
        A = torch.randn(8, 1, 6, 1)
        B = torch.randn(7, 1, 5)
        scale_A = 3.0
        zero_point_A = 7
        scale_B = 5.0
        zero_point_B = 127

        scale_C = 0.5
        zero_point_C = 5

        qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point_A,
                                       dtype=torch.quint8)
        qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point_B,
                                       dtype=torch.quint8)

        # mul ground truth
        C = (qA.dequantize() * qB.dequantize()).numpy()
        qC = _quantize(C, scale_C, zero_point_C)
        qC_hat = mul(qA, qB, scale=scale_C, zero_point=zero_point_C)
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized multiplication failed.")

    """Tests that quantized add works with broadcasting"""
    def test_qadd_broadcast(self):
        A = torch.randn(1, 1, 4, 4)
        B = torch.randn(2, 1, 4, 4)
        qA = torch.quantize_per_tensor(A, 0.02, 0, torch.quint8)
        qB = torch.quantize_per_tensor(B, 0.04, 2, torch.quint8)

        output_scale = 0.01
        output_zp = 1

        # ground truth
        C = qA.dequantize() + qB.dequantize()
        qC = torch.quantize_per_tensor(C, output_scale, output_zp, torch.quint8)

        # quantized
        qC_hat_1 = torch.ops.quantized.add(qA, qB, output_scale, output_zp)
        qC_hat_2 = torch.ops.quantized.add(qB, qA, output_scale, output_zp)

        self.assertTrue(torch.allclose(qC.dequantize(), qC_hat_1.dequantize()))
        self.assertTrue(torch.allclose(qC.dequantize(), qC_hat_2.dequantize()))

    """Tests channel shuffle operation on quantized tensors."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=2, max_side=32, max_numel=10**5),
                       qparams=hu.qparams(dtypes=[torch.quint8])),
           groups=st.integers(2, 6))
    def test_channel_shuffle(self, X, groups):
        X, (scale, zero_point, torch_type) = X
        channels = X.shape[-3]
        iH, iW = X.shape[-2:]
        assume(channels % groups == 0)

        a = torch.from_numpy(X)
        a = torch.rand(a.shape)
        a_out = torch.nn.functional.channel_shuffle(a, groups)

        a_ref = torch.quantize_per_tensor(a_out, scale=scale,
                                          zero_point=zero_point, dtype=torch_type)
        a_ref = a_ref.dequantize()
        qa = torch.quantize_per_tensor(a, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        a_hat = torch.nn.functional.channel_shuffle(qa, groups)
        self.assertEqual(a_ref, a_hat.dequantize(),
                         msg="torch.nn.functional.channel_shuffle results are off")

    """Tests 1D max pool operation on quantized tensors."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=2, max_dims=3,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           kernel=st.sampled_from((3, 5, 7)),
           stride=st.sampled_from((None, 1, 2)),
           dilation=st.integers(1, 2),
           padding=st.integers(0, 2),
           ceil_mode=st.booleans())
    def test_max_pool1d(self, X, kernel, stride, dilation, padding, ceil_mode):
        X, (scale, zero_point, torch_type) = X
        # Check constraints
        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        iW = X.shape[-1]
        oW = pool_output_shape(iW, kernel, padding, stride, dilation, ceil_mode)
        assume(oW > 0)

        a = torch.from_numpy(X)
        a_pool = torch.nn.functional.max_pool1d(a, kernel_size=kernel,
                                                stride=stride,
                                                padding=padding,
                                                dilation=dilation,
                                                ceil_mode=ceil_mode)
        a_ref = torch.quantize_per_tensor(a_pool, scale=scale,
                                          zero_point=zero_point, dtype=torch_type)
        a_ref = a_ref.dequantize()
        qa = torch.quantize_per_tensor(a, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        ops_under_test = {
            "torch": torch.max_pool1d,
            "nn.functional": torch.nn.functional.max_pool1d,
            "nn.quantized.functional": torch.nn.quantized.functional.max_pool1d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.max_pool1d,
        }

        for name, op in ops_under_test.items():
            a_hat = op(qa, kernel_size=kernel, stride=stride, padding=padding,
                       dilation=dilation, ceil_mode=ceil_mode)
            self.assertEqual(a_ref, a_hat.dequantize(),
                             msg="{} results are off".format(name))
        # Test the ops.quantized separately, because None is not treated.
        a_hat = torch.ops.quantized.max_pool1d(
            qa, kernel_size=_single(kernel),
            stride=_single(kernel if stride is None else stride),
            padding=_single(padding), dilation=_single(dilation),
            ceil_mode=ceil_mode)
        self.assertEqual(a_ref, a_hat.dequantize(),
                         msg="ops.quantized.max_pool1d results are off")

    # TODO: merge this test with test_max_pool2d when USE_EXPERIMENTAL_CUDNN_V8_API flag is enabled in CI
    """Tests 2D cudnn max pool operation on quantized tensors."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                              min_side=1, max_side=10),
                       # cudnn's support for quantized pooling is limited to
                       # int8 currently
                       qparams=hu.qparams(dtypes=[torch.qint8])),
           kernel=st.sampled_from((3, 5, 7)),
           stride=st.sampled_from((None, 1, 2)),
           # currently there is no support for dilation for cudnn
           # pooling
           dilation=st.integers(1, 1),
           padding=st.integers(0, 2),
           ceil_mode=st.booleans())
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    @unittest.skip("Local only - currently the qconv2d_cudnn op is bulid "
                   "with USE_EXPERIMENTAL_CUDNN_V8_API, we can enable the test "
                   "after it is built by default")
    def test_max_pool2d_cudnn(self, X, kernel, stride, dilation, padding, ceil_mode):
        X, (scale, zero_point, torch_type) = X
        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        iH, iW = X.shape[-2:]
        oH = pool_output_shape(iH, kernel, padding, stride, dilation, ceil_mode)
        assume(oH > 0)
        oW = pool_output_shape(iW, kernel, padding, stride, dilation, ceil_mode)
        assume(oW > 0)

        a = torch.from_numpy(X).to(device="cuda")
        a_pool = torch.nn.functional.max_pool2d(a, kernel_size=kernel,
                                                stride=stride,
                                                padding=padding, dilation=dilation,
                                                ceil_mode=ceil_mode)
        a_ref = torch.quantize_per_tensor(a_pool, scale=scale,
                                          zero_point=zero_point, dtype=torch_type)
        a_ref = a_ref.dequantize()
        qa = torch.quantize_per_tensor(a, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # Test the ops.quantized separately, because None is not treated.
        a_hat = torch.ops.quantized.max_pool2d(
            qa, kernel_size=_pair(kernel),
            stride=_pair(kernel if stride is None else stride),
            padding=_pair(padding), dilation=_pair(dilation), ceil_mode=ceil_mode)
        self.assertEqual(a_ref, a_hat.dequantize(),
                         msg="ops.quantized.max_pool2d results are off")

    """Tests 2D max pool operation on quantized tensors."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           kernel=st.sampled_from((3, 5, 7)),
           stride=st.sampled_from((None, 1, 2)),
           dilation=st.integers(1, 2),
           padding=st.integers(0, 2),
           ceil_mode=st.booleans())
    def test_max_pool2d(self, X, kernel, stride, dilation, padding, ceil_mode):
        X, (scale, zero_point, torch_type) = X
        # Check constraints
        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        iH, iW = X.shape[-2:]
        oH = pool_output_shape(iH, kernel, padding, stride, dilation, ceil_mode)
        assume(oH > 0)
        oW = pool_output_shape(iW, kernel, padding, stride, dilation, ceil_mode)
        assume(oW > 0)

        a = torch.from_numpy(X)
        a_pool = torch.nn.functional.max_pool2d(a, kernel_size=kernel,
                                                stride=stride,
                                                padding=padding, dilation=dilation,
                                                ceil_mode=ceil_mode)
        a_ref = torch.quantize_per_tensor(a_pool, scale=scale,
                                          zero_point=zero_point, dtype=torch_type)
        a_ref = a_ref.dequantize()
        qa = torch.quantize_per_tensor(a, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        ops_under_test = {
            "torch": torch.max_pool2d,
            "nn.functional": torch.nn.functional.max_pool2d,
            "nn.quantized.functional": torch.nn.quantized.functional.max_pool2d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.max_pool2d,
        }

        for name, op in ops_under_test.items():
            a_hat = op(qa, kernel_size=kernel, stride=stride, padding=padding,
                       dilation=dilation, ceil_mode=ceil_mode)
            self.assertEqual(a_ref, a_hat.dequantize(),
                             msg="{} results are off".format(name))
        # Test the ops.quantized separately, because None is not treated.
        a_hat = torch.ops.quantized.max_pool2d(
            qa, kernel_size=_pair(kernel),
            stride=_pair(kernel if stride is None else stride),
            padding=_pair(padding), dilation=_pair(dilation), ceil_mode=ceil_mode)
        self.assertEqual(a_ref, a_hat.dequantize(),
                         msg="ops.quantized.max_pool2d results are off")

    """Tests max pool operation on NHWC quantized tensors."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           kernel=st.sampled_from((3, 5, 7)),
           stride=st.sampled_from((None, 1, 2)),
           dilation=st.integers(1, 2),
           padding=st.integers(0, 2),
           ceil_mode=st.booleans())
    def test_max_pool2d_nhwc(self, X, kernel, stride, dilation, padding, ceil_mode):
        X, (scale, zero_point, torch_type) = X
        # Ensure we hit the vectorized paths
        # 176 = 128 + 32 + 16
        # 128 hits the interleaved path
        # 32 hits the non-interleaved path
        # 16 hits the scalar path
        if X.shape[1] < 176:
            X = np.repeat(X, 176 / X.shape[1], 1)
        # Check constraints
        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        iH, iW = X.shape[-2:]
        oH = pool_output_shape(iH, kernel, padding, stride, dilation, ceil_mode)
        assume(oH > 0)
        oW = pool_output_shape(iW, kernel, padding, stride, dilation, ceil_mode)
        assume(oW > 0)

        X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 1]))
        a = torch.from_numpy(X_nchw).permute([0, 3, 1, 2])
        a_pool = torch.nn.functional.max_pool2d(a, kernel_size=kernel,
                                                stride=stride,
                                                padding=padding, dilation=dilation,
                                                ceil_mode=ceil_mode)
        a_ref = torch.quantize_per_tensor(a_pool, scale=scale,
                                          zero_point=zero_point, dtype=torch_type)
        a_ref = a_ref.dequantize()
        qa = torch.quantize_per_tensor(torch.from_numpy(X_nchw), scale=scale, zero_point=zero_point,
                                       dtype=torch_type).permute([0, 3, 1, 2])
        self.assertTrue(qa.stride() != sorted(qa.stride()))

        ops_under_test = {
            "torch": torch.max_pool2d,
            "nn.functional": torch.nn.functional.max_pool2d,
            "nn.quantized.functional": torch.nn.quantized.functional.max_pool2d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.max_pool2d,
        }

        for name, op in ops_under_test.items():
            a_hat = op(qa, kernel_size=kernel, stride=stride, padding=padding,
                       dilation=dilation, ceil_mode=ceil_mode)
            self.assertTrue(a_hat.stride() != sorted(a_hat.stride()))
            self.assertEqual(a_ref, a_hat.dequantize(),
                             msg="{} results are off".format(name))
        # Test the ops.quantized separately, because None is not treated.
        a_hat = torch.ops.quantized.max_pool2d(
            qa, kernel_size=_pair(kernel),
            stride=_pair(kernel if stride is None else stride),
            padding=_pair(padding), dilation=_pair(dilation), ceil_mode=ceil_mode)
        self.assertEqual(a_ref, a_hat.dequantize(),
                         msg="ops.quantized.max_pool2d results are off")

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                              min_side=5, max_side=10),
                       qparams=hu.qparams(dtypes=torch.quint8)),
           kernel=st.sampled_from((3, 5)),
           stride=st.sampled_from((None, 1, 2)),
           padding=st.integers(0, 2),
           ceil_mode=st.sampled_from((True, False)),
           count_include_pad=st.sampled_from((True, False)),
           divisor_override=st.sampled_from((None, None)))
    def test_avg_pool2d(self, X, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override):
        """
        Note: we currently cannot test the divisor_override, because quantized op will clamp the result
        within range. However, the float op will not.
        """
        X, (scale, zero_point, torch_type) = X

        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        iH, iW = X.shape[-2:]
        oH = pool_output_shape(iH, kernel, padding, stride, dilation=1)
        assume(oH > 0)
        oW = pool_output_shape(iW, kernel, padding, stride, dilation=1)
        assume(oW > 0)
        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)
        X = qX.dequantize()
        # Run reference on float tensor and then quantize the result for comparison
        X_ref = torch.nn.functional.avg_pool2d(
            X, kernel_size=kernel, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)
        ops_under_test = {
            "nn.functional": torch.nn.functional.avg_pool2d,
            "nn.quantized.functional": torch.nn.quantized.functional.avg_pool2d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.avg_pool2d,
        }
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            qX_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad, divisor_override=divisor_override)
            qX_ref = torch.quantize_per_tensor(X_ref, scale=qX_hat.q_scale(), zero_point=qX_hat.q_zero_point(),
                                               dtype=torch_type)

            self.assertEqual(qX_ref.int_repr().to(torch.double), qX_hat.int_repr().to(torch.double), atol=1.0, rtol=0,
                             msg=error_message.format(name, qX_ref.int_repr(), qX_hat.int_repr()))
            self.assertEqual(scale, qX_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale,
                                                      qX_hat.q_zero_point()))

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=5, max_side=10),
                       qparams=hu.qparams(dtypes=torch.qint8)),
           kernel=st.sampled_from((4, 5)),
           stride=st.sampled_from((None, 1, 2)),
           padding=st.integers(0, 2),
           ceil_mode=st.sampled_from((True, False)),
           count_include_pad=st.sampled_from((True, False)),
           divisor_override=st.sampled_from((None, None)))
    def test_avg_pool2d_nhwc(self, X, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override):
        """
        Note: 1) we currently cannot test the divisor_override, because quantized op will clamp the result
        within range. However, the float op will not.
        2) we cannot test the qint32, since the float point precision is much lower than int32 for big number,
        which will make the test be very flaky.
        """
        X, (scale, zero_point, torch_type) = X
        H, W = X.shape[-2:]


        if X.shape[1] < 176:
            X = np.repeat(X, 176 / X.shape[1], 1)

        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        iH, iW = X.shape[-2:]
        oH = pool_output_shape(iH, kernel, padding, stride, dilation=1)
        assume(oH > 0)
        oW = pool_output_shape(iW, kernel, padding, stride, dilation=1)
        assume(oW > 0)

        X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 1]))

        qX = torch.quantize_per_tensor(torch.from_numpy(X_nchw), scale=scale,
                                       zero_point=zero_point, dtype=torch_type).permute([0, 3, 1, 2])
        X = qX.dequantize()

        # Run reference on int_repr + round to avoid double rounding error.
        X_ref = torch.nn.functional.avg_pool2d(
            X, kernel_size=kernel, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)

        self.assertTrue(qX.stride() != sorted(qX.stride()))
        ops_under_test = {
            "nn.functional": torch.nn.functional.avg_pool2d,
            "nn.quantized.functional": torch.nn.quantized.functional.avg_pool2d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.avg_pool2d,
        }
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            X_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                       count_include_pad=count_include_pad, divisor_override=divisor_override)
            self.assertTrue(X_hat.stride() != sorted(X_hat.stride()))
            qX_ref = torch.quantize_per_tensor(X_ref, scale=X_hat.q_scale(), zero_point=X_hat.q_zero_point(),
                                               dtype=torch_type)

            self.assertEqual(qX_ref.int_repr().to(torch.double), X_hat.int_repr().to(torch.double), atol=1.0, rtol=0,
                             msg=error_message.format(name, qX_ref.int_repr(), X_hat.int_repr()))
            self.assertEqual(scale, X_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, X_hat.q_scale()))
            self.assertEqual(zero_point, X_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale,
                             X_hat.q_zero_point()))

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=5, max_dims=5,
                                              min_side=5, max_side=10),
                       qparams=hu.qparams(dtypes=torch.quint8)),
           kernel=st.sampled_from((3, 5)),
           stride=st.sampled_from((None, 1, 2)),
           padding=st.integers(0, 2),
           ceil_mode=st.sampled_from((True, False)),
           count_include_pad=st.sampled_from((True, False)),
           divisor_override=st.sampled_from((None, None)))
    def test_avg_pool3d(self, X, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override):
        """
        Note: we currently cannot test the divisor_override, because quantized op will clamp the result
        within range. However, the float op will not.
        """
        X, (scale, zero_point, torch_type) = X

        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        iD, iH, iW = X.shape[-3:]
        oD = pool_output_shape(iD, kernel, padding, stride, dilation=1)
        assume(oD > 0)
        oH = pool_output_shape(iH, kernel, padding, stride, dilation=1)
        assume(oH > 0)
        oW = pool_output_shape(iW, kernel, padding, stride, dilation=1)
        assume(oW > 0)

        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)
        X = qX.dequantize()
        # Run reference on float tensor and then quantize the result for comparison
        X_ref = torch.nn.functional.avg_pool3d(
            X, kernel_size=kernel, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)

        ops_under_test = {
            "nn.functional": torch.nn.functional.avg_pool3d,
            "nn.quantized.functional": torch.nn.quantized.functional.avg_pool3d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.avg_pool3d,
        }
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            qX_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad, divisor_override=divisor_override)
            qX_ref = torch.quantize_per_tensor(X_ref, scale=qX_hat.q_scale(), zero_point=qX_hat.q_zero_point(),
                                               dtype=torch_type)
            self.assertEqual(qX_ref.int_repr().to(torch.double), qX_hat.int_repr().to(torch.double), atol=1.0, rtol=0,
                             msg=error_message.format(name, qX_ref.int_repr(), qX_hat.int_repr()))
            self.assertEqual(scale, qX_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale,
                                                      qX_hat.q_zero_point()))

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=5, max_dims=5,
                                              min_side=5, max_side=10),
                       qparams=hu.qparams(dtypes=torch.qint8)),
           kernel=st.sampled_from((4, 5)),
           stride=st.sampled_from((None, 1, 2)),
           padding=st.integers(0, 2),
           ceil_mode=st.sampled_from((True, False)),
           count_include_pad=st.sampled_from((True, False)),
           divisor_override=st.sampled_from((None, None)))
    def test_avg_pool3d_nhwc(self, X, kernel, stride, padding, ceil_mode, count_include_pad, divisor_override):
        """
        Note: 1) we currently cannot test the divisor_override, because quantized op will clamp the result
        within range. However, the float op will not.
        2) we cannot test the qint32, since the float point precision is much lower than int32 for big number,
        which will make the test be very flaky.
        """
        X, (scale, zero_point, torch_type) = X
        D, H, W = X.shape[-3:]


        if X.shape[1] < 176:
            X = np.repeat(X, 176 / X.shape[1], 1)

        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        iD, iH, iW = X.shape[-3:]
        oD = pool_output_shape(iD, kernel, padding, stride, dilation=1)
        assume(oD > 0)
        oH = pool_output_shape(iH, kernel, padding, stride, dilation=1)
        assume(oH > 0)
        oW = pool_output_shape(iW, kernel, padding, stride, dilation=1)
        assume(oW > 0)

        X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 4, 1]))

        qX = torch.quantize_per_tensor(torch.from_numpy(X_nchw), scale=scale,
                                       zero_point=zero_point, dtype=torch_type).permute([0, 4, 1, 2, 3])
        X = qX.dequantize()

        # Run reference on int_repr + round to avoid double rounding error.
        X_ref = torch.nn.functional.avg_pool3d(
            X, kernel_size=kernel, stride=stride, padding=padding,
            ceil_mode=ceil_mode, count_include_pad=count_include_pad, divisor_override=divisor_override)

        self.assertTrue(qX.stride() != sorted(qX.stride()))
        ops_under_test = {
            "nn.functional": torch.nn.functional.avg_pool3d,
            "nn.quantized.functional": torch.nn.quantized.functional.avg_pool3d,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.avg_pool3d,
        }
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            X_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                       count_include_pad=count_include_pad, divisor_override=divisor_override)
            self.assertTrue(X_hat.stride() != sorted(X_hat.stride()))
            qX_ref = torch.quantize_per_tensor(X_ref, scale=X_hat.q_scale(), zero_point=X_hat.q_zero_point(),
                                               dtype=torch_type)

            self.assertEqual(qX_ref.int_repr().to(torch.double), X_hat.int_repr().to(torch.double), atol=1.0, rtol=0,
                             msg=error_message.format(name, qX_ref.int_repr(), X_hat.int_repr()))
            self.assertEqual(scale, X_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, X_hat.q_scale()))
            self.assertEqual(zero_point, X_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale,
                             X_hat.q_zero_point()))

    """Tests adaptive average pool operation on NHWC quantized tensors."""
    def test_adaptive_avg_pool2d_nhwc(self):
        side_lens = (range(1, 10))
        dim_lens = (range(3, 4))
        torch_type = torch.qint8
        zero_points = (0, 1)
        combined = [side_lens, dim_lens, zero_points]
        test_cases = itertools.product(*combined)
        for test_case in test_cases:
            output_size_h = random.randint(1, 10)
            output_size_w = random.randint(1, 10)
            side_len, dim_len, zero_point = test_case
            shapes = [side_len] * dim_len
            X, X_scale, X_zero_point = \
                _get_random_tensor_and_q_params(shapes, 1.0, zero_point)
            X = np.array(X)
            scale = 1
            H, W = X.shape[-2:]
            output_size_h = output_size_h if (output_size_h <= H) else H
            output_size_w = output_size_w if (output_size_w <= W) else W
            if output_size_h == output_size_w:
                output_size = output_size_h
            else:
                output_size = (output_size_h, output_size_w)

            if X.shape[1] < 176:
                X = np.repeat(X, 176 / X.shape[1], 1)

            if X.ndim == 4:
                X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 1]))
                X = torch.from_numpy(X_nchw).permute([0, 3, 1, 2])
                qX = torch.quantize_per_tensor(torch.from_numpy(X_nchw),
                                               scale=scale,
                                               zero_point=zero_point,
                                               dtype=torch_type).permute([0, 3, 1, 2])
            else:  # ndim == 3
                X_nchw = np.ascontiguousarray(X.transpose([1, 2, 0]))
                X = torch.from_numpy(X_nchw).permute([2, 0, 1])
                qX = torch.quantize_per_tensor(torch.from_numpy(X_nchw),
                                               scale=scale,
                                               zero_point=zero_point,
                                               dtype=torch_type).permute([2, 0, 1])

            # Run reference on int_repr + round to avoid double rounding error.
            X_ref = torch.nn.functional.adaptive_avg_pool2d(qX.int_repr().to(torch.double), output_size).round()

            self.assertTrue(qX.stride() != sorted(qX.stride()))

            ops_under_test = {
                "nn.functional": torch.nn.functional.adaptive_avg_pool2d,
                "nn.quantized.functional":
                    torch.nn.quantized.functional.adaptive_avg_pool2d,
                "ao.nn.quantized.functional":
                    torch.ao.nn.quantized.functional.adaptive_avg_pool2d,
            }
            error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
            for name, op in ops_under_test.items():
                X_hat = op(qX, output_size=output_size)
                self.assertTrue(X_hat.stride() != sorted(X_hat.stride()))
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(X_ref, X_hat.int_repr(), atol=1.0, rtol=0,
                                           msg=error_message.format(name, X_ref, X_hat.int_repr()))
                self.assertEqual(scale, X_hat.q_scale(),
                                 msg=error_message.format(name + '.scale', scale, X_hat.q_scale()))
                self.assertEqual(zero_point, X_hat.q_zero_point(),
                                 msg=error_message.format(name + '.zero_point', scale,
                                 X_hat.q_zero_point()))

    def test_adaptive_avg_pool(self):

        side_lens = (range(1, 10))
        dim_lens = (range(3, 5))
        torch_type = torch.qint8
        zero_points = (0, 1)
        combined = [side_lens, dim_lens, zero_points]
        test_cases = itertools.product(*combined)
        for test_case in test_cases:
            output_size_d = random.randint(1, 10)
            output_size_h = random.randint(1, 10)
            output_size_w = random.randint(1, 10)
            side_len, dim_len, zero_point = test_case
            shapes = [side_len] * dim_len
            X, X_scale, X_zero_point = \
                _get_random_tensor_and_q_params(shapes, 1.0, zero_point)
            X = np.array(X)
            scale = 1
            ndim = X.ndim
            dim_to_check = []
            if ndim <= 4:
                dim_to_check.append(2)
            if ndim >= 4:
                dim_to_check.append(3)

            D, H, W = X.shape[-3:]
            output_size_d = output_size_d if (output_size_d <= D) else D
            output_size_h = output_size_h if (output_size_h <= H) else H
            output_size_w = output_size_w if (output_size_w <= W) else W

            X = torch.from_numpy(X)
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

            for dim in dim_to_check:
                if dim == 2:
                    if output_size_h == output_size_w:
                        output_size = output_size_h
                    else:
                        output_size = (output_size_h, output_size_w)
                elif dim == 3:
                    if output_size_d == output_size_h == output_size_w:
                        output_size = output_size_h
                    else:
                        output_size = (output_size_d, output_size_h, output_size_w)

                # Run reference on int_repr + round to avoid double rounding error.
                ref_op = getattr(torch.nn.functional, 'adaptive_avg_pool{}d'.format(dim))
                X_ref = ref_op(qX.int_repr().to(torch.float), output_size).round()

                ops_under_test = {
                    "nn.functional":
                        getattr(torch.nn.functional, 'adaptive_avg_pool{}d'.format(dim)),
                    "nn.quantized.functional":
                        getattr(torch.nn.quantized.functional, 'adaptive_avg_pool{}d'.format(dim)),
                    "ao.nn.quantized.functional":
                        getattr(torch.ao.nn.quantized.functional, 'adaptive_avg_pool{}d'.format(dim))
                }

                error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"

                for name, op in ops_under_test.items():
                    # TODO: torch.cuda.is_available() should be swapped for a flag that checks if cudnn
                    # is enabled in the build when cudnn supports adaptive average pooling
                    devices = ["cpu", "cuda"] if (dim == 2 and torch.cuda.is_available()) else ["cpu"]
                    for device in devices:
                        qX_hat = op(qX.to(device=device), output_size=output_size)
                        # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                        self.assertEqualIgnoreType(
                            X_ref, qX_hat.int_repr(), atol=1.0,
                            rtol=0, msg=error_message.format(name, X_ref, qX_hat))
                        self.assertEqual(
                            scale, qX_hat.q_scale(),
                            msg=error_message.format(name + '.scale', scale,
                                                     qX_hat.q_scale()))
                        self.assertEqual(
                            zero_point, qX_hat.q_zero_point(),
                            msg=error_message.format(name + '.zero_point', scale,
                                                     qX_hat.q_zero_point()))

    """Tests adaptive average pool operation on NHWC quantized tensors."""
    def test_adaptive_avg_pool3d_ndhwc(self):
        side_lens = (range(1, 10))
        dim_lens = (range(4, 5))
        torch_type = torch.qint8
        zero_point = 0
        combined = [side_lens, dim_lens]
        test_cases = itertools.product(*combined)
        for test_case in test_cases:
            output_size_d = random.randint(1, 10)
            output_size_h = random.randint(1, 10)
            output_size_w = random.randint(1, 10)
            side_len, dim_len = test_case
            shapes = [side_len] * dim_len
            X, X_scale, X_zero_point = \
                _get_random_tensor_and_q_params(shapes, 1.0, zero_point)
            X = np.array(X)
            scale = 1
            D, H, W = X.shape[-3:]
            output_size_d = output_size_d if (output_size_d <= D) else D
            output_size_h = output_size_h if (output_size_h <= H) else H
            output_size_w = output_size_w if (output_size_w <= W) else W
            if output_size_d == output_size_h == output_size_w:
                output_size = output_size_h
            else:
                output_size = (output_size_d, output_size_h, output_size_w)

            if X.shape[1] < 176:
                X = np.repeat(X, 176 / X.shape[1], 1)

            if X.ndim == 5:
                X_ncdhw = np.ascontiguousarray(X.transpose([0, 2, 3, 4, 1]))
                X = torch.from_numpy(X_ncdhw).permute([0, 4, 1, 2, 3])
                qX = torch.quantize_per_tensor(torch.from_numpy(X_ncdhw),
                                               scale=scale,
                                               zero_point=zero_point,
                                               dtype=torch_type).permute([0, 4, 1, 2, 3])
            else:  # ndim == 4
                X_ncdhw = np.ascontiguousarray(X.transpose([1, 2, 3, 0]))
                X = torch.from_numpy(X_ncdhw).permute([3, 0, 1, 2])
                qX = torch.quantize_per_tensor(torch.from_numpy(X_ncdhw),
                                               scale=scale,
                                               zero_point=zero_point,
                                               dtype=torch_type).permute([3, 0, 1, 2])

            # Run reference on int_repr + round to avoid double rounding error.
            X_ref = torch.nn.functional.adaptive_avg_pool3d(
                qX.int_repr().to(torch.double), output_size).round()

            self.assertTrue(qX.stride() != sorted(qX.stride()))

            ops_under_test = {
                "nn.functional": torch.nn.functional.adaptive_avg_pool3d,
                "nn.quantized.functional":
                    torch.nn.quantized.functional.adaptive_avg_pool3d,
                "ao.nn.quantized.functional":
                    torch.ao.nn.quantized.functional.adaptive_avg_pool3d,
            }
            error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
            for name, op in ops_under_test.items():
                X_hat = op(qX, output_size=output_size)
                self.assertTrue(X_hat.stride() != sorted(X_hat.stride()))
                # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                self.assertEqualIgnoreType(X_ref, X_hat.int_repr(), atol=1.0, rtol=0,
                                           msg=error_message.format(name, X_ref, X_hat.int_repr()))
                self.assertEqual(scale, X_hat.q_scale(),
                                 msg=error_message.format(name + '.scale', scale, X_hat.q_scale()))
                self.assertEqual(zero_point, X_hat.q_zero_point(),
                                 msg=error_message.format(name + '.zero_point', scale,
                                 X_hat.q_zero_point()))

    def test_qtopk(self):
        x_dims = [3, 4]  # Num elements in the shape
        sides = [3, 5]  # Side of the tensor generated
        dims = [0, 1, 2, 3]  # dimension over which to perform topk
        largest = [False, True]  # Return largest or smallest element
        sorted = [False, True]  # Return sorted or not
        dtypes = [torch.qint8, torch.quint8]
        is_nhwc = [False, True]  # Is input in the NHWC format?

        test_cases = itertools.product(x_dims, sides, dims, largest, sorted, dtypes, is_nhwc)
        k = 2
        for x_dim, side, dim, larg, sort, dtype, nhwc in test_cases:
            if nhwc and x_dim != 4:  # NHWC requires 4 dimensions
                continue
            if dim >= x_dim:  # Dimension to find top-k for should exist
                continue
            shape = [side] * x_dim
            X, scale, zp = _get_random_tensor_and_q_params(shape, 1.0, dtype)
            qX = torch.quantize_per_tensor(X, scale, zp, dtype)

            if nhwc:
                qX = qX.permute([0, 3, 1, 2])
                X = np.transpose(X, [0, 3, 1, 2])

            unquantized_out = torch.topk(qX.dequantize(), k, dim=dim, largest=larg, sorted=sort)

            values = torch.quantize_per_tensor(X, scale, zp, dtype)
            indices = torch.tensor(X).long()

            quantized_out = torch.topk(qX, k, dim=dim, largest=larg, sorted=sort)

            assert(len(unquantized_out) == len(quantized_out))
            torch.testing.assert_close(quantized_out[0].dequantize(), unquantized_out[0])
            torch.testing.assert_close(quantized_out[1], unquantized_out[1])

    """Tests quantize concatenation (both fused and not)."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           num=st.integers(1, 4),
           dim=st.integers(1, 4),
           relu=st.booleans())
    def test_cat(self, X, num, dim, relu):
        tensors_q = []
        tensors_ref = []
        X, (scale, zero_point, torch_type) = X
        assume(dim < X.ndim)
        X = torch.from_numpy(X)
        new_shape = np.array(X.shape)
        new_shape[dim] = 0
        for idx in range(num):
            tensors_q.append(torch.quantize_per_tensor(X, scale, zero_point,
                                                       torch_type))
            tensors_ref.append(X)
            new_shape[dim] += tensors_ref[-1].shape[dim]

        cat_ref = torch.cat(tensors_ref, dim=dim)
        cat_ref = torch.quantize_per_tensor(cat_ref, scale, zero_point, torch_type)
        cat_ref = cat_ref.dequantize()

        if relu:
            cat_ref = F.relu(cat_ref)
            q_cat_op = torch.ops.quantized.cat_relu
            q_cat_out_op = torch.ops.quantized.cat_relu_out
        else:
            q_cat_op = torch.ops.quantized.cat
            q_cat_out_op = torch.ops.quantized.cat_out

        cat_q = q_cat_op(tensors_q, dim=dim, scale=scale,
                         zero_point=zero_point)
        cat_q = cat_q.dequantize()
        np.testing.assert_equal(cat_ref.numpy(), cat_q.numpy())

        cat_q_out = torch._empty_affine_quantized(
            list(new_shape), scale=scale,
            zero_point=zero_point, dtype=torch_type)
        q_cat_out_op(tensors_q, dim=dim, out=cat_q_out)
        cat_q_out = cat_q_out.dequantize()
        np.testing.assert_equal(cat_ref.numpy(), cat_q_out.numpy())

        # Test the cat on per-channel quantized tensor.
        ch_axis = 1
        scales = torch.from_numpy(np.array([1.0] * X.shape[ch_axis]))
        scales = scales.to(torch.float64)
        zero_points = torch.from_numpy(np.array([0] * X.shape[ch_axis]))
        zero_points = zero_points.to(torch.long)
        tensors_q[0] = torch.quantize_per_channel(
            X, scales, zero_points, axis=ch_axis, dtype=torch_type)
        with self.assertRaisesRegex(RuntimeError, "supported.*cat"):
            cat_q = q_cat_op(tensors_q, dim=ch_axis, scale=scale,
                             zero_point=zero_point)

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=5, max_side=10),
                       qparams=hu.qparams()),
           size=st.sampled_from((1, 3, 5, 10)),
           mode=st.sampled_from(("bilinear", "nearest", "nearest-exact")),
           scale_factor=st.sampled_from((None, 1.5, 2.0)),
           align_corners=st.sampled_from((True, False)),
           nhwc_layout=st.sampled_from((True, False)))
    def test_interpolate(self, X, size, mode, scale_factor, align_corners, nhwc_layout):
        """
        This test cover upsample_nearest2d and upsample_bilinear2d
        """
        X, (scale, zero_point, torch_type) = X

        if scale_factor is not None:
            size = None
        if mode in ("nearest", "nearest-exact"):
            align_corners = None

        if nhwc_layout:
            if X.shape[1] < 176:
                X = np.repeat(X, 176 / X.shape[1], 1)

            X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 1]))
            X = torch.from_numpy(X_nchw).permute([0, 3, 1, 2])

            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type).permute([0, 3, 1, 2])
        else:
            X = torch.from_numpy(X)
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

        X_ref = torch.nn.functional.interpolate(
            qX.int_repr().to(torch.float), size=size, scale_factor=scale_factor,
            mode=mode, align_corners=align_corners)

        ops_under_test = {
            "nn.functional": torch.nn.functional.interpolate,
            "nn.quantized.functional": torch.nn.quantized.functional.interpolate,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.interpolate,
        }
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            qX_hat = op(qX, size=size, scale_factor=scale_factor,
                        mode=mode, align_corners=align_corners)
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(X_ref, qX_hat.int_repr(), atol=1.0, rtol=0,
                                       msg="{} results are off: qX_hat={} X_ref={}"
                                           .format(name, qX_hat.int_repr(), X_ref))
            self.assertEqual(scale, qX_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale,
                                                      qX_hat.q_zero_point()))

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=5, max_dims=5,
                                              min_side=5, max_side=10),
                       qparams=hu.qparams()),
           size=st.sampled_from((1, 3, 5, 5, 10)),
           mode=st.sampled_from(("nearest", "nearest-exact")),
           scale_factor=st.sampled_from((None, 1.5, 2.0)),
           align_corners=st.sampled_from((True, False)),
           nhwc_layout=st.sampled_from((True, False)))
    def test_interpolate3d(self, X, size, mode, scale_factor, align_corners, nhwc_layout):
        """
        This test cover upsample_nearest3d
        """
        X, (scale, zero_point, torch_type) = X
        if scale_factor is not None:
            size = None

        align_corners = None

        if nhwc_layout:
            if X.shape[1] < 176:
                X = np.repeat(X, 176 / X.shape[1], 1)

            X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 4, 1]))
            X = torch.from_numpy(X_nchw).permute([0, 4, 1, 2, 3])

            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type).permute([0, 4, 1, 2, 3])
        else:
            X = torch.from_numpy(X)
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
        X_ref = torch.nn.functional.interpolate(
            qX.int_repr().to(torch.float), size=size, scale_factor=scale_factor,
            mode=mode, align_corners=align_corners)

        ops_under_test = {
            "nn.functional": torch.nn.functional.interpolate,
            "nn.quantized.functional": torch.nn.quantized.functional.interpolate,
            "ao.nn.quantized.functional": torch.ao.nn.quantized.functional.interpolate,
        }

        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            qX_hat = op(qX, size=size, scale_factor=scale_factor,
                        mode=mode, align_corners=align_corners)
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(X_ref, qX_hat.int_repr(), atol=1.0, rtol=0,
                                       msg="{} results are off: qX_hat={}, X_ref={}"
                                           .format(name, qX_hat.int_repr(), X_ref))
            self.assertEqual(scale, qX_hat.q_scale(),
                             msg=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             msg=error_message.format(name + '.zero_point', scale,
                                                      qX_hat.q_zero_point()))

    """Tests quantize concatenation (both fused and not)."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           relu=st.booleans())
    def test_cat_nhwc(self, X, relu):
        # X is NHWC
        X, (scale, zero_point, torch_type) = X

        # Tile out X so # channels is > 64
        X = np.repeat(X, 70 / X.shape[3], 3)
        X = torch.from_numpy(np.ascontiguousarray(X))
        Y = X.clone()
        Y = torch.from_numpy(np.ascontiguousarray(Y))
        # We add a fast path in qcat: when inputs share the same scale and zero_point,
        # it will go direct memcpy instead of dequant-cat-quant.
        for scaleX, scaleY in ((scale, scale), (scale, scale * 1.1)):
            # Here, we quantize and get quantized tensors in NHWC for both dims and strides. The
            # permute switches it so that the tensor looks like NCHW but it laid out in memory as
            # NHWC.
            qX = torch.quantize_per_tensor(X, scaleX, zero_point, torch_type).permute([0, 3, 1, 2])
            qY = torch.quantize_per_tensor(Y, scaleY, zero_point, torch_type).permute([0, 3, 1, 2])

            ref = torch.cat([qX.dequantize(), qY.dequantize()], dim=1)
            if relu:
                ref[ref < 0] = 0.0
            ref = torch.quantize_per_tensor(ref, scale=scale, zero_point=zero_point, dtype=torch_type)

            if relu:
                out = torch.ops.quantized.cat_relu(
                    [qX, qY], dim=1, scale=scale, zero_point=zero_point)
            else:
                out = torch.ops.quantized.cat([qX, qY], dim=1, scale=scale, zero_point=zero_point)

            torch.testing.assert_close(out.dequantize(), ref.dequantize())
            self.assertNotEqual(out.stride(), sorted(out.stride()))

    @override_qengines
    def test_mean(self):
        scale_list = (1, 0.25)
        zero_point_list = (0, 2)
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4), (4, 4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        dims = ((), (-1,), (0,), (1,), (2,), (3,), (0, 1), (1, 2), (3, 4))
        test_cases = itertools.product(scale_list, zero_point_list, shapes, dtypes, dims)
        op = torch.mean
        for scale, zp, shape, dtype, dim in test_cases:
            if not all([d < len(shape) for d in dim]):
                continue
            X = torch.randn(*shape) * 10
            qX = torch.quantize_per_tensor(X, scale, zp, dtype)
            Y = op(qX.dequantize(), dim)
            Y = torch.quantize_per_tensor(Y, scale, zp, dtype).dequantize()
            qY = op(qX, dim)
            self.assertEqual(Y, qY.dequantize())

    @skipIfNoQNNPACK
    @given(keep=st.booleans())
    def test_quantized_mean_qnnpack(self, keep):
        with override_quantized_engine("qnnpack"):
            # using multiple of 4 sizes to satisfy pytorch_q8gavgpool_ukernel_up8xm__sse2() 4-byte alignment demand under ASAN
            in_dim = (4, 4, 4, 4)
            if keep:
                out_dim = (4, 4, 1, 1)
            else:
                out_dim = (4, 4)
            X = torch.ones(in_dim)
            Y = torch.ones(out_dim)
            XQ = torch.quantize_per_tensor(X, scale=0.2, zero_point=0, dtype=torch.quint8)
            YQ = torch.quantize_per_tensor(Y, scale=0.2, zero_point=0, dtype=torch.quint8)
            MQ = XQ.mean((2, 3), keepdim=keep)
            self.assertTrue(torch.equal(MQ, YQ))

    @override_qengines
    def test_std(self):
        scale_list = (1, 0.25)
        zero_point_list = (0, 2)
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4), (4, 4, 4, 4, 4))
        dtypes = (torch.quint8, torch.qint8)
        dims = ((), (-1,), (0,), (1,), (2,), (3,), (0, 1), (1, 2), (3, 4))
        unbiased_list = (True, False)
        keep_dim_list = (True, False)
        test_cases = itertools.product(scale_list, zero_point_list, shapes,
                                       dtypes, dims, unbiased_list, keep_dim_list)
        op = torch.std
        for scale, zp, shape, dtype, dim, unbiased, keep_dim in test_cases:
            if not all([d < len(shape) for d in dim]):
                continue
            X = torch.randn(*shape) * 10
            qX = torch.quantize_per_tensor(X, scale, zp, dtype)
            Y = op(qX.dequantize(), dim, unbiased, keep_dim)
            Y = torch.quantize_per_tensor(Y, scale, zp, dtype).dequantize()
            qY = op(qX, dim, unbiased, keep_dim)
            self.assertEqual(Y, qY.dequantize())

    """Tests the correctness of the quantized equal op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()),
           X2=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                        qparams=hu.qparams()),
           X_per_channel=st.booleans(),
           X2_per_channel=st.booleans())
    def test_equal(self, X, X2, X_per_channel, X2_per_channel):
        X, X_params = X
        (scale, zero_point, torch_type) = X_params
        X2, X2_params = X2
        (scale2, zero_point2, torch_type2) = X2_params

        X = torch.from_numpy(X)
        if X_per_channel:
            X_scheme = 'per_channel'
            channels = X.shape[-1]
            qX = torch.quantize_per_channel(
                X,
                scales=torch.tensor([scale] * channels),
                zero_points=torch.tensor([zero_point] * channels),
                dtype=torch_type,
                axis=X.ndim - 1)
        else:
            X_scheme = 'per_tensor'
            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
        X2 = torch.from_numpy(X2)
        if X2_per_channel:
            X2_scheme = 'per_channel'
            channels = X2.shape[-1]
            qX2 = torch.quantize_per_channel(
                X2,
                scales=torch.tensor([scale2] * channels),
                zero_points=torch.tensor([zero_point2] * channels),
                dtype=torch_type2,
                axis=X2.ndim - 1)
        else:
            X2_scheme = 'per_tensor'
            qX2 = torch.quantize_per_tensor(X2, scale=scale2, zero_point=zero_point2,
                                            dtype=torch_type2)

        def equal_ref(qX, qX2):
            if qX.qscheme() != qX2.qscheme():
                return False
            if qX.shape != qX2.shape:
                return False
            if qX.dtype != qX2.dtype:
                return False
            if qX.qscheme() == torch.per_tensor_affine:
                if qX.q_scale() != qX2.q_scale():
                    return False
                if qX.q_zero_point() != qX2.q_zero_point():
                    return False
            elif qX.qscheme() == torch.per_channel_affine:
                if (qX.q_per_channel_scales() !=
                   qX2.q_per_channel_scales()).any():
                    return False
                if (qX.q_per_channel_zero_points() !=
                   qX2.q_per_channel_zero_points()).any():
                    return False
            else:
                raise NotImplementedError("Don't know what to do with",
                                          qX.qscheme())
            if (qX.int_repr().to(float) != qX2.int_repr().to(float)).any():
                return False
            return True

        self.assertEqual(qX.equal(qX), equal_ref(qX, qX))
        self.assertEqual(qX.equal(qX2), equal_ref(qX, qX2))

    @skipIfNoFBGEMM
    def test_group_norm(self):
        # hypothesis is flaky for this test, create test cases manually
        batches_list = (1, 7)
        num_groups_list = (1, 4)
        channels_per_groups = (1, 36, 72)
        elements_per_channels = (8, 128, 1024)
        torch_types = (torch.qint8, torch.quint8)
        y_scales = (0.1, 4.23)
        y_zero_points = (0, 1)
        channels_last_list = [True, False]
        affine_list = [True, False]
        combined = [batches_list, num_groups_list, channels_per_groups, elements_per_channels,
                    torch_types, y_scales, y_zero_points, channels_last_list, affine_list]
        test_cases = itertools.product(*combined)

        with override_quantized_engine("fbgemm"):
            for test_case in test_cases:

                batches, num_groups, channels_per_group, elements_per_channel, \
                    torch_type, Y_scale, Y_zero_point, channels_last, \
                    affine = test_case
                num_channels = num_groups * channels_per_group
                # minimum rank for channels_last
                shapes = (batches, num_channels, elements_per_channel, 1)

                # In the FP kernel, sums and sums of squares are calculated in floating point.
                # In the int8 and uint8 versions of the quantized kernel, they are
                # calculated in integer arithmetic (which is exact).
                # Because of this, the numerics do not always match exactly which is
                # expected and acceptable. We do the following to allow this failure
                # in this test:
                # 1. do not use Hypothesis to generate the input tensor.  Hypothesis
                #    favors homogeneous inputs in its search strategies which isn't
                #    representative of the inputs we care about, and tends to maximize
                #    this particular numerics difference.
                # 2. allow a small % of off by Y_scale errors.  Even when the
                #    variance of the input is high, there can be off by one errors
                #    in the result if the input value happens to fall exactly on
                #    the bin boundary of the output scale.
                #
                # If we want the numerics to match we could switch to calculating
                # mean+var in floating point in the future, at the cost of speed.
                X, X_scale, X_zero_point = \
                    _get_random_tensor_and_q_params(shapes, 1.0, torch_type)

                # Initialize the weights non-randomly for reproducibility
                if affine:
                    weight = torch.ones(num_channels).float() * 0.5
                    bias = torch.ones(num_channels).float()
                    for i in range(num_channels):
                        weight[i] *= i
                        bias[i] *= i
                else:
                    weight = None
                    bias = None

                eps = 0.001

                qX = torch.quantize_per_tensor(X, X_scale, X_zero_point, torch_type)
                if channels_last:
                    qX = qX.contiguous(memory_format=torch.channels_last)
                dqX = qX.dequantize()

                # Enforce non-homogeneous inputs
                for batch_idx in range(batches):
                    for group_idx in range(num_groups):
                        ch_start = group_idx * channels_per_group
                        ch_end = ch_start + channels_per_group
                        group_vals = dqX[batch_idx][ch_start:ch_end]
                        assume(
                            float(torch.unique(group_vals).shape[0]) / group_vals.numel() > 0.001
                            or group_vals.numel() < 5)

                qY = torch.ops.quantized.group_norm(qX, num_groups, weight, bias, eps, Y_scale, Y_zero_point)

                dqY_hat = F.group_norm(dqX, num_groups=num_groups, weight=weight, bias=bias, eps=eps)
                qY_hat = torch.quantize_per_tensor(dqY_hat, Y_scale, Y_zero_point, torch_type)

                # Due to the numerics difference mentioned above between calculating
                # the variance in float vs int, the results can still be slightly
                # different.
                dqY = qY.dequantize()
                dqY_hat = qY_hat.dequantize()
                diff = dqY - dqY_hat

                # off-by-one errors are magnitude of Y_scale
                num_diff = torch.sum(diff > Y_scale * 1.0001)
                pct_diff = float(num_diff) / (diff.numel() + 1e-5)
                num_diff_off_by_one = torch.sum((diff > 0) * (diff <= Y_scale))
                pct_diff_off_by_one = float(num_diff_off_by_one) / (diff.numel() + 1e-5)

                self.assertTrue(pct_diff < 1e-6)
                self.assertTrue(pct_diff_off_by_one < 0.01)

    @skipIfNoFBGEMM
    def test_instance_norm(self):
        max_sides = (4, 5)
        shape_list = ([2, 2, 2, 2], [8, 8, 8, 8], [11, 11, 11, 11])
        torch_types = (torch.qint8, torch.quint8)
        y_scales = (0.1, 4.23)
        y_zero_points = (0, 1)
        channels_last_list = (True, False)
        affine_list = (True, False)
        combined = [shape_list, torch_types, y_scales, y_zero_points, channels_last_list, affine_list]
        test_cases_product = itertools.product(*combined)
        test_cases = list(test_case for test_case in test_cases_product)
        # add just one test case to test overflow
        test_cases.append([
            [1, 4, 224, 224, 160],  # shape,
            torch.qint8,  # torch_type
            0.1,  # scale
            0,  # zero_point
            False,   # channels_last
            True,  # affine
        ])
        with override_quantized_engine("fbgemm"):
            for test_case in test_cases:

                shapes, torch_type, Y_scale, Y_zero_point, channels_last, affine = test_case
                if channels_last and shapes.__len__() >= 5:
                    # required rank 4 tensor to use channels_last format
                    continue

                # In the FP kernel, sums and sums of squares are calculated in floating point.
                # In the int8 and uint8 versions of the quantized kernel, they are
                # calculated in integer arithmetic (which is exact).
                # Because of this, the numerics do not always match exactly which is
                # expected and acceptable. We do the following to allow this failure
                # in this test:
                # 1. do not use Hypothesis to generate the input tensor.  Hypothesis
                #    favors homogeneous inputs in its search strategies which isn't
                #    representative of the inputs we care about, and tends to maximize
                #    this particular numerics difference.
                # 2. allow a small % of off by Y_scale errors.  Even when the
                #    variance of the input is high, there can be off by one errors
                #    in the result if the input value happens to fall exactly on
                #    the bin boundary of the output scale.
                #
                # If we want the numerics to match we could switch to calculating
                # mean+var in floating point in the future, at the cost of speed.
                X, X_scale, X_zero_point = \
                    _get_random_tensor_and_q_params(shapes, 1.0, torch_type)

                num_channels = shapes[1]
                if affine:
                    weight = torch.rand(num_channels).float() * 0.5
                    bias = torch.rand(num_channels).float()
                    for i in range(num_channels):
                        weight[i] *= i
                        bias[i] *= i
                else:
                    weight = None
                    bias = None
                eps = 0.001

                qX = torch.quantize_per_tensor(X, X_scale, X_zero_point, torch_type)
                if channels_last:
                    qX = qX.contiguous(memory_format=torch.channels_last)
                dqX = qX.dequantize()

                # Enforce non-homogeneous inputs
                batches = shapes[0]
                for batch_idx in range(batches):
                    for ch_idx in range(num_channels):
                        ch_vals = dqX[batch_idx][ch_idx]
                        assume(
                            float(torch.unique(ch_vals).shape[0]) / ch_vals.numel() > 0.01
                            or ch_vals.numel() < 5 or ch_vals.numel() > 25600)

                qY = torch.ops.quantized.instance_norm(qX, weight, bias, eps, Y_scale, Y_zero_point)

                dqY_hat = F.instance_norm(dqX, weight=weight, bias=bias, eps=eps)
                qY_hat = torch.quantize_per_tensor(dqY_hat, Y_scale, Y_zero_point, torch_type)

                # Due to the numerics difference mentioned above between calculating
                # the variance in float vs int, the results can still be slightly
                # different.
                dqY = qY.dequantize()
                dqY_hat = qY_hat.dequantize()
                diff = dqY - dqY_hat

                # off-by-one errors are magnitude of Y_scale
                num_diff = torch.sum(diff > Y_scale * 1.0001)
                pct_diff = float(num_diff) / (diff.numel() + 1e-5)
                num_diff_off_by_one = torch.sum((diff > 0) * (diff <= Y_scale))
                pct_diff_off_by_one = float(num_diff_off_by_one) / (diff.numel() + 1e-5)

                self.assertTrue(pct_diff < 1e-6)
                self.assertTrue(pct_diff_off_by_one < 0.01)

    @skipIfNoFBGEMM
    def test_batch_norm_relu(self):
        # hypothesis too slow for this test, create test cases manually
        max_sides = (2, 3, 4, 5)
        side_lens = (1, 8, 11)
        torch_types = (torch.qint8, torch.quint8)
        combined = [max_sides, side_lens, torch_types]
        test_cases = itertools.product(*combined)

        with override_quantized_engine("fbgemm"):
            for test_case in test_cases:
                max_side, side_len, torch_type = test_case
                Y_zero_point = 1
                Y_scale = 0.5

                shapes = [side_len] * max_side
                X, scale_x, zero_point_x = \
                    _get_random_tensor_and_q_params(shapes, 1.0, torch_type)
                dtype_x = torch_type

                c = X.shape[1]
                mean = torch.rand(c).float()
                var = torch.rand(c).float()
                weight = torch.rand(c).float()
                bias = torch.rand(c).float()
                eps = 0.001
                qx = torch.quantize_per_tensor(X, scale_x, zero_point_x, dtype_x)
                if len(X.shape) == 2 or len(X.shape) == 3:
                    qy = torch.ops.quantized.batch_norm1d_relu(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)
                elif len(X.shape) == 4:
                    qy = torch.ops.quantized.batch_norm2d_relu(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)
                else:
                    qy = torch.ops.quantized.batch_norm3d_relu(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)


                float_ref = F.batch_norm(qx.dequantize(), weight=weight, bias=bias,
                                         running_mean=mean, running_var=var,
                                         training=False, momentum=0, eps=eps).numpy()

                float_ref_relu = float_ref.copy()
                float_ref_relu[float_ref < 0] = 0
                quantize_ref = torch.quantize_per_tensor(
                    torch.from_numpy(float_ref_relu), Y_scale, Y_zero_point, dtype_x)
                self.assertEqual(
                    qy.int_repr().numpy(),
                    quantize_ref.int_repr().numpy(),
                    msg="{} vs {}".format(qy, quantize_ref))

    @skipIfNoFBGEMM
    def test_batch_norm(self):
        # hypothesis too slow for this test, create test cases manually
        max_sides = (2, 3, 4, 5)
        side_lens = (1, 8, 11)
        torch_types = (torch.qint8, torch.quint8)
        combined = [max_sides, side_lens, torch_types]
        test_cases = itertools.product(*combined)

        with override_quantized_engine("fbgemm"):
            for test_case in test_cases:
                max_side, side_len, torch_type = test_case
                Y_zero_point = 1
                Y_scale = 0.5

                shapes = [side_len] * max_side
                X, scale_x, zero_point_x = \
                    _get_random_tensor_and_q_params(shapes, 1.0, torch_type)
                dtype_x = torch_type

                c = X.shape[1]
                mean = torch.rand(c).float()
                var = torch.rand(c).float()
                weight = torch.rand(c).float()
                bias = torch.rand(c).float()
                eps = 0.001
                qx = torch.quantize_per_tensor(X, scale_x, zero_point_x, dtype_x)
                if len(X.shape) == 2 or len(X.shape) == 3:
                    qy = torch.ops.quantized.batch_norm1d(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)
                elif len(X.shape) == 4:
                    qy = torch.ops.quantized.batch_norm2d(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)
                elif len(X.shape) == 5:
                    qy = torch.ops.quantized.batch_norm3d(
                        qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)

                float_ref = F.batch_norm(qx.dequantize(), weight=weight, bias=bias,
                                         running_mean=mean, running_var=var, training=False,
                                         momentum=0, eps=eps)
                quantize_ref = torch.quantize_per_tensor(float_ref, Y_scale, Y_zero_point, dtype_x)
                self.assertEqual(
                    qy.int_repr().numpy(), quantize_ref.int_repr().numpy(),
                    msg="{} vs {}".format(qy, quantize_ref))

    @override_qengines
    def test_empty_batch(self):
        scale = 1.0
        zero_point = 0
        X = torch.ones((0, 2, 4, 4), dtype=torch.float32)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch.quint8)

        # upsample_nearest2d
        qY = torch.nn.functional.upsample_nearest(qX, scale_factor=2)
        np.testing.assert_equal(qY.size(), (0, 2, 8, 8),
                                "Quantized upsample_nearsest2d with batch size 0 failed.")

        # relu
        qY = torch.nn.functional.relu(qX)
        np.testing.assert_equal(qY.size(), qX.size(),
                                "Quantized relu with batch size 0 failed.")

        # tanh
        qY = torch.tanh(qX)
        np.testing.assert_equal(qY.size(), qX.size(),
                                "Quantized tanh with batch size 0 failed.")
        # sigmoid
        qY = torch.sigmoid(qX)
        np.testing.assert_equal(qY.size(), qX.size(),
                                "Quantized sigmoid with batch size 0 failed.")

        # interpolate
        op = torch.ao.nn.quantized.functional.interpolate
        for mode in ["nearest", "bilinear", "nearest-exact"]:
            qY = op(qX, scale_factor=2, mode=mode)
            np.testing.assert_equal(qY.size(), (0, 2, 8, 8),
                                    "Quantized interpolate with batch size 0 failed.")

        # avg_pool
        kernel = (2, 2)
        stride = (1, 1)
        padding = (0, 0)
        op = torch.ao.nn.quantized.functional.avg_pool2d
        qY = op(qX, kernel, stride, padding)
        np.testing.assert_equal(qY.size(), (0, 2, 3, 3),
                                "Quantized avg_pool2d with batch size 0 failed.")

        # adaptive_avg_pool
        op = torch.ao.nn.quantized.functional.adaptive_avg_pool2d
        qY = op(qX, (3, 3))
        np.testing.assert_equal(qY.size(), (0, 2, 3, 3),
                                "Quantized adaptive_avg_pool2d with batch size 0 failed.")

        # max_pool
        dilation = (1, 1)
        qY = torch.ops.quantized.max_pool2d(qX, kernel, stride, padding, dilation, ceil_mode=False)
        oH = pool_output_shape(4, 2, 0, 1, 1)
        oW = pool_output_shape(4, 2, 0, 1, 1)
        np.testing.assert_equal(qY.size(), (0, 2, oH, oW),
                                "Quantized maxpool2d with batch size 0 failed.")

        # hardtanh
        qY = torch.ao.nn.quantized.functional.hardtanh(qX, -1, 6)
        np.testing.assert_equal(qY.size(), qX.size(),
                                "Quantized hardtanh with batch size 0 failed.")

        # mul
        qY = torch.ops.quantized.mul(qX, qX, 1.0, 0)
        np.testing.assert_equal(qY.size(), qX.size(),
                                "Quantized mul with batch size 0 failed.")
        # add
        qY = torch.ops.quantized.add(qX, qX, 1.0, 0)
        np.testing.assert_equal(qY.size(), qX.size(),
                                "Quantized addition with batch size 0 failed.")

        # conv
        w = torch.randn((2, 2, 2, 2), dtype=torch.float)
        qw = torch.quantize_per_tensor(w, scale=1.0, zero_point=0, dtype=torch.qint8)
        bias_float = torch.ones(2, dtype=torch.float)
        strides = [1, 1]
        pads = [0, 0]
        dilations = [1, 1]

        w_packed = torch.ops.quantized.conv2d_prepack(qw, bias_float, strides, pads, dilations, 1)
        result = torch.ops.quantized.conv2d(qX, w_packed, 1.0, 0)
        self.assertEqual(result.shape, (0, 2, 3, 3))

        # linear
        X = torch.ones((0, 2), dtype=torch.float32)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch.quint8)
        w = torch.randn((2, 2), dtype=torch.float)
        qw = torch.quantize_per_tensor(w, scale=1.0, zero_point=0, dtype=torch.qint8)
        w_packed = torch.ops.quantized.linear_prepack(qw, bias_float)
        result = torch.ops.quantized.linear(qX, w_packed, 1.0, 0)
        self.assertEqual(result.shape, (0, 2))

        # dynamic linear
        result = torch.ops.quantized.linear_dynamic(X, w_packed)
        self.assertEqual(result.shape, (0, 2))

    @override_qengines
    def test_linear_bias_unpack(self):
        """
        Verifies the correctness of bias() and unpack() API for LinearPackedParamBase.
        """
        bias_float = torch.ones(2, dtype=torch.float)
        w = torch.randn((2, 2), dtype=torch.float)
        qw = torch.quantize_per_tensor(w, scale=1.0, zero_point=0, dtype=torch.qint8)
        w_packed = torch.ops.quantized.linear_prepack(qw, bias_float)
        # test bias()
        self.assertEqual(w_packed.bias(), bias_float)
        # test unpack()
        self.assertEqual(w_packed.unpack()[0], qw)

    def test_advanced_indexing(self):
        """
        Verifies that the x[:, [0], :, :] syntax works for quantized tensors.
        """
        for dtype in (torch.qint8, torch.quint8, torch.qint32):
            scale = 0.1
            zp = 0
            x_q = torch.quantize_per_tensor(
                torch.randn(1, 4, 4, 4), scale, zp, dtype)
            # reference
            x_fp32 = x_q.dequantize()

            # single dim, single index
            x_q_s1 = x_q[:, [0], :, :]
            x_fp32_s1 = x_fp32[:, [0], :, :]
            x_fp32_s1_ref = \
                torch.quantize_per_tensor(x_fp32_s1, scale, zp, dtype)
            self.assertEqual(x_q_s1, x_fp32_s1_ref)

            # multiple dim, single index
            x_q_s2 = x_q[:, [0], [2], :]
            x_fp32_s2 = x_fp32[:, [0], [2], :]
            x_fp32_s2_ref = \
                torch.quantize_per_tensor(x_fp32_s2, scale, zp, dtype)
            self.assertEqual(x_q_s2, x_fp32_s2_ref)

            # single dim, multiple indices
            x_q_s3 = x_q[:, [2, 0, 1], :, :]
            x_fp32_s3 = x_fp32[:, [2, 0, 1], :, :]
            x_fp32_s3_ref = \
                torch.quantize_per_tensor(x_fp32_s3, scale, zp, dtype)
            self.assertEqual(x_q_s3, x_fp32_s3_ref)

            # multiple dim, multiple indices
            x_q_s4 = x_q[:, [2, 0, 1], :, [1]]
            x_fp32_s4 = x_fp32[:, [2, 0, 1], :, [1]]
            x_fp32_s4_ref = \
                torch.quantize_per_tensor(x_fp32_s4, scale, zp, dtype)
            self.assertEqual(x_q_s4, x_fp32_s4_ref)

    @override_qengines
    def test_custom_module_lstm(self):
        qengine = torch.backends.quantized.engine

        batch_size = 4
        seq_len = 8
        input_size = 12

        hidden_size = 8
        num_layers = 2

        dropout = 0  # This is not supported

        Bias = [False, True]
        Batch_first = [False, True]
        Bidirectional = [False, True]

        dtype = np.uint8
        qtype = torch.quint8

        x = np.random.randn(seq_len, batch_size, input_size)
        scale, zero_point = _calculate_dynamic_qparams(x, dtype=dtype)
        x = torch.from_numpy(x).to(torch.float)
        qx = torch.quantize_per_tensor(x, scale=scale, zero_point=zero_point,
                                       dtype=qtype)
        x = qx.dequantize()

        with torch.no_grad():
            for bias, batch_first, bidirectional in itertools.product(
                    Bias, Batch_first, Bidirectional):
                # Assume 12dB is sufficient for functional equivalence
                # Without the bias, linear performs poorly
                min_power = 10 if bias else 5
                max_mse = 5e-6 if bias else 5e-1

                if batch_first:
                    x = x.reshape(batch_size, seq_len, input_size)
                    qx = qx.reshape(batch_size, seq_len, input_size)
                else:
                    x = x.reshape(seq_len, batch_size, input_size)
                    qx = qx.reshape(seq_len, batch_size, input_size)

                lstm = torch.nn.Sequential(
                    torch.nn.LSTM(input_size, hidden_size,
                                  num_layers=num_layers,
                                  bias=bias, batch_first=batch_first,
                                  dropout=dropout,
                                  bidirectional=bidirectional))
                lstm.eval()
                y_ref = lstm(x)

                # Prepare
                lstm.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                lstm_prepared = torch.ao.quantization.prepare(lstm)
                self.assertTrue(hasattr(lstm_prepared[0], 'layers'))
                self.assertEqual(num_layers, len(lstm_prepared[0].layers))
                assert type(lstm_prepared[0]) == torch.nn.quantizable.LSTM

                # Calibrate
                y = lstm_prepared(x)
                self.assertEqual(y_ref, y)

                # Quantize
                lstm_quantized = torch.ao.quantization.convert(lstm_prepared)
                assert type(lstm_quantized[0]) == torch.nn.quantized.LSTM
                qy = lstm_quantized(qx)

                snr = _snr(y, qy)
                snr = [snr[0]] + snr[1]

                for signal, mse, power in snr:
                    self.assertTrue(
                        power > min_power or mse < max_mse,
                        msg=(f"Error is too high: SNR(dB): {power}, "
                             f"Signal: {signal}, MSE: {mse}"))

                # Trace
                jit_qmodule = torch.jit.trace(lstm_quantized, qx)

                # Script
                jit_qmodule = torch.jit.script(lstm_quantized)

    @override_qengines
    def test_custom_module_multi_head_attention(self):
        class MultiheadAttentionModel(torch.nn.Module):
            def __init__(self, *args, **kwargs):
                super(MultiheadAttentionModel, self).__init__()
                self.layer = torch.nn.MultiheadAttention(*args, **kwargs)

            def forward(
                self,
                query,
                key,
                value,
                key_padding_mask: Optional[torch.Tensor] = None,
                need_weights: bool = True,
                attn_mask: Optional[torch.Tensor] = None,
            ):
                return self.layer(query, key, value, key_padding_mask, need_weights, attn_mask)

        qengine = torch.backends.quantized.engine

        min_power = 30
        max_mse = 2

        num_heads = 16
        batch_size = 4
        target_seq_length = 128
        source_seq_length = 64
        qembed_dim = 512  # Must be divisible by the number of heads
        kembed_dim = 128
        vembed_dim = 256

        dropout = 0.0  # This is not supported

        Bias = [False, True]
        Add_bias_kv = [False, True]
        Add_zero_attn = [False, True]

        dtype = np.uint8
        qtype = torch.quint8

        for kdim, vdim in ((kembed_dim, vembed_dim), (None, None)):
            fp_data = [
                torch.randn(target_seq_length, batch_size, qembed_dim),  # Q
                torch.randn(source_seq_length, batch_size,
                            qembed_dim if kdim is None else kembed_dim),  # K
                torch.randn(source_seq_length, batch_size,
                            qembed_dim if vdim is None else vembed_dim)   # V
            ]

            q_data = []
            reduce_range = (qengine in ('x86', 'fbgemm', 'onednn'))
            for idx, x in enumerate(fp_data):
                scale, zero_point = _calculate_dynamic_qparams(
                    x, dtype=dtype, reduce_range=reduce_range)
                x = x.to(torch.float)
                qx = torch.quantize_per_tensor(x, scale=scale,
                                               zero_point=zero_point, dtype=qtype)
                q_data.append(qx)

                # Dequantize the data back for reference
                fp_data[idx] = qx.dequantize()

            with torch.no_grad():
                for bias, add_bias_kv, add_zero_attn in itertools.product(
                        Bias, Add_bias_kv, Add_zero_attn):
                    mha = MultiheadAttentionModel(qembed_dim, num_heads, dropout,
                                                  bias, add_bias_kv, add_zero_attn,
                                                  kdim=kdim, vdim=vdim)
                    mha.eval()

                    # Prepare
                    if qengine_is_onednn():
                        # `reduce_range` is False by default for ONEDNN backend
                        # but the test fails on earlier CPUs without VNNI.
                        # So we use a default qconfig with `reduce_range=True` here
                        mha.qconfig = torch.ao.quantization.get_default_qconfig()
                    else:
                        mha.qconfig = torch.ao.quantization.get_default_qconfig(qengine)
                    mha_prepared = torch.ao.quantization.prepare(
                        mha)

                    # Calibrate
                    y = mha_prepared(*fp_data)
                    y_ref = mha(*fp_data)
                    # Check the result of the prepare
                    self.assertEqual(y_ref[0], y[0])  # Attention
                    self.assertEqual(y_ref[1], y[1])  # Weight

                    # Quantize
                    mha_quantized = torch.ao.quantization.convert(mha_prepared)
                    qy = mha_quantized(*q_data)

                    # Reference result
                    mha.layer = mha_quantized.layer.dequantize()
                    y_ref = mha(*fp_data)

                    snr = _snr(y, qy)
                    for signal, mse, power in snr:
                        self.assertTrue(
                            power > min_power or mse < max_mse,
                            msg=(f"Error is too high: SNR(dB): {power}, "
                                 f"Signal: {signal}, MSE: {mse}; "
                                 f"Run with bias={bias}, "
                                 f"add_bias_kv={add_bias_kv}, "
                                 f"add_zero_attn={add_zero_attn}"))

                    # Verify the result is scriptable
                    mha_quantized_scripted = torch.jit.script(mha_quantized)


class TestDynamicQuantizedOps(TestCase):
    """Tests the correctness of the dynamic quantized linear and linear_relu op."""
    @override_qengines
    @given(
        batch_size=st.integers(1, 4),
        input_channels=st.integers(16, 32),
        output_channels=st.integers(4, 8),
        use_bias=st.booleans(),
        use_relu=st.booleans(),
        use_multi_dim_input=st.booleans(),
        use_channelwise=st.booleans(),
        reduce_range=st.booleans())
    def test_qlinear(self, batch_size, input_channels, output_channels,
                     use_bias, use_relu, use_multi_dim_input, use_channelwise, reduce_range):
        if torch.backends.quantized.engine == 'qnnpack':
            reduce_range = False

        qlinear_prepack = torch.ops.quantized.linear_prepack
        if use_relu:
            qlinear_dynamic = torch.ops.quantized.linear_relu_dynamic
        else:
            qlinear_dynamic = torch.ops.quantized.linear_dynamic

        if use_multi_dim_input:
            batch_size *= 3  # Test the multi-dim input tensor

        X_scale = 1.0
        X_zp = 0
        X_value_min = 0
        X_value_max = 255
        if reduce_range:
            X_value_max = 127
        X_q0 = np.round(np.random.rand(batch_size, input_channels) *
                        (X_value_max - X_value_min) + X_value_min).astype(np.uint8)
        X_q0[0, 0] = X_value_min
        X_q0[0, 1] = X_value_max

        # W_scale = 1.0
        # W_zp = 0
        W_scales = np.ones(output_channels)
        W_zps = np.zeros(output_channels).astype(np.int)
        W_value_min = -128
        W_value_max = 127
        W_q0 = np.round(
            np.random.rand(output_channels, input_channels)
            * (W_value_max - W_value_min)
            + W_value_min
        ).astype(np.int8)
        W_q0[0, 0] = W_value_min
        W_q0[1, 0] = W_value_max

        b_value_min = -10
        b_value_max = 10
        b_q0 = np.round(
            np.random.rand(output_channels) *
            (b_value_max - b_value_min) + b_value_min
        ).astype(np.int32) if use_bias else None

        if torch.backends.quantized.engine in ('x86', 'fbgemm', 'onednn'):
            avoid_vpmaddubsw_overflow_linear(
                batch_size,
                input_channels,
                output_channels,
                X_q0,
                X_value_min,
                X_value_max,
                W_q0,
                W_value_min,
                W_value_max,
            )

        X_fp32 = torch.from_numpy(_dequantize(X_q0, X_scale, X_zp)).to(dtype=torch.float)
        if use_multi_dim_input:
            X_fp32 = X_fp32.view(3, int(batch_size / 3), input_channels)

        # W_scale, W_zp = _calculate_dynamic_qparams(W_fp32, torch.qint8)
        # We currently only check the case where W_scale = 1.0, W_zp = 0.

        if use_channelwise:
            W_fp32 = torch.from_numpy(_dequantize(W_q0, W_scales.reshape(
                (-1, 1)), W_zps.reshape((-1, 1)))).to(dtype=torch.float)
            W_q = torch.quantize_per_channel(W_fp32, scales=torch.from_numpy(W_scales),
                                             zero_points=torch.from_numpy(W_zps), axis=0, dtype=torch.qint8)
            b_fp32 = torch.from_numpy(
                _dequantize(b_q0, X_scale * W_scales, 0)
            ).to(dtype=torch.float) if use_bias else None
        else:
            W_fp32 = torch.from_numpy(_dequantize(
                W_q0, W_scales[0], W_zps[0])).to(dtype=torch.float)
            W_q = torch.quantize_per_tensor(W_fp32, scale=W_scales[0], zero_point=(
                W_zps[0].astype(int).item()), dtype=torch.qint8)
            b_fp32 = torch.from_numpy(
                _dequantize(b_q0, X_scale * int(W_scales[0].item()), 0)
            ).to(dtype=torch.float) if use_bias else None

        # Observe X_fp32 and determine X_scale and X_zero_point, this should match
        # internals of dynamic linear.
        X_scale, X_zp = _calculate_dynamic_qparams(X_fp32, torch.quint8, reduce_range)
        X_q = torch.quantize_per_tensor(X_fp32, scale=X_scale, zero_point=X_zp, dtype=torch.quint8)

        # Weight prepacking operator for dynamic quantized Linear
        W_prepack = qlinear_prepack(W_q, b_fp32)
        # Dynamic quantized Linear operator with prepacked weight
        Y_fp32 = qlinear_dynamic(X_q.dequantize(), W_prepack, reduce_range)
        # Y_fp32 = qlinear_dynamic(X_fp32, W_prepack, b_fp32)

        Y_fp32_ref = F.linear(X_q.dequantize(), W_q.dequantize(), b_fp32)
        # Y_fp32_ref = F.linear(X_fp32, W_fp32, b_fp32)
        # if use_multi_dim_input:
        #     Y_fp32_ref = Y_fp32_ref.view(3, int(batch_size / 3), output_channels)

        if use_relu:
            Y_fp32_ref[Y_fp32_ref < 0.0] = 0.0
        self.assertEqual(Y_fp32, Y_fp32_ref,
                         msg="torch.ops.quantized.linear_dynamic results are off")

    @skipIfNoFBGEMM
    @given(
        batch_size=st.integers(1, 4),
        input_channels=st.integers(16, 32),
        output_channels=st.integers(4, 8),
    )
    def test_qlinear_legacy(self, batch_size, input_channels, output_channels):
        X_scale = 1.0
        X_zp = 0
        X_value_min = 0
        X_value_max = 255
        X_q0 = np.round(np.random.rand(batch_size, input_channels) * (
            X_value_max - X_value_min) + X_value_min
        ).astype(np.uint8)
        X_q0[0, 0] = X_value_min
        X_q0[0, 1] = X_value_max

        W_scale = 1.0
        W_zp = 0
        W_value_min = -128
        W_value_max = 127
        W_q0 = np.round(
            np.random.rand(output_channels, input_channels)
            * (W_value_max - W_value_min)
            + W_value_min
        ).astype(np.int8)
        W_q0[0, 0] = W_value_min
        W_q0[1, 0] = W_value_max

        b_value_min = -10
        b_value_max = 10
        b_q0 = np.round(
            np.random.rand(output_channels) * (b_value_max - b_value_min) +
            b_value_min
        ).astype(np.int32)

        avoid_vpmaddubsw_overflow_linear(
            batch_size,
            input_channels,
            output_channels,
            X_q0,
            X_value_min,
            X_value_max,
            W_q0,
            W_value_min,
            W_value_max,
        )

        X_fp32 = torch.from_numpy(_dequantize(X_q0, X_scale, X_zp)).to(dtype=torch.float)
        W_fp32 = torch.from_numpy(_dequantize(W_q0, W_scale, W_zp)).to(dtype=torch.float)
        b_fp32 = torch.from_numpy(
            _dequantize(b_q0, X_scale * W_scale, 0)
        ).to(dtype=torch.float)

        W_scale, W_zp = _calculate_dynamic_qparams(W_fp32, torch.qint8)
        W_q = torch.quantize_per_tensor(W_fp32, scale=W_scale, zero_point=W_zp, dtype=torch.qint8)

        # Observe X_fp32 and determine X_scale and X_zero_point, this should match
        # internals of dynamic linear.
        X_scale, X_zp = _calculate_dynamic_qparams(X_fp32, torch.quint8)
        X_q = torch.quantize_per_tensor(X_fp32, scale=X_scale, zero_point=X_zp, dtype=torch.quint8)

        W_int8, col_offsets, W_scale, W_zp = torch.fbgemm_linear_quantize_weight(W_q.dequantize())
        W_prepack = torch.fbgemm_pack_quantized_matrix(W_int8.clone(), W_int8.size(1), W_int8.size(0))
        # Quantized Linear operator with prepacked weight
        Y_fp32 = torch.fbgemm_linear_int8_weight(
            X_q.dequantize(), W_q.dequantize(), W_prepack, col_offsets,
            W_scale, W_zp, b_fp32)

        Y_fp32_ref = F.linear(X_q.dequantize(), W_q.dequantize(), b_fp32)
        # Y_fp32_ref = F.linear(X_fp32, W_fp32, b_fp32)

        self.assertEqual(Y_fp32, Y_fp32_ref,
                         msg="torch.ops.quantized.fbgemm_linear_dynamic results are off")

    @skipIfNoFBGEMM
    @given(
        input_channels=st.integers(16, 32),
        output_channels=st.integers(4, 8),
        exponent=st.integers(0, 8))
    def test_linear_prepack_fp16_numerics(self, input_channels, output_channels, exponent):
        w = torch.randn(output_channels, input_channels) * 10**exponent
        bias = None
        w_packed_fp16 = torch.ops.quantized.linear_prepack_fp16(w, bias)
        w_unpacked_fp16 = torch.ops.quantized.linear_unpack_fp16(w_packed_fp16)
        w_fp16 = w.to(torch.float16).to(torch.float32)
        self.assertTrue(torch.equal(w_fp16, w_unpacked_fp16[0]))

    @skipIfNoFBGEMM
    def test_qlinear_dynamic_fp16(self):

        options = itertools.product(
            (2, 4),         # batch_size
            (4, 5, 12),     # input_channels
            (4, 7, 8),      # output_channels
            (True, False),  # use_bias
            (True, False),  # use_relu
        )
        for batch_size, input_channels, output_channels, use_bias, use_relu in options:
            qlinear_prepack = torch.ops.quantized.linear_prepack_fp16
            if use_relu:
                qlinear_dynamic = torch.ops.quantized.linear_relu_dynamic_fp16
            else:
                qlinear_dynamic = torch.ops.quantized.linear_dynamic_fp16

            x = torch.randn(batch_size, input_channels)
            w = torch.randn(output_channels, input_channels)
            bias = torch.randn(output_channels) if use_bias else None

            w_packed = qlinear_prepack(w, bias)
            out = qlinear_dynamic(x, w_packed)

            # qlinear_dynamic_fp16 uses FP32 activation tensors and FP16 weight tensors
            # output is FP32
            w_fp16 = w.to(torch.float16).to(torch.float32)
            ref = F.linear(x, w_fp16, bias)
            if use_relu:
                ref.relu_()

            self.assertEqual(out, ref)

    """Tests the correctness of the dynamic quantized lstm/gru."""

    def _get_rnn_inputs(self, seq_len, num_batches, input_size, hidden_size, num_directions, reduce_range):
        # For Input (seq_len, batch, input_size)
        X = torch.randn(seq_len, num_batches, input_size)
        s, z = _calculate_dynamic_qparams(X, torch.quint8, reduce_range)
        Xq = torch.quantize_per_tensor(X, s, z, torch.quint8)

        # For H and C: (num_layers(1) * num_directions, batch, hidden_size)

        if num_directions == 1:
            H = torch.randn(num_directions, num_batches, hidden_size)
            C = torch.randn(num_directions, num_batches, hidden_size)
        else:
            H = torch.zeros(num_directions, num_batches, hidden_size)
            C = torch.zeros(num_directions, num_batches, hidden_size)

        s, z = _calculate_dynamic_qparams(H, torch.quint8, reduce_range)
        Hq = torch.quantize_per_tensor(H, s, z, torch.quint8)
        s, z = _calculate_dynamic_qparams(C, torch.quint8, reduce_range)
        Cq = torch.quantize_per_tensor(C, s, z, torch.quint8)
        return Xq, Hq, Cq

    def _get_rnn_weights_and_bias(self, input_size, hidden_size, num_directions, per_channel_quant, rnn_type):
        hidden_mult_map = {'LSTM': 4, 'LSTMCell': 4, 'GRU': 3, 'GRUCell': 3, 'RNNTanh': 2, 'RNNReLU': 2}
        hidden_mult = hidden_mult_map[rnn_type]
        weights1 = torch.randn(hidden_mult * hidden_size, input_size)
        weights2 = torch.randn(hidden_mult * hidden_size, hidden_size)
        scale1 = 0.1 * torch.ones([weights1.size()[0]])
        scale2 = 0.3 * torch.ones([weights2.size()[0]])
        zero_point1 = torch.zeros(scale1.size()).to(int)
        zero_point2 = torch.zeros(scale2.size()).to(int)
        b1 = torch.zeros(hidden_mult * hidden_size)
        if per_channel_quant:
            Wq1 = torch.quantize_per_channel(weights1, scale1, zero_point1, 0, torch.qint8)
            Wq2 = torch.quantize_per_channel(weights2, scale2, zero_point2, 0, torch.qint8)

        else:
            Wq1 = torch.quantize_per_tensor(weights1, float(scale1[0]), int(zero_point1[0]), torch.qint8)
            Wq2 = torch.quantize_per_tensor(weights2, float(scale2[0]), int(zero_point2[0]), torch.qint8)
        return Wq1, Wq2, b1, b1

    @given(
        num_batches=st.integers(1, 4),
        input_size=st.integers(16, 32),
        hidden_size=st.integers(4, 8),
        num_directions=st.integers(1, 2),
        per_channel_quant=st.booleans())
    @override_qengines
    def test_qlstmGRU(self, num_batches, input_size, hidden_size,
                      num_directions, per_channel_quant):
        # We test only for seq length of 1 and num layers of 1 as dynamic quantization occurs multiple times
        # within the LSTM op and we do not model the quantization between multiple calls of the linear op within the
        # lstm op
        seq_len = 1

        for rnn_type in ['LSTM', 'GRU']:
            for dtype in [torch.qint8, torch.float16]:
                # Fp16 quantization is not supported for qnnpack or onednn
                if torch.backends.quantized.engine in ('qnnpack', 'onednn') and dtype == torch.float16:
                    continue

                if torch.backends.quantized.engine == 'qnnpack':
                    reduce_range = False
                else:
                    reduce_range = True
                Xq, Hq, Cq = self._get_rnn_inputs(seq_len, num_batches, input_size,
                                                  hidden_size, num_directions, reduce_range)
                Wq1, Wq2, b1, b2 = self._get_rnn_weights_and_bias(input_size,
                                                                  hidden_size,
                                                                  num_directions,
                                                                  per_channel_quant,
                                                                  rnn_type)
                if dtype == torch.qint8:
                    packed_ih = torch.ops.quantized.linear_prepack(Wq1, b1)
                    packed_hh = torch.ops.quantized.linear_prepack(Wq2, b2)
                    cell_params = torch.ops.quantized.make_quantized_cell_params_dynamic(
                        packed_ih, packed_hh, b1, b2, reduce_range)
                    W_ref1 = Wq1.dequantize()
                    W_ref2 = Wq2.dequantize()

                else:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(Wq1.dequantize(), b1)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(Wq2.dequantize(), b2)
                    cell_params = torch.ops.quantized.make_quantized_cell_params_fp16(packed_ih, packed_hh)
                    W_ref1 = Wq1.dequantize().to(torch.float16).to(torch.float32)
                    W_ref2 = Wq2.dequantize().to(torch.float16).to(torch.float32)

                if rnn_type == 'LSTM':
                    if num_directions > 1:
                        result_ref = _VF.lstm(Xq.dequantize(),
                                              (Hq.dequantize(), Cq.dequantize()),
                                              [W_ref1, W_ref2, b1, b2, W_ref1, W_ref2, b1, b2],
                                              True,
                                              1,
                                              0,
                                              False,
                                              num_directions > 1,
                                              False)

                        result_dynamic = torch.quantized_lstm(Xq.dequantize(),
                                                              (Hq.dequantize(), Cq.dequantize()),
                                                              ([cell_params, cell_params]),
                                                              True,
                                                              1,
                                                              0,
                                                              False,
                                                              True,
                                                              False,
                                                              dtype=torch.qint8,
                                                              use_dynamic=True)
                    else:
                        result_ref = _VF.lstm(Xq.dequantize(),
                                              (Hq.dequantize(), Cq.dequantize()),
                                              [W_ref1, W_ref2, b1, b2],
                                              True,
                                              1,
                                              0,
                                              False,
                                              num_directions > 1,
                                              False)

                        result_dynamic = torch.quantized_lstm(Xq.dequantize(),
                                                              (Hq.dequantize(), Cq.dequantize()),
                                                              ([cell_params]),
                                                              True,
                                                              1,
                                                              0,
                                                              False,
                                                              num_directions > 1,
                                                              False,
                                                              dtype=torch.qint8,
                                                              use_dynamic=True)

                if rnn_type == 'GRU':
                    if num_directions > 1:
                        result_ref = _VF.gru(Xq.dequantize(),
                                             Hq.dequantize(),
                                             [W_ref1, W_ref2, b1, b2, W_ref1, W_ref2, b1, b2],
                                             True,
                                             1,
                                             0,
                                             False,
                                             True,
                                             False)

                        result_dynamic = torch.quantized_gru(Xq.dequantize(),
                                                             Hq.dequantize(),
                                                             ([cell_params, cell_params]),
                                                             True,
                                                             1,
                                                             0,
                                                             False,
                                                             True,
                                                             False)
                    else:
                        result_ref = _VF.gru(Xq.dequantize(),
                                             Hq.dequantize(),
                                             [W_ref1, W_ref2, b1, b2],
                                             True,
                                             1,
                                             0,
                                             False,
                                             False,
                                             False)

                        result_dynamic = torch.quantized_gru(Xq.dequantize(),
                                                             Hq.dequantize(),
                                                             ([cell_params]),
                                                             True,
                                                             1,
                                                             0,
                                                             False,
                                                             False,
                                                             False)

                self.assertEqual(result_ref[0], result_dynamic[0], msg="torch.quantized_lstm results are off")

    @given(
        num_batches=st.integers(1, 4),
        input_size=st.integers(16, 32),
        hidden_size=st.integers(4, 8),
        per_channel_quant=st.booleans())
    @override_qengines
    def test_qrnncell(self, num_batches, input_size, hidden_size, per_channel_quant):
        # We test only for seq length of 1 and num layers of 1 as dynamic quantization occurs multiple times
        # within the LSTM op and we do not model the quantization between multiple calls of the linear op within the
        # lstm op
        seq_len = 1

        for rnn_type in ['LSTMCell', 'GRUCell', 'RNNTanh', 'RNNReLU']:
            for dtype in [torch.qint8, torch.float16]:
                # Fp16 quantization is not supported for qnnpack or onednn
                if torch.backends.quantized.engine in ('qnnpack', 'onednn') and dtype == torch.float16:
                    continue

                if torch.backends.quantized.engine == 'qnnpack':
                    reduce_range = False
                else:
                    reduce_range = True

                Xq, Hq, Cq = self._get_rnn_inputs(seq_len, num_batches, input_size, hidden_size, 1, reduce_range)
                Wq1, Wq2, b1, b2 = self._get_rnn_weights_and_bias(
                    input_size, hidden_size, 1, per_channel_quant, rnn_type)
                if dtype == torch.qint8:
                    packed_ih = torch.ops.quantized.linear_prepack(Wq1, b1)
                    packed_hh = torch.ops.quantized.linear_prepack(Wq2, b2)
                    W_ref1 = Wq1.dequantize()
                    W_ref2 = Wq2.dequantize()
                else:
                    packed_ih = torch.ops.quantized.linear_prepack_fp16(Wq1.dequantize(), b1)
                    packed_hh = torch.ops.quantized.linear_prepack_fp16(Wq2.dequantize(), b2)
                    W_ref1 = Wq1.dequantize().to(torch.float16).to(torch.float32)
                    W_ref2 = Wq2.dequantize().to(torch.float16).to(torch.float32)

                state = {'LSTMCell': (Hq.dequantize()[0], Cq.dequantize()[0]),
                         'GRUCell': Hq.dequantize()[0],
                         'RNNTanh': Hq.dequantize()[0],
                         'RNNReLU': Hq.dequantize()[0]}
                fn_dict = {'LSTMCell': torch._VF.lstm_cell,
                           'GRUCell': torch._VF.gru_cell,
                           'RNNTanh': torch._VF.rnn_tanh_cell,
                           'RNNReLU': torch._VF.rnn_relu_cell}
                qfn_dict = {'LSTMCell': torch.ops.quantized.quantized_lstm_cell_dynamic,
                            'GRUCell': torch.ops.quantized.quantized_gru_cell_dynamic,
                            'RNNTanh': torch.ops.quantized.quantized_rnn_tanh_cell_dynamic,
                            'RNNReLU': torch.ops.quantized.quantized_rnn_relu_cell_dynamic}
                W_ref_dict = {torch.float16: (Wq1.dequantize().to(torch.float16).to(torch.float32),
                                              Wq2.dequantize().to(torch.float16).to(torch.float32)),
                              torch.qint8: (Wq1.dequantize(), Wq2.dequantize())}

                result_ref = fn_dict[rnn_type](Xq.dequantize()[0], state[rnn_type], W_ref1, W_ref2, b1, b2)
                result_dynamic = qfn_dict[rnn_type](Xq.dequantize()[0], state[rnn_type], packed_ih, packed_hh, b1, b2)
                self.assertEqual(result_ref[0], result_dynamic[0], msg="torch.quantized_rnncell results are off")

    def _test_qconv_op_impl(self, q_mod, dq_op, dim, dtype):
        # The goal here is to show that the dynamic op is the same as
        # calc params->quantize_input->quantized op->dequantize output

        if qengine_is_qnnpack() and (IS_PPC or TEST_WITH_UBSAN):
            return  # not supported by QNNPACK

        if qengine_is_qnnpack():
            reduce_range = False
        else:
            reduce_range = True

        X_fp32 = torch.randn(*([2] * dim))
        s, z = _calculate_dynamic_qparams(X_fp32, dtype, reduce_range)

        quantized_module = q_mod(2, 3, 1)
        packed_params = quantized_module._packed_params

        quantized_module.scale, quantized_module.zero_point = s, z

        X_q = torch.quantize_per_tensor(X_fp32, s, z, dtype)
        Y_q_ref = quantized_module(X_q)
        Y_ref = torch.dequantize(Y_q_ref)

        X_dq = torch.dequantize(X_q)
        Y = dq_op(X_dq, packed_params, reduce_range)

        self.assertEqual(Y, Y_ref)

    @override_qengines
    def test_dynamic_conv1d(self):
        q_mod = torch.ao.nn.quantized.Conv1d
        dq_op = torch.ops.quantized.conv1d_dynamic
        dim = 3
        dtype = torch.quint8

        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)

    @override_qengines
    def test_dynamic_conv2d(self):
        q_mod = torch.ao.nn.quantized.Conv2d
        dq_op = torch.ops.quantized.conv2d_dynamic
        dim = 4
        dtype = torch.quint8

        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)

    @override_qengines
    def test_dynamic_conv3d(self):
        q_mod = torch.ao.nn.quantized.Conv3d
        dq_op = torch.ops.quantized.conv3d_dynamic
        dim = 5
        dtype = torch.quint8

        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)

    @override_qengines
    def test_dynamic_convtranspose1d(self):
        q_mod = torch.ao.nn.quantized.ConvTranspose1d
        dq_op = torch.ops.quantized.conv_transpose1d_dynamic
        dim = 3
        dtype = torch.quint8

        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)

    @override_qengines
    def test_dynamic_convtranspose2d(self):
        q_mod = torch.ao.nn.quantized.ConvTranspose2d
        dq_op = torch.ops.quantized.conv_transpose2d_dynamic
        dim = 4
        dtype = torch.quint8

        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)

    @override_qengines
    def test_dynamic_convtranspose3d(self):
        q_mod = torch.ao.nn.quantized.ConvTranspose3d
        dq_op = torch.ops.quantized.conv_transpose3d_dynamic
        dim = 5
        dtype = torch.quint8

        if qengine_is_qnnpack():
            return  # TODO: fix MakeDeConvOutputShape overflowing for convT3d with qnnpack
        self._test_qconv_op_impl(q_mod, dq_op, dim, dtype)


class TestQuantizedLinear(TestCase):
    def _test_qlinear_impl(self, batch_size, input_channels, output_channels, use_bias,
                           post_op, use_multi_dim_input, use_channelwise, **post_op_kwargs):
        decimal_val = 4
        dtypes = [torch.quint8]
        if torch.backends.quantized.engine == 'qnnpack':
            # QNNPACK supports uint8 in the kernels. In the op we shift the int8
            # weight values to uint8 to be on par with fbgemm. However, this causes
            # some rounding issues in rare cases. So, we relax the check to allow
            # off by one results.
            decimal_val = 0

            # only qnnpack qengine supports qint8 when xnnpack is available
            if torch.backends.xnnpack.enabled:
                dtypes.append(torch.qint8)

        for dtype in dtypes:
            # No support for channelwise in xnnpack (int8)
            # ONEDNN does not support qint8
            if dtype == torch.qint8 and (use_channelwise or qengine_is_onednn()):
                return

            nptype = np_dtype[dtype]
            qlinear_prepack = torch.ops.quantized.linear_prepack
            if post_op == 'relu':
                qlinear = torch.ops.quantized.linear_relu
            elif post_op == 'leaky_relu':
                qlinear = torch.ops.quantized.linear_leaky_relu
            else:
                qlinear = torch.ops.quantized.linear
            if use_multi_dim_input:
                batch_size *= 3  # Test the multi-dim input tensor
            X_scale = 1.5
            X_zp = 5
            X_value_min = -128 if dtype == torch.qint8 else 0
            X_value_max = 127 if dtype == torch.qint8 else 255
            X_q0 = np.round(
                np.random.rand(batch_size, input_channels) *
                (X_value_max - X_value_min)
                + X_value_min
            ).astype(nptype)

            W_scales = np.random.rand(output_channels)
            # xnnpack forces W_zp to 0 when using symmetric quantization
            # ONEDNN only supports symmetric quantization of weight
            if dtype == torch.qint8 or qengine_is_onednn():
                W_zps = np.zeros(output_channels).astype(np.int)
            else:
                W_zps = np.round(np.random.rand(output_channels) * 100 - 50).astype(np.int)
            # when using symmetric quantization
            # special restriction for xnnpack fully connected op weight
            # [-127, 127] instead of [-128, 127]
            W_value_min = -127 if dtype == torch.qint8 else -128
            W_value_max = 127
            W_q0 = np.round(
                np.random.rand(output_channels, input_channels)
                * (W_value_max - W_value_min)
                + W_value_min
            ).astype(np.int8)  # weight is always int8_t
            b_value_min = -10
            b_value_max = 10
            b_q0 = np.round(
                np.random.rand(output_channels) *
                (b_value_max - b_value_min) + b_value_min
            ).astype(np.int32) if use_bias else None
            if torch.backends.quantized.engine in ('x86', 'fbgemm', 'onednn'):
                avoid_vpmaddubsw_overflow_linear(
                    batch_size,
                    input_channels,
                    output_channels,
                    X_q0,
                    X_value_min,
                    X_value_max,
                    W_q0,
                    W_value_min,
                    W_value_max,
                )
            X = torch.from_numpy(_dequantize(
                X_q0, X_scale, X_zp)).to(dtype=torch.float)
            X_q = torch.quantize_per_tensor(
                X, scale=X_scale, zero_point=X_zp, dtype=dtype)
            if use_channelwise:
                W = torch.from_numpy(_dequantize(W_q0, W_scales.reshape(
                    (-1, 1)), W_zps.reshape((-1, 1)))).to(dtype=torch.float)
                W_q = torch.quantize_per_channel(W, scales=torch.from_numpy(W_scales),
                                                 zero_points=torch.from_numpy(W_zps), axis=0, dtype=torch.qint8)
                b = torch.from_numpy(_dequantize(
                    b_q0, X_scale * W_scales, 0)).to(dtype=torch.float) if use_bias else None
                b_q = torch.quantize_per_channel(b, scales=torch.from_numpy(X_scale * W_scales),
                                                 zero_points=torch.zeros(output_channels, dtype=torch.long),
                                                 axis=0, dtype=torch.qint32) if use_bias else None
            else:
                W = torch.from_numpy(_dequantize(
                    W_q0, W_scales[0], W_zps[0])).to(dtype=torch.float)
                W_q = torch.quantize_per_tensor(W, scale=W_scales[0], zero_point=(
                    W_zps[0].astype(int).item()), dtype=torch.qint8)
                b = torch.from_numpy(_dequantize(
                    b_q0, X_scale * (W_scales[0].item()), 0)).to(dtype=torch.float) if use_bias else None
                b_q = torch.quantize_per_tensor(
                    b, scale=X_scale * (W_scales[0].item()), zero_point=0, dtype=torch.qint32) if use_bias else None
            # Compare X_scale * W_scale * input_channels * X_value_max * W_value_max with
            # Y_scale * 255 (max for uint8).
            Y_scale = 125.1234
            Y_zp = 5
            # Weight prepacking operator for quantized Linear
            float_bias = b if use_bias else None
            W_prepack = qlinear_prepack(W_q, float_bias)
            if use_multi_dim_input:
                X_q = X_q.view(3, int(batch_size / 3), input_channels)
            # Quantized Linear operator with prepacked weight
            Y_q = qlinear(X_q, W_prepack, Y_scale, Y_zp, **post_op_kwargs)
            if not use_channelwise and post_op in ('none', 'relu'):
                # Test the per-tensor quantization only
                # Reference quantized Linear operator
                Y_q_ref = qlinear_ref(X_q0, X_scale, X_zp, W_q0,
                                      W_scales[0], W_zps[0], b_q0, Y_scale, Y_zp, dtype=nptype)
                if post_op == 'relu':
                    Y_q_ref[Y_q_ref < Y_zp] = Y_zp
                if use_multi_dim_input:
                    Y_q_ref = np.reshape(
                        Y_q_ref, (3, int(batch_size / 3), output_channels))
                # Assert equal
                np.testing.assert_array_almost_equal(Y_q_ref, Y_q.int_repr().numpy(), decimal=decimal_val)
            # Test both per-tensor and per-channel quantization
            # Reference quantized result from PyTorch Linear operator
            W_fp32 = W_q.dequantize().to(dtype=torch.float)
            X_fp32 = X_q.dequantize().to(dtype=torch.float)
            b_fp32 = b_q.dequantize().to(dtype=torch.float) if use_bias else None
            Y_fp32_ref = F.linear(X_fp32, W_fp32, b_fp32)
            if post_op == 'relu':
                Y_fp32_ref[Y_fp32_ref < 0.0] = 0.0
            elif post_op == 'leaky_relu':
                Y_fp32_ref = F.leaky_relu(Y_fp32_ref, **post_op_kwargs)
            Y_q_ref2 = torch.quantize_per_tensor(
                Y_fp32_ref, Y_scale, Y_zp, dtype)
            # Assert equal
            np.testing.assert_array_almost_equal(
                Y_q_ref2.int_repr().numpy(), Y_q.int_repr().numpy(), decimal=decimal_val)

    """Tests the correctness of the quantized linear and linear_relu op."""
    @override_qengines
    def test_qlinear(self):
        batch_size_list = [1, 4]
        input_channels_list = [16, 32]
        output_channels_list = [4, 8]
        use_bias_list = [True, False]
        post_op_list = ['none', 'relu']
        use_multi_dim_input_list = [True, False]
        use_channelwise_list = [True, False]
        cases = itertools.product(batch_size_list, input_channels_list, output_channels_list,
                                  use_bias_list, post_op_list, use_multi_dim_input_list,
                                  use_channelwise_list)
        for batch_size, input_channels, output_channels, use_bias, post_op,\
                use_multi_dim_input, use_channelwise in cases:
            self._test_qlinear_impl(batch_size, input_channels, output_channels,
                                    use_bias, post_op, use_multi_dim_input, use_channelwise)

    @given(batch_size=st.integers(1, 4),
           # in cudnn v. 8.4.0, there is a limitation that input channels
           # should be a multiple of 4 for int8 tensors. in cudnn v.8.3.3
           # this should be a multiple of 16
           input_channels=st.sampled_from([4, 8, 12, 16, 32]),
           # constraints on output channels appear to be relax, as it seems we can use any positive integer here
           # except 1. It is not clear why 1 will not work. TODO: check with Yang
           output_channels=st.integers(2, 36),
           use_bias=st.booleans(),
           use_relu=st.booleans(),
           use_multi_dim_input=st.booleans(),
           use_channelwise=st.sampled_from([False]))  # channelwise currently not supported for qlinear cudnn
    @skipIfNoFBGEMM
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    @unittest.skip("Local only - currently the qlinear_cudnn op is bulid "
                   "with USE_EXPERIMENTAL_CUDNN_V8_API, we can enable the test "
                   "after it is built by default")
    # TODO: check with yang regarding CUDNN flags
    def test_qlinear_cudnn(self, batch_size, input_channels, output_channels, use_bias,
                           use_relu, use_multi_dim_input, use_channelwise):
        qlinear_prepack = torch.ops.quantized.linear_prepack
        if use_relu:
            qlinear_op = torch.ops.quantized.linear_relu
        else:
            qlinear_op = torch.ops.quantized.linear
        X_scale = 1.5
        X_zp = 0
        X_value_min = -128
        X_value_max = 127
        X_q0 = np.round(
            np.random.rand(batch_size, input_channels) *
            (X_value_max - X_value_min)
            + X_value_min).astype(np.int8)
        W_scale = 2.5
        W_zp = 0
        W_value_min = -128
        W_value_max = 127
        W_q0 = np.round(
            np.random.rand(output_channels, input_channels)
            * (W_value_max - W_value_min)
            + W_value_min
        ).astype(np.int8)
        b_value_min = -10
        b_value_max = 10
        b_q0 = np.round(
            np.random.rand(output_channels) *
            (b_value_max - b_value_min) + b_value_min
        ).astype(np.int32) if use_bias else None
        if use_bias:
            b_value_min = -10
            b_value_max = 10
            b_q0 = np.round(
                np.random.rand(output_channels) *
                (b_value_max - b_value_min) + b_value_min
            ).astype(np.int32)
        else:
            bias = None
        avoid_vpmaddubsw_overflow_linear(
            batch_size,
            input_channels,
            output_channels,
            X_q0,
            X_value_min,
            X_value_max,
            W_q0,
            W_value_min,
            W_value_max,
        )
        quant_dtype = torch.qint8
        X = torch.from_numpy(_dequantize(
            X_q0, X_scale, X_zp)).to(dtype=torch.float).to(device="cuda")
        X_q = torch.quantize_per_tensor(
            X, scale=X_scale, zero_point=X_zp, dtype=quant_dtype)
        W = torch.from_numpy(_dequantize(
            W_q0, W_scale, W_zp)).to(dtype=torch.float).to(device="cuda")
        W_q = torch.quantize_per_tensor(W, scale=W_scale, zero_point=W_zp, dtype=quant_dtype)
        b = torch.from_numpy(_dequantize(
            b_q0, X_scale * (W_zp), 0)).to(dtype=torch.float).to(device="cuda") if use_bias else None
        b_q = torch.quantize_per_tensor(
            b, scale=X_scale * W_scale, zero_point=0, dtype=quant_dtype) if use_bias else None
        Y_scale = 0.5
        Y_zp = 0
        # Weight prepacking operator for quantized Linear
        float_bias = b if use_bias else None
        W_prepack = qlinear_prepack(W_q, float_bias if use_bias else None)
        # Quantized Linear operator with prepacked weight
        Y_q = qlinear_op(X_q, W_prepack, Y_scale, Y_zp).to(device="cpu")
        Y_q_ref = qlinear_ref(X_q0, X_scale, X_zp, W_q0,
                              W_scale, W_zp, b_q0, Y_scale, Y_zp, dtype=np.int8)
        if use_relu:
            Y_q_ref[Y_q_ref < Y_zp] = Y_zp
        decimal_val = 0
        np.testing.assert_array_almost_equal(Y_q_ref, Y_q.int_repr().numpy(), decimal=decimal_val)

    """Tests the correctness of the quantized::linear_unpack op."""
    @given(W=hu.tensor(shapes=hu.array_shapes(2, 2,),
                       qparams=hu.qparams(dtypes=torch.qint8)),
           use_channelwise=st.booleans())
    @override_qengines
    def test_qlinear_unpack(self, W, use_channelwise):
        W, (W_scale, W_zp, torch_type) = W
        if use_channelwise:
            output_channels = W.shape[0]
            W_scales = torch.rand(output_channels).to(torch.double)
            W_zps = torch.round(torch.rand(output_channels)
                                * 100 - 50).to(torch.int64)
        qlinear_prepack = torch.ops.quantized.linear_prepack
        qlinear_unpack = torch.ops.quantized.linear_unpack

        # ONEDNN only supports symmetric quantization of weight
        if qengine_is_onednn():
            if use_channelwise:
                W_zps = torch.zeros(output_channels).to(torch.int64)
            else:
                W_zp = 0

        W = torch.from_numpy(W)
        if use_channelwise:
            W_q = torch.quantize_per_channel(
                W, W_scales, W_zps, 0, dtype=torch_type)
        else:
            W_q = torch.quantize_per_tensor(W, scale=W_scale, zero_point=W_zp,
                                            dtype=torch_type)
        # Weight prepacking operator for quantized Linear
        W_prepack = qlinear_prepack(W_q)
        # Weight unpack operator for quantized Linear (Used for serialization)
        W_q_origin = qlinear_unpack(W_prepack)[0]
        # Assert equal
        np.testing.assert_equal(W_q.int_repr(), W_q_origin.int_repr().numpy())
        if use_channelwise:
            np.testing.assert_array_almost_equal(np.float32(W_q.q_per_channel_scales().numpy()),
                                                 np.float32(
                                                     W_q_origin.q_per_channel_scales().numpy()),
                                                 decimal=4)
            np.testing.assert_equal(W_q.q_per_channel_zero_points(
            ).numpy(), W_q_origin.q_per_channel_zero_points().numpy())
        else:
            np.testing.assert_equal(np.float32(
                W_q.q_scale()), np.float32(W_q_origin.q_scale()))
            np.testing.assert_equal(
                W_q.q_zero_point(), W_q_origin.q_zero_point())

    @skipIfNoONEDNN
    def test_qlinear_leaky_relu(self):
        with override_quantized_engine('onednn'):
            batch_size_list = [1, 4]
            input_channels_list = [16, 32]
            output_channels_list = [4, 8]
            use_bias_list = [True, False]
            use_multi_dim_input_list = [True, False]
            use_channelwise_list = [True, False]
            negative_slopes_list = [0.01, 0.05]
            post_op = 'leaky_relu'
            cases = itertools.product(batch_size_list, input_channels_list, output_channels_list,
                                      use_bias_list, use_multi_dim_input_list,
                                      use_channelwise_list, negative_slopes_list)
            for batch_size, input_channels, output_channels, use_bias,\
                    use_multi_dim_input, use_channelwise, neg_slope in cases:
                self._test_qlinear_impl(batch_size, input_channels, output_channels,
                                        use_bias, post_op, use_multi_dim_input,
                                        use_channelwise, negative_slope=neg_slope)

@unittest.skipIf(IS_MACOS, "Known test failure on Mac.")
class TestQuantizedEmbeddingOps(TestCase):

    def _test_embedding_bag_unpack_impl(self, pack_fn, unpack_fn, bit_rate, optimized_qparams, weights):
        data_type = weights.dtype

        qtype = torch.quint8
        if bit_rate == 8:
            w_packed = pack_fn(weights)
        else:
            w_packed = pack_fn(weights, optimized_qparams=optimized_qparams)
        w_unpacked = unpack_fn(w_packed)

        if (bit_rate == 8 or bit_rate == 4) and data_type != torch.float16:
            # torch.quantize_per_channel does not support float16 yet.

            obs_weights = weights
            # Combine 3D embeddings (e.g. stacked combination of embeddings)
            # in a dimension orthogonal to channels.
            if (len(obs_weights.shape) > 2):
                stacked_shape = list(weights.size())
                stacked_shape[1] *= stacked_shape[0]
                obs_weights = weights.reshape(stacked_shape[1:])

            # Check numerics of prepack function that accepts qtensor as input.
            # We use min-max observer to mimic the quantization performed in the original function.
            obs = PerChannelMinMaxObserver(dtype=torch.quint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
            obs(obs_weights)
            # Get the scale and zero point for the weight tensor
            qparams = obs.calculate_qparams()
            if bit_rate == 4:
                qtype = torch.quint4x2
            # Quantize the weights to 8bits
            qweight = torch.quantize_per_channel(obs_weights, qparams[0], qparams[1], axis=0, dtype=qtype)
            real_packed_weight = torch.ops.quantized.embedding_bag_prepack(qweight)
            self.assertEqual(isinstance(real_packed_weight, torch._C.ScriptObject), True)
            unpacked_weight = torch.ops.quantized.embedding_bag_unpack(real_packed_weight)
            self.assertEqual(unpacked_weight.int_repr().numpy(), qweight.int_repr().numpy())
            self.assertEqual(unpacked_weight.q_per_channel_scales(), qweight.q_per_channel_scales())
            self.assertEqual(unpacked_weight.q_per_channel_zero_points(), qweight.q_per_channel_zero_points())

        # compare against C2 to ensure numerical equivalency.
        from caffe2.python import core, workspace
        conversion_op = "FloatToFused8BitRowwiseQuantized" if data_type == torch.float32 else "HalfFloatToFused8BitRowwiseQuantized"
        reverse_conversion_op = None
        if bit_rate == 4:
            conversion_op = "FloatToFused4BitRowwiseQuantized" if data_type == torch.float32 else "HalfToFused4BitRowwiseQuantized"
            reverse_conversion_op = "Fused4BitRowwiseQuantizedToFloat"
        elif bit_rate == 2:
            conversion_op = "FloatToFused2BitRowwiseQuantized" if data_type == torch.float32 else "HalfToFused2BitRowwiseQuantized"
            reverse_conversion_op = "Fused2BitRowwiseQuantizedToFloat"

        def get_c2_weights(weights, engine_str):
            workspace.ResetWorkspace()

            workspace.FeedBlob("weights", weights)
            workspace.RunOperatorOnce(
                core.CreateOperator(
                    conversion_op, ["weights"], ["quantized_weights"], engine=engine_str
                )
            )
            emb_q = workspace.FetchBlob("quantized_weights")
            if bit_rate == 4 or bit_rate == 2:
                workspace.RunOperatorOnce(
                    core.CreateOperator(
                        reverse_conversion_op, ["quantized_weights"], ["dequantized_weights"]
                    )
                )
                dequantized_data = torch.from_numpy(workspace.FetchBlob("dequantized_weights"))
            else:
                dequantized_data = torch.ops._caffe2.Fused8BitRowwiseQuantizedToFloat(
                    torch.tensor(emb_q)
                )
            return torch.from_numpy(emb_q), dequantized_data

        if optimized_qparams:
            engine = "GREEDY"
        else:
            engine = ""

        # C2 quantization needs the memory format of Tensor to be `continuous`, otherwise it will
        # throw exceptions. torch.clone() will make the memory format to be `continuous`
        c2_copy = torch.clone(weights)
        w_packed_c2, w_unpacked_c2 = get_c2_weights(c2_copy, engine)

        # Compare packed weights against C2.
        np.testing.assert_allclose(w_packed.numpy(), w_packed_c2.numpy(), atol=1e-6, rtol=1e-6)
        # Compare unpacked weights against C2
        np.testing.assert_allclose(w_unpacked.numpy(), w_unpacked_c2.numpy(), atol=1e-6, rtol=1e-6)


    def _test_embedding_bag_unpack_fn(self, pack_fn, unpack_fn, num_embeddings, embedding_dim, bit_rate,
                                      optimized_qparams, num_batches, data_type=np.float32):

        # when num_batches = 1, it will create a 2D tensor
        unsplit_weight = torch.from_numpy((np.random.random_sample((
            num_batches, num_embeddings, embedding_dim)).squeeze() + 1).astype(np.float32))

        # test unsplit weight (memory format is `contiguous`)
        self._test_embedding_bag_unpack_impl(pack_fn, unpack_fn, bit_rate, optimized_qparams, unsplit_weight)

        # test split weights (memory format is not `contiguous`)
        split_dim = len(unsplit_weight.shape) - 2
        split_weights = torch.split(unsplit_weight, 1, dim=split_dim)
        for weight in split_weights:
            self._test_embedding_bag_unpack_impl(pack_fn, unpack_fn, bit_rate, optimized_qparams, weight)


    """ Tests the correctness of the embedding_bag_8bit pack/unpack op against C2 """
    @unittest.skipIf(not BUILD_WITH_CAFFE2, "Test needs Caffe2")
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
           num_batches=st.integers(1, 5),
           data_type=st.sampled_from([np.float32, np.float16]),)
    def test_embedding_bag_byte_unpack(self, num_embeddings, embedding_dim, num_batches, data_type):
        pack_fn = torch.ops.quantized.embedding_bag_byte_prepack
        unpack_fn = torch.ops.quantized.embedding_bag_byte_unpack

        self._test_embedding_bag_unpack_fn(
            pack_fn, unpack_fn, num_embeddings, embedding_dim, 8, False, num_batches, data_type=data_type)

    """ Tests the correctness of the embedding_bag_4bit pack/unpack op against C2 """
    @unittest.skipIf(not BUILD_WITH_CAFFE2, "Test needs Caffe2")
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
           optimized_qparams=st.booleans(),
           data_type=st.sampled_from([np.float32, np.float16]),)
    def test_embedding_bag_4bit_unpack(self, num_embeddings, embedding_dim, optimized_qparams, data_type):
        pack_fn = torch.ops.quantized.embedding_bag_4bit_prepack
        unpack_fn = torch.ops.quantized.embedding_bag_4bit_unpack

        # 4bit and 2bit quantization right now only works for 2D Tensor so we set the num_batches to 1
        self._test_embedding_bag_unpack_fn(
            pack_fn, unpack_fn, num_embeddings, embedding_dim, 4, optimized_qparams, 1, data_type=data_type)

    """ Tests the correctness of the embedding_bag_2bit pack/unpack op against C2 """
    @unittest.skipIf(not BUILD_WITH_CAFFE2, "Test needs Caffe2")
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 8 == 0),
           optimized_qparams=st.booleans(),
           data_type=st.sampled_from([np.float32, np.float16]),)
    def test_embedding_bag_2bit_unpack(self, num_embeddings, embedding_dim, optimized_qparams, data_type):
        pack_fn = torch.ops.quantized.embedding_bag_2bit_prepack
        unpack_fn = torch.ops.quantized.embedding_bag_2bit_unpack

        # 4bit and 2bit quantization right now only works for 2D Tensor so we set the num_batches to 1
        self._test_embedding_bag_unpack_fn(
            pack_fn, unpack_fn, num_embeddings, embedding_dim, 2, optimized_qparams, 1, data_type=data_type)


    def embedding_bag_rowwise_offsets_run(
            self, bit_rate, num_embeddings,
            embedding_dim, num_offsets,
            use_32bit_indices, use_32bit_offsets,
            enable_per_sample_weights,
            include_last_offset, fallback_to_no_sparse, sparsity, atol, rtol):
        pt_op = torch.ops.quantized.embedding_bag_byte_rowwise_offsets
        pt_prepack_op = torch.ops.quantized.embedding_bag_byte_prepack
        if bit_rate == 4:
            pt_op = torch.ops.quantized.embedding_bag_4bit_rowwise_offsets
            pt_prepack_op = torch.ops.quantized.embedding_bag_4bit_prepack
        elif bit_rate == 2:
            pt_op = torch.ops.quantized.embedding_bag_2bit_rowwise_offsets
            pt_prepack_op = torch.ops.quantized.embedding_bag_2bit_prepack

        weights = torch.from_numpy((np.random.random_sample((
            num_embeddings, embedding_dim)) + 1).astype(np.float32))

        max_segments = 5
        max_segment_length = 20
        num_lengths = np.random.randint(1, max_segments + 1)
        lengths = np.random.randint(0, max_segment_length + 1,
                                    size=num_lengths).astype(np.int32)
        num_indices = np.sum(lengths)

        def lengths_to_offsets(t, offset_type=np.int64, use_begin_offset=True):
            """
            Convert lengths to offsets
            """
            tt = np.zeros((t.shape[0] + 1,), dtype=offset_type)
            tt[1:] = t
            tt = torch.from_numpy(np.cumsum(tt, dtype=offset_type))
            if use_begin_offset:
                return tt[:-1]
            return tt[1:]

        offsets = lengths_to_offsets(lengths)
        indices = torch.from_numpy(np.random.randint(
            low=0, high=num_embeddings, size=num_indices, dtype=np.int64))

        q_weights = pt_prepack_op(weights)
        per_sample_weights = torch.from_numpy(np.random.uniform(
            low=0.01, high=0.5, size=[len(indices)]).astype(np.float32)) if \
            enable_per_sample_weights else None
        if include_last_offset:
            offsets = torch.cat(
                (offsets, torch.tensor([indices.size(0)], dtype=torch.long)), 0
            )

        # Reference result will be the floating point torch.nn.EmbeddingBag.
        def get_reference_result(
                num_embeddings, embedding_dim,
                include_last_offset, weights, per_sample_weights,
                indices, offsets):
            embedding_bag = torch.nn.EmbeddingBag(
                num_embeddings=num_embeddings,
                embedding_dim=embedding_dim,
                include_last_offset=include_last_offset, _weight=weights,
                scale_grad_by_freq=False, mode='sum'
            )
            return embedding_bag(indices, offsets,
                                 per_sample_weights=per_sample_weights)

        mapping_table = np.zeros(num_embeddings, dtype=np.int32)
        pruned_weights = weights
        prune_weights = sparsity > 0
        if prune_weights:
            if fallback_to_no_sparse:
                # Testing that prune_weight with mapping_table {0} will
                # fallback to non sparse embedding look up kernel.
                mapping_table = np.zeros(1, dtype=np.int32)
            else:
                # Prune and generate mapping table
                num_compressed_rows = 0
                unpruned_ids = []
                for i in range(num_embeddings):
                    if np.random.uniform() < sparsity:
                        mapping_table[i] = -1
                        q_weights[i, :] = 0
                        weights[i, :] = 0
                    else:
                        mapping_table[i] = num_compressed_rows
                        num_compressed_rows += 1
                        unpruned_ids.append(i)
                q_weights = q_weights[unpruned_ids]
                pruned_weights = weights[unpruned_ids]

        result = pt_op(q_weights,
                       indices.int() if use_32bit_indices else indices,
                       offsets.int() if use_32bit_offsets else offsets,
                       mode=0,
                       pruned_weights=prune_weights,
                       per_sample_weights=per_sample_weights,
                       compressed_indices_mapping=torch.tensor(mapping_table),
                       include_last_offset=include_last_offset)

        reference_result = get_reference_result(
            num_embeddings, embedding_dim, include_last_offset, weights,
            per_sample_weights, indices, offsets)

        torch.testing.assert_close(reference_result, result, atol=atol, rtol=rtol)


        if bit_rate == 8 or bit_rate == 4:
            # Test operator that accepts TorchBind packed weights.
            if bit_rate == 4:
                qdtype = torch.quint4x2
                op = torch.ops.quantized.embedding_bag_4bit
            else:
                qdtype = torch.quint8
                op = torch.ops.quantized.embedding_bag_byte
            obs = PerChannelMinMaxObserver(dtype=qdtype, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
            obs(pruned_weights)
            # Get the scale and zero point for the weight tensor
            qparams = obs.calculate_qparams()
            # Quantize the weights to 8bits
            qweight = torch.quantize_per_channel(pruned_weights, qparams[0], qparams[1], axis=0, dtype=qdtype)
            packed_weight = torch.ops.quantized.embedding_bag_prepack(qweight)
            result = op(packed_weight, indices, offsets, mode=0,
                        pruned_weights=prune_weights,
                        per_sample_weights=per_sample_weights,
                        compressed_indices_mapping=torch.tensor(mapping_table),
                        include_last_offset=include_last_offset)
            torch.testing.assert_close(reference_result, result, atol=atol, rtol=rtol)

    """ Tests the correctness of the embedding_bag_8bit quantized operator """
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
           num_offsets=st.integers(1, 20),
           use_32bit_indices=st.booleans(),
           use_32bit_offsets=st.booleans(),
           enable_per_sample_weights=st.booleans(),
           include_last_offset=st.booleans(),
           fallback_to_no_sparse=st.booleans(),
           sparsity=st.sampled_from([0.0, 0.5, 0.7]))
    def test_embedding_bag_byte(self, num_embeddings,
                                embedding_dim, num_offsets,
                                use_32bit_indices,
                                use_32bit_offsets,
                                enable_per_sample_weights,
                                include_last_offset,
                                fallback_to_no_sparse,
                                sparsity):
        self.embedding_bag_rowwise_offsets_run(
            8, num_embeddings, embedding_dim, num_offsets,
            use_32bit_indices, use_32bit_offsets,
            enable_per_sample_weights, include_last_offset,
            fallback_to_no_sparse,
            sparsity=sparsity, atol=0.005, rtol=1e-3)

    """ Tests the correctness of the embedding_bag_4bit quantized operator """
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0),
           num_offsets=st.integers(1, 20),
           use_32bit_indices=st.booleans(),
           use_32bit_offsets=st.booleans(),
           enable_per_sample_weights=st.booleans(),
           include_last_offset=st.booleans(),
           fallback_to_no_sparse=st.booleans(),
           sparsity=st.sampled_from([0.0, 0.5, 0.7]))
    def test_embedding_bag_4bit(self, num_embeddings,
                                embedding_dim, num_offsets,
                                use_32bit_indices,
                                use_32bit_offsets,
                                enable_per_sample_weights,
                                include_last_offset,
                                fallback_to_no_sparse,
                                sparsity):
        self.embedding_bag_rowwise_offsets_run(4, num_embeddings,
                                               embedding_dim, num_offsets,
                                               use_32bit_indices, use_32bit_offsets,
                                               enable_per_sample_weights,
                                               include_last_offset,
                                               fallback_to_no_sparse,
                                               sparsity=sparsity,
                                               atol=0.1, rtol=1e-2)

    """ Tests the correctness of the embedding_bag_2bit quantized operator """
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 8 == 0),
           num_offsets=st.integers(1, 20),
           use_32bit_indices=st.booleans(),
           use_32bit_offsets=st.booleans(),
           enable_per_sample_weights=st.booleans(),
           include_last_offset=st.booleans(),
           fallback_to_no_sparse=st.booleans(),
           sparsity=st.sampled_from([0.0, 0.5, 0.7]))
    def test_embedding_bag_2bit(self, num_embeddings,
                                embedding_dim, num_offsets,
                                use_32bit_indices,
                                use_32bit_offsets,
                                enable_per_sample_weights,
                                include_last_offset,
                                fallback_to_no_sparse,
                                sparsity):
        self.embedding_bag_rowwise_offsets_run(2, num_embeddings,
                                               embedding_dim, num_offsets,
                                               use_32bit_indices, use_32bit_offsets,
                                               enable_per_sample_weights,
                                               include_last_offset,
                                               fallback_to_no_sparse,
                                               sparsity=sparsity,
                                               atol=1.0, rtol=1e-1)

    """ Tests the correctness of the quantized 8 bit embedding lookup operator """
    @given(num_embeddings=st.integers(10, 100),
           embedding_dim=st.integers(5, 50).filter(lambda x: x % 4 == 0))
    def test_embedding(self, num_embeddings, embedding_dim):
        dtypes = [torch.quint8, torch.quint4x2]
        quant_ops = [torch.ops.quantized.embedding_byte, torch.ops.quantized.embedding_4bit]
        atols = [0.005, 0.1]
        rtols = [1e-3, 1e-2]
        prepack_op = torch.ops.quantized.embedding_bag_prepack
        for quant_op, dtype, atol, rtol in zip(quant_ops, dtypes, atols, rtols):
            weights = torch.from_numpy((np.random.random_sample((
                num_embeddings, embedding_dim)) + 1).astype(np.float32))

            obs = PerChannelMinMaxObserver(dtype=dtype, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
            obs(weights)
            # Get the scale and zero point for the weight tensor
            qparams = obs.calculate_qparams()

            # Quantize the weights to 8bits
            qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=dtype)
            max_segments = 5
            max_segment_length = 20
            num_lengths = np.random.randint(1, max_segments + 1)
            lengths = np.random.randint(1, max_segment_length + 1,
                                        size=num_lengths).astype(np.int32)
            num_indices = np.sum(lengths)
            indices = torch.from_numpy(np.random.randint(
                low=0, high=num_embeddings, size=num_indices, dtype=np.int64))

            packed_weight = prepack_op(qweight)
            qresult = quant_op(packed_weight, indices, pruned_weights=False)

            ref = torch.embedding(weights, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False)
            torch.testing.assert_close(ref, qresult, atol=atol, rtol=rtol)

    def test_embedding_2d_indices(self):
        """
        Tests the case where 2D indices are passed into the operator
        In this case the operator computes the correct offsets argument.
        Output shape is dependent on the indices dimension.
        """
        quant_op = torch.ops.quantized.embedding_byte
        prepack_op = torch.ops.quantized.embedding_bag_prepack

        indices = torch.tensor([[9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8], [3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3]])
        weights = torch.randn(10, 12, dtype=torch.float32)

        ref = torch.embedding(weights, indices, padding_idx=-1, scale_grad_by_freq=False, sparse=False)
        obs = PerChannelMinMaxObserver(dtype=torch.quint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
        obs(weights)
        qparams = obs.calculate_qparams()

        qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=torch.quint8)
        packed_weight = prepack_op(qweight)
        qresult = quant_op(packed_weight, indices, pruned_weights=False)
        torch.testing.assert_close(ref, qresult, atol=0.05, rtol=1e-3)

    def test_embedding_bag_2d_indices(self):
        """
        Tests the case where 2D indices are passed into the operator
        In this case the operator computes the correct offsets argument.
        """
        indices = torch.tensor([[9, 6, 5, 7, 8, 8, 9, 2, 8, 6, 6, 9, 1, 6, 8, 8], [3, 2, 3, 6, 3, 6, 5, 7, 0, 8, 4, 6, 5, 8, 2, 3]])
        weights = torch.randn(10, 12, dtype=torch.float32)

        embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings=10,
            embedding_dim=12,
            include_last_offset=False, _weight=weights,
            scale_grad_by_freq=False, mode='sum'
        )
        result = embedding_bag(indices)

        pt_op = torch.ops.quantized.embedding_bag_byte_rowwise_offsets
        pt_prepack_op = torch.ops.quantized.embedding_bag_byte_prepack
        q_weights = pt_prepack_op(weights)
        qresult = pt_op(q_weights, indices, mode=0, pruned_weights=False)
        torch.testing.assert_close(result, qresult, atol=0.05, rtol=1e-3)

        # Test TorchBind based embedding_bag operator
        obs = PerChannelMinMaxObserver(dtype=torch.quint8, qscheme=torch.per_channel_affine_float_qparams, ch_axis=0)
        obs(weights)
        # Get the scale and zero point for the weight tensor
        qparams = obs.calculate_qparams()

        # Quantize the weights to 8bits
        qweight = torch.quantize_per_channel(weights, qparams[0], qparams[1], axis=0, dtype=torch.quint8)

        packed_weight = torch.ops.quantized.embedding_bag_prepack(qweight)
        qresult = torch.ops.quantized.embedding_bag_byte(packed_weight, indices, mode=0)

        torch.testing.assert_close(result, qresult, atol=0.05, rtol=1e-3)


class TestQuantizedConv(TestCase):
    def _test_qconv_unpack_impl(self, qconv_prepack_fn, qconv_unpack_fn, inputs,
                                strides, i_pads, o_pads, channelwise):
        (X_data, W_data, bias_data, groups, transposed) = inputs
        (X, (X_scale, X_zero_point, X_qtype)) = X_data
        (W, (W_scale, W_zero_point, W_qtype)) = W_data
        (bias, (bias_scale, bias_zero_point, bias_qtype)) = bias_data

        W = torch.from_numpy(W).float()
        bias = torch.from_numpy(bias).float()
        if channelwise and transposed:
            # currently transposed conv and per-channel per quantization does not work
            return
        # ONEDNN only supports symmetric quantization of weight and zero output padding
        if qengine_is_onednn():
            W_zero_point = 0
            o_pads = len(o_pads) * [0] if o_pads is not None else None
        if channelwise:
            if transposed:
                output_channels = W.shape[1]  # IC OC/G
            else:
                output_channels = W.shape[0]  # OC IC/G
            W_scale = torch.tensor([W_scale] * output_channels)
            W_zero_point = torch.tensor([W_zero_point] * output_channels)
            W_q = torch.quantize_per_channel(
                W, scales=W_scale, zero_points=W_zero_point,
                axis=int(transposed), dtype=W_qtype)
        else:
            W_q = torch.quantize_per_tensor(
                W, scale=W_scale, zero_point=W_zero_point, dtype=W_qtype)

        if isinstance(strides, int):
            dilations = [1]
        else:
            dilations = (1,) * len(strides)

        if transposed:
            W_packed = qconv_prepack_fn(W_q, bias, strides, i_pads, o_pads,
                                        dilations, groups)
        else:
            W_packed = qconv_prepack_fn(W_q, bias, strides, i_pads, dilations,
                                        groups)
        (W_unpacked, bias) = qconv_unpack_fn(W_packed)

        # Assert equal
        np.testing.assert_equal(W_q.int_repr().numpy(),
                                W_unpacked.int_repr().numpy())
        if channelwise:
            np.testing.assert_array_almost_equal(
                np.float32(W_q.q_per_channel_scales().numpy()),
                np.float32(W_unpacked.q_per_channel_scales().numpy()),
                decimal=4)
            np.testing.assert_equal(W_q.q_per_channel_zero_points(
            ).numpy(), W_unpacked.q_per_channel_zero_points().numpy())
        else:
            np.testing.assert_equal(np.float32(
                W_q.q_scale()), np.float32(W_unpacked.q_scale()))
            np.testing.assert_equal(
                W_q.q_zero_point(), W_unpacked.q_zero_point())

    def _make_qconv_tensors(
        self, batch_size, input_channels_per_group, input_feature_map_shape,
        output_channels_per_group, groups, kernels, strides, pads, dilations,
        X_scale, X_zero_point, W_scale, W_zero_point,
        use_bias, use_channelwise, use_transpose,
        device=torch.device("cpu"),
        input_dtype=torch.quint8,
        weight_dtype=torch.qint8,
    ):
        assert not (use_channelwise and use_transpose), \
               "Cannot generate channelwise qconv_transpose_tensors "
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        # Padded input size should be at least as big as dilated kernel
        kernels = _single(kernels)
        strides = _single(strides)
        pads = _single(pads)
        dilations = _single(dilations)
        for i in range(len(kernels)):
            assume(input_feature_map_shape[i] + 2 * pads[i]
                   >= dilations[i] * (kernels[i] - 1) + 1)
        W_scale = W_scale * output_channels
        W_zero_point = W_zero_point * output_channels
        # Resize W_scale and W_zero_points arrays equal to output_channels
        W_scale = W_scale[:output_channels]
        W_zero_point = W_zero_point[:output_channels]
        # For testing, we use small values for weights and for activations
        # so that no overflow occurs in vpmaddubsw instruction. If the
        # overflow occurs in qconv implementation and if there is no
        # overflow
        # In reference we can't exactly match the results with reference.
        # Please see the comment in qconv implementation file
        # aten/src/ATen/native/quantized/cpu/qconv.cpp for more details.
        (W_value_min, W_value_max) = (-5, 5)
        # the operator expects them in the format
        # (output_channels, input_channels/groups, kernel_d, kernel_h, kernel_w)
        # (input_channels, output_channels/groups, kernel_d, kernel_h, kernel_w)
        if use_transpose:
            output_shape = (input_channels, output_channels_per_group,)
        else:
            output_shape = (output_channels, input_channels_per_group,)
        W_init = torch.randint(
            W_value_min,
            W_value_max,
            output_shape + kernels,
            device=device,
        )
        b_init = torch.randint(0, 10, (output_channels,), device=device)

        (X_value_min, X_value_max) = (0, 4)
        X_init = torch.randint(
            X_value_min,
            X_value_max,
            (batch_size, input_channels,) + input_feature_map_shape,
            device=device
        )
        X = X_scale * (X_init - X_zero_point).float()

        if use_channelwise:
            W_shape = (-1, 1) + (1,) * len(kernels)
            W_scales_tensor = torch.tensor(W_scale, dtype=torch.float, device=device)
            W_zero_points_tensor = torch.tensor(W_zero_point, dtype=torch.float, device=device)
            W = W_scales_tensor.reshape(*W_shape) * (
                W_init.float() - W_zero_points_tensor.reshape(*W_shape)).float()
            b = X_scale * W_scales_tensor * b_init.float()
        else:
            W = W_scale[0] * (W_init - W_zero_point[0]).float()
            b = X_scale * W_scale[0] * b_init.float()

        X_q = torch.quantize_per_tensor(
            X, scale=X_scale, zero_point=X_zero_point, dtype=input_dtype)
        if use_channelwise:
            W_q = torch.quantize_per_channel(
                W, W_scales_tensor, W_zero_points_tensor.long(), 0,
                dtype=weight_dtype)
        else:
            W_q = torch.quantize_per_tensor(
                W, scale=W_scale[0], zero_point=W_zero_point[0],
                dtype=weight_dtype)

        bias_float = b if use_bias else None

        return (X, W), (X_q, W_q), bias_float

    def _test_qconv_impl(
        self, qconv_fn, qconv_prepack_fn, conv_op, batch_size,
        input_channels_per_group, input_feature_map_shape,
        output_channels_per_group, groups, kernels, strides, pads, o_pads,
        dilations, X_scale, X_zero_point, W_scale, W_zero_point, Y_scale,
        Y_zero_point, use_bias, use_relu, use_channelwise, use_transpose,
        device=torch.device("cpu"),
        input_dtype=torch.quint8,
        weight_dtype=torch.qint8,
        output_dtype=torch.quint8,
    ):
        # ONEDNN only supports symmetric quantization of weight
        if qengine_is_onednn() and W_zero_point is not None:
            W_zero_point = len(W_zero_point) * [0]
        (X, W), (X_q, W_q), bias_float = self._make_qconv_tensors(
            batch_size, input_channels_per_group, input_feature_map_shape,
            output_channels_per_group, groups, kernels,
            strides, pads, dilations, X_scale, X_zero_point, W_scale,
            W_zero_point, use_bias, use_channelwise, use_transpose,
            device=device, input_dtype=input_dtype, weight_dtype=weight_dtype)
        if bias_float is not None:
            bias_float = bias_float.to(device)
        # Assign weights
        W = W_q.dequantize()
        X = X_q.dequantize()
        conv_op.weight = torch.nn.Parameter(W, requires_grad=False)
        conv_op.bias = torch.nn.Parameter(
            bias_float, requires_grad=False) if use_bias else None
        result_ref = conv_op(X)
        if use_relu:
            assert not use_transpose, "Cannot fuse ReLU with ConvTranspose"
            relu = torch.nn.ReLU()
            result_ref = relu(result_ref)

        # Quantize reference results for comparison
        result_ref_q = torch.quantize_per_tensor(
            result_ref, scale=Y_scale, zero_point=Y_zero_point,
            dtype=output_dtype)

        if qconv_prepack_fn is not None:
            if use_transpose:
                W_prepack = qconv_prepack_fn(
                    W_q, bias_float, strides, pads, o_pads, dilations, groups)
            else:
                W_prepack = qconv_prepack_fn(
                    W_q, bias_float, strides, pads, dilations, groups)
            Y_q = qconv_fn(
                X_q,
                W_prepack,
                Y_scale,
                Y_zero_point,
            )
        else:
            # quantized conv op without prepacking
            Y_q = qconv_fn(X_q, W_q, bias_float, strides, pads, dilations, groups, Y_scale, Y_zero_point)

        # Make sure the results match
        # assert_array_almost_equal compares using the following formula:
        #     abs(desired-actual) < 1.5 * 10**(-decimal)
        # (https://docs.scipy.org/doc/numpy/reference/generated/numpy.testing.assert_almost_equal.html)
        # We use decimal = 0 to ignore off-by-1 differences between
        # reference and test. Off-by-1 differences arise due to the order of
        # round and zero_point addition operation, i.e., if addition
        # followed by round is used by reference and round followed by
        # addition is used by test, the results may differ by 1.
        # For example, the result of round(2.5) + 1 is 3 while
        # round(2.5 + 1) is 4 assuming the rounding mode is
        # round-to-nearest, ties-to-even.
        np.testing.assert_array_almost_equal(
            result_ref_q.int_repr().cpu().numpy(), Y_q.int_repr().cpu().numpy(), decimal=0,
            err_msg=f'''X: {X_q}, W: {W_q}, b: {bias_float}, strides: {strides},
            pads: {pads}, o_pads: {o_pads}, dilations: {dilations},
            groups: {groups}, y_s: {Y_scale}, y_zp: {Y_zero_point}''')

        # Return the quantized data for later reuse
        return X_q, W_q, bias_float

    """Tests the correctness of quantized convolution op."""
    @given(batch_size=st.integers(1, 3),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(10, 16),
           width=st.integers(7, 14),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 300),
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_relu=st.booleans(),
           use_channelwise=st.booleans())
    @override_qengines
    def test_qconv2d(
            self,
            batch_size,
            input_channels_per_group,
            height,
            width,
            output_channels_per_group,
            groups,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation,
            X_scale,
            X_zero_point,
            W_scale,
            W_zero_point,
            Y_scale,
            Y_zero_point,
            use_bias,
            use_relu,
            use_channelwise,
    ):
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        pads = (pad_h, pad_w)
        dilations = (dilation, dilation)

        qconv = torch.ops.quantized.conv2d
        if use_relu:
            qconv = torch.ops.quantized.conv2d_relu
        qconv_prepack = torch.ops.quantized.conv2d_prepack
        conv_op = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernels,
            strides,
            pads,
            dilations,
            groups,
        )

        act_qdtypes = [torch.quint8]
        # Only qnnpack qengine supportes qint8
        if qengine_is_qnnpack() and torch.backends.xnnpack.enabled:
            act_qdtypes.append(torch.qint8)

        for X_qdtype in act_qdtypes:
            if X_qdtype == torch.qint8:
                W_zero_point = [0 for i in range(len(W_zero_point))]

            self._test_qconv_impl(
                qconv, qconv_prepack, conv_op, batch_size,
                input_channels_per_group, (height, width),
                output_channels_per_group, groups, kernels, strides, pads, None,
                dilations, X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, use_relu, use_channelwise, False, input_dtype=X_qdtype, output_dtype=X_qdtype)

    # TODO: merge this test with test_qconv2d when CUDNN runtime flags becomes available
    """Tests the correctness of quantized 2D convolution cudnn op."""
    @given(batch_size=st.integers(1, 3),
           # cudnn only supports multiples of 4, but we have explicitly added padding on the backend
           input_channels_per_group=st.integers(1, 32),
           height=st.integers(10, 16),
           width=st.integers(7, 14),
           # cudnn only supports multiples of 4, but we have explicitly added padding on the backend
           output_channels_per_group=st.integers(1, 32),
           groups=st.integers(1, 1),  # currently padding only supports groups=1
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           # result for dilation == 2 is not correct
           # dilation=st.integers(1, 2),
           # currently cudnn has only been verified to work for dilation = 1
           # TODO: check backend works for dilation > 1
           dilation=st.integers(1, 1),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.sampled_from([0]),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(0, 0), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.sampled_from([0]),
           use_bias=st.booleans(),
           use_relu=st.booleans(),
           # TODO: enable channelwise
           use_channelwise=st.sampled_from([False]))
    @skipIfNoFBGEMM
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    @unittest.skip("Local only - currently the qconv2d_cudnn op is bulid "
                   "with USE_EXPERIMENTAL_CUDNN_V8_API, we can enable the test "
                   "after it is built by default")
    def test_qconv2d_cudnn(
            self,
            batch_size,
            input_channels_per_group,
            height,
            width,
            output_channels_per_group,
            groups,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            dilation,
            X_scale,
            X_zero_point,
            W_scale,
            W_zero_point,
            Y_scale,
            Y_zero_point,
            use_bias,
            use_relu,
            use_channelwise,
    ):
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        pads = (pad_h, pad_w)
        dilations = (dilation, dilation)

        if use_relu:
            qconv = torch.ops.quantized.conv2d_relu
        else:
            qconv = torch.ops.quantized.conv2d
        conv_op = torch.nn.Conv2d(
            input_channels,
            output_channels,
            kernels,
            strides,
            pads,
            dilations,
            groups,
        ).to(torch.device("cuda"))
        self._test_qconv_impl(
            qconv, torch.ops.quantized.conv2d_prepack, conv_op, batch_size,
            input_channels_per_group, (height, width),
            output_channels_per_group, groups, kernels, strides, pads, None,
            dilations, X_scale, X_zero_point, W_scale, W_zero_point,
            Y_scale, Y_zero_point, use_bias, use_relu, use_channelwise, False,
            device=torch.device("cuda"),
            input_dtype=torch.qint8, weight_dtype=torch.qint8, output_dtype=torch.qint8)

    @unittest.skip("used for local benchmarking, comment when we want to run it")
    def test_benchmark(self):
        batch_size = 16
        in_channel = 64
        out_channel = 64
        kernel_size = 3
        height = 256
        width = 256
        print(
            "parameters:",
            "batch_size:", batch_size,
            "in_channel:", in_channel,
            "out_channel:", out_channel,
            "kernel_size:", kernel_size,
            "height:", height,
            "widht:", width
        )
        conv = torch.nn.Conv2d(in_channel, out_channel, kernel_size).cuda()
        input = torch.randn((batch_size, in_channel, height, width), device='cuda')
        weight = conv.weight.detach()
        stride = (1, 1)
        padding = (0, 0)
        dilation = (1, 1)
        groups = 1
        conv_op = torch.nn.functional.conv2d
        # profile
        from torch.profiler import profile, ProfilerActivity

        def trace_handler(p):
            output = p.key_averages().table(sort_by="self_cpu_time_total", row_limit=10)
            p.export_chrome_trace("/tmp/trace_" + str(p.step_num) + ".json")

        my_schedule = torch.profiler.schedule(
            wait=5,
            warmup=5,
            active=20)

        # fp32 benchmark
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=my_schedule,
                on_trace_ready=trace_handler) as prof:
            for i in range(30):
                conv_op(input, weight, None, stride, padding, dilation, groups)
                prof.step()

        print("fp32 benchmark result:")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

        # fp16 benchmark
        input_fp16 = input.to(torch.float16)
        weight_fp16 = input.to(torch.float16)

        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=my_schedule,
                on_trace_ready=trace_handler) as prof:
            for i in range(30):
                conv_op(input_fp16, weight_fp16, None, stride, padding, dilation, groups)
                prof.step()

        print("fp16 benchmark result:")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

        input_int8 = torch.quantize_per_tensor(input, 1, 0, torch.qint8).contiguous(memory_format=torch.channels_last)
        weight_int8 = torch.quantize_per_tensor(weight, 1, 0, torch.qint8).contiguous(memory_format=torch.channels_last)
        scale = 1.0
        zero_point = 0
        conv_op = torch.ops.quantized.conv2d
        weight_prepacked = torch.ops.quantized.conv2d_prepack(weight_int8, None, stride, padding, dilation, groups)
        with profile(
                activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
                schedule=my_schedule,
                on_trace_ready=trace_handler) as prof:
            for i in range(30):
                conv_op(input_int8, weight_prepacked, scale, zero_point)
                prof.step()

        print("int8 benchmark result:")
        print(prof.key_averages().table(sort_by="self_cpu_time_total", row_limit=10))

    """Tests the correctness of quantized convolution op."""
    @override_qengines
    def test_qconv_transpose1d(self):
        if not qengine_is_qnnpack():
            return  # Currently only the QNNPACK is supported
        if qengine_is_qnnpack() and (IS_PPC or TEST_WITH_UBSAN):
            return  # QNNPACK doesn't support these
        batch_size = 2
        input_channels_per_group_list = [2, 32]
        width = 14
        output_channels_per_group_list = [2, 8]
        groups_list = [1, 3]
        kernel_list = [1, 7]
        stride_list = [1, 2]
        pad = 2
        o_pad = 0
        dilation = 1
        X_scale = 1.2
        X_zero_point = 1
        W_scale = [1.2]
        W_zero_point = [1]
        Y_scale = 4.2
        Y_zero_point = 2
        use_bias_list = [True, False]

        test_cases = itertools.product(
            input_channels_per_group_list, output_channels_per_group_list,
            groups_list, kernel_list, stride_list, use_bias_list)
        for input_channels_per_group, output_channels_per_group, \
                groups, kernel, stride, use_bias in test_cases:

            input_channels = input_channels_per_group * groups
            output_channels = output_channels_per_group * groups
            kernels = (kernel,)
            strides = (stride,)
            pads = (pad,)
            o_pads = (o_pad,)
            dilations = (dilation,)

            qconv = torch.ops.quantized.conv_transpose1d
            qconv_prepack = torch.ops.quantized.conv_transpose1d_prepack
            conv_op = torch.nn.ConvTranspose1d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernels,
                stride=strides,
                padding=pads,
                output_padding=o_pads,
                groups=groups,
                dilation=dilations,
                bias=use_bias
            )

            act_qdtypes = [torch.quint8]
            # Only qnnpack qengine supportes qint8
            if qengine_is_qnnpack() and torch.backends.xnnpack.enabled:
                act_qdtypes.append(torch.qint8)

            for X_qdtype in act_qdtypes:
                if X_qdtype == torch.qint8:
                    W_zero_point = [0 for i in range(len(W_zero_point))]

                X_q, W_q, bias_float = self._test_qconv_impl(
                    qconv, qconv_prepack, conv_op, batch_size,
                    input_channels_per_group, (width, ),
                    output_channels_per_group, groups, kernels, strides, pads, o_pads,
                    dilations, X_scale, X_zero_point, W_scale, W_zero_point,
                    Y_scale, Y_zero_point, use_bias, use_relu=False,
                    use_channelwise=False, use_transpose=True, input_dtype=X_qdtype, output_dtype=X_qdtype)

                # check that this doesn't error
                test_conv = torch.ao.nn.quantized.ConvTranspose1d(input_channels, output_channels, 1)
                test_conv.scale = Y_scale
                test_conv(X_q)

                # Test the module implementation
                qconv_op = torch.ao.nn.quantized.ConvTranspose1d(
                    in_channels=input_channels,
                    out_channels=output_channels,
                    kernel_size=kernels,
                    stride=strides,
                    padding=pads,
                    output_padding=o_pads,
                    groups=groups,
                    dilation=dilations,
                    bias=use_bias
                )
                qconv_op.scale = Y_scale
                qconv_op.zero_point = Y_zero_point
                qconv_op.set_weight_bias(W_q, bias_float)

                Y_dq_ref = conv_op(X_q.dequantize())
                Y_q_ref = torch.quantize_per_tensor(Y_dq_ref, scale=Y_scale,
                                                    zero_point=Y_zero_point,
                                                    dtype=X_qdtype)
                Y_q = qconv_op(X_q)
                self.assertEqual(Y_q_ref, Y_q)


    """Tests the correctness of quantized convolution op."""
    @given(batch_size=st.integers(1, 3),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(10, 16),
           width=st.integers(7, 14),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 300),
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           o_pad_h=st.integers(0, 2),
           o_pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans())
    @override_qengines
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    def test_qconv_transpose2d(
            self,
            batch_size,
            input_channels_per_group,
            height,
            width,
            output_channels_per_group,
            groups,
            kernel_h,
            kernel_w,
            stride_h,
            stride_w,
            pad_h,
            pad_w,
            o_pad_h,
            o_pad_w,
            dilation,
            X_scale,
            X_zero_point,
            W_scale,
            W_zero_point,
            Y_scale,
            Y_zero_point,
            use_bias):
        if qengine_is_qnnpack() and (IS_PPC or TEST_WITH_UBSAN):
            return  # QNNPACK doesn't support these
        # ONEDNN does not support output paddings
        if qengine_is_onednn() and (o_pad_h, o_pad_w) != (0, 0):
            return
        assume(o_pad_h < stride_h and o_pad_h < dilation)
        assume(o_pad_w < stride_w and o_pad_w < dilation)

        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        pads = (pad_h, pad_w)
        o_pads = (o_pad_h, o_pad_w)
        dilations = (dilation, dilation)

        qconv = torch.ops.quantized.conv_transpose2d
        qconv_prepack = torch.ops.quantized.conv_transpose2d_prepack
        conv_op = torch.nn.ConvTranspose2d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernels,
            stride=strides,
            padding=pads,
            output_padding=o_pads,
            groups=groups,
            dilation=dilations,
            bias=use_bias
        )
        act_qdtypes = [torch.quint8]
        # Only qnnpack qengine supportes qint8
        if qengine_is_qnnpack() and torch.backends.xnnpack.enabled:
            act_qdtypes.append(torch.qint8)

        for X_qdtype in act_qdtypes:
            if X_qdtype == torch.qint8:
                W_zero_point = [0 for i in range(len(W_zero_point))]

            X_q, W_q, bias_float = self._test_qconv_impl(
                qconv, qconv_prepack, conv_op, batch_size,
                input_channels_per_group, (height, width),
                output_channels_per_group, groups, kernels, strides, pads, o_pads,
                dilations, X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, use_relu=False,
                use_channelwise=False, use_transpose=True, input_dtype=X_qdtype, output_dtype=X_qdtype)

            # check that this doesn't error
            test_conv = torch.ao.nn.quantized.ConvTranspose2d(input_channels, output_channels, 1)
            test_conv.scale = Y_scale
            test_conv(X_q)

            # Test the module implementation
            qconv_op = torch.ao.nn.quantized.ConvTranspose2d(
                in_channels=input_channels,
                out_channels=output_channels,
                kernel_size=kernels,
                stride=strides,
                padding=pads,
                output_padding=o_pads,
                groups=groups,
                dilation=dilations,
                bias=use_bias
            )
            qconv_op.scale = Y_scale
            qconv_op.zero_point = Y_zero_point
            qconv_op.set_weight_bias(W_q, bias_float)

            Y_dq_ref = conv_op(X_q.dequantize())
            Y_q_ref = torch.quantize_per_tensor(Y_dq_ref, scale=Y_scale,
                                                zero_point=Y_zero_point,
                                                dtype=X_qdtype)
            Y_q = qconv_op(X_q)
            self.assertEqual(Y_q_ref, Y_q)

    """Tests the correctness of quantized convolution op."""
    @given(batch_size=st.integers(1, 3),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           time=st.integers(2, 5),
           height=st.integers(10, 16),
           width=st.integers(7, 14),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 300),
           kernel_t=st.integers(1, 7),
           kernel_h=st.integers(1, 7),
           kernel_w=st.integers(1, 7),
           stride_t=st.integers(1, 2),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_t=st.integers(0, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           o_pad_t=st.integers(0, 2),
           o_pad_h=st.integers(0, 2),
           o_pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans())
    @override_qengines
    @unittest.skip(
        "this is broken without changes to any relevant code, "
        "we need to remove hypothesis testing in CI")
    def test_qconv_transpose3d(
            self,
            batch_size,
            input_channels_per_group,
            time,
            height,
            width,
            output_channels_per_group,
            groups,
            kernel_t,
            kernel_h,
            kernel_w,
            stride_t,
            stride_h,
            stride_w,
            pad_t,
            pad_h,
            pad_w,
            o_pad_t,
            o_pad_h,
            o_pad_w,
            dilation,
            X_scale,
            X_zero_point,
            W_scale,
            W_zero_point,
            Y_scale,
            Y_zero_point,
            use_bias):
        if qengine_is_qnnpack():
            return  # QNNPACK doesn't support this
        # ONEDNN doesn't support output paddings
        if qengine_is_onednn() and (o_pad_t, o_pad_h, o_pad_w) != (0, 0, 0):
            return
        assume(o_pad_t < stride_t or o_pad_t < dilation)
        assume(o_pad_h < stride_h or o_pad_h < dilation)
        assume(o_pad_w < stride_w or o_pad_w < dilation)

        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_t, kernel_h, kernel_w)
        strides = (stride_t, stride_h, stride_w)
        pads = (pad_t, pad_h, pad_w)
        o_pads = (o_pad_t, o_pad_h, o_pad_w)
        dilations = (dilation, dilation, dilation)

        qconv = torch.ops.quantized.conv_transpose3d
        qconv_prepack = torch.ops.quantized.conv_transpose3d_prepack
        conv_op = torch.nn.ConvTranspose3d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernels,
            stride=strides,
            padding=pads,
            output_padding=o_pads,
            groups=groups,
            dilation=dilations,
            bias=use_bias
        )
        X_q, W_q, bias_float = self._test_qconv_impl(
            qconv, qconv_prepack, conv_op, batch_size,
            input_channels_per_group, (time, height, width),
            output_channels_per_group, groups, kernels, strides, pads, o_pads,
            dilations, X_scale, X_zero_point, W_scale, W_zero_point,
            Y_scale, Y_zero_point, use_bias, use_relu=False,
            use_channelwise=False, use_transpose=True)

        # check that this doesn't error
        test_conv = torch.ao.nn.quantized.ConvTranspose3d(input_channels, output_channels, 1)
        test_conv.scale = Y_scale
        test_conv(X_q)

        # Test the module implementation
        qconv_op = torch.ao.nn.quantized.ConvTranspose3d(
            in_channels=input_channels,
            out_channels=output_channels,
            kernel_size=kernels,
            stride=strides,
            padding=pads,
            output_padding=o_pads,
            groups=groups,
            dilation=dilations,
            bias=use_bias
        )
        qconv_op.scale = Y_scale
        qconv_op.zero_point = Y_zero_point
        qconv_op.set_weight_bias(W_q, bias_float)

        Y_dq_ref = conv_op(X_q.dequantize())
        Y_q_ref = torch.quantize_per_tensor(Y_dq_ref, scale=Y_scale,
                                            zero_point=Y_zero_point,
                                            dtype=torch.quint8)
        Y_q = qconv_op(X_q)
        self.assertEqual(Y_q_ref, Y_q)

    @given(
        inputs=hu.tensor_conv(
            spatial_dim=1, batch_size_range=(1, 3),
            input_channels_per_group_range=(1, 4),
            output_channels_per_group_range=(1, 4), feature_map_range=(4, 8),
            kernel_range=(1, 4), max_groups=4,
            can_be_transposed=False,
            qparams=[hu.qparams(dtypes=torch.quint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint32,
                                zero_point_min=0,
                                zero_point_max=0)]),
        stride=st.integers(1, 3),
        pad=st.integers(1, 2),
        o_pad=st.integers(1, 2),
        channelwise=st.booleans())
    @override_qengines
    def test_qconv1d_unpack(self, inputs, stride, pad, o_pad, channelwise):
        transposed = inputs[-1]
        qengine = torch.backends.quantized.engine
        if qengine not in supported_qengines:
            return
        if qengine == 'qnnpack':
            assume(not channelwise)  # QNNPACK doesn't support channelwise
        else:
            assume(not transposed)  # Only QNNPACK supports transposed conv
        if transposed:
            qconv_prepack = torch.ops.quantized.conv_transpose1d_prepack
            qconv_unpack = torch.ops.quantized.conv_transpose1d_unpack
        else:
            qconv_prepack = torch.ops.quantized.conv1d_prepack
            qconv_unpack = torch.ops.quantized.conv1d_unpack
        self._test_qconv_unpack_impl(
            qconv_prepack, qconv_unpack, inputs, [stride],
            [pad], [o_pad], channelwise)

    @given(
        inputs=hu.tensor_conv(
            spatial_dim=2, batch_size_range=(1, 3),
            input_channels_per_group_range=(1, 4),
            output_channels_per_group_range=(1, 4), feature_map_range=(4, 8),
            kernel_range=(1, 4), max_groups=4,
            can_be_transposed=True,
            qparams=[hu.qparams(dtypes=torch.quint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint32,
                                zero_point_min=0,
                                zero_point_max=0)]),
        stride=st.integers(1, 3),
        pad=st.integers(0, 2),
        o_pad=st.integers(0, 2),
        channelwise=st.booleans())
    @override_qengines
    def test_qconv2d_unpack(self, inputs, stride, pad, o_pad, channelwise):
        transposed = inputs[-1]
        qengine = torch.backends.quantized.engine
        if qengine not in supported_qengines:
            return
        if qengine == 'qnnpack':
            assume(not channelwise)  # QNNPACK doesn't support channelwise
        if transposed:
            qconv_prepack = torch.ops.quantized.conv_transpose2d_prepack
            qconv_unpack = torch.ops.quantized.conv_transpose2d_unpack
        else:
            qconv_prepack = torch.ops.quantized.conv2d_prepack
            qconv_unpack = torch.ops.quantized.conv2d_unpack
        self._test_qconv_unpack_impl(
            qconv_prepack, qconv_unpack, inputs, [stride, stride],
            [pad, pad], [o_pad, o_pad], channelwise)

    """Tests the correctness of quantized 1D convolution op."""
    @given(batch_size=st.integers(1, 6),
           input_channels_per_group=st.sampled_from((2, 4, 5, 8, 16, 32)),
           output_channels_per_group=st.sampled_from((2, 4, 5, 8, 16, 32)),
           groups=st.integers(1, 3),
           length=st.integers(4, 16),
           kernel=st.integers(1, 7),
           stride=st.integers(1, 2),
           pad=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_relu=st.booleans(),
           use_channelwise=st.booleans())
    @override_qengines
    def test_qconv1d(
        self,
        batch_size,
        input_channels_per_group,
        output_channels_per_group,
        groups,
        length,
        kernel,
        stride,
        pad,
        dilation,
        X_scale,
        X_zero_point,
        W_scale,
        W_zero_point,
        Y_scale,
        Y_zero_point,
        use_bias,
        use_relu,
        use_channelwise,
    ):
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        if torch.backends.quantized.engine == 'qnnpack':
            use_channelwise = False
        conv1d = torch.nn.Conv1d(
            input_channels,
            output_channels,
            kernel,
            stride,
            pad,
            dilation,
            groups,
        )
        qconv_prepack = torch.ops.quantized.conv1d_prepack
        qconv = torch.ops.quantized.conv1d
        if use_relu:
            qconv = torch.ops.quantized.conv1d_relu

        act_qdtypes = [torch.quint8]
        # Only qnnpack qengine supportes qint8
        if qengine_is_qnnpack() and torch.backends.xnnpack.enabled:
            act_qdtypes.append(torch.qint8)

        for X_qdtype in act_qdtypes:
            if X_qdtype == torch.qint8:
                W_zero_point = [0 for i in range(len(W_zero_point))]

            self._test_qconv_impl(
                qconv, qconv_prepack, conv1d, batch_size,
                input_channels_per_group, (length, ),
                output_channels_per_group, groups, kernel, [stride], [pad], None,
                [dilation], X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, use_relu, use_channelwise, False,
                input_dtype=X_qdtype, output_dtype=X_qdtype)

    # TODO: merge this test with test_qconv1d when CUDNN runtime flags becomes available
    """Tests the correctness of quantized 1D convolution cudnn op."""
    @given(batch_size=st.integers(1, 6),
           # cudnn only supports multiples of 4, but we have explicitly added padding on the backend
           input_channels_per_group=st.integers(1, 32),
           # cudnn only supports multiples of 4, but we have explicitly added padding on the backend
           output_channels_per_group=st.integers(1, 32),
           groups=st.integers(1, 1),  # currently padding only supports groups=1
           length=st.integers(4, 16),
           kernel=st.integers(1, 7),
           stride=st.integers(1, 2),
           pad=st.integers(0, 2),
           # currently cudnn has only been verified to work for dilation = 1
           # TODO: check backend works for dilation > 1
           dilation=st.integers(1, 1),
           X_scale=st.floats(1.2, 1.6),
           # currently conv cudnn backend is only implemented for int8 symmetric
           X_zero_point=st.sampled_from([0]),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           # currently conv cudnn backend is only implemented for int8 symmetric
           W_zero_point=st.lists(st.integers(0, 0), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           # currently conv cudnn backend is only implemented for int8 symmetric
           Y_zero_point=st.sampled_from([0]),
           use_bias=st.booleans(),
           use_relu=st.booleans(),
           # TODO: enable channelwise
           use_channelwise=st.sampled_from([False]))
    @skipIfNoFBGEMM
    @unittest.skipIf(not TEST_CUDNN, "cudnn is not enabled.")
    @unittest.skip("Local only - currently the qconv1d_cudnn op is bulid "
                   "with USE_EXPERIMENTAL_CUDNN_V8_API, we can enable the test "
                   "after it is built by default")
    def test_qconv1d_cudnn(
        self,
        batch_size,
        input_channels_per_group,
        output_channels_per_group,
        groups,
        length,
        kernel,
        stride,
        pad,
        dilation,
        X_scale,
        X_zero_point,
        W_scale,
        W_zero_point,
        Y_scale,
        Y_zero_point,
        use_bias,
        use_relu,
        use_channelwise,
    ):
        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups

        conv1d = torch.nn.Conv1d(
            input_channels,
            output_channels,
            kernel,
            stride,
            pad,
            dilation,
            groups,
        ).to(torch.device("cuda"))
        qconv_prepack = torch.ops.quantized.conv1d_prepack
        if use_relu:
            qconv = torch.ops.quantized.conv1d_relu
        else:
            qconv = torch.ops.quantized.conv1d

        self._test_qconv_impl(
            qconv, qconv_prepack, conv1d, batch_size,
            input_channels_per_group, (length, ),
            output_channels_per_group, groups, kernel, [stride], [pad], None,
            [dilation], X_scale, X_zero_point, W_scale, W_zero_point,
            Y_scale, Y_zero_point, use_bias, use_relu, use_channelwise, False,
            device=torch.device("cuda"),
            input_dtype=torch.qint8, weight_dtype=torch.qint8, output_dtype=torch.qint8)

    @given(batch_size=st.integers(1, 4),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16]),
           D=st.integers(4, 8),
           H=st.integers(4, 8),
           W=st.integers(4, 8),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16]),
           groups=st.integers(1, 3),
           kernel_d=st.integers(1, 4),
           kernel_h=st.integers(1, 4),
           kernel_w=st.integers(1, 4),
           stride_d=st.integers(1, 2),
           stride_h=st.integers(1, 2),
           stride_w=st.integers(1, 2),
           pad_d=st.integers(0, 2),
           pad_h=st.integers(0, 2),
           pad_w=st.integers(0, 2),
           dilation=st.integers(1, 2),
           X_scale=st.floats(1.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.lists(st.floats(0.2, 1.6), min_size=1, max_size=2),
           W_zero_point=st.lists(st.integers(-5, 5), min_size=1, max_size=2),
           Y_scale=st.floats(4.2, 5.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_relu=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_qconv3d(
        self,
        batch_size,
        input_channels_per_group,
        D,
        H,
        W,
        output_channels_per_group,
        groups,
        kernel_d,
        kernel_h,
        kernel_w,
        stride_d,
        stride_h,
        stride_w,
        pad_d,
        pad_h,
        pad_w,
        dilation,
        X_scale,
        X_zero_point,
        W_scale,
        W_zero_point,
        Y_scale,
        Y_zero_point,
        use_bias,
        use_relu,
        use_channelwise,
        qengine
    ):
        if qengine not in supported_qengines:
            return

        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_d, kernel_h, kernel_w)
        strides = (stride_d, stride_h, stride_w)
        pads = (pad_d, pad_h, pad_w)
        dilations = (dilation, dilation, dilation)

        with override_quantized_engine(qengine):
            qconv = torch.ops.quantized.conv3d
            if use_relu:
                qconv = torch.ops.quantized.conv3d_relu
            qconv_prepack = torch.ops.quantized.conv3d_prepack
            conv_op = torch.nn.Conv3d(
                input_channels,
                output_channels,
                kernels,
                strides,
                pads,
                dilations,
                groups,
            )
            self._test_qconv_impl(
                qconv, qconv_prepack, conv_op, batch_size,
                input_channels_per_group, (D, H, W), output_channels_per_group,
                groups, kernels, strides, pads, None, dilations, X_scale,
                X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
                use_bias, use_relu, use_channelwise, use_transpose=False)

    """Tests the correctness of the quantized::qconv3d_unpack op."""
    @given(
        inputs=hu.tensor_conv(
            spatial_dim=3, batch_size_range=(1, 3),
            input_channels_per_group_range=(1, 3),
            output_channels_per_group_range=(1, 3), feature_map_range=(3, 6),
            kernel_range=(1, 3), max_groups=3,
            qparams=[hu.qparams(dtypes=torch.quint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint32,
                                zero_point_min=0,
                                zero_point_max=0)]),
        stride_d=st.integers(1, 2), stride_h=st.integers(1, 2),
        stride_w=st.integers(1, 2),
        pad_d=st.integers(1, 2), pad_h=st.integers(1, 2),
        pad_w=st.integers(1, 2),
        o_pad=st.integers(0, 2),
        channelwise=st.booleans())
    @override_qengines
    def test_qconv3d_unpack(
        self, inputs, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w, o_pad,
        channelwise
    ):
        if qengine_is_qnnpack():
            return  # QNNPACK doesn't support this
        transposed = inputs[-1]
        if transposed:
            qconv_prepack = torch.ops.quantized.conv_transpose3d_prepack
            qconv_unpack = torch.ops.quantized.conv_transpose3d_unpack
        else:
            qconv_prepack = torch.ops.quantized.conv3d_prepack
            qconv_unpack = torch.ops.quantized.conv3d_unpack
        self._test_qconv_unpack_impl(
            qconv_prepack, qconv_unpack, inputs,
            (stride_d, stride_h, stride_w), (pad_d, pad_h, pad_w), (o_pad, o_pad, o_pad),
            channelwise)

class TestPadding(TestCase):
    @given(batch_size=st.integers(1, 64),
           channels=st.integers(1, 64),
           width=st.integers(16, 128),
           qtype=st.sampled_from(hu._ALL_QINT_TYPES))
    def test_reflection_pad1d(self, batch_size, channels, width, qtype):
        padding = width // 4

        x = torch.arange(batch_size * channels * width).to(torch.float)
        x = x.resize(batch_size, channels, width)
        # Per-Tensor test
        scale, zp = _calculate_dynamic_qparams(x, qtype)
        qx = torch.quantize_per_tensor(x, scale, zp, qtype)

        padding_op = torch.nn.ReflectionPad1d(padding)

        y_ref = padding_op(x)
        qy_ref = torch.quantize_per_tensor(y_ref, scale, zp, qtype)
        qy_hat = padding_op(qx)
        self.assertEqual(qy_ref, qy_hat)

        # Out variant
        qy_hat = torch._C._nn.reflection_pad1d(qx, padding, out=qy_hat)
        self.assertEqual(qy_ref, qy_hat)

    @given(batch_size=st.integers(1, 64),
           channels=st.integers(1, 64),
           height=st.integers(16, 128),
           width=st.integers(16, 128),
           qtype=st.sampled_from(hu._ALL_QINT_TYPES))
    def test_reflection_pad2d(self, batch_size, channels, height, width, qtype):
        padding = (width // 4, width // 4, height // 4, height // 4)

        x = torch.arange(batch_size * channels * height * width).to(torch.float)
        x = x.resize(batch_size, channels, height, width)
        # Per-Tensor test
        scale, zp = _calculate_dynamic_qparams(x, qtype)
        qx = torch.quantize_per_tensor(x, scale, zp, qtype)

        padding_op = torch.nn.ReflectionPad2d(padding)

        y_ref = padding_op(x)
        qy_ref = torch.quantize_per_tensor(y_ref, scale, zp, qtype)
        qy_hat = padding_op(qx)
        self.assertEqual(qy_ref, qy_hat)

        # Out variant
        qy_hat = torch._C._nn.reflection_pad2d(qx, padding, out=qy_hat)
        self.assertEqual(qy_ref, qy_hat)

    @given(batch_size=st.integers(1, 64),
           channels=st.integers(1, 64),
           hwd=st.integers(1, 16),  # For 3D, max input size would be 16x16x16
           d=st.sampled_from([1, 2, 3]),
           value=st.floats(-5, 5, allow_nan=False, allow_infinity=False),
           qtype=st.sampled_from(hu._ALL_QINT_TYPES))
    def test_constant_padNd(self, batch_size, channels, d, hwd, value, qtype):
        padding = hwd // 4

        shape = [batch_size, channels, hwd]
        op = torch.nn.ConstantPad1d
        if d >= 2:
            shape.append(hwd)
            op = torch.nn.ConstantPad2d
        if d == 3:
            shape.append(hwd)
            op = torch.nn.ConstantPad3d
        numel = np.prod(shape)

        x = torch.arange(numel).to(torch.float)
        x = x.resize(*shape)
        # Per-Tensor test
        scale, zp = _calculate_dynamic_qparams(x, qtype)
        qx = torch.quantize_per_tensor(x, scale, zp, qtype)

        padding_op = op(padding, value)

        y_ref = padding_op(x)
        qy_ref = torch.quantize_per_tensor(y_ref, scale, zp, qtype)
        qy_hat = padding_op(qx)

        self.assertEqual(qy_ref, qy_hat)


@unittest.skipUnless('qnnpack' in supported_qengines,
                     "This Pytorch Build has not been built with or does not support QNNPACK")
class TestQNNPackOps(TestCase):
    """Tests the correctness of the quantized::qnnpack_relu op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams(dtypes=torch.quint8,
                                          zero_point_min=0,
                                          zero_point_max=0)))
    def test_qnnpack_relu(self, X):
        with override_quantized_engine('qnnpack'):
            X, (scale, zero_point, torch_type) = X
            relu = torch.nn.functional.relu
            X = torch.from_numpy(X)
            Y = X.clone()

            qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point, dtype=torch_type)
            qY_hat = relu(qX)

            Y[Y < 0] = 0
            qY = torch.quantize_per_tensor(Y, scale=scale, zero_point=zero_point, dtype=torch_type)
            self.assertEqual(qY, qY_hat)

    """Tests the correctness of the quantized::qnnpack_tanh op."""
    @skipIfNoFBGEMM
    def test_qnnpack_tanh(self):
        # Note: In QNNPACK the output scale and zero_point can only be
        #       2.0/256, 128 respectively, as it uses a LUT with 256 bins.

        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        memory_formats = (torch.channels_last, torch.contiguous_format)
        test_cases = itertools.product(shapes, memory_formats)
        for shape, memory_format in test_cases:
            X, scale, zero_point, torch_type = torch.randn(*shape), 1.0, 0, torch.quint8
            if memory_format == torch.channels_last and len(shape) != 4:
                continue
            X = X.to(memory_format=memory_format)
            qX = torch.quantize_per_tensor(X, scale=scale,
                                           zero_point=zero_point,
                                           dtype=torch_type)

            # Floating point reference
            Y = torch.tanh(qX.dequantize())
            qY = torch.quantize_per_tensor(Y, scale=1.0 / 128, zero_point=128,
                                           dtype=torch.quint8)
            with override_quantized_engine('fbgemm'):
                qYserver = torch.tanh(qX)
            with override_quantized_engine('qnnpack'):
                qY_hat = torch.tanh(qX)
                self.assertEqual(
                    qY, qY_hat,
                    msg="QNNPACK TanH failed (FP ref), memory_format {}".format(memory_format))
                self.assertEqual(
                    qYserver, qY_hat,
                    msg="QNNPACK TanH failed (FBGEMM ref), memory_format {}".format(memory_format))

    """Tests the correctness of the quantized::qnnpack_sigmoid op."""
    @skipIfNoFBGEMM
    def test_qnnpack_sigmoid(self):
        # Note: In QNNPACK the output scale and zero_point can only be
        #       1.0/256, 0 respectively, as it uses a LUT with 256 bins.
        shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
        memory_formats = (torch.channels_last, torch.contiguous_format)
        test_cases = itertools.product(shapes, memory_formats)
        for shape, memory_format in test_cases:
            X, scale, zero_point, torch_type = torch.randn(*shape), 1.0, 0, torch.quint8
            if memory_format == torch.channels_last and len(shape) != 4:
                continue
            X = X.to(memory_format=memory_format)
            qX = torch.quantize_per_tensor(X, scale=scale,
                                           zero_point=zero_point,
                                           dtype=torch_type)

            # Floating point reference
            Y = torch.sigmoid(qX.dequantize())
            qY = torch.quantize_per_tensor(Y, scale=1.0 / 256, zero_point=0,
                                           dtype=torch.quint8)
            with override_quantized_engine('fbgemm'):
                qYserver = torch.sigmoid(qX)
            with override_quantized_engine('qnnpack'):
                qY_hat = torch.sigmoid(qX)
                self.assertEqual(
                    qY, qY_hat,
                    msg="QNNPACK Sigmoid failed (FP ref), memory_format {}".format(memory_format))
                self.assertEqual(
                    qYserver, qY_hat,
                    msg="QNNPACK Sigmoid failed (FBGEMM ref), memory_format {}".format(memory_format))

    @skipIfNoFBGEMM
    def test_qnnpack_sigmoid_sweep(self):
        # Input parameters
        f_min = -4.0
        f_max = 4.0
        scale = (f_max - f_min) / 256.0
        zero_point = 128
        dtype = torch.quint8

        step = scale / 2.0
        x = np.arange(f_min, f_max + step, step)
        X = torch.from_numpy(x).to(torch.float32)
        qX = torch.quantize_per_tensor(X, scale=scale,
                                       zero_point=zero_point,
                                       dtype=dtype)

        dqX = qX.dequantize()
        # Floating point reference
        Y = torch.sigmoid(dqX)
        qY = torch.quantize_per_tensor(Y, scale=1.0 / 256, zero_point=0,
                                       dtype=torch.quint8)
        with override_quantized_engine('fbgemm'):
            qYserver = torch.sigmoid(qX)
        with override_quantized_engine('qnnpack'):
            qY_hat = torch.sigmoid(qX)
            self.assertEqual(qY, qY_hat,
                             msg="QNNPACK Sigmoid failed (FP ref)!")
            self.assertEqual(qYserver, qY_hat,
                             msg="QNNPACK Sigmoid failed (FBGEMM ref)!")

    """Tests the correctness of the quantized::add (qnnpack) op."""
    @settings(suppress_health_check=(HealthCheck.filter_too_much,))
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams(dtypes=[torch.quint8, torch.qint8])),
           zero_point=st.sampled_from([0, 2, 5, 15, 127]),
           scale_A=st.sampled_from([0.001, 0.057, 0.889, 12.3]),
           scale_B=st.sampled_from([0.008, 0.0821, 0.67, 7]),
           scale_C=st.sampled_from([0.003, 0.07821, 0.457, 7.34]),)
    def test_qnnpack_add(self, A, zero_point, scale_A, scale_B, scale_C):
        with override_quantized_engine('qnnpack'):
            A_temp = A
            for channels_last in [True, False]:
                if channels_last and len(A_temp[0].shape) != 4:
                    continue
                A, (scale_a, zero_point_A, torch_type) = A_temp
                B, (scale_b, zero_point_B, torch_type) = A_temp
                A = torch.from_numpy(A)
                B = torch.from_numpy(B)

                if torch_type == torch.qint8 and not torch.backends.xnnpack.enabled:
                    continue

                if channels_last:
                    A = A.to(memory_format=torch.channels_last)
                    B = B.to(memory_format=torch.channels_last)
                assume(scale_A // scale_C >= 2**-14)
                assume(scale_A // scale_C < 2**8)
                assume(scale_B // scale_C >= 2**-14)
                assume(scale_B // scale_C < 2**8)

                zero_point_C = 127
                np_dtype = np.uint8

                if torch_type == torch.qint8:
                    zero_point_C = 0
                    np_dtype = np.int8

                qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point,
                                               dtype=torch_type)
                qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point,
                                               dtype=torch_type)

                # Add ground truth
                C = (qA.dequantize() + qB.dequantize()).numpy()

                qC = _quantize(C, scale_C, zero_point_C, dtype=np_dtype)

                qC_qnnp = torch.ops.quantized.add(qA, qB, scale_C, zero_point_C)

                np.testing.assert_equal(qC, qC_qnnp.int_repr(),
                                        "Quantized addition failed.")

                Crelu = C.copy()
                Crelu[C < 0] = 0
                qCrelu = torch.quantize_per_tensor(torch.from_numpy(Crelu), scale_C,
                                                   zero_point_C, dtype=torch_type)
                qCrelu_hat = torch.ops.quantized.add_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)
                np.testing.assert_equal(qCrelu.int_repr().numpy(), qCrelu_hat.int_repr(),
                                        "Quantized addition with ReLU failed.")

        """Tests the correctness of the quantized::add (qnnpack) mul."""
    @settings(suppress_health_check=(HealthCheck.filter_too_much,))
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams(dtypes=[torch.quint8, torch.qint8])),
           zero_point=st.sampled_from([0, 2, 5, 15, 127]),
           scale_A=st.sampled_from([0.3, 0.57, 0.889]),
           scale_B=st.sampled_from([0.8, 0.821, 0.67]),
           scale_C=st.sampled_from([0.3, 0.7821, 0.457]),)
    def test_qnnpack_mul(self, A, zero_point, scale_A, scale_B, scale_C):
        with override_quantized_engine('qnnpack'):
            A_temp = A
            for channels_last in [True, False]:
                if channels_last and len(A_temp[0].shape) != 4:
                    continue
                A, (scale_a, zero_point_A, torch_type) = A_temp
                B, (scale_b, zero_point_B, torch_type) = A_temp
                A = torch.from_numpy(A)
                B = torch.from_numpy(B)

                if torch_type == torch.qint8 and not torch.backends.xnnpack.enabled:
                    continue

                if channels_last:
                    A = A.to(memory_format=torch.channels_last)
                    B = B.to(memory_format=torch.channels_last)
                assume(scale_A // scale_C >= 2**-14)
                assume(scale_A // scale_C < 2**8)
                assume(scale_B // scale_C >= 2**-14)
                assume(scale_B // scale_C < 2**8)

                zero_point_C = 127
                np_dtype = np.uint8

                if torch_type == torch.qint8:
                    zero_point_C = 0
                    np_dtype = np.int8

                qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point,
                                               dtype=torch_type)
                qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point,
                                               dtype=torch_type)

                # Add ground truth
                C = (qA.dequantize() * qB.dequantize()).numpy()

                qC = _quantize(C, scale_C, zero_point_C, dtype=np_dtype)
                qC_qnnp = torch.ops.quantized.mul(qA, qB, scale_C, zero_point_C)

                np.testing.assert_equal(qC, qC_qnnp.int_repr(),
                                        "Quantized addition failed.")

                Crelu = C.copy()
                Crelu[C < 0] = 0
                qCrelu = torch.quantize_per_tensor(torch.from_numpy(Crelu), scale_C,
                                                   zero_point_C, dtype=torch_type)
                qCrelu_hat = torch.ops.quantized.mul_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)
                np.testing.assert_equal(qCrelu.int_repr().numpy(), qCrelu_hat.int_repr(),
                                        "Quantized addition with ReLU failed.")


    """Tests that quantized add works with broadcasting """
    def test_qnnpack_add_broadcast(self):
        def _run_test(A, B):
            qA = torch.quantize_per_tensor(A, 0.02, 0, dtype)
            qB = torch.quantize_per_tensor(B, 0.04, 2, dtype)

            output_scale = 0.01
            output_zp = 1

            # ground truth
            C = qA.dequantize() + qB.dequantize()
            qC = torch.quantize_per_tensor(C, output_scale, output_zp, dtype)

            # quantized
            qC_hat_1 = torch.ops.quantized.add(qA, qB, output_scale, output_zp)
            qC_hat_2 = torch.ops.quantized.add(qB, qA, output_scale, output_zp)

            self.assertTrue(torch.allclose(qC.dequantize(), qC_hat_1.dequantize()))
            self.assertTrue(torch.allclose(qC.dequantize(), qC_hat_2.dequantize()))

        with override_quantized_engine("qnnpack"):
            for dtype in (torch.qint8, torch.quint8):
                if dtype == torch.qint8 and not torch.backends.xnnpack.enabled:
                    continue

                for channels_last in [True, False]:
                    # 4d
                    A = torch.randn(1, 3, 4, 4)
                    B = torch.randn(1, 1, 1, 1)
                    if channels_last:
                        A = A.to(memory_format=torch.channels_last)
                        B = B.to(memory_format=torch.channels_last)
                    _run_test(A, B)

                    # 5d
                    C = torch.randn(1, 3, 4, 4, 4)
                    D = torch.randn(1, 1, 1, 1, 1)
                    if channels_last:
                        C = C.to(memory_format=torch.channels_last_3d)
                        D = D.to(memory_format=torch.channels_last_3d)
                    _run_test(C, D)

    """Tests the correctness of quantized::qnnpack_maxpool2d op."""
    @given(A=hu.tensor(shapes=hu.array_shapes(4, 4, 3, 5),
                       qparams=hu.qparams(dtypes=torch.quint8)),
           kernel=st.sampled_from([2, 4]),
           stride=st.sampled_from([1, 2]),
           padding=st.sampled_from([1, 2]))
    def test_qnnpack_maxpool2d(self, A, kernel, stride, padding):
        import torch.nn.functional as F

        with override_quantized_engine('qnnpack'):
            A, (scale, zero_point, torch_type) = A
            X = torch.from_numpy(A)
            np_type = np.uint8
            dilation = 1

            # Check constraints
            assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!

            iH, iW = X.shape[-2:]

            oH = pool_output_shape(iH, kernel, padding, stride, dilation)
            assume(oH > 0)
            oW = pool_output_shape(iW, kernel, padding, stride, dilation)
            assume(oW > 0)

            k = (kernel, kernel)
            s = (stride, stride)
            d = (dilation, dilation)
            p = (padding, padding)

            q_max_pool = torch.ops.quantized.max_pool2d

            a = scale * (X - zero_point).to(dtype=torch.float)
            qa = torch.quantize_per_tensor(a, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

            a_ref = qa.dequantize()

            a_pool = F.max_pool2d(a_ref, kernel_size=k, stride=s, padding=p,
                                  dilation=d)

            a_pool_nhwc = a_pool.permute([0, 2, 3, 1])

            qa_pool = q_max_pool(qa, k, s, p, d, ceil_mode=False)

            qa_pool_int = qa_pool.dequantize()
            np.testing.assert_equal(a_pool.numpy(), qa_pool_int.numpy())

    @given(batch_size=st.integers(1, 5),
           channels=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(4, 10),
           width=st.integers(4, 10),
           kernel=st.integers(2, 5),
           stride=st.integers(1, 2),
           padding=st.integers(1, 2),
           scale=st.floats(0.2, 1.6),
           zero_point=st.integers(0, 25)
           )
    def test_avg_pool2d(
            self,
            batch_size,
            channels,
            height,
            width,
            kernel,
            stride,
            padding,
            scale,
            zero_point

    ):
        with override_quantized_engine('qnnpack'):
            import torch.nn.functional as F
            X_init = torch.from_numpy(np.random.randint(
                0, 50, (batch_size, channels, height, width)))

            X = scale * (X_init - zero_point).to(dtype=torch.float)

            # Check constraints
            assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!

            iH, iW = X.shape[-2:]

            oH = pool_output_shape(iH, kernel, padding, stride, 1)
            assume(oH > 0)
            oW = pool_output_shape(iW, kernel, padding, stride, 1)
            assume(oW > 0)
            k = (kernel, kernel)
            s = (stride, stride)
            p = (padding, padding)

            q_avg_pool = torch.ao.nn.quantized.functional.avg_pool2d

            x_q = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                            dtype=torch.quint8)

            a_pool = F.avg_pool2d(x_q.dequantize().to(torch.float), kernel_size=k, stride=s, padding=p)
            qa_pool = q_avg_pool(x_q, k, s, p)
            # Quantize Ref Output
            a_pool_q = torch.quantize_per_tensor(a_pool, scale=scale, zero_point=zero_point,
                                                 dtype=torch.quint8)
            np.testing.assert_array_almost_equal(a_pool_q.int_repr().numpy(),
                                                 qa_pool.int_repr().numpy(), decimal=0)


    @given(batch_size=st.integers(1, 5),
           channels=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(4, 20),
           width=st.integers(4, 20),
           output_height=st.integers(2, 10),
           output_width=st.integers(2, 10),
           scale=st.floats(0.2, 1.6),
           zero_point=st.integers(0, 25)
           )
    def test_adaptive_avg_pool2d(
            self,
            batch_size,
            channels,
            height,
            width,
            output_height,
            output_width,
            scale,
            zero_point

    ):
        with override_quantized_engine('qnnpack'):
            # Check constraints
            assume(height >= output_height)
            assume(width >= output_width)

            import torch.nn.functional as F
            X_init = torch.from_numpy(np.random.randint(
                0, 50, (batch_size, channels, height, width)))

            X = scale * (X_init - zero_point).to(dtype=torch.float)

            iH, iW = X.shape[-2:]

            q_avg_pool = torch.ao.nn.quantized.functional.adaptive_avg_pool2d

            x_q = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                            dtype=torch.quint8)

            a_pool = F.adaptive_avg_pool2d(x_q.dequantize().to(torch.float), (output_height, output_width))
            qa_pool = q_avg_pool(x_q, (output_height, output_width))
            # Quantize Ref Output
            a_pool_q = torch.quantize_per_tensor(a_pool, scale=scale, zero_point=zero_point,
                                                 dtype=torch.quint8)
            np.testing.assert_array_almost_equal(a_pool_q.int_repr().numpy(),
                                                 qa_pool.int_repr().numpy(), decimal=0)


    @given(batch_size=st.integers(1, 5),
           channels=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(4, 10),
           width=st.integers(4, 10),
           scale=st.floats(0.02, 2.6),
           zero_point=st.integers(0, 25))
    def test_mean(self, batch_size, channels, height, width, scale, zero_point):
        with override_quantized_engine('qnnpack'):
            dim = (2, 3)
            X_init = torch.from_numpy(np.random.randint(
                0, 50, (batch_size, channels, height, width)))
            X = scale * (X_init - zero_point).to(dtype=torch.float)

            qX = torch.quantize_per_tensor(X, scale, zero_point, torch.quint8)
            Y = torch.mean(qX.dequantize(), dim)
            Y = torch.quantize_per_tensor(Y, scale, zero_point, torch.quint8)
            qY = torch.mean(qX, dim)
            np.testing.assert_array_almost_equal(Y.int_repr().numpy(), qY.int_repr().numpy(), decimal=0)

    """Tests the correctness of the quantized::hardtanh op."""
    def test_hardtanh(self):
        if 'qnnpack' not in torch.backends.quantized.supported_engines:
            return
        with override_quantized_engine('qnnpack'):
            shapes = ((4,), (4, 4), (4, 4, 4), (4, 4, 4, 4))
            memory_formats = (torch.channels_last, torch.contiguous_format)
            min_vals = (-0.5, -0.3, 0.5)
            max_vals = (-0.3, 0.3, 0.7)
            test_cases = itertools.product(shapes, memory_formats, min_vals, max_vals)
            for shape, memory_format, min_val, max_val in test_cases:
                X, scale, zero_point, torch_type = torch.randn(*shape), 1.0, 0, torch.quint8
                if memory_format == torch.channels_last and len(shape) != 4:
                    continue

                Y = X.clone()
                Y[Y < min_val] = min_val
                Y[Y > max_val] = max_val
                qY = torch.quantize_per_tensor(Y, scale=scale,
                                               zero_point=zero_point, dtype=torch_type)
                qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                               dtype=torch_type)

                qY_hat = torch.ao.nn.quantized.functional.hardtanh(qX, min_val, max_val)
                self.assertEqual(
                    qY, qY_hat,
                    msg="hardtanh failed:\nactual {}\nexpected {}\nmemory_format {}".format(qY_hat, qY, memory_format))

"""Tests the correctness of the tensor comparators."""
class TestComparatorOps(TestCase):
    """Tests the element-wise equality ops."""
    @given(A=hu.tensor(shapes=((3, 4, 5),),
                       qparams=hu.qparams()),
           B=hu.tensor(shapes=((5,), (1, 5), (1, 1, 5), (4, 5), (3, 4, 5)),
                       qparams=hu.qparams()))
    def test_compare_tensor_tensor(self, A, B):
        A, (scale_a, zero_point_a, dtype_a) = A
        B, (scale_b, zero_point_b, dtype_b) = B
        tA = torch.from_numpy(A)
        tB = torch.from_numpy(B)

        qA = torch.quantize_per_tensor(tA, scale=scale_a, zero_point=zero_point_a,
                                       dtype=dtype_a)
        qB = torch.quantize_per_tensor(tB, scale=scale_b, zero_point=zero_point_b,
                                       dtype=dtype_b)
        dqA = qA.dequantize()
        dqB = qB.dequantize()

        ops_under_test = ('__eq__', '__ne__', '__ge__', '__le__', '__gt__',
                          '__lt__', 'eq', 'ne', 'ge', 'le', 'gt', 'lt')

        for op in ops_under_test:
            result_ref = getattr(dqA, op)(dqB)
            result = getattr(qA, op)(qB)
            self.assertEqual(result_ref, result,
                             msg="'tensor.{}(tensor)'' failed".format(op))
            # Reversed broadcasting.
            result_ref = getattr(dqB, op)(dqA)
            result = getattr(qB, op)(qA)
            self.assertEqual(result_ref, result,
                             msg="'tensor.{}(tensor)'' failed".format(op))

    @given(A=hu.tensor(shapes=((3, 4, 5),),
                       qparams=hu.qparams()),
           b=hu.floats(allow_infinity=False, allow_nan=False))
    def test_compare_tensor_scalar(self, A, b):
        A, (scale_a, zero_point_a, dtype_a) = A
        tA = torch.from_numpy(A)

        qA = torch.quantize_per_tensor(tA, scale=scale_a, zero_point=zero_point_a,
                                       dtype=dtype_a)
        dqA = qA.dequantize()

        ops_under_test_reversible = ('__eq__', '__ne__', '__ge__', '__le__',
                                     '__gt__', '__lt__')
        ops_under_test_nonreversible = ('eq', 'ne', 'ge', 'le', 'gt', 'lt')

        for op in ops_under_test_reversible:
            result_ref = getattr(dqA, op)(b)
            result = getattr(qA, op)(b)
            note("result_ref 1: {}".format(result_ref))
            note("result 1: {}".format(result))
            self.assertEqual(result_ref, result,
                             msg="'tensor.{}(scalar)'' failed".format(op))
            # Reversed broadcasting.
            result_ref = getattr(b, op)(dqA)
            result = getattr(b, op)(qA)
            note("result_ref 2: {}".format(result_ref))
            note("result 2: {}".format(result))
            self.assertEqual(result_ref, result,
                             msg="'scalar.{}(tensor)'' failed".format(op))

        for op in ops_under_test_nonreversible:
            result_ref = getattr(dqA, op)(b)
            result = getattr(qA, op)(b)
            note("result_ref 3: {}".format(result_ref))
            note("result 3: {}".format(result))
            self.assertEqual(result_ref, result,
                             msg="'tensor.{}(scalar)'' failed".format(op))
