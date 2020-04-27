from __future__ import division
from builtins import round

import numpy as np
import unittest

import torch
import torch.jit
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair

from hypothesis import settings, HealthCheck
from hypothesis import assume, given, note
from hypothesis import strategies as st
import torch.testing._internal.hypothesis_utils as hu
hu.assert_deadline_disabled()

from torch.testing._internal.common_utils import TEST_WITH_UBSAN, TestCase, IS_PPC, IS_MACOS
from torch.testing._internal.common_quantized import _quantize, _dequantize, _calculate_dynamic_qparams, \
    override_quantized_engine

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
def qlinear_ref(X_q, X_scale, X_zp, W_q, W_scale, W_zp, b_q, Y_scale, Y_zp):
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
    Y_q_ref = _quantize(Prod_XqWq_ref, Y_scale / (X_scale * W_scale), Y_zp)
    return Y_q_ref

"""Computes the output shape given pooling parameters."""
def pool_output_shape(input_size, kernel_size, padding, stride,
                      dilation, ceiling_mode=False):
    if stride is None:
        stride = kernel_size
    output_size = (
        (input_size + 2 * padding - dilation * (kernel_size - 1) - 1
         + (stride - 1 if ceiling_mode else 0)) // stride + 1)
    if (padding > 0 and
            ((output_size - 1) * stride >= input_size + padding)):
        output_size += 1
    return output_size

"""Common logic for hardswish testing, called from fbgemm and qnnpack testers"""
def _test_hardswish(self, X, Y_scale, Y_zero_point, engine):
    if engine not in torch.backends.quantized.supported_engines:
        return
    with override_quantized_engine(engine):
        X, (X_scale, X_zero_point, torch_type) = X
        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=X_scale, zero_point=X_zero_point,
                                       dtype=torch_type)
        dqX = qX.dequantize()

        dqY_hat = F.hardswish(dqX)
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=Y_scale,
                                           zero_point=Y_zero_point,
                                           dtype=torch_type)

        qY = torch.nn.quantized.functional.hardswish(
            qX, scale=Y_scale, zero_point=Y_zero_point)
        self.assertEqual(
            qY, qY_hat,
            message="Hardswish failed: {} vs {}".format(qY, qY_hat))

"""Common logic for hardswish testing, called from fbgemm and qnnpack testers"""
def _test_hardsigmoid(self, X, engine):
    if engine not in torch.backends.quantized.supported_engines:
        return
    with override_quantized_engine(engine):
        X, (scale, zero_point, torch_type) = X

        X = torch.from_numpy(X)

        qX = torch.quantize_per_tensor(X, scale=scale,
                                       zero_point=zero_point,
                                       dtype=torch_type)
        dqX = qX.dequantize()


        # Quantize the reference to account for max error.
        # Note that the output scale has +1, because we use scale of 1.0/2^BITS
        # in the implementations.
        f_min, f_max = 0.0, 1.0
        q_min, q_max = torch.iinfo(torch_type).min, torch.iinfo(torch_type).max
        output_scale = (f_max - f_min) / (q_max - q_min + 1.0)
        output_zero_point = 0 if torch_type == torch.qint32 else q_min
        dqY_hat = F.hardsigmoid(dqX)
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=output_scale,
                                           zero_point=output_zero_point,
                                           dtype=torch_type)

        qY = torch.nn.quantized.functional.hardsigmoid(qX)
        self.assertEqual(qY, qY_hat,
                         message="Hardsigmoid failed: {} vs. {}".format(qY, qY_hat))

class TestQuantizedOps(TestCase):

    """Tests the correctness of the quantized::relu op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_qrelu(self, X):
        X, (scale, zero_point, torch_type) = X

        Y = X.copy()
        Y[Y < 0] = 0
        qY = torch.quantize_per_tensor(torch.from_numpy(Y), scale=scale,
                                       zero_point=zero_point, dtype=torch_type)
        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        ops_under_test = {
            'native': torch.relu,
            'nn.functional': torch.nn.functional.relu,
        }

        for name, op in ops_under_test.items():
            qY_hat = op(qX)
            self.assertEqual(qY, qY_hat, message="{} relu failed".format(name))

        ops_under_test_inplace = {
            'inplace native': torch.relu_,
            'inplace nn.functional': torch.nn.functional.relu_,
        }

        for name, op_ in ops_under_test_inplace.items():
            qY_hat = qX.clone()
            op_(qY_hat)
            self.assertEqual(qY, qY_hat, message="{} relu failed".format(name))

    """Tests the correctness of the quantized::relu op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_qrelu6(self, X):
        X, (scale, zero_point, torch_type) = X

        Y = X.copy()
        Y[Y < 0] = 0
        Y[Y > 6.0] = 6.0
        qY = torch.quantize_per_tensor(torch.from_numpy(Y), scale=scale,
                                       zero_point=zero_point, dtype=torch_type)
        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        ops_under_test = {
            'ops.quantized': torch.ops.quantized.relu6,
            'module': torch.nn.quantized.ReLU6(),
        }

        for name, op in ops_under_test.items():
            for inplace in (True, False):
                if hasattr(op, 'inplace'):
                    op.inplace = inplace
                    qY_hat = op(qX)
                else:
                    qY_hat = op(qX, inplace=inplace)
                self.assertEqual(qY, qY_hat,
                                 message="{} relu failed".format(name))

    """Tests the correctness of the quantized::relu op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()),
           alpha=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False))
    def test_qrelu_leaky(self, X, alpha):
        X, (scale, zero_point, torch_type) = X

        X = torch.from_numpy(X)
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
                         message="F.leaky_relu failed ({} vs {})".format(qY, qY_hat))

    """Tests the correctness of the quantized::elu op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams()),
           alpha=st.floats(0.01, 10.0, allow_nan=False, allow_infinity=False))
    def test_qelu(self, X, alpha):
        X, (scale, zero_point, torch_type) = X

        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)
        op = torch.nn.quantized.functional.elu

        # calculate ELU(dqX) and quantize
        dqX = qX.dequantize()
        dqY_hat = dqX.clone()
        dqY_hat[dqX < 0] = alpha * (torch.exp(dqY_hat[dqX < 0]) - 1.)
        qY_hat = torch.quantize_per_tensor(dqY_hat, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)

        # test regular
        qY = op(qX, alpha=alpha)
        self.assertEqual(qY, qY_hat,
                         message="F.elu failed ({} vs {})".format(qY, qY_hat))

        # test inplace
        qXcopy = qX.clone()
        op(qXcopy, alpha=alpha, inplace=True)
        self.assertEqual(qXcopy, qY_hat,
                         message="F.elu_ failed ({} vs {})".format(qXcopy, qY_hat))

        # test explicit scale and zp
        qYout = op(qX, alpha=alpha, scale=scale, zero_point=zero_point)
        self.assertEqual(qYout, qY_hat,
                         message="F.elu.out failed ({} vs {})".format(qY, qY_hat))

    """Tests the correctness of the quantized::qnnpack_sigmoid op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_qsigmoid(self, X):
        # Note: QNNPACK is tested separately in TestQNNPackOps
        X, (scale, zero_point, torch_type) = X

        X = torch.from_numpy(X)
        Y = torch.sigmoid(X)

        qX = torch.quantize_per_tensor(X, scale=scale,
                                       zero_point=zero_point,
                                       dtype=torch_type)

        # Quantize the reference to account for max error.
        # Note that the output scale has +1, because we use scale of 1.0/2^BITS
        # in the implementations.
        f_min, f_max = 0.0, 1.0
        q_min, q_max = torch.iinfo(torch_type).min, torch.iinfo(torch_type).max
        output_scale = (f_max - f_min) / (q_max - q_min + 1.0)
        output_zero_point = output_zero_point = 0 if torch_type == torch.qint32 else q_min
        qY = torch.quantize_per_tensor(Y, scale=output_scale,
                                       zero_point=output_zero_point,
                                       dtype=torch_type)
        qY_hat = torch.sigmoid(qX)
        self.assertEqual(qY, qY_hat,
                         message="Sigmoid failed: {} vs. {}".format(qY, qY_hat))

    """Tests the correctness of the quantized::qhardsigmoid op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       elements=hu.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams()))
    def test_qhardsigmoid(self, X):
        _test_hardsigmoid(self, X, 'fbgemm')


    """Tests the correctness of the quantized::qlayer_norm op."""
    @given(shapes=hu.array_shapes(3, 5, 1, 32),
           torch_type=st.sampled_from((torch.qint8, torch.quint8, torch.qint32)),
           X_rand_scale=st.floats(0.01, 1e3),
           Y_scale=st.floats(0.2, 2.6),
           Y_zero_point=st.integers(0, 5))
    def test_qlayer_norm(self, shapes, torch_type, X_rand_scale, Y_scale, Y_zero_point):
        if "fbgemm" not in torch.backends.quantized.supported_engines:
            return

        with override_quantized_engine("fbgemm"):

            # In the FP kernel, mean and variance are calculated in floating point.
            # In the quantized kernel, they are calculated in integer arithmetic.
            # Because of this, the numerics do not always match exactly which is
            # expected and acceptable. We do two things to whitelist this failure
            # in this test:
            # 1. do not use Hypothesis to generate the input tensor.  Hypothesis
            #    favors homogeneous inputs in its search strategies which isn't
            #    representative of the inputs we care about, and tends to maximize
            #    this particular numerics difference.
            # 2. whitelist a small % of off by Y_scale errors.  Even when the
            #    variance of the input is high, there can be off by one errors
            #    in the result if the input value happens to fall exactly on
            #    the bin boundary of the output scale.
            #
            # If we want the numerics to match we could switch to calculating
            # mean+var in floating point in the future, at the cost of speed.

            X = (np.random.rand(*shapes).astype(np.float32) - 0.5) * X_rand_scale

            # Calculate reasonable quantization params
            min_val = np.min(X)
            max_val = np.max(X)
            if torch_type == torch.qint32:
                X_zero_point = 0
                num_bins = 2 ** 32
                X_scale = float(max_val - min_val) / num_bins
            elif torch_type == torch.qint8:
                X_zero_point = 0
                num_bins = 2 ** 8
                X_scale = float(max_val - min_val) / num_bins
            else:  # torch.quint8
                X_zero_point = 127
                num_bins = 2 ** 8
                X_scale = float(max_val - min_val) / num_bins
            if X_scale == 0:
                X_scale = 1e-10

            X = torch.from_numpy(X)
            qX = torch.quantize_per_tensor(X, scale=X_scale,
                                           zero_point=X_zero_point,
                                           dtype=torch_type)
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
            weight = torch.ones(*qX.size()[1:], dtype=torch.float) * 0.5
            bias = torch.ones(*qX.size()[1:], dtype=torch.float) * 1
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

            note("LayerNorm failed:\n {} input vs\n {} actual vs \n{} expected"
                 .format(X, qY, qY_hat))
            note("Pct diff: {}".format(pct_diff))
            note("Pct diff off by one: {}".format(pct_diff_off_by_one))

            self.assertTrue(pct_diff < 1e-6)
            self.assertTrue(pct_diff_off_by_one < 0.01)


    """Tests the correctness of the quantized::qnnpack_tanh op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
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
                         message="TanH failed: {} vs. {}".format(qY, qY_hat))

    """Tests the correctness of the quantized::clamp op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 8, 1, 8),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False),
                       qparams=hu.qparams()),
           min_val=hu.floats(-1e6, 1e6, allow_nan=False),
           max_val=hu.floats(-1e6, 1e6, allow_nan=False))
    def test_qclamp(self, X, min_val, max_val):
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
            'ops.quantized': torch.ops.quantized.clamp,
        }

        for name, op in ops_under_test.items():
            qY_hat = op(qX, min_val, max_val)
            self.assertEqual(qY, qY_hat, message="{} qclamp failed".format(name))

    """Tests the correctness of the quantized::hardtanh op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 8, 1, 8),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams()),
           min_val=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
           max_val=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
    def test_hardtanh(self, X, min_val, max_val):
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
        }

        for name, op in ops_under_test.items():
            qY_hat = op(qX, min_val, max_val)
            self.assertEqual(qY, qY_hat, message="{} hardtanh failed".format(name))

        ops_under_test_inplace = {
            'inplace nn.quantized.functional.hardtanh':
                torch.nn.quantized.functional.hardtanh,
        }

        for name, op_ in ops_under_test_inplace.items():
            qY_hat = qX.clone()
            op_(qY_hat, min_val, max_val, inplace=True)
            self.assertEqual(qY, qY_hat, message="{} hardtanh failed".format(name))

    """Tests the correctness of the quantized::hardswish op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 8, 1, 8),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams()),
           Y_scale=st.floats(1e-6, 1e6),
           Y_zero_point=st.integers(0, 10))
    def test_hardswish(self, X, Y_scale, Y_zero_point):
        _test_hardswish(self, X, Y_scale, Y_zero_point, 'fbgemm')

    """Tests the correctness of the scalar addition."""
    @unittest.skip("Failing on MacOS")
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 4, 1, 5),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False),
                       qparams=hu.qparams()),
           b=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False))
    def test_qadd_scalar_relu(self, A, b):
        import copy
        add_scalar = torch.ops.quantized.add_scalar
        add_scalar_relu = torch.ops.quantized.add_scalar_relu

        A, (scale, zero_point, dtype) = A
        A = A.astype(np.float32)
        qA = torch.quantize_per_tensor(torch.from_numpy(A), scale, zero_point, dtype)

        C = qA.dequantize() + round(b / scale) * scale
        C_relu = copy.deepcopy(C)
        C_relu[C_relu < 0] = 0

        C_hat = add_scalar(qA, b)
        C_ref = torch.quantize_per_tensor(C, C_hat.q_scale(), C_hat.q_zero_point(), dtype)
        C_relu_hat = add_scalar_relu(qA, b)
        C_relu_ref = torch.quantize_per_tensor(
            C_relu, C_relu_hat.q_scale(), C_relu_hat.q_zero_point(), dtype)

        self.assertEqual(C_ref.dequantize(), C_hat.dequantize(),
                         message="Scalar add results don't match:\
                         {} vs {}".format(C_ref.dequantize(), C_hat.dequantize()))
        self.assertEqual(C_relu_ref.dequantize(), C_relu_hat.dequantize(),
                         message="Scalar add relu results don't match:\
                         {} vs {}".format(C_relu_ref.dequantize(), C_relu_hat.dequantize()))

    """Tests the correctness of the add and add_relu op."""
    def test_qadd_relu_same_qparams(self):
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            add_relu = torch.ops.quantized.add_relu
            add = torch.ops.quantized.add
            add_out = torch.ops.quantized.add_out
            add_relu_out = torch.ops.quantized.add_relu_out

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
            self.assertEqual(qC_hat, qC_out_hat, message="Add.out failed")

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
                             message="AddReLU.out failed")


    """Tests the correctness of the add and add_relu op."""
    def test_qadd_relu_different_qparams(self):
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            add_relu = torch.ops.quantized.add_relu
            add = torch.ops.quantized.add
            add_out = torch.ops.quantized.add_out
            add_relu_out = torch.ops.quantized.add_relu_out

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
            self.assertEqual(qC_hat, qC_out_hat, message="Add.out failed")

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
                             message="AddReLU.out failed")

    """Tests the correctness of the mul and mul_relu op."""
    def test_qmul_relu_same_qparams(self):
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            mul_relu = torch.ops.quantized.mul_relu
            mul = torch.ops.quantized.mul
            mul_out = torch.ops.quantized.mul_out
            mul_relu_out = torch.ops.quantized.mul_relu_out

            A = torch.arange(-100, 100, dtype=torch.float)
            B = torch.arange(-100, 100, dtype=torch.float)
            scale = 2.0
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
            self.assertEqual(qC_hat, qC_out_hat, message="mul.out failed")

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
                             message="mulReLU.out failed")

            # Scalar multiplication
            for b in B:
                C_ref = qA.dequantize().numpy() * b.item()
                qC_hat = torch.ops.quantized.mul_scalar(qA, b.item())

                self.assertEqual(C_ref, qC_hat.dequantize())

            # Scalar multiplication + relu
            for b in B:
                C_ref = qA.dequantize().numpy() * b.item()
                C_ref[C_ref < 0] = 0
                qC_hat = torch.ops.quantized.mul_scalar_relu(qA, b.item())

                self.assertEqual(C_ref, qC_hat.dequantize())

    """Tests the correctness of the mul and mul_relu op."""
    def test_qmul_relu_different_qparams(self):
        for dtype in [torch.quint8, torch.qint8, torch.qint32]:
            mul_relu = torch.ops.quantized.mul_relu
            mul = torch.ops.quantized.mul
            mul_out = torch.ops.quantized.mul_out
            mul_relu_out = torch.ops.quantized.mul_relu_out

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
            self.assertEqual(qC_hat, qC_out_hat, message="mul.out failed")

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
                             message="mulReLU.out failed")

    """Tests the correctness of the mul and mul_relu op."""
    def test_qmul_broadcast(self):
        mul_relu = torch.ops.quantized.mul_relu
        mul = torch.ops.quantized.mul
        mul_out = torch.ops.quantized.mul_out
        mul_relu_out = torch.ops.quantized.mul_relu_out

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

    """Tests channel shuffle operation on quantized tensors."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=2, max_side=32),
                       qparams=hu.qparams()),
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
                                          zero_point=zero_point, dtype=torch.quint8)
        a_ref = a_ref.dequantize()
        qa = torch.quantize_per_tensor(a, scale=scale, zero_point=zero_point,
                                       dtype=torch.quint8)

        a_hat = torch.nn.functional.channel_shuffle(qa, groups)
        self.assertEqual(a_ref, a_hat.dequantize(),
                         message="torch.nn.functional.channel_shuffle results are off")

    """Tests max pool operation on quantized tensors."""
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
            "nn.quantized.functional": torch.nn.quantized.functional.max_pool2d
        }

        for name, op in ops_under_test.items():
            a_hat = op(qa, kernel_size=kernel, stride=stride, padding=padding,
                       dilation=dilation, ceil_mode=ceil_mode)
            self.assertEqual(a_ref, a_hat.dequantize(),
                             message="{} results are off".format(name))
        # Test the ops.quantized separately, because None is not treated.
        a_hat = torch.ops.quantized.max_pool2d(
            qa, kernel_size=_pair(kernel),
            stride=_pair(kernel if stride is None else stride),
            padding=_pair(padding), dilation=_pair(dilation), ceil_mode=ceil_mode)
        self.assertEqual(a_ref, a_hat.dequantize(),
                         message="ops.quantized.max_pool2d results are off")

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
            "nn.quantized.functional": torch.nn.quantized.functional.max_pool2d
        }

        for name, op in ops_under_test.items():
            a_hat = op(qa, kernel_size=kernel, stride=stride, padding=padding,
                       dilation=dilation, ceil_mode=ceil_mode)
            self.assertTrue(a_hat.stride() != sorted(a_hat.stride()))
            self.assertEqual(a_ref, a_hat.dequantize(),
                             message="{} results are off".format(name))
        # Test the ops.quantized separately, because None is not treated.
        a_hat = torch.ops.quantized.max_pool2d(
            qa, kernel_size=_pair(kernel),
            stride=_pair(kernel if stride is None else stride),
            padding=_pair(padding), dilation=_pair(dilation), ceil_mode=ceil_mode)
        self.assertEqual(a_ref, a_hat.dequantize(),
                         message="ops.quantized.max_pool2d results are off")

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
            "nn.quantized.functional": torch.nn.quantized.functional.avg_pool2d
        }
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            qX_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad, divisor_override=divisor_override)
            qX_ref = torch.quantize_per_tensor(X_ref, scale=qX_hat.q_scale(), zero_point=qX_hat.q_zero_point(),
                                               dtype=torch_type)

            self.assertEqual(qX_ref.int_repr().to(torch.double), qX_hat.int_repr().to(torch.double), atol=1.0,
                             message=error_message.format(name, qX_hat.int_repr(), qX_ref.int_repr()))
            self.assertEqual(scale, qX_hat.q_scale(),
                             message=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             message=error_message.format(name + '.zero_point', scale,
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
            "nn.quantized.functional": torch.nn.quantized.functional.avg_pool2d
        }
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            X_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                       count_include_pad=count_include_pad, divisor_override=divisor_override)
            self.assertTrue(X_hat.stride() != sorted(X_hat.stride()))
            qX_ref = torch.quantize_per_tensor(X_ref, scale=X_hat.q_scale(), zero_point=X_hat.q_zero_point(),
                                               dtype=torch_type)

            self.assertEqual(qX_ref.int_repr().to(torch.double), X_hat.int_repr().to(torch.double), atol=1.0,
                             message=error_message.format(name, X_hat.int_repr(), qX_ref.int_repr()))
            self.assertEqual(scale, X_hat.q_scale(),
                             message=error_message.format(name + '.scale', scale, X_hat.q_scale()))
            self.assertEqual(zero_point, X_hat.q_zero_point(),
                             message=error_message.format(name + '.zero_point', scale,
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
            "nn.quantized.functional": torch.nn.quantized.functional.avg_pool3d
        }
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            qX_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                        count_include_pad=count_include_pad, divisor_override=divisor_override)
            qX_ref = torch.quantize_per_tensor(X_ref, scale=qX_hat.q_scale(), zero_point=qX_hat.q_zero_point(),
                                               dtype=torch_type)
            self.assertEqual(qX_ref.int_repr().to(torch.double), qX_hat.int_repr().to(torch.double), atol=1.0,
                             message=error_message.format(name, qX_hat.int_repr(), qX_ref.int_repr()))
            self.assertEqual(scale, qX_hat.q_scale(),
                             message=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             message=error_message.format(name + '.zero_point', scale,
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
            "nn.quantized.functional": torch.nn.quantized.functional.avg_pool3d
        }
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            X_hat = op(qX, kernel_size=kernel, stride=stride, padding=padding, ceil_mode=ceil_mode,
                       count_include_pad=count_include_pad, divisor_override=divisor_override)
            self.assertTrue(X_hat.stride() != sorted(X_hat.stride()))
            qX_ref = torch.quantize_per_tensor(X_ref, scale=X_hat.q_scale(), zero_point=X_hat.q_zero_point(),
                                               dtype=torch_type)

            self.assertEqual(qX_ref.int_repr().to(torch.double), X_hat.int_repr().to(torch.double), atol=1.0,
                             message=error_message.format(name, X_hat.int_repr(), qX_ref.int_repr()))
            self.assertEqual(scale, X_hat.q_scale(),
                             message=error_message.format(name + '.scale', scale, X_hat.q_scale()))
            self.assertEqual(zero_point, X_hat.q_zero_point(),
                             message=error_message.format(name + '.zero_point', scale,
                             X_hat.q_zero_point()))

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams(dtypes=torch.quint8)),
           output_size_h=st.integers(1, 10),
           output_size_w=st.integers(1, 10))
    def test_adaptive_avg_pool2d(self, X, output_size_h, output_size_w):
        X, (scale, zero_point, torch_type) = X

        H, W = X.shape[-2:]
        assume(output_size_h <= H)
        assume(output_size_w <= W)
        if output_size_h == output_size_w:
            output_size = output_size_h
        else:
            output_size = (output_size_h, output_size_w)
        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale, zero_point=zero_point,
                                       dtype=torch_type)

        # Run reference on int_repr + round to avoid double rounding error.
        X_ref = torch.nn.functional.adaptive_avg_pool2d(
            qX.int_repr().to(torch.float), output_size).round()

        ops_under_test = {
            "nn.functional": torch.nn.functional.adaptive_avg_pool2d,
            "nn.quantized.functional":
                torch.nn.quantized.functional.adaptive_avg_pool2d
        }

        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"

        for name, op in ops_under_test.items():
            qX_hat = op(qX, output_size=output_size)
            self.assertEqual(X_ref, qX_hat.int_repr(), atol=1.0,
                             message=error_message.format(name, X_ref, qX_hat))
            self.assertEqual(scale, qX_hat.q_scale(),
                             message=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             message=error_message.format(name + '.zero_point', scale,
                                                          qX_hat.q_zero_point()))

    """Tests adaptive average pool operation on NHWC quantized tensors."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams(dtypes=torch.qint8)),
           output_size_h=st.integers(1, 10),
           output_size_w=st.integers(1, 10))
    def test_adaptive_avg_pool2d_nhwc(self, X, output_size_h, output_size_w):
        X, (scale, zero_point, torch_type) = X
        H, W = X.shape[-2:]
        assume(output_size_h <= H)
        assume(output_size_w <= W)
        if output_size_h == output_size_w:
            output_size = output_size_h
        else:
            output_size = (output_size_h, output_size_w)

        if X.shape[1] < 176:
            X = np.repeat(X, 176 / X.shape[1], 1)

        X_nchw = np.ascontiguousarray(X.transpose([0, 2, 3, 1]))
        X = torch.from_numpy(X_nchw).permute([0, 3, 1, 2])
        qX = torch.quantize_per_tensor(torch.from_numpy(X_nchw), scale=scale,
                                       zero_point=zero_point, dtype=torch_type).permute([0, 3, 1, 2])

        # Run reference on int_repr + round to avoid double rounding error.
        X_ref = torch.nn.functional.adaptive_avg_pool2d(qX.int_repr().to(torch.double), output_size).round()

        self.assertTrue(qX.stride() != sorted(qX.stride()))

        ops_under_test = {
            "nn.functional": torch.nn.functional.adaptive_avg_pool2d,
            "nn.quantized.functional":
                torch.nn.quantized.functional.adaptive_avg_pool2d
        }
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            X_hat = op(qX, output_size=output_size)
            self.assertTrue(X_hat.stride() != sorted(X_hat.stride()))
            self.assertEqual(X_ref, X_hat.int_repr(), atol=1.0,
                             message="{} results are off".format(name))
            self.assertEqual(scale, X_hat.q_scale(),
                             message=error_message.format(name + '.scale', scale, X_hat.q_scale()))
            self.assertEqual(zero_point, X_hat.q_zero_point(),
                             message=error_message.format(name + '.zero_point', scale,
                                                          X_hat.q_zero_point()))

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           k=st.integers(1, 10),
           dim=st.integers(1, 4),
           largest=st.booleans(),
           sorted=st.booleans())
    def test_qtopk(self, X, k, dim, largest, sorted):
        X, (scale, zero_point, torch_type) = X
        qX = torch.quantize_per_tensor(torch.from_numpy(X), scale, zero_point, torch_type)
        assume(dim < X.ndim)
        assume(k < X.shape[dim])

        unquantized_out = torch.topk(qX.dequantize(), k, dim=dim, largest=largest, sorted=sorted)

        values = torch.quantize_per_tensor(torch.from_numpy(X), scale, zero_point, torch_type)
        indices = torch.tensor(torch.from_numpy(X)).long()

        quantized_out = torch.topk(qX, k, dim=dim, largest=largest, sorted=sorted)

        assert(len(unquantized_out) == len(quantized_out))
        torch.testing.assert_allclose(quantized_out[0].dequantize(), unquantized_out[0])
        torch.testing.assert_allclose(quantized_out[1], unquantized_out[1])

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           k=st.integers(1, 10),
           dim=st.integers(1, 4),
           largest=st.booleans(),
           sorted=st.booleans())
    def test_qtopk_nhwc(self, X, k, dim, largest, sorted):
        # X is NHWC, we permute to view as NCHW but keep NHWC in memory
        X, (scale, zero_point, torch_type) = X
        qX = torch.quantize_per_tensor(torch.from_numpy(X), scale, zero_point, torch_type).permute([0, 3, 1, 2])
        X = np.transpose(X, [0, 3, 1, 2])
        assume(dim < X.ndim)
        assume(k < X.shape[dim])

        unquantized_out = torch.topk(qX.dequantize(), k, dim=dim, largest=largest, sorted=sorted)

        values = torch.quantize_per_tensor(torch.from_numpy(X), scale, zero_point, torch_type)
        indices = torch.tensor(torch.from_numpy(X)).long()

        quantized_out = torch.topk(qX, k, dim=dim, largest=largest, sorted=sorted)

        assert(len(unquantized_out) == len(quantized_out))
        torch.testing.assert_allclose(quantized_out[0].dequantize(), unquantized_out[0])
        torch.testing.assert_allclose(quantized_out[1], unquantized_out[1])


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
           mode=st.sampled_from(("bilinear", "nearest")),
           scale_factor=st.sampled_from((None, 1.5, 2.0)),
           align_corners=st.sampled_from((True, False)),
           nhwc_layout=st.sampled_from((True, False)))
    def test_interpolate(self, X, size, mode, scale_factor, align_corners, nhwc_layout):
        """
        This test cover upsample_nearest2d and upsample_bilinear2d
        """
        X, (scale, zero_point, torch_type) = X
        H, W = X.shape[-2:]

        if scale_factor is not None:
            size = None
        if mode == "nearest":
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
            "nn.quantized.functional": torch.nn.quantized.functional.interpolate
        }
        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            qX_hat = op(qX, size=size, scale_factor=scale_factor,
                        mode=mode, align_corners=align_corners)
            self.assertEqual(X_ref, qX_hat.int_repr(), atol=1.0,
                             message="{} results are off".format(name, qX_hat.int_repr(), X_ref))
            self.assertEqual(scale, qX_hat.q_scale(),
                             message=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             message=error_message.format(name + '.zero_point', scale,
                                                          qX_hat.q_zero_point()))

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=5, max_dims=5,
                                              min_side=5, max_side=10),
                       qparams=hu.qparams()),
           size=st.sampled_from((1, 3, 5, 5, 10)),
           scale_factor=st.sampled_from((None, 1.5, 2.0)),
           align_corners=st.sampled_from((True, False)),
           nhwc_layout=st.sampled_from((True, False)))
    def test_interpolate3d(self, X, size, scale_factor, align_corners, nhwc_layout):
        """
        This test cover upsample_nearest2d and upsample_bilinear2d
        """
        X, (scale, zero_point, torch_type) = X
        D, H, W = X.shape[-3:]
        mode = "nearest"
        if scale_factor is not None:
            size = None
        if mode == "nearest":
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
            "nn.quantized.functional": torch.nn.quantized.functional.interpolate
        }

        error_message = r"Results are off for {}:\n\tExpected:\n{}\n\tGot:\n{}"
        for name, op in ops_under_test.items():
            qX_hat = op(qX, size=size, scale_factor=scale_factor,
                        mode=mode, align_corners=align_corners)
            self.assertEqual(X_ref, qX_hat.int_repr(), atol=1.0,
                             message="{} results are off".format(name, qX_hat.int_repr(), X_ref))
            self.assertEqual(scale, qX_hat.q_scale(),
                             message=error_message.format(name + '.scale', scale, qX_hat.q_scale()))
            self.assertEqual(zero_point, qX_hat.q_zero_point(),
                             message=error_message.format(name + '.zero_point', scale,
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
        # Here, we quantize and get quantized tensors in NHWC for both dims and strides. The
        # permute switches it so that the tensor looks like NCHW but it laid out in memory as
        # NHWC.
        qX = torch.quantize_per_tensor(X, scale, zero_point, torch_type).permute([0, 3, 1, 2])
        qY = torch.quantize_per_tensor(Y, scale, zero_point, torch_type).permute([0, 3, 1, 2])

        ref = torch.cat([qX.dequantize(), qY.dequantize()], dim=1)
        if relu:
            ref[ref < 0] = 0.0
        ref = torch.quantize_per_tensor(ref, scale=scale, zero_point=zero_point, dtype=torch_type)

        if relu:
            out = torch.ops.quantized.cat_relu(
                [qX, qY], dim=1, scale=scale, zero_point=zero_point)
        else:
            out = torch.ops.quantized.cat([qX, qY], dim=1, scale=scale, zero_point=zero_point)

        torch.testing.assert_allclose(out.dequantize(), ref.dequantize())
        self.assertNotEqual(out.stride(), sorted(out.stride()))

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=3,
                                              min_side=1, max_side=2),
                       qparams=hu.qparams()),
           dim=st.integers(1, 2))
    def test_mean(self, X, dim):
        X, (scale, zero_point, torch_type) = X
        qX = torch.quantize_per_tensor(torch.tensor(X).float(), scale, zero_point, torch_type)

        Y = torch.mean(qX.dequantize(), dim)
        Y = torch.quantize_per_tensor(Y, scale, zero_point, torch_type).dequantize()
        qY = torch.mean(qX, dim)

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


    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=4,
                                              min_side=1, max_side=32),
                       qparams=hu.qparams()),
           Y_scale=st.floats(0.2, 2.6),
           Y_zero_point=st.integers(0, 5))
    def test_batch_norm2d(self, X, Y_scale, Y_zero_point):
        if "fbgemm" not in torch.backends.quantized.supported_engines:
            return

        with override_quantized_engine("fbgemm"):
            X, (scale_x, zero_point_x, dtype_x) = X

            X = torch.from_numpy(X)
            c = X.shape[1]

            mean = torch.rand(c).float()
            var = torch.rand(c).float()
            weight = torch.rand(c).float()
            bias = torch.rand(c).float()
            eps = 0.001
            qx = torch.quantize_per_tensor(X, scale_x, zero_point_x, dtype_x)
            qy = torch.ops.quantized.batch_norm2d(qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)

            float_ref = F.batch_norm(qx.dequantize(), weight=weight, bias=bias,
                                     running_mean=mean, running_var=var, training=False, momentum=0, eps=eps)
            quantize_ref = torch.quantize_per_tensor(float_ref, Y_scale, Y_zero_point, dtype_x)
            self.assertEqual(qy.int_repr().numpy(), quantize_ref.int_repr().numpy())

    @unittest.skip("Takes 20+ min to finish in many configurations")
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=4, max_dims=5,
                                              min_side=1, max_side=32),
                       qparams=hu.qparams()),
           Y_scale=st.floats(0.2, 2.6),
           Y_zero_point=st.integers(0, 5))
    def test_batch_norm2d_relu(self, X, Y_scale, Y_zero_point):
        if "fbgemm" not in torch.backends.quantized.supported_engines:
            return

        with override_quantized_engine("fbgemm"):
            X, (scale_x, zero_point_x, dtype_x) = X

            X = torch.from_numpy(X)
            c = X.shape[1]

            mean = torch.rand(c).float()
            var = torch.rand(c).float()
            weight = torch.rand(c).float()
            bias = torch.rand(c).float()
            eps = 0.001
            qx = torch.quantize_per_tensor(X, scale_x, zero_point_x, dtype_x)
            if len(X.shape) == 4:
                qy = torch.ops.quantized.batch_norm2d_relu(qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)
            else:
                qy = torch.ops.quantized.batch_norm3d_relu(qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)


            float_ref = F.batch_norm(qx.dequantize(), weight=weight, bias=bias,
                                     running_mean=mean, running_var=var, training=False, momentum=0, eps=eps).numpy()

            float_ref_relu = float_ref.copy()
            float_ref_relu[float_ref < 0] = 0
            quantize_ref = torch.quantize_per_tensor(torch.from_numpy(float_ref_relu), Y_scale, Y_zero_point, dtype_x)
            self.assertEqual(qy.int_repr().numpy(), quantize_ref.int_repr().numpy())

    @unittest.skip("Takes 20+ min to finish in many configurations")
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=5, max_dims=5,
                                              min_side=1, max_side=32),
                       qparams=hu.qparams()),
           Y_scale=st.floats(0.2, 2.6),
           Y_zero_point=st.integers(0, 5))
    def test_batch_norm3d(self, X, Y_scale, Y_zero_point):
        if "fbgemm" not in torch.backends.quantized.supported_engines:
            return

        with override_quantized_engine("fbgemm"):
            X, (scale_x, zero_point_x, dtype_x) = X

            X = torch.from_numpy(X)
            c = X.shape[1]

            mean = torch.rand(c).float()
            var = torch.rand(c).float()
            weight = torch.rand(c).float()
            bias = torch.rand(c).float()
            eps = 0.001
            qx = torch.quantize_per_tensor(X, scale_x, zero_point_x, dtype_x)
            qy = torch.ops.quantized.batch_norm3d(qx, weight, bias, mean, var, eps, Y_scale, Y_zero_point)

            float_ref = F.batch_norm(qx.dequantize(), weight=weight, bias=bias,
                                     running_mean=mean, running_var=var, training=False, momentum=0, eps=eps)
            quantize_ref = torch.quantize_per_tensor(float_ref, Y_scale, Y_zero_point, dtype_x)
            self.assertEqual(qy.int_repr().numpy(), quantize_ref.int_repr().numpy())

@unittest.skipUnless('fbgemm' in torch.backends.quantized.supported_engines,
                     " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
                     " with instruction set support avx2 or newer.")
class TestDynamicQuantizedLinear(TestCase):
    """Tests the correctness of the dynamic quantized linear and linear_relu op."""
    @given(
        batch_size=st.integers(1, 4),
        input_channels=st.integers(16, 32),
        output_channels=st.integers(4, 8),
        use_bias=st.booleans(),
        use_relu=st.booleans(),
        use_multi_dim_input=st.booleans(),
        use_channelwise=st.booleans(),
        qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_qlinear(self, batch_size, input_channels, output_channels,
                     use_bias, use_relu, use_multi_dim_input, use_channelwise, qengine):

        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN or IS_MACOS:
                return
            use_channelwise = False
            use_relu = False

        with override_quantized_engine(qengine):
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
            X_q0 = np.round(np.random.rand(batch_size, input_channels) *
                            (X_value_max - X_value_min)
                            + X_value_min
                            ).astype(np.uint8)
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

            if qengine == 'fbgemm':
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
            X_scale, X_zp = _calculate_dynamic_qparams(X_fp32, torch.quint8)
            X_q = torch.quantize_per_tensor(X_fp32, scale=X_scale, zero_point=X_zp, dtype=torch.quint8)

            # Weight prepacking operator for dynamic quantized Linear
            W_prepack = qlinear_prepack(W_q, b_fp32)
            # Dynamic quantized Linear operator with prepacked weight
            Y_fp32 = qlinear_dynamic(X_q.dequantize(), W_prepack)
            # Y_fp32 = qlinear_dynamic(X_fp32, W_prepack, b_fp32)

            Y_fp32_ref = F.linear(X_q.dequantize(), W_q.dequantize(), b_fp32)
            # Y_fp32_ref = F.linear(X_fp32, W_fp32, b_fp32)
            # if use_multi_dim_input:
            #     Y_fp32_ref = Y_fp32_ref.view(3, int(batch_size / 3), output_channels)

            if use_relu:
                Y_fp32_ref[Y_fp32_ref < 0.0] = 0.0

            self.assertEqual(Y_fp32, Y_fp32_ref,
                             message="torch.ops.quantized.linear_dynamic (fbgemm) results are off")

    """Tests the correctness of the legacy dynamic quantized linear op."""
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
                         message="torch.ops.quantized.fbgemm_linear_dynamic results are off")

class TestQuantizedLinear(unittest.TestCase):
    """Tests the correctness of the quantized linear and linear_relu op."""
    @given(batch_size=st.integers(1, 4),
           input_channels=st.integers(16, 32),
           output_channels=st.integers(4, 8),
           use_bias=st.booleans(),
           use_relu=st.booleans(),
           use_multi_dim_input=st.booleans(),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_qlinear(self, batch_size, input_channels, output_channels, use_bias,
                     use_relu, use_multi_dim_input, use_channelwise, qengine):
        if qengine not in torch.backends.quantized.supported_engines:
            return
        decimal_val = 4
        if qengine == 'qnnpack':
            # QNNPACK qlinear is flaky on MACOS. Issue #27326
            if IS_PPC or TEST_WITH_UBSAN or IS_MACOS:
                return
            use_channelwise = False
            use_multi_dim_input = False
            # QNNPACK supports uint8 in the kernels. In the op we shift the int8
            # weight values to uint8 to be on par with fbgemm. However, this causes
            # some rounding issues in rare cases. So, we relax the check to allow
            # off by one results.
            decimal_val = 0

        with override_quantized_engine(qengine):
            qlinear_prepack = torch.ops.quantized.linear_prepack
            if use_relu:
                qlinear = torch.ops.quantized.linear_relu
            else:
                qlinear = torch.ops.quantized.linear
            if use_multi_dim_input:
                batch_size *= 3  # Test the multi-dim input tensor
            X_scale = 1.5
            X_zp = 5
            X_value_min = 0
            X_value_max = 225
            X_q0 = np.round(
                np.random.rand(batch_size, input_channels) *
                (X_value_max - X_value_min)
                + X_value_min
            ).astype(np.uint8)
            W_scales = np.random.rand(output_channels)
            W_zps = np.round(np.random.rand(output_channels) * 100 - 50).astype(np.int)
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
                X, scale=X_scale, zero_point=X_zp, dtype=torch.quint8)
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
            Y_q = qlinear(X_q, W_prepack, Y_scale, Y_zp)
            if not use_channelwise:
                # Test the per-tensor quantization only
                # Reference quantized Linear operator
                Y_q_ref = qlinear_ref(X_q0, X_scale, X_zp, W_q0,
                                      W_scales[0], W_zps[0], b_q0, Y_scale, Y_zp)
                if use_relu:
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
            if use_relu:
                Y_fp32_ref[Y_fp32_ref < 0.0] = 0.0
            Y_q_ref2 = torch.quantize_per_tensor(
                Y_fp32_ref, Y_scale, Y_zp, torch.quint8)
            # Assert equal
            np.testing.assert_array_almost_equal(
                Y_q_ref2.int_repr().numpy(), Y_q.int_repr().numpy(), decimal=decimal_val)

    """Tests the correctness of the quantized::linear_unpack op."""
    @given(W=hu.tensor(shapes=hu.array_shapes(2, 2,),
                       qparams=hu.qparams(dtypes=torch.qint8)),
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_qlinear_unpack(self, W, use_channelwise, qengine):
        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return
            use_channelwise = False

        with override_quantized_engine(qengine):
            W, (W_scale, W_zp, torch_type) = W
            if use_channelwise:
                output_channels = W.shape[0]
                W_scales = torch.rand(output_channels).to(torch.double)
                W_zps = torch.round(torch.rand(output_channels)
                                    * 100 - 50).to(torch.int64)
            qlinear_prepack = torch.ops.quantized.linear_prepack
            qlinear_unpack = torch.ops.quantized.linear_unpack

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

class TestQuantizedConv(unittest.TestCase):
    def _test_qconv_unpack_impl(
        self, qconv_prepack_fn, qconv_unpack_fn, inputs, strides, pads,
        channelwise
    ):
        (X_data, W_data, bias_data, groups) = inputs
        (X, (X_scale, X_zero_point, X_qtype)) = X_data
        (W, (W_scale, W_zero_point, W_qtype)) = W_data
        (bias, (bias_scale, bias_zero_point, bias_qtype)) = bias_data
        if channelwise:
            output_channels = W.shape[0]
            W_scale = torch.tensor([W_scale] * output_channels)
            W_zero_point = torch.tensor([W_zero_point] * output_channels)

        W = torch.from_numpy(W).float()
        bias = torch.from_numpy(bias).float()
        if channelwise:
            W_q = torch.quantize_per_channel(
                W, scales=W_scale, zero_points=W_zero_point, axis=0,
                dtype=W_qtype)
        else:
            W_q = torch.quantize_per_tensor(
                W, scale=W_scale, zero_point=W_zero_point, dtype=W_qtype)

        dilations = (1,) * len(strides)
        W_packed = qconv_prepack_fn(W_q, bias, strides, pads, dilations, groups)
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
        self, batch_size,
        input_channels_per_group, input_feature_map_shape,
        output_channels_per_group, groups, kernels, strides, pads, dilations,
        X_scale, X_zero_point, W_scale, W_zero_point,
        use_bias, use_channelwise
    ):
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
        # (output_channels, input_channels/groups,
        #  kernel_d, kernel_h, kernel_w)
        W_init = torch.randint(
            W_value_min,
            W_value_max,
            (output_channels, input_channels_per_group,) + kernels,
        )
        b_init = torch.randint(0, 10, (output_channels,))

        (X_value_min, X_value_max) = (0, 4)
        X_init = torch.randint(
            X_value_min,
            X_value_max,
            (batch_size, input_channels,) + input_feature_map_shape,
        )
        X = X_scale * (X_init - X_zero_point).float()

        if use_channelwise:
            W_shape = (-1, 1) + (1,) * len(kernels)
            W_scales_tensor = torch.tensor(W_scale, dtype=torch.float)
            W_zero_points_tensor = torch.tensor(W_zero_point, dtype=torch.float)
            W = W_scales_tensor.reshape(*W_shape) * (
                W_init.float() - W_zero_points_tensor.reshape(*W_shape)).float()
            b = X_scale * W_scales_tensor * b_init.float()
        else:
            W = W_scale[0] * (W_init - W_zero_point[0]).float()
            b = X_scale * W_scale[0] * b_init.float()

        X_q = torch.quantize_per_tensor(
            X, scale=X_scale, zero_point=X_zero_point, dtype=torch.quint8)
        if use_channelwise:
            W_q = torch.quantize_per_channel(
                W, W_scales_tensor, W_zero_points_tensor.long(), 0,
                dtype=torch.qint8)
        else:
            W_q = torch.quantize_per_tensor(
                W, scale=W_scale[0], zero_point=W_zero_point[0],
                dtype=torch.qint8)

        bias_float = b if use_bias else None

        return (X, W), (X_q, W_q), bias_float

    def _test_qconv_impl(
        self, qconv_fn, qconv_prepack_fn, conv_op, batch_size,
        input_channels_per_group, input_feature_map_shape,
        output_channels_per_group, groups, kernels, strides, pads, dilations,
        X_scale, X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
        use_bias, use_relu, use_channelwise
    ):
        (X, W), (X_q, W_q), bias_float = self._make_qconv_tensors(
            batch_size, input_channels_per_group, input_feature_map_shape,
            output_channels_per_group, groups, kernels,
            strides, pads, dilations, X_scale, X_zero_point, W_scale,
            W_zero_point, use_bias, use_channelwise)
        # Assign weights
        conv_op.weight = torch.nn.Parameter(W, requires_grad=False)
        conv_op.bias = torch.nn.Parameter(
            bias_float, requires_grad=False) if use_bias else None
        result_ref = conv_op(X)
        if use_relu:
            relu = torch.nn.ReLU()
            result_ref = relu(result_ref)

        # Quantize reference results for comparison
        result_ref_q = torch.quantize_per_tensor(
            result_ref, scale=Y_scale, zero_point=Y_zero_point,
            dtype=torch.quint8)

        W_prepack = qconv_prepack_fn(
            W_q, bias_float, strides, pads, dilations, groups)
        Y_q = qconv_fn(
            X_q,
            W_prepack,
            strides,
            pads,
            dilations,
            groups,
            Y_scale,
            Y_zero_point,
        )

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
            result_ref_q.int_repr().numpy(), Y_q.int_repr().numpy(), decimal=0)

    """Tests the correctness of quantized convolution op."""
    @given(batch_size=st.integers(1, 3),
           input_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           height=st.integers(10, 16),
           width=st.integers(7, 14),
           output_channels_per_group=st.sampled_from([2, 4, 5, 8, 16, 32]),
           groups=st.integers(1, 3),
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
           use_channelwise=st.booleans(),
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_qconv(
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
            qengine
    ):
        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            # QNNPACK qconv is flaky on MACOS. Issue #27326
            if IS_PPC or TEST_WITH_UBSAN or IS_MACOS:
                return
            use_channelwise = False

        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups
        kernels = (kernel_h, kernel_w)
        strides = (stride_h, stride_w)
        pads = (pad_h, pad_w)
        dilations = (dilation, dilation)

        with override_quantized_engine(qengine):
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
            self._test_qconv_impl(
                qconv, qconv_prepack, conv_op, batch_size,
                input_channels_per_group, (height, width),
                output_channels_per_group, groups, kernels, strides, pads,
                dilations, X_scale, X_zero_point, W_scale, W_zero_point,
                Y_scale, Y_zero_point, use_bias, use_relu, use_channelwise)

    """Tests the correctness of the quantized::qconv_unpack op."""
    @given(
        inputs=hu.tensor_conv(
            spatial_dim=2, batch_size_range=(1, 3),
            input_channels_per_group_range=(1, 4),
            output_channels_per_group_range=(1, 4), feature_map_range=(4, 8),
            kernel_range=(1, 4), max_groups=4,
            qparams=[hu.qparams(dtypes=torch.quint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint8,
                                zero_point_min=0,
                                zero_point_max=0),
                     hu.qparams(dtypes=torch.qint32,
                                zero_point_min=0,
                                zero_point_max=0)]),
        stride_h=st.integers(1, 3), stride_w=st.integers(1, 3),
        pad_h=st.integers(1, 2), pad_w=st.integers(1, 2),
        channelwise=st.booleans(),
        qengine=st.sampled_from(("qnnpack", "fbgemm")))
    def test_qconv_unpack(
        self, inputs, stride_h, stride_w, pad_h, pad_w, channelwise, qengine
    ):
        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            if IS_PPC or TEST_WITH_UBSAN:
                return
            channelwise = False

        with override_quantized_engine(qengine):
            qconv_prepack = torch.ops.quantized.conv2d_prepack
            qconv_unpack = torch.ops.quantized.conv2d_unpack
            self._test_qconv_unpack_impl(
                qconv_prepack, qconv_unpack, inputs, (stride_h, stride_w),
                (pad_h, pad_w), channelwise)

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
           qengine=st.sampled_from(("qnnpack", "fbgemm")))
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
        qengine,
    ):
        if qengine not in torch.backends.quantized.supported_engines:
            return
        if qengine == 'qnnpack':
            # QNNPACK qconv is flaky on MACOS. Issue #27326
            if IS_PPC or TEST_WITH_UBSAN or IS_MACOS:
                return

        input_channels = input_channels_per_group * groups
        output_channels = output_channels_per_group * groups

        (X, W), (X_q, W_q), bias_float = self._make_qconv_tensors(
            batch_size, input_channels_per_group, (length,),
            output_channels_per_group, groups, kernel, stride, pad,
            dilation, X_scale, X_zero_point, W_scale, W_zero_point,
            use_bias, False)

        true_conv1d = torch.nn.Conv1d(
            input_channels,
            output_channels,
            kernel,
            stride,
            pad,
            dilation,
            groups,
        )
        true_conv1d.weight = torch.nn.Parameter(W)
        true_conv1d.bias = torch.nn.Parameter(bias_float) if use_bias else None
        true_outp = true_conv1d(X)
        q_result_ref = torch.quantize_per_tensor(
            true_outp, scale=Y_scale, zero_point=Y_zero_point,
            dtype=torch.quint8)

        with override_quantized_engine(qengine):
            conv_op = torch.nn.quantized.Conv1d(
                input_channels,
                output_channels,
                kernel,
                stride,
                pad,
                dilation,
                groups,
            )
            # Get the quantized weights and the output quantization params.
            conv_op.set_weight_bias(W_q, bias_float)
            conv_op.scale = float(Y_scale)
            conv_op.zero_point = int(Y_zero_point)

            q_outp = conv_op(X_q)

            np.testing.assert_array_almost_equal(
                q_result_ref.int_repr().numpy(),
                q_outp.int_repr().numpy(),
                decimal=0)

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
           qengine=st.sampled_from(("fbgemm",)))
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
        if qengine not in torch.backends.quantized.supported_engines:
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
                groups, kernels, strides, pads, dilations, X_scale,
                X_zero_point, W_scale, W_zero_point, Y_scale, Y_zero_point,
                use_bias, use_relu, use_channelwise)

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
        channelwise=st.booleans(),
        qengine=st.sampled_from(("fbgemm",)))
    def test_qconv3d_unpack(
        self, inputs, stride_d, stride_h, stride_w, pad_d, pad_h, pad_w,
        channelwise, qengine
    ):
        if qengine not in torch.backends.quantized.supported_engines:
            return

        with override_quantized_engine(qengine):
            qconv3d_prepack = torch.ops.quantized.conv3d_prepack
            qconv3d_unpack = torch.ops.quantized.conv3d_unpack
            self._test_qconv_unpack_impl(
                qconv3d_prepack, qconv3d_unpack, inputs,
                (stride_d, stride_h, stride_w), (pad_d, pad_h, pad_w),
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


@unittest.skipUnless('qnnpack' in torch.backends.quantized.supported_engines,
                     "This Pytorch Build has not been built with QNNPACK")
@unittest.skipIf(IS_PPC, "QNNPACK is not currently supported on ppc64le")
@unittest.skipIf(TEST_WITH_UBSAN,
                 "QNNPACK does not play well with UBSAN at the moment,"
                 " so we skip the test if we are in a UBSAN environment.")
@unittest.skipIf(IS_MACOS, "QNNPACK tests are flaky on MacOS currently - Issue #29326")
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
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_qnnpack_tanh(self, X):
        # Note: In QNNPACK the output scale and zero_point can only be
        #       2.0/256, 128 respectively, as it uses a LUT with 256 bins.
        X, (scale, zero_point, torch_type) = X
        X = torch.from_numpy(X)
        qX = torch.quantize_per_tensor(X, scale=scale,
                                       zero_point=zero_point,
                                       dtype=torch_type)

        # Floating point reference
        Y = torch.tanh(X)
        qY = torch.quantize_per_tensor(Y, scale=1.0 / 128, zero_point=128,
                                       dtype=torch.quint8)
        with override_quantized_engine('fbgemm'):
            qYserver = torch.tanh(qX)
        with override_quantized_engine('qnnpack'):
            qY_hat = torch.tanh(qX)
            self.assertEqual(qY, qY_hat,
                             message="QNNPACK TanH failed (FP ref)!")
            self.assertEqual(qYserver, qY_hat,
                             message="QNNPACK TanH failed (FBGEMM ref)!")

    """Tests the correctness of the quantized::qnnpack_sigmoid op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_qnnpack_sigmoid(self, X):
        # Note: In QNNPACK the output scale and zero_point can only be
        #       1.0/256, 0 respectively, as it uses a LUT with 256 bins.
        X, (scale, zero_point, torch_type) = X
        X = torch.from_numpy(X).to(torch.float32)
        qX = torch.quantize_per_tensor(X, scale=scale,
                                       zero_point=zero_point,
                                       dtype=torch_type)

        # Floating point reference
        Y = torch.sigmoid(X)
        qY = torch.quantize_per_tensor(Y, scale=1.0 / 256, zero_point=0,
                                       dtype=torch.quint8)
        with override_quantized_engine('fbgemm'):
            qYserver = torch.sigmoid(qX)
        with override_quantized_engine('qnnpack'):
            qY_hat = torch.sigmoid(qX)
            self.assertEqual(qY, qY_hat,
                             message="QNNPACK Sigmoid failed (FP ref)!")
            self.assertEqual(qYserver, qY_hat,
                             message="QNNPACK Sigmoid failed (FBGEMM ref)!")

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
                             message="QNNPACK Sigmoid failed (FP ref)!")
            self.assertEqual(qYserver, qY_hat,
                             message="QNNPACK Sigmoid failed (FBGEMM ref)!")

    """Tests the correctness of the quantized::add (qnnpack) op."""
    @settings(suppress_health_check=(HealthCheck.filter_too_much,))
    @given(A=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams(dtypes=torch.quint8)),
           zero_point=st.sampled_from([0, 2, 5, 15, 127]),
           scale_A=st.sampled_from([0.001, 0.057, 0.889, 12.3]),
           scale_B=st.sampled_from([0.008, 0.0821, 0.67, 7]),
           scale_C=st.sampled_from([0.003, 0.07821, 0.457, 7.34]),)
    def test_qnnpack_add(self, A, zero_point, scale_A, scale_B, scale_C):
        with override_quantized_engine('qnnpack'):
            A_temp = A
            A, (scale_a, zero_point_A, torch_type) = A_temp
            B, (scale_b, zero_point_B, torch_type) = A_temp
            A = torch.from_numpy(A)
            B = torch.from_numpy(B)

            assume(scale_A // scale_C >= 2**-14)
            assume(scale_A // scale_C < 2**8)
            assume(scale_B // scale_C >= 2**-14)
            assume(scale_B // scale_C < 2**8)

            zero_point_C = 127
            qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point,
                                           dtype=torch.quint8)
            qB = torch.quantize_per_tensor(B, scale=scale_B, zero_point=zero_point,
                                           dtype=torch.quint8)

            # Add ground truth
            C = (qA.dequantize() + qB.dequantize()).numpy()

            qC = _quantize(C, scale_C, zero_point_C)

            qC_qnnp = torch.ops.quantized.add(qA, qB, scale_C, zero_point_C)

            np.testing.assert_equal(qC, qC_qnnp.int_repr(),
                                    "Quantized addition failed.")

            Crelu = C.copy()
            Crelu[C < 0] = 0
            qCrelu = torch.quantize_per_tensor(torch.from_numpy(Crelu), scale_C,
                                               zero_point_C, dtype=torch.quint8)
            qCrelu_hat = torch.ops.quantized.add_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)
            np.testing.assert_equal(qCrelu.int_repr().numpy(), qCrelu_hat.int_repr(),
                                    "Quantized addition with ReLU failed.")

            A = torch.ones((0, 2), dtype=torch.float32)
            qA = torch.quantize_per_tensor(A, scale=scale_A, zero_point=zero_point_A,
                                           dtype=torch.quint8)
            qC = torch.ops.quantized.add(qA, qA, scale_C, zero_point_C)
            np.testing.assert_equal(qC.size(), qA.size(),
                                    "Quantized addition with batch size 0 failed.")

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

            A = torch.ones((0, 2, 4, 4), dtype=torch.float32)
            qa = torch.quantize_per_tensor(A, scale=scale, zero_point=zero_point,
                                           dtype=torch_type)
            qc = q_max_pool(qa, k, s, p, d, ceil_mode=False)
            oH = pool_output_shape(4, kernel, padding, stride, dilation)
            oW = pool_output_shape(4, kernel, padding, stride, dilation)
            np.testing.assert_equal(qc.size(), (0, 2, oH, oW),
                                    "Quantized maxpool2d with batch size 0 failed.")

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

            q_avg_pool = torch.nn.quantized.functional.avg_pool2d

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

            q_avg_pool = torch.nn.quantized.functional.adaptive_avg_pool2d

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

    """Tests the correctness of the quantized::hardswish op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 8, 1, 8),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=(torch.quint8))),
           Y_scale=st.floats(1e-6, 1e6),
           Y_zero_point=st.integers(0, 10))
    def test_hardswish(self, X, Y_scale, Y_zero_point):
        _test_hardswish(self, X, Y_scale, Y_zero_point, 'qnnpack')

    """Tests the correctness of the quantized::hardsigmoid op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 8, 1, 8),
                       elements=hu.floats(-1e6, 1e6, allow_nan=False, allow_infinity=False),
                       qparams=hu.qparams(dtypes=(torch.quint8))))
    def test_qhardsigmoid(self, X):
        _test_hardsigmoid(self, X, 'qnnpack')

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
                             "'tensor.{}(tensor)'' failed".format(op))
            # Reversed broadcasting.
            result_ref = getattr(dqB, op)(dqA)
            result = getattr(qB, op)(qA)
            self.assertEqual(result_ref, result,
                             "'tensor.{}(tensor)'' failed".format(op))

    @unittest.skip("FIXME: Failing due to overflow error without width option")
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
            self.assertEqual(result_ref, result,
                             "'tensor.{}(scalar)'' failed".format(op))
            # Reversed broadcasting.
            result_ref = getattr(b, op)(dqA)
            result = getattr(b, op)(qA)
            self.assertEqual(result_ref, result,
                             "'scalar.{}(tensor)'' failed".format(op))

        for op in ops_under_test_nonreversible:
            result_ref = getattr(dqA, op)(b)
            result = getattr(qA, op)(b)
            self.assertEqual(result_ref, result,
                             "'tensor.{}(scalar)'' failed".format(op))
