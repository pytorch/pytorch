from __future__ import absolute_import, division, print_function, unicode_literals

import numpy as np
import unittest

import torch
import torch.jit
import torch.nn.functional as F
from torch.nn.modules.utils import _pair

from hypothesis import assume, given
from hypothesis import strategies as st
import hypothesis_utils as hu

from common_utils import TEST_WITH_UBSAN, TestCase, run_tests, IS_WINDOWS, IS_PPC
from common_quantized import _quantize, _dequantize, _calculate_dynamic_qparams


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

class TestQuantizedOps(TestCase):
    """Computes the output shape given pooling parameters."""
    def _pool_output_shape(self, input_size, kernel_size, padding, stride,
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

    """Tests the correctness of the quantized::relu op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams()))
    def test_qrelu(self, X):
        X, (scale, zero_point, torch_type) = X

        Y = X.copy()
        Y[Y < 0] = 0
        qY = torch.quantize_linear(torch.from_numpy(Y), scale=scale,
                                   zero_point=zero_point, dtype=torch_type)
        X = torch.from_numpy(X)
        qX = torch.quantize_linear(X, scale=scale, zero_point=zero_point,
                                   dtype=torch_type)

        ops_under_test = {
            'ops.quantized': torch.ops.quantized.relu,
            'native': torch.relu,
            'nn.functional': torch.nn.functional.relu
        }

        for name, op in ops_under_test.items():
            qY_hat = op(qX)
            self.assertEqual(qY, qY_hat, message="{} relu failed".format(name))

    """Tests the correctness of the add and add_relu op."""
    def test_qadd_relu_same_qparams(self):
        add_relu = torch.ops.quantized.add_relu
        add = torch.ops.quantized.add

        A = torch.arange(-25, 25, dtype=torch.float)
        B = torch.arange(-25, 25, dtype=torch.float)
        scale = 2.0
        zero_point = 127
        qA = torch.quantize_linear(A, scale=scale, zero_point=zero_point,
                                   dtype=torch.quint8)
        qB = torch.quantize_linear(B, scale=scale, zero_point=zero_point,
                                   dtype=torch.quint8)

        # Add ReLU ground truth
        C = (qA.dequantize() + qB.dequantize()).numpy()
        qC = _quantize(C, scale, zero_point)
        qC_hat = add(qA, qB, scale=scale, zero_point=zero_point)
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized addition failed.")

        # Add + ReLU ground truth
        Crelu = C.copy()
        Crelu[C < 0] = 0
        qCrelu = _quantize(Crelu, scale, zero_point)
        qCrelu_hat = add_relu(qA, qB, scale=scale, zero_point=zero_point)
        np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                "Quantized addition with ReLU failed.")

    """Tests the correctness of the add and add_relu op."""
    def test_qadd_relu_different_qparams(self):
        add_relu = torch.ops.quantized.add_relu
        add = torch.ops.quantized.add

        A = torch.arange(-25, 25, dtype=torch.float)
        B = torch.arange(-25, 25, dtype=torch.float)
        scale_A = 3.0
        zero_point_A = 7
        scale_B = 5.0
        zero_point_B = 127

        scale_C = 0.5
        zero_point_C = 5

        qA = torch.quantize_linear(A, scale=scale_A, zero_point=zero_point_A,
                                   dtype=torch.quint8)
        qB = torch.quantize_linear(B, scale=scale_B, zero_point=zero_point_B,
                                   dtype=torch.quint8)

        # Add ground truth
        C = (qA.dequantize() + qB.dequantize()).numpy()
        qC = _quantize(C, scale_C, zero_point_C)
        qC_hat = add(qA, qB, scale=scale_C, zero_point=zero_point_C)
        np.testing.assert_equal(qC, qC_hat.int_repr(),
                                "Quantized addition failed.")

        # Add + ReLU ground truth
        Crelu = C.copy()
        Crelu[C < 0] = 0
        qCrelu = _quantize(Crelu, scale_C, zero_point_C)
        qCrelu_hat = add_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)
        np.testing.assert_equal(qCrelu, qCrelu_hat.int_repr(),
                                "Quantized addition with ReLU failed.")

    """Tests max pool operation on quantized tensors."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           kernel=st.sampled_from((3, 5, 7)),
           stride=st.sampled_from((None, 1, 2)),
           dilation=st.integers(1, 2),
           padding=st.integers(0, 2))
    def test_max_pool2d(self, X, kernel, stride, dilation, padding):
        X, (scale, zero_point, torch_type) = X
        # Check constraints
        assume(kernel // 2 >= padding)  # Kernel cannot be overhanging!
        iH, iW = X.shape[-2:]
        oH = self._pool_output_shape(iH, kernel, padding, stride, dilation)
        assume(oH > 0)
        oW = self._pool_output_shape(iW, kernel, padding, stride, dilation)
        assume(oW > 0)

        a = torch.from_numpy(X)
        a_pool = torch.nn.functional.max_pool2d(a, kernel_size=kernel,
                                                stride=stride,
                                                padding=padding, dilation=dilation)
        a_ref = torch.quantize_linear(a_pool, scale=scale,
                                      zero_point=zero_point, dtype=torch_type)
        a_ref = a_ref.dequantize()
        qa = torch.quantize_linear(a, scale=scale, zero_point=zero_point,
                                   dtype=torch_type)

        ops_under_test = {
            "torch": torch.max_pool2d,
            "nn.functional": torch.nn.functional.max_pool2d,
            "nn.quantized.functional": torch.nn.quantized.functional.max_pool2d
        }

        for name, op in ops_under_test.items():
            a_hat = op(qa, kernel_size=kernel, stride=stride, padding=padding,
                       dilation=dilation)
            self.assertEqual(a_ref, a_hat.dequantize(),
                             message="{} results are off".format(name))
        # Test the ops.quantized separately, because None is not treated.
        a_hat = torch.ops.quantized.max_pool2d(
            qa, kernel_size=_pair(kernel),
            stride=_pair(kernel if stride is None else stride),
            padding=_pair(padding), dilation=_pair(dilation))
        self.assertEqual(a_ref, a_hat.dequantize(),
                         message="ops.quantized.max_pool2d results are off")

    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
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
        qX = torch.quantize_linear(X, scale=scale, zero_point=zero_point,
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
            qX_repr = qX_hat.int_repr()
            self.assertEqual(X_ref, qX_repr,
                             message=error_message.format(name, X_ref, qX_repr))


    """Tests quantize concatenation (both fused and not)."""
    @given(X=hu.tensor(shapes=hu.array_shapes(min_dims=3, max_dims=4,
                                              min_side=1, max_side=10),
                       qparams=hu.qparams()),
           num=st.integers(1, 4),
           axis=st.integers(1, 4),
           relu=st.booleans())
    def test_cat(self, X, num, axis, relu):
        tensors_q = []
        tensors_ref = []
        X, (scale, zero_point, torch_type) = X
        assume(axis < X.ndim)
        X = torch.from_numpy(X)
        new_shape = np.array(X.shape)
        new_shape[axis] = 0
        for idx in range(num):
            tensors_q.append(torch.quantize_linear(X, scale, zero_point,
                                                   torch_type))
            tensors_ref.append(X)
            new_shape[axis] += tensors_ref[-1].shape[axis]

        cat_ref = torch.cat(tensors_ref, axis=axis)
        cat_ref = torch.quantize_linear(cat_ref, scale, zero_point, torch_type)
        cat_ref = cat_ref.dequantize()

        if relu:
            cat_ref = F.relu(cat_ref)
            q_cat_op = torch.ops.quantized.cat_relu
            q_cat_out_op = torch.ops.quantized.cat_relu_out
        else:
            q_cat_op = torch.ops.quantized.cat
            q_cat_out_op = torch.ops.quantized.cat_out

        cat_q = q_cat_op(tensors_q, axis=axis, scale=scale,
                         zero_point=zero_point)
        cat_q = cat_q.dequantize()
        np.testing.assert_equal(cat_ref.numpy(), cat_q.numpy())

        cat_q_out = torch._empty_affine_quantized(
            list(new_shape), scale=scale,
            zero_point=zero_point, dtype=torch_type)
        q_cat_out_op(tensors_q, axis=axis, out=cat_q_out)
        cat_q_out = cat_q_out.dequantize()
        np.testing.assert_equal(cat_ref.numpy(), cat_q_out.numpy())

        # Test the cat on per-channel quantized tensor.
        ch_axis = 1
        scales = torch.from_numpy(np.array([1.0] * X.shape[ch_axis]))
        scales = scales.to(torch.float64)
        zero_points = torch.from_numpy(np.array([0] * X.shape[ch_axis]))
        zero_points = zero_points.to(torch.long)
        tensors_q[0] = torch.quantize_linear_per_channel(
            X, scales, zero_points, axis=[ch_axis], dtype=torch_type)
        with self.assertRaisesRegex(RuntimeError, "supported.*cat"):
            cat_q = q_cat_op(tensors_q, axis=axis, scale=scale,
                             zero_point=zero_point)


@unittest.skipIf(
    TEST_WITH_UBSAN or not torch.fbgemm_is_cpu_supported(),
    " Quantized Linear requires FBGEMM. FBGEMM does not play"
    " well with UBSAN at the moment, so we skip the test if"
    " we are in a UBSAN environment.",
)
class TestDynamicQuantizedLinear(TestCase):
    """Tests the correctness of the dynamic quantized linear and linear_relu op."""
    @given(
        batch_size=st.integers(1, 4),
        input_channels=st.integers(16, 32),
        output_channels=st.integers(4, 8),
        use_bias=st.booleans(),
        use_relu=st.booleans(),
    )
    def test_qlinear(self, batch_size, input_channels, output_channels, use_bias, use_relu):
        qlinear_prepack = torch.ops.quantized.fbgemm_linear_prepack
        if use_relu:
            qlinear_dynamic = torch.ops.quantized.fbgemm_linear_relu_dynamic
        else:
            qlinear_dynamic = torch.ops.quantized.fbgemm_linear_dynamic

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

        X_fp32 = torch.from_numpy(_dequantize(X_q0, X_scale, X_zp)).to(dtype=torch.float)
        W_fp32 = torch.from_numpy(_dequantize(W_q0, W_scale, W_zp)).to(dtype=torch.float)
        b_fp32 = torch.from_numpy(
            _dequantize(b_q0, X_scale * W_scale, 0)
        ).to(dtype=torch.float) if use_bias else None

        W_scale, W_zp = _calculate_dynamic_qparams(W_fp32, torch.qint8)
        W_q = torch.quantize_linear(W_fp32, scale=W_scale, zero_point=W_zp, dtype=torch.qint8)

        # Observe X_fp32 and determine X_scale and X_zero_point, this should match
        # internals of dynamic linear.
        X_scale, X_zp = _calculate_dynamic_qparams(X_fp32, torch.quint8)
        X_q = torch.quantize_linear(X_fp32, scale=X_scale, zero_point=X_zp, dtype=torch.quint8)

        # Weight prepacking operator for dynamic quantized Linear
        W_prepack = qlinear_prepack(W_q)
        # Dynamic quantized Linear operator with prepacked weight
        Y_fp32 = qlinear_dynamic(X_q.dequantize(), W_prepack, b_fp32)
        # Y_fp32 = qlinear_dynamic(X_fp32, W_prepack, b_fp32)

        Y_fp32_ref = F.linear(X_q.dequantize(), W_q.dequantize(), b_fp32)
        # Y_fp32_ref = F.linear(X_fp32, W_fp32, b_fp32)

        if use_relu:
            Y_fp32_ref[Y_fp32_ref < 0.0] = 0.0

        self.assertEqual(Y_fp32, Y_fp32_ref,
                         message="torch.ops.quantized.fbgemm_linear_dynamic results are off")


@unittest.skipIf(
    not torch.fbgemm_is_cpu_supported(),
    " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
    " with instruction set support avx2 or newer.",
)
class TestQuantizedLinear(unittest.TestCase):
    """Tests the correctness of the quantized linear and linear_relu op."""
    @given(batch_size=st.integers(1, 4),
           input_channels=st.integers(16, 32),
           output_channels=st.integers(4, 8),
           use_bias=st.booleans(),
           use_relu=st.booleans())
    def test_qlinear(self, batch_size, input_channels, output_channels, use_bias, use_relu):
        qlinear_prepack = torch.ops.quantized.fbgemm_linear_prepack
        if use_relu:
            qlinear = torch.ops.quantized.fbgemm_linear_relu
        else:
            qlinear = torch.ops.quantized.fbgemm_linear

        X_scale = 1.5
        X_zp = 5
        X_value_min = 0
        X_value_max = 225
        X_q0 = np.round(
            np.random.rand(batch_size, input_channels) * (X_value_max - X_value_min)
            + X_value_min
        ).astype(np.uint8)

        W_scale = 0.4
        W_zp = 2
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
            np.random.rand(output_channels) * (b_value_max - b_value_min) + b_value_min
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

        X = torch.from_numpy(_dequantize(X_q0, X_scale, X_zp)).to(dtype=torch.float)
        W = torch.from_numpy(_dequantize(W_q0, W_scale, W_zp)).to(dtype=torch.float)
        b = torch.from_numpy(_dequantize(b_q0, X_scale * W_scale, 0)).to(dtype=torch.float) if use_bias else None

        X_q = torch.quantize_linear(X, scale=X_scale, zero_point=X_zp, dtype=torch.quint8)
        W_q = torch.quantize_linear(W, scale=W_scale, zero_point=W_zp, dtype=torch.qint8)
        b_q = torch.quantize_linear(b, scale=X_scale * W_scale, zero_point=0, dtype=torch.qint32) if use_bias else None

        # Compare X_scale * W_scale * input_channels * X_value_max * W_value_max with
        # Y_scale * 255 (max for uint8).
        Y_scale = 125.1234
        Y_zp = 5

        # Reference quantized Linear operator
        Y_q_ref = qlinear_ref(X_q0, X_scale, X_zp, W_q0, W_scale, W_zp, b_q0, Y_scale, Y_zp)
        if use_relu:
            Y_q_ref[Y_q_ref < Y_zp] = Y_zp

        # Weight prepacking operator for quantized Linear
        W_prepack = qlinear_prepack(W_q)
        # Quantized Linear operator with prepacked weight
        Y_q = qlinear(X_q, W_prepack, b_q, Y_scale, Y_zp)

        # Y_q_ref_real = _dequantize(Y_q_ref, Y_scale, Y_zp)
        # Y_q_real = Y_q.dequantize()

        # Assert equal
        np.testing.assert_equal(Y_q_ref, Y_q.int_repr().numpy())

        # Reference quantized result from PyTorch Linear operator
        W_fp32 = W_q.dequantize().to(dtype=torch.float)
        X_fp32 = X_q.dequantize().to(dtype=torch.float)
        b_fp32 = b_q.dequantize().to(dtype=torch.float) if use_bias else None
        Y_fp32_ref = F.linear(X_fp32, W_fp32, b_fp32)
        if use_relu:
            Y_fp32_ref[Y_fp32_ref < 0.0] = 0.0
        Y_q_ref2 = torch.quantize_linear(Y_fp32_ref, Y_scale, Y_zp, torch.quint8)

        # Assert equal
        np.testing.assert_equal(Y_q_ref2.int_repr().numpy(), Y_q.int_repr().numpy())

    """Tests the correctness of the quantized::fbgemm_linear_unpack op."""
    @given(W=hu.tensor(shapes=hu.array_shapes(2, 2,),
                       qparams=hu.qparams(dtypes=torch.qint8)))
    def test_qlinear_unpack(self, W):
        W, (W_scale, W_zp, torch_type) = W
        qlinear_prepack = torch.ops.quantized.fbgemm_linear_prepack
        qlinear_unpack = torch.ops.quantized.fbgemm_linear_unpack

        W = torch.from_numpy(W)
        W_q = torch.quantize_linear(W, scale=W_scale, zero_point=W_zp,
                                    dtype=torch_type)

        # Weight prepacking operator for quantized Linear
        W_prepack = qlinear_prepack(W_q)
        # Weight unpack operator for quantized Linear (Used for serialization)
        W_q_origin = qlinear_unpack(W_prepack)

        # Assert equal
        np.testing.assert_equal(W_q.int_repr(), W_q_origin.int_repr().numpy())
        np.testing.assert_equal(W_q.q_scale(), W_q_origin.q_scale())
        np.testing.assert_equal(W_q.q_zero_point(), W_q_origin.q_zero_point())

@unittest.skipIf(
    not torch.fbgemm_is_cpu_supported(),
    " Quantized operations require FBGEMM. FBGEMM is only optimized for CPUs"
    " with instruction set support avx2 or newer.",
)
class TestQuantizedConv(unittest.TestCase):
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
           dilation=st.integers(1, 1),
           X_scale=st.floats(0.2, 1.6),
           X_zero_point=st.integers(0, 4),
           W_scale=st.floats(0.2, 1.6),
           W_zero_point=st.integers(-5, 5),
           Y_scale=st.floats(0.2, 1.6),
           Y_zero_point=st.integers(0, 4),
           use_bias=st.booleans(),
           use_relu=st.booleans())
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
            use_relu
    ):

        qconv = torch.ops.quantized.fbgemm_conv2d
        if use_relu:
            qconv = torch.ops.quantized.fbgemm_conv2d_relu
        qconv_prepack = torch.ops.quantized.fbgemm_conv_prepack

        # C
        input_channels = input_channels_per_group * groups
        # K
        output_channels = output_channels_per_group * groups

        dilation_h = dilation_w = dilation

        # For testing, we use small values for weights and for activations so that no overflow occurs
        # in vpmaddubsw instruction. If the overflow occurs in qconv implementation and if there is no overflow
        # in reference we can't exactly match the results with reference.
        # Please see the comment in qconv implementation file (aten/src/ATen/native/quantized/cpu/qconv.cpp)
        # for more details.
        W_value_min = -5
        W_value_max = 5

        # the operator expects them in the format (output_channels, input_channels/groups, kernel_h, kernel_w)
        W_init = torch.from_numpy(
            np.random.randint(
                W_value_min,
                W_value_max,
                (output_channels, int(input_channels / groups), kernel_h, kernel_w)),
        )


        b_init = torch.from_numpy(np.random.randint(0, 10, (output_channels,)))

        stride = [stride_h, stride_w]
        pad = [pad_h, pad_w]
        dilation = [dilation_h, dilation_w]

        X_value_min = 0
        X_value_max = 4
        X_init = torch.from_numpy(np.random.randint(
            X_value_min, X_value_max, (batch_size, input_channels, height, width)))

        X = X_scale * (X_init - X_zero_point).to(dtype=torch.float)

        W = W_scale * (W_init - W_zero_point).to(dtype=torch.float)

        b = X_scale * W_scale * (b_init - 0).to(dtype=torch.float)

        # Existing floating point conv operator
        conv_op = torch.nn.Conv2d(input_channels,
                                  output_channels,
                                  (kernel_h, kernel_w),
                                  (stride_h, stride_w),
                                  (pad_h, pad_w),
                                  (dilation_h, dilation_w),
                                  groups)

        # assign weights
        conv_op.weight = torch.nn.Parameter(W, requires_grad=False)

        conv_op.bias = torch.nn.Parameter(b, requires_grad=False) if use_bias else None

        result_ref = conv_op(X)
        if use_relu:
            relu = torch.nn.ReLU()
            result_ref = relu(result_ref)
        # quantize reference results for comparision
        result_ref_q = torch.quantize_linear(result_ref, scale=Y_scale, zero_point=Y_zero_point, dtype=torch.quint8)

        # reformat X_init and W_init in the required format by qconv operator
        # NCHW -> NHWC
        X_NHWC = X.permute([0, 2, 3, 1]).contiguous()
        # K(C/G)RS -> KRS(C/G)
        W_KRSC = W.permute([0, 2, 3, 1]).contiguous()

        X_q = torch.quantize_linear(X_NHWC, scale=X_scale, zero_point=X_zero_point, dtype=torch.quint8)
        W_q = torch.quantize_linear(W_KRSC, scale=W_scale, zero_point=W_zero_point, dtype=torch.qint8)
        b_q = torch.quantize_linear(b, scale=X_scale * W_scale, zero_point=0, dtype=torch.qint32) if use_bias else None

        W_prepack = qconv_prepack(W_q, stride, pad, dilation, groups)

        Y_q = qconv(
            X_q,
            W_prepack,
            b_q,
            stride,
            pad,
            dilation,
            groups,
            Y_scale,
            Y_zero_point,
        )

        # Back to NCHW format
        Y_q = Y_q.permute([0, 3, 1, 2]).contiguous()


        # Make sure the results match
        # assert_array_almost_equal compares using the following formula:
        #     abs(desired-actual) < 1.5 * 10**(-decimal)
        # (https://docs.scipy.org/doc/numpy/reference/generated/numpy.testing.assert_almost_equal.html)

        # We use decimal = 0 to ignore off-by-1 differences between reference and
        # test. Off-by-1 differences arise due to the order of round and
        # zero_point addition operation, i.e., if addition followed by round is
        # used by reference and round followed by addition is used by test, the
        # results may differ by 1.

        # For example, the result of round(2.5) + 1 is 3 while round(2.5 + 1) is 4
        # assuming the rounding mode is round-to-nearest, ties-to-even.
        np.testing.assert_array_almost_equal(result_ref_q.int_repr().numpy(), Y_q.int_repr().numpy(), decimal=0)

    """Tests the correctness of the quantized::fbgemm_qconv_unpack op."""
    @given(X=hu.tensor_conv2d(min_batch=1, max_batch=3,
                              min_in_channels=1, max_in_channels=7,
                              min_out_channels=1, max_out_channels=7,
                              H_range=(6, 12), W_range=(6, 12),
                              kH_range=(3, 5), kW_range=(3, 5),
                              max_groups=4,
                              qparams=[hu.qparams(dtypes=torch.quint8,
                                                  zero_point_min=0,
                                                  zero_point_max=0),
                                       hu.qparams(dtypes=torch.qint8,
                                                  zero_point_min=0,
                                                  zero_point_max=0),
                                       hu.qparams(dtypes=torch.qint32,
                                                  zero_point_min=0,
                                                  zero_point_max=0)]),
           strideH=st.integers(1, 3), strideW=st.integers(1, 3),
           padH=st.integers(1, 2), padW=st.integers(1, 2))
    def test_qconv_unpack(self, X, strideH, strideW, padH, padW):
        (inputs, filters, bias, groups) = X
        inputs, (inputs_scale, inputs_zero_point, inputs_qtype) = inputs
        filters, (filters_scale, filters_zero_point, filters_qtype) = filters
        bias, (bias_scale, bias_zero_point, bias_qtype) = bias

        qconv_prepack = torch.ops.quantized.fbgemm_conv_prepack
        qconv_unpack = torch.ops.quantized.fbgemm_conv_unpack

        # Orig tensor is assumed to be in K(C/G)RS format
        W = torch.from_numpy(filters).to(torch.float)
        # K(C/G)RS -> KRS(C/G)
        W_KRSC = W.permute([0, 2, 3, 1]).contiguous()
        W_q = torch.quantize_linear(W_KRSC, scale=filters_scale, zero_point=filters_zero_point, dtype=filters_qtype)

        # Pack weights using weight packing operator
        strides = [strideH, strideW]
        paddings = [padH, padW]
        dilations = [1, 1]
        W_packed = qconv_prepack(W_q, strides, paddings, dilations, groups)
        # Unpack weights weight unpacking operator (Used for serialization)
        W_unpacked = qconv_unpack(W_packed)

        # Assert equal
        np.testing.assert_equal(W_q.int_repr().numpy(), W_unpacked.int_repr().numpy())
        np.testing.assert_equal(W_q.q_scale(), W_unpacked.q_scale())
        np.testing.assert_equal(W_q.q_zero_point(), W_unpacked.q_zero_point())


@unittest.skipIf(IS_WINDOWS, "QNNPACK has not been built for Windows")
@unittest.skipIf(IS_PPC, "QNNPACK is not currently supported on ppc64le")
@unittest.skipIf(TEST_WITH_UBSAN,
                 "QNNPACK does not play well with UBSAN at the moment,"
                 " so we skip the test if we are in a UBSAN environment.")
class TestQNNPackOps(TestCase):
    """Tests the correctness of the quantized::qnnpack_relu op."""
    @given(X=hu.tensor(shapes=hu.array_shapes(1, 5, 1, 5),
                       qparams=hu.qparams(dtypes=torch.quint8,
                                          zero_point_min=0,
                                          zero_point_max=0)))
    def test_qnnpack_relu(self, X):
        X, (scale, zero_point, torch_type) = X
        relu = torch.ops.quantized.qnnpack_relu

        X = torch.from_numpy(X)
        Y = X.clone()

        qX = torch.quantize_linear(X, scale=scale, zero_point=zero_point, dtype=torch_type)
        qY_hat = relu(qX)

        Y[Y < 0] = 0
        qY = torch.quantize_linear(Y, scale=scale, zero_point=zero_point, dtype=torch_type)
        self.assertEqual(qY, qY_hat)

    """Tests the correctness of the quantized::qnnpack_linear op."""
    @given(output_channels=st.sampled_from([2, 4, 5, 8, 16, 32]),
           X=hu.tensor(shapes=hu.array_shapes(2, 3, 8, 15),
                       qparams=hu.qparams(dtypes=torch.quint8)))
    def test_qnnpack_linear(self, output_channels, X):
        X, (X_scale, X_zp, torch_type) = X
        qmin = torch.iinfo(torch_type).min
        qmax = torch.iinfo(torch_type).max

        input_channels = X.shape[X.ndim - 1]

        input_rows = 1

        for x in range(X.ndim - 1):
            input_rows *= X.shape[x]

        qnnpack_linear = torch.ops.quantized.qnnpack_linear

        X_q0 = np.round(X * (qmin - qmax) + qmin).astype(np.uint8)

        W_scale = 0.4
        W_zp = 0
        W_value_min = 0
        W_value_max = 255
        W_q0 = np.round(
            np.random.rand(output_channels, input_channels)
            * (W_value_max - W_value_min)
            + W_value_min
        ).astype(np.uint8)

        b_value_min = -10
        b_value_max = 10
        b_q0 = np.round(
            np.random.rand(output_channels) * (b_value_max - b_value_min) + b_value_min
        ).astype(np.int32)

        X_scale = 10
        X_zp = 0
        X = torch.from_numpy(_dequantize(X_q0, X_scale, X_zp)).to(dtype=torch.float)
        W = torch.from_numpy(_dequantize(W_q0, W_scale, W_zp)).to(dtype=torch.float)
        b = torch.from_numpy(_dequantize(b_q0, X_scale * W_scale, 0)).to(dtype=torch.float)

        X_q = torch.quantize_linear(X, scale=X_scale, zero_point=X_zp, dtype=torch.quint8)
        W_q = torch.quantize_linear(W, scale=W_scale, zero_point=W_zp, dtype=torch.quint8)
        b_q = torch.quantize_linear(b, scale=X_scale * W_scale, zero_point=0, dtype=torch.qint32)

        Y_scale = 5.4  # This makes sure that the max output value does not exceed 255.
        Y_zp = 0

        # Reference quantized Linear operator
        Y_q_ref = qlinear_ref(X_q0, X_scale, X_zp, W_q0, W_scale, W_zp, b_q0, Y_scale, Y_zp)
        Y_q_ref_float = _dequantize(Y_q_ref, Y_scale, Y_zp)

        # Quantized linear operator
        Y_q = qnnpack_linear(X_q, W_q, b_q, Y_scale, Y_zp)

        # Assert equal
        np.testing.assert_array_almost_equal(Y_q_ref_float, Y_q.dequantize().numpy(), decimal=4)

        # Reference quantized result from PyTorch Linear operator

        W_fp32 = W_q.dequantize().to(dtype=torch.float)
        X_fp32 = X_q.dequantize().to(dtype=torch.float)
        b_fp32 = b_q.dequantize().to(dtype=torch.float)
        Y_fp32_ref = F.linear(X_fp32, W_fp32, b_fp32)
        Y_fp32_ref = Y_fp32_ref.view(-1, output_channels)
        Y_q_ref2 = torch.quantize_linear(Y_fp32_ref, Y_scale, Y_zp, torch.quint8)

        # Assert equal
        np.testing.assert_array_almost_equal(Y_q_ref2.dequantize().numpy(), Y_q.dequantize().numpy(), decimal=4)

if __name__ == "__main__":
    run_tests()
