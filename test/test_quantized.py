from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.jit
import numpy as np
import unittest
from common_utils import TEST_WITH_UBSAN, TestCase, run_tests, skipIfNotRegistered
import torch.nn.functional as F


def canonical(graph):
    return str(torch._C._jit_pass_canonicalize(graph))


def _quantize(x, scale, zero_point, qmin=0, qmax=255, qtype=np.uint8):
    """Quantizes a numpy array."""
    qx = np.round(x / scale + zero_point)
    qx = np.clip(qx, qmin, qmax).astype(qtype)
    return qx


def _dequantize(qx, scale, zero_point):
    """Dequantizes a numpy array."""
    x = (qx.astype(np.float) - zero_point) * scale
    return x


# Make sure we won't have overflows from vpmaddubsw instruction used in FBGEMM.
# On the current Intel x86 architecture, we need to utilize vpmaddubsw instruction
# for the 8-bit int multiplication. This instruction vertically multiplies each
# unsigned 8-bit integer from a with the corresponding signed 8-bit integer from
# b, producing intermediate signed 16-bit integers. This function modifies the
# weights to eliminate the overflow on the signed 16-bit integers.
def avoid_vpmaddubsw_overflow_fc(
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


# Reference quantized FC operator
def qfc_ref(X_q, X_scale, X_zp, W_q, W_scale, W_zp, b_q, Y_scale, Y_zp):
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
    Y_q_ref = _quantize(Prod_XqWq_ref + b_q, Y_scale / (X_scale * W_scale), Y_zp)
    return Y_q_ref


@skipIfNotRegistered("Relu_ENGINE_FBGEMM",
                     "fbgemm-based Caffe2 ops are not linked")
class TestQuantized(TestCase):
    def test_relu(self):
        a = (torch.tensor([4, 6, 1, 10], dtype=torch.uint8), 0.01, 5)
        r = torch.ops.c10.quantized_relu(a)
        np.testing.assert_equal(
            r[0].numpy(), torch.tensor([5, 6, 5, 10], dtype=torch.uint8).numpy()
        )
        np.testing.assert_almost_equal(0.01, r[1])
        self.assertEqual(5, r[2])

    def test_quantize(self):
        a = (torch.tensor([4, 6, 1, 10], dtype=torch.uint8), 0.01, 5)
        r = torch.ops.c10.dequantize(a)
        np.testing.assert_almost_equal(r.numpy(), [-0.01, 0.01, -0.04, 0.05])
        # default args
        q_def = torch.ops.c10.quantize(r)
        # specified
        q = torch.ops.c10.quantize(r, scale=0.01, zero_point=5)
        np.testing.assert_equal(q[0].numpy(), a[0].numpy())
        np.testing.assert_almost_equal(q[1], a[1])
        self.assertEqual(q[2], a[2])

    def test_script(self):
        @torch.jit.script
        def foo(x):
            # type: (Tuple[Tensor, float, int]) -> Tuple[Tensor, float, int]
            return torch.ops.c10.quantized_relu(x)

        self.assertExpectedInline(
            canonical(foo.graph),
            """\
graph(%x : (Tensor, float, int)):
  %1 : (Tensor, float, int) = c10::quantized_relu(%x)
  return (%1)
""",
        )


class TestQuantizedOps(unittest.TestCase):
    """Tests the correctness of the quantized::relu op."""

    def test_qrelu(self):
        relu = torch.ops.quantized.relu

        X = torch.arange(-5, 5, dtype=torch.float)
        scale = 2.0
        zero_point = 1
        qX = X.quantize_linear(scale=scale, zero_point=zero_point)
        # print("X:\n{}".format(X))
        # print("\nQuantized:\n{}\nFake:\n{}".format(qX.int_repr(), _quantize(X.numpy(), scale, zero_point)))

        Y = X.numpy().copy()
        Y[Y < 0] = 0
        qY = _quantize(Y, scale, zero_point)
        qY_hat = relu(qX)
        np.testing.assert_equal(qY, qY_hat.int_repr())

    """Tests the correctness of the quantized::sum_relu op."""
    def test_qsumrelu_same_qparams(self):
        sum_relu = torch.ops.quantized.sum_relu

        A = torch.arange(-25, 25, dtype=torch.float)
        B = torch.arange(-25, 25, dtype=torch.float)
        scale = 2.0
        zero_point = 127
        qA = A.quantize_linear(scale=scale, zero_point=zero_point)
        qB = A.quantize_linear(scale=scale, zero_point=zero_point)

        # Sum + ReLU ground truth
        C = (qA.dequantize() + qB.dequantize()).numpy()
        C[C < 0] = 0
        qC = _quantize(C, scale, zero_point)

        qC_hat = sum_relu(qA, qB, scale=scale, zero_point=zero_point)
        np.testing.assert_equal(qC, qC_hat.int_repr())

    """Tests the correctness of the quantized::sum_relu op."""
    def test_qsumrelu_different_qparams(self):
        sum_relu = torch.ops.quantized.sum_relu

        A = torch.arange(-25, 25, dtype=torch.float)
        B = torch.arange(-25, 25, dtype=torch.float)
        scale_A = 3.0
        zero_point_A = 7
        scale_B = 5.0
        zero_point_B = 127

        scale_C = 0.5
        zero_point_C = 5

        qA = A.quantize_linear(scale=scale_A, zero_point=zero_point_A)
        qB = A.quantize_linear(scale=scale_B, zero_point=zero_point_B)

        # Sum + ReLU ground truth
        C = (qA.dequantize() + qB.dequantize()).numpy()
        C[C < 0] = 0
        qC = _quantize(C, scale_C, zero_point_C)

        qC_hat = sum_relu(qA, qB, scale=scale_C, zero_point=zero_point_C)
        np.testing.assert_equal(qC, qC_hat.int_repr())


@unittest.skipIf(
    TEST_WITH_UBSAN or not torch.fbgemm_is_cpu_supported(),
    " Quantized FC requires FBGEMM. FBGEMM does not play"
    " well with UBSAN at the moment, so we skip the test if"
    " we are in a UBSAN environment.",
)
class TestQuantizedFC(unittest.TestCase):
    """Tests the correctness of the quantized::fc op."""

    def test_qfc(self):
        qfc_prepack = torch.ops.quantized.fbgemm_linear_prepack
        qfc = torch.ops.quantized.fbgemm_linear

        batch_size = 4
        input_channels = 16
        output_channels = 8

        X_scale = 1.5
        X_zp = 5
        X_value_min = 0
        X_value_max = 225
        X_q = np.round(
            np.random.rand(batch_size, input_channels) * (X_value_max - X_value_min)
            + X_value_min
        ).astype(np.uint8)

        W_scale = 0.4
        W_zp = 2
        W_value_min = -128
        W_value_max = 127
        W_q = np.round(
            np.random.rand(output_channels, input_channels)
            * (W_value_max - W_value_min)
            + W_value_min
        ).astype(np.int8)

        b_q = np.round(np.random.randn(output_channels) * 10 - 10).astype(np.int32)

        # Compare X_scale * W_scale * input_channels * X_value_max * W_value_max with
        # Y_scale * 255 (max for uint8).
        Y_scale = 125.1234
        Y_zp = 5

        avoid_vpmaddubsw_overflow_fc(
            batch_size,
            input_channels,
            output_channels,
            X_q,
            X_value_min,
            X_value_max,
            W_q,
            W_value_min,
            W_value_max,
        )

        # Reference quantized FC operator
        Y_q_ref = qfc_ref(X_q, X_scale, X_zp, W_q, W_scale, W_zp, b_q, Y_scale, Y_zp)

        # Weight prepacking operator for quantized FC
        W_prepack = qfc_prepack(torch.from_numpy(W_q), W_zp)
        [Y_q, Y_scale, Y_zp] = qfc(
            torch.from_numpy(X_q),
            X_scale,
            X_zp,
            W_prepack,
            W_scale,
            W_zp,
            torch.from_numpy(b_q),
            Y_scale,
            Y_zp,
        )

        # Assert equal
        np.testing.assert_equal(Y_q_ref, Y_q.numpy())

        # Reference quantized result from PyTorch Linear operator
        W_fp32 = _dequantize(W_q, W_scale, W_zp).astype(np.float)
        X_fp32 = _dequantize(X_q, X_scale, X_zp).astype(np.float)
        b_fp32 = _dequantize(b_q, W_scale * X_scale, 0).astype(np.float)
        Y_fp32_ref = F.linear(torch.from_numpy(X_fp32), torch.from_numpy(W_fp32),
                              torch.from_numpy(b_fp32)).numpy()
        Y_q_ref2 = _quantize(Y_fp32_ref, Y_scale, Y_zp)

        # Assert equal
        np.testing.assert_equal(Y_q_ref2, Y_q.numpy())


if __name__ == "__main__":
    run_tests()
