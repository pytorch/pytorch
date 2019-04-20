from __future__ import absolute_import, division, print_function, unicode_literals

import torch
import torch.jit
import numpy as np
import unittest
from common_utils import TEST_WITH_UBSAN, TestCase, run_tests, skipIfNotRegistered


def canonical(graph):
    return str(torch._C._jit_pass_canonicalize(graph))


def _quantize(x, scale, zero_point):
    """Quantizes a numpy array."""
    qx = np.round(x / scale).astype(np.uint8) + zero_point
    return qx


def _dequantize(qx, scale, zero_point):
    """Dequantizes a numpy array."""
    x = (qx - zero_point) * scale
    return x


# Make sure we won't have overflows from vpmaddubsw instruction used in fbgemm)
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

    Y_q_ref = (
        (np.round((Prod_XqWq_ref + b_q) * (Y_scale / X_scale / W_scale)) + Y_zp)
        .clip(0, 255)
        .astype(np.uint8)
    )
    return Y_q_ref


@skipIfNotRegistered("Relu_ENGINE_DNNLOWP",
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


class TestQuantizedRelu(unittest.TestCase):
    """Tests the correctness of the quantized::relu op."""

    def test_qrelu(self):
        relu = torch.ops.quantized.relu

        X_tensor = np.arange(0, 10, dtype=np.uint8)
        scale = 255.0
        zero_point = 5

        Y_tensor = X_tensor.copy()
        Y_tensor[X_tensor < zero_point] = zero_point

        X = (torch.from_numpy(X_tensor), scale, zero_point)

        Y_hat = relu(*X)
        Y_hat_tensor = Y_hat[0].numpy()

        np.testing.assert_equal(Y_tensor, Y_hat_tensor)


@unittest.skipIf(
    TEST_WITH_UBSAN or not torch.fbgemm_is_cpu_supported(),
    " Quantized FC requires FBGEMM. FBGEMM does not play"
    " well with UBSAN at the moment, so we skip the test if"
    " we are in a UBSAN environment.",
)
class TestQuantizedFC(unittest.TestCase):
    """Tests the correctness of the quantized::fc op."""

    def test_qfc(self):
        qfc_prepack = torch.ops.quantized.fc_prepack
        qfc = torch.ops.quantized.fc

        batch_size = 4
        input_channels = 16
        output_channels = 8

        X_scale = 1.5
        X_zp = 5
        X_value_min = 0
        X_value_max = 15
        X_q = np.round(
            np.random.rand(batch_size, input_channels) * (X_value_max - X_value_min)
            + X_value_min
        ).astype(np.uint8)

        W_scale = 0.4
        W_zp = 2
        W_value_min = -10
        W_value_max = 15
        W_q = np.round(
            np.random.rand(output_channels, input_channels)
            * (W_value_max - W_value_min)
            + W_value_min
        ).astype(np.int8)

        b_q = np.round(np.random.randn(output_channels) * 10 - 10).astype(np.int32)

        Y_scale = 0.1234
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

        # print(Y_q_ref)
        # print(Y_q)

        # Assert equal
        np.testing.assert_equal(Y_q_ref, Y_q.numpy())


if __name__ == "__main__":
    run_tests()
