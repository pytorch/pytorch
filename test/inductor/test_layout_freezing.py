# Owner(s): ["module: inductor"]
"""
Tests for the FlexibleLayout freezing discipline: strides of an unfrozen
FlexibleLayout are provisional (freezing may reorder or pad them), so they
must be frozen before being persisted anywhere. See #192575.
"""

import torch
import torch._inductor.config as inductor_config
from torch._inductor import ir, metrics
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.inductor_utils import GPU_TYPE, requires_gpu


class TestLayoutFreezing(TestCase):
    def _flexible_buffer(self, size=(8, 128)):
        layout = ir.FlexibleLayout(
            device=torch.device("cpu"), dtype=torch.float32, size=list(size)
        )
        return ir.Buffer(name="buf_test", layout=layout)

    def test_reinterpret_view_requires_frozen_base(self):
        buf = self._flexible_buffer()
        view_layout = ir.FixedLayout(
            device=torch.device("cpu"), dtype=torch.float32, size=[1024], stride=[1]
        )
        with self.assertRaisesRegex(AssertionError, "flexible layout"):
            ir.ReinterpretView(data=ir.StorageBox(buf), layout=view_layout)
        buf.freeze_layout()
        ir.ReinterpretView(data=ir.StorageBox(buf), layout=view_layout)

    def test_strict_mode_blocks_stride_reads(self):
        buf = self._flexible_buffer()
        layout = buf.get_layout()
        with inductor_config.patch(strict_flexible_layout_strides=True):
            with self.assertRaisesRegex(AssertionError, "unfrozen FlexibleLayout"):
                layout.stride
            with self.assertRaisesRegex(AssertionError, "unfrozen FlexibleLayout"):
                buf.get_stride()
            self.assertEqual(list(layout.stride_hint()), [128, 1])
            self.assertEqual(list(buf.get_stride_hint()), [128, 1])
            str(layout)  # repr must not raise
            buf.freeze_layout()
            self.assertEqual(list(buf.get_stride()), [128, 1])

    @requires_gpu()
    def test_noncontiguous_reshape_of_padded_buffer(self):
        # Regression test for #192575: View.create baked an unfrozen buffer's
        # placeholder strides into a ReinterpretView; comprehensive padding
        # then changed the real strides, silently corrupting gradients.
        width = 342  # 3 * 342 = 1026 crosses the padding threshold, unaligned

        def fn(x, offset, weight):
            query, key, value = (
                part.view(2, 3, 1, width) + offset
                for part in (x @ weight.T).chunk(3, -1)
            )
            query_sig = torch.sigmoid(query).transpose(1, 2)
            query_tanh = torch.tanh(query).transpose(1, 2)
            key_sig = torch.sigmoid(key).transpose(1, 2)
            key_tanh = torch.tanh(key).transpose(1, 2)
            scores = (
                query_sig @ key_sig.transpose(-2, -1)
                + query_tanh @ key_tanh.transpose(-2, -1)
                - query_sig @ key_tanh.transpose(-2, -1)
            )
            return scores @ value.transpose(1, 2)

        torch.manual_seed(0)
        args = tuple(
            torch.randn(*shape, device=GPU_TYPE, requires_grad=True)
            for shape in [(2, 3, 1), (2, 3, 1, width), (3 * width, 1)]
        )
        with inductor_config.patch(
            comprehensive_padding=True, padding_stride_threshold=1024
        ):
            metrics.reset()
            expected = fn(*args)
            expected_grads = torch.autograd.grad(expected.sum(), args)
            actual = torch.compile(fn)(*args)
            actual_grads = torch.autograd.grad(actual.sum(), args)
        self.assertTrue(metrics.num_comprehensive_padding > 0)
        self.assertEqual(actual, expected, atol=1e-4, rtol=1e-4)
        self.assertEqual(actual_grads, expected_grads, atol=2e-3, rtol=1e-4)

    @requires_gpu()
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_select_of_padded_buffer(self):
        # select with an unbacked index persists the input's strides/offset
        # into DynamicSelectStorageOffset; the layout must be frozen first.
        def fn(x, idx_t):
            idx = idx_t.item()
            y = torch.sin(x)
            return y.select(0, idx) + 1

        x = torch.randn(16, 1030, device=GPU_TYPE)
        idx_t = torch.tensor(3, device=GPU_TYPE)
        with inductor_config.patch(
            comprehensive_padding=True, padding_stride_threshold=0
        ):
            expected = fn(x, idx_t)
            actual = torch.compile(fn)(x, idx_t)
        self.assertEqual(actual, expected)

    @requires_gpu()
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_slice_of_padded_buffer(self):
        def fn(x, start_t):
            start = start_t.item()
            y = torch.sin(x)
            return y[start:, :] + 1

        x = torch.randn(16, 1030, device=GPU_TYPE)
        start_t = torch.tensor(2, device=GPU_TYPE)
        with inductor_config.patch(
            comprehensive_padding=True, padding_stride_threshold=0
        ):
            expected = fn(x, start_t)
            actual = torch.compile(fn)(x, start_t)
        self.assertEqual(actual, expected)

    @requires_gpu()
    def test_strict_mode_end_to_end(self):
        # Common compile paths must not read unfrozen strides.
        def fn(x, y):
            a = (x @ y).relu()
            b = torch.cat([a, a.transpose(0, 1)[: a.size(0)]], dim=1)
            return b.reshape(b.size(0), -1, 2).sum(-1)

        x = torch.randn(64, 1030, device=GPU_TYPE)
        y = torch.randn(1030, 64, device=GPU_TYPE)
        with inductor_config.patch(strict_flexible_layout_strides=True):
            expected = fn(x, y)
            actual = torch.compile(fn)(x, y)
        self.assertEqual(actual, expected)


if __name__ == "__main__":
    run_tests()
