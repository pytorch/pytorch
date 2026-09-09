# Owner(s): ["module: inductor"]
"""
Tests for the FlexibleLayout freezing discipline: strides of an unfrozen
FlexibleLayout are provisional (freezing may reorder or pad them), so they
must be frozen before being persisted anywhere. See #192575.
"""

from unittest.mock import Mock, patch

import torch
import torch._inductor.config as inductor_config
from torch._inductor import ir, metrics
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch.testing._internal.common_device_type import (
    instantiate_device_type_tests,
    onlyAccelerator,
    onlyCPU,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
)


@instantiate_parametrized_tests
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

    def test_analysis_mode_allows_stride_reads(self):
        buf = self._flexible_buffer()
        layout = buf.get_layout()
        with inductor_config.patch(strict_flexible_layout_strides=True):
            with ir.allow_layout_analysis():
                self.assertEqual(list(layout.stride), [128, 1])
                # compositional: evaluate a region against hypothetical strides
                with patch.object(layout, "_stride", [1, 8]):
                    self.assertEqual(list(layout.stride), [1, 8])
            self.assertIsInstance(buf.get_layout(), ir.FlexibleLayout)
            with self.assertRaisesRegex(AssertionError, "unfrozen FlexibleLayout"):
                layout.stride

    @parametrize("frozen", [False, True])
    @inductor_config.patch(strict_flexible_layout_strides=True)
    def test_mutation_stride_hint(self, frozen):
        buf = self._flexible_buffer((2, 3))
        buf.get_layout().stride = [1, 2]
        if frozen:
            buf.freeze_layout()
        with V.set_graph_handler(Mock(sizevars=SizeVarAllocator())):
            layout = ir.MutationLayoutSHOULDREMOVE(ir.StorageBox(buf))
            self.assertFalse(layout.is_contiguous())
        mutation = ir.TensorBox.create(ir.Buffer(name="mutation", layout=layout))
        self.assertEqual(layout.stride_hint(), [1, 2])
        self.assertEqual(mutation.get_stride_hint(), [1, 2])

    @parametrize("frozen", [False, True])
    @inductor_config.patch(strict_flexible_layout_strides=True)
    def test_permute_stride_hint(self, frozen):
        buf = self._flexible_buffer((2, 3))
        if frozen:
            buf.freeze_layout()
        view = ir.PermuteView(data=ir.StorageBox(buf), dims=[1, 0])
        self.assertEqual(view.get_stride_hint(), [1, 3])
        self.assertEqual(ir.TensorBox(view).get_stride_hint(), [1, 3])

    def test_generic_view_has_no_stride_hint(self):
        buf = self._flexible_buffer((2, 3))
        view = ir.GenericView(
            data=ir.StorageBox(buf), size=[6], reindex=lambda i: [i[0] // 3, i[0] % 3]
        )
        self.assertIsNone(view.maybe_get_stride_hint())
        self.assertIsNone(ir.TensorBox(view).maybe_get_stride_hint())

    @parametrize("layout", ["flexible", "contiguous", "padded"])
    def test_flex_cpu_layout_requirement_does_not_copy(self, layout):
        from torch._inductor.kernel.flex.common import contiguous_last_dim

        buf = self._flexible_buffer((8, 16))
        if layout == "padded":
            buf.get_layout().stride = [32, 1]
        if layout != "flexible":
            buf.freeze_layout()
        x = ir.TensorBox.create(buf)
        with V.set_graph_handler(Mock(sizevars=SizeVarAllocator())):
            self.assertIs(contiguous_last_dim(x), x)
        self.assertIsInstance(buf.get_layout(), ir.FixedLayout)
        self.assertEqual(x.get_stride(), [32, 1] if layout == "padded" else [16, 1])

    def test_flex_cpu_preserves_outer_stride_order(self):
        from torch._inductor.kernel.flex.common import contiguous_last_dim

        buf = self._flexible_buffer((2, 3, 4, 16))
        strides = [192, 16, 48, 1]
        buf.get_layout().stride = strides
        x = ir.TensorBox.create(buf)
        with V.set_graph_handler(Mock(sizevars=SizeVarAllocator())):
            self.assertIs(contiguous_last_dim(x), x)
        self.assertIsInstance(buf.get_layout(), ir.FixedLayout)
        self.assertEqual(x.get_stride(), strides)

    @parametrize("strides", [[16, 1], [1, 8]])
    def test_significant_strides_keep_layout_flexible(self, strides):
        buf = ir.ComputedBuffer(
            name="input",
            layout=ir.FlexibleLayout(torch.device("cpu"), torch.float32, [8, 16]),
            data=ir.Pointwise(
                device=torch.device("cpu"),
                dtype=torch.float32,
                ranges=[8, 16],
                inner_fn=lambda index: ir.ops.constant(0, torch.float32),
            ),
        )
        with (
            V.set_graph_handler(Mock(sizevars=SizeVarAllocator())),
            patch.object(buf, "get_fill_order", return_value=None),
        ):
            self.assertIs(ir.try_match_insignificant_strides(buf, strides), buf)
        self.assertIsInstance(buf.get_layout(), ir.FlexibleLayout)


class TestLayoutFreezingDevice(TestCase):
    @onlyCPU
    @parametrize("input_index", [0, 1, 2])
    def test_flex_attention_strided_last_dim(self, device, input_index):
        from torch.nn.attention.flex_attention import flex_attention

        args = [torch.randn(1, 2, 32, 16, device=device) for _ in range(3)]
        args[input_index] = torch.randn(1, 2, 32, 32, device=device)[..., ::2]
        expected = torch.nn.functional.scaled_dot_product_attention(*args)
        actual = torch.compile(flex_attention, fullgraph=True)(*args)
        self.assertEqual(actual, expected)

    @onlyCPU
    @parametrize("input_index", [0, 1, 2])
    def test_flex_attention_computed_transpose(self, device, input_index):
        from torch.nn.attention.flex_attention import flex_attention

        def fn(x, q, k, v):
            args = [q, k, v]
            args[input_index] = x.sin().transpose(-1, -2)
            return flex_attention(*args)

        x = torch.randn(1, 2, 16, 32, device=device)
        q, k, v = [torch.randn(1, 2, 32, 16, device=device) for _ in range(3)]
        args = [q, k, v]
        args[input_index] = x.sin().transpose(-1, -2)
        expected = torch.nn.functional.scaled_dot_product_attention(*args)
        actual = torch.compile(fn, fullgraph=True)(x, q, k, v)
        self.assertEqual(actual, expected)

    @onlyAccelerator
    @parametrize("width", [342, 347])
    def test_noncontiguous_reshape_of_padded_buffer(self, device, width):
        # Regression test for #192575: View.create baked an unfrozen buffer's
        # placeholder strides into a ReinterpretView; comprehensive padding
        # then changed the real strides, silently corrupting gradients.

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
            torch.randn(*shape, device=device, requires_grad=True)
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

    @onlyAccelerator
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_select_of_padded_buffer(self, device):
        # select with an unbacked index persists the input's strides/offset
        # into DynamicSelectStorageOffset; the layout must be frozen first.
        def fn(x, idx_t):
            idx = idx_t.item()
            y = torch.sin(x)
            return y.select(0, idx) + 1

        x = torch.randn(16, 1030, device=device)
        idx_t = torch.tensor(3, device=device)
        with inductor_config.patch(
            comprehensive_padding=True, padding_stride_threshold=0
        ):
            expected = fn(x, idx_t)
            actual = torch.compile(fn)(x, idx_t)
        self.assertEqual(actual, expected)

    @onlyAccelerator
    @torch._dynamo.config.patch(capture_scalar_outputs=True)
    def test_unbacked_slice_of_padded_buffer(self, device):
        def fn(x, start_t):
            start = start_t.item()
            y = torch.sin(x)
            return y[start:, :] + 1

        x = torch.randn(16, 1030, device=device)
        start_t = torch.tensor(2, device=device)
        with inductor_config.patch(
            comprehensive_padding=True, padding_stride_threshold=0
        ):
            expected = fn(x, start_t)
            actual = torch.compile(fn)(x, start_t)
        self.assertEqual(actual, expected)

    @onlyAccelerator
    def test_sdpa_computed_bias_with_reuse(self, device):
        # A realized-but-flexible bias (realized by a second consumer) must
        # take the require_* path, which freezes it into the wanted order in
        # place; the alignment check used to peek at its unfrozen strides and
        # skip that path.
        from torch.nn.attention import sdpa_kernel, SDPBackend

        def fn(q, k, v, b):
            bias = (b * 2.0).sin()
            aux = bias * 3.0
            with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION]):
                out = torch.nn.functional.scaled_dot_product_attention(
                    q, k, v, attn_mask=bias
                )
            return out, aux

        torch.manual_seed(0)
        q, k, v = (torch.randn(2, 4, 128, 64, device=device) for _ in range(3))
        b = torch.randn(2, 4, 128, 128, device=device)
        expected = fn(q, k, v, b)
        with inductor_config.patch(realize_opcount_threshold=1):
            actual = torch.compile(fn)(q, k, v, b)
        self.assertEqual(actual, expected, atol=2e-4, rtol=2e-4)

    def test_strict_mode_end_to_end(self, device):
        # Common compile paths must not read unfrozen strides.
        def fn(x, y):
            a = (x @ y).relu()
            b = torch.cat([a, a.transpose(0, 1)[: a.size(0)]], dim=1)
            return b.reshape(b.size(0), -1, 2).sum(-1)

        x = torch.randn(64, 1030, device=device)
        y = torch.randn(1030, 64, device=device)
        with inductor_config.patch(strict_flexible_layout_strides=True):
            expected = fn(x, y)
            actual = torch.compile(fn)(x, y)
        self.assertEqual(actual, expected)


instantiate_device_type_tests(
    TestLayoutFreezingDevice, globals(), allow_xpu=True, allow_mps=True
)


if __name__ == "__main__":
    run_tests()
