# Owner(s): ["module: inductor"]
"""
Tests for the FlexibleLayout freezing discipline: strides of an unfrozen
FlexibleLayout are provisional (freezing may reorder or pad them), so they
must be frozen before being persisted anywhere. See #192575.
"""

from unittest.mock import Mock, patch

import torch
import torch._inductor.config as inductor_config
from torch._inductor import ir
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
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

    @parametrize("transposed", [False, True])
    @parametrize("aten_fallback", [False, True])
    @inductor_config.patch(
        strict_flexible_layout_strides=True,
        comprehensive_padding=True,
        padding_stride_threshold=0,
        padding_alignment_bytes=64,
    )
    def test_tma_orientation_preserves_layout_freedom(self, transposed, aten_fallback):
        from torch._inductor.heuristics.template import triton as tma_heuristics
        from torch._inductor.kernel_inputs import MMKernelInputs
        from torch._inductor.select_algorithm import TritonTemplateKernel
        from torch._inductor.utils import tma_inner_dim

        buf = ir.ComputedBuffer(
            name="matrix",
            layout=self._flexible_buffer((8, 24)).get_layout(),
            data=ir.Pointwise(
                device=torch.device("cpu"),
                dtype=torch.float32,
                ranges=[8, 24],
                inner_fn=lambda index: ir.ops.constant(0, torch.float32),
            ),
        )
        if transposed:
            buf.get_layout().stride = [1, 8]
        weight = ir.InputBuffer(
            name="weight",
            layout=ir.FixedLayout(torch.device("cpu"), torch.float32, [24, 8]),
        )
        inputs = MMKernelInputs([ir.StorageBox(buf), ir.StorageBox(weight)])
        graph = Mock(sizevars=SizeVarAllocator(), buffer_layout_constraints={})
        base = tma_heuristics.MMTemplateConfigMixin
        heuristic = tma_heuristics.TMATemplateConfigMixin()
        layout_error = AssertionError("early layout decision")
        with (
            V.set_graph_handler(graph),
            patch.object(buf, "get_fill_order", side_effect=layout_error),
            patch.object(base, "_get_template_configs_impl", return_value=iter([{}])),
            patch.object(tma_heuristics, "get_num_sms", return_value=1),
            patch(
                "torch._inductor.select_algorithm.use_aten_gemm_kernels",
                return_value=aten_fallback,
            ),
        ):
            (options,) = heuristic._get_template_configs_impl(inputs, "mm")
            self.assertIsInstance(buf.get_layout(), ir.FlexibleLayout)
            kernel = Mock(always_freeze_layout=False)
            strides = TritonTemplateKernel.get_stride_and_maybe_freeze_layout(
                kernel, inputs.nodes()[0]
            )
            self.assertEqual(options["A_ROW_MAJOR"], tma_inner_dim(strides) == 1)
            self.assertEqual(strides, [1, 16] if transposed else [32, 1])
            if aten_fallback:
                self.assertIsInstance(buf.get_layout(), ir.FlexibleLayout)
                layout = graph.buffer_layout_constraints[buf.get_name()]
                self.assertEqual(layout.stride, strides)
            else:
                self.assertIsInstance(buf.get_layout(), ir.FixedLayout)


if __name__ == "__main__":
    run_tests()
