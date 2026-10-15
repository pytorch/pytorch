# Owner(s): ["module: inductor"]

from contextlib import ExitStack
from unittest.mock import Mock, patch

import torch
from torch._inductor import config, ir
from torch._inductor.autotune_process import TensorMeta
from torch._inductor.codegen.rocm.ck_tile_universal_gemm_template import (
    CKTileGemmTemplate,
)
from torch._inductor.codegen.rocm.ck_universal_gemm_template import CKGemmTemplate
from torch._inductor.codegen.rocm.rocm_benchmark_request import ROCmBenchmarkRequest
from torch._inductor.codegen.rocm.rocm_kernel import ROCmTemplateCaller
from torch._inductor.select_algorithm import AlgorithmSelectorCache, NoValidChoicesError
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


@instantiate_parametrized_tests
class TestROCmLayoutConstraints(TestCase):
    def setUp(self):
        super().setUp()
        self.graph = Mock(sizevars=SizeVarAllocator(), buffer_layout_constraints={})
        self.graph.get_dtype.return_value = torch.float32
        stack = ExitStack()
        self.addCleanup(stack.close)
        stack.enter_context(V.set_graph_handler(self.graph))
        stack.enter_context(
            config.patch(
                strict_flexible_layout_strides=True,
                comprehensive_padding=True,
                padding_stride_threshold=0,
                padding_alignment_bytes=64,
            )
        )

    def make_choice(self, offset=0):
        device = torch.device("cpu")
        x = ir.Buffer(
            name="x", layout=ir.FlexibleLayout(device, torch.float32, [8, 17])
        )
        x.get_layout().offset = offset
        w = ir.Buffer(name="w", layout=ir.FixedLayout(device, torch.float32, [17, 8]))
        layout = ir.FixedLayout(device, torch.float32, [8, 8])
        template = CKTileGemmTemplate([x, w], layout)
        bmreq = Mock(spec=ROCmBenchmarkRequest, workspace_size=0)
        bmreq.input_tensor_meta = [
            TensorMeta(
                device=device,
                dtype=torch.float32,
                sizes=node.get_size(),
                strides=node.get_stride(),
                offset=node.get_layout().offset,
                name=node.get_name(),
            )
            for node in template.input_nodes
        ]
        choice = ROCmTemplateCaller(
            "ck", "ck", [x, w], layout, Mock(), bmreq, template, {}
        )
        return choice, x

    def test_unselected_choice_preserves_layout_freedom(self):
        choice, x = self.make_choice()
        self.assertIsInstance(x.get_layout(), ir.FlexibleLayout)
        self.assertEqual(x.get_stride_hint(), [17, 1])
        with patch.object(ir.ExternKernel, "copy_input") as copy_input:
            result = ir.ExternKernel.require_stride_order(x, [0, 1])
        copy_input.assert_not_called()
        self.assertIs(result, x)
        self.assertEqual(x.get_stride(), [1, 8])
        self.assertFalse(choice.is_layout_compatible())

    def test_selected_choice_freezes_padded_layout(self):
        choice, x = self.make_choice()
        self.assertEqual(choice.template.input_nodes[0].get_stride(), [32, 1])
        output = ir.Buffer(name="output", layout=choice.layout)
        with patch(
            "torch._inductor.codegen.rocm.rocm_kernel.ROCmTemplateBuffer",
            return_value=output,
        ):
            choice.output_node()
        self.assertIsInstance(x.get_layout(), ir.FixedLayout)
        self.assertEqual(x.get_stride(), [32, 1])
        self.assertTrue(choice.is_layout_compatible())

    @parametrize("deferred", [False, True])
    def test_incompatible_choice_is_not_selected(self, deferred):
        choice, x = self.make_choice()
        x.freeze_layout_with_stride_order([0, 1])
        fallback = Mock(spec=ir.ChoiceCaller)
        fallback.is_layout_compatible.return_value = True
        if deferred:
            multi = Mock(choice_timings=Mock(return_value={choice: 1.0, fallback: 2.0}))
            selected, timing = ir.MultiTemplateBuffer.get_min_choice(multi)
            self.assertEqual(timing, 2.0)
        else:
            _, selected = AlgorithmSelectorCache()(
                "mm", [choice, fallback], choice.input_nodes, choice.layout
            )
        self.assertIs(selected, fallback)
        self.assertEqual(x.get_stride(), [1, 8])

    def test_incompatible_choice_cannot_be_materialized(self):
        choice, x = self.make_choice()
        x.freeze_layout_with_stride_order([0, 1])
        with self.assertRaisesRegex(AssertionError, "layouts changed"):
            choice.output_node()
        multi = Mock(choice_timings=Mock(return_value={choice: 1.0}))
        with self.assertRaisesRegex(NoValidChoicesError, "compatible"):
            ir.MultiTemplateBuffer.get_min_choice(multi)

    def test_layout_change_during_autotuning_rejects_choice(self):
        choice, x = self.make_choice()
        fallback = Mock(spec=ir.ChoiceCaller)
        fallback.is_layout_compatible.return_value = True

        def autotune(*args, **kwargs):
            x.freeze_layout_with_stride_order([0, 1])
            return {choice: 1.0, fallback: 2.0}

        selector = AlgorithmSelectorCache()
        with (
            config.patch(autotune_in_subproc=False),
            patch.object(selector, "make_precompile_fn"),
            patch.object(selector, "do_autotuning", side_effect=autotune),
        ):
            _, selected = selector(
                "mm", [choice, fallback], choice.input_nodes, choice.layout
            )
        self.assertIs(selected, fallback)

    @parametrize("constrained", [False, True])
    @parametrize("offset", [0, 5])
    def test_benchmark_adapts_input_without_changing_values(self, constrained, offset):
        choice, x = self.make_choice(offset)
        strides = [48, 1] if constrained else [17, 1]
        if constrained:
            self.graph.buffer_layout_constraints[x.get_name()] = ir.FixedLayout(
                torch.device("cpu"), torch.float32, [8, 17], strides, offset
            )
        storage_size = torch._prims_common.compute_required_storage_length(
            (8, 17), strides, offset
        )
        value = torch.empty(storage_size).as_strided((8, 17), strides)
        logical_value = value.as_strided((8, 17), strides, offset)
        logical_value.copy_(torch.arange(8 * 17).view(8, 17))
        weight = torch.randn(17, 8)
        output = torch.empty(8, 8)
        choice.bmreq.benchmark.return_value = 1.0
        self.assertEqual(choice.benchmark(value, weight, out=output), 1.0)
        actual, actual_weight = choice.bmreq.benchmark.call_args.args
        self.assertEqual(actual.stride(), (32, 1))
        self.assertEqual(actual.storage_offset(), 0)
        self.assertEqual(actual.as_strided((8, 17), (32, 1), offset), logical_value)
        self.assertIs(actual_weight, weight)
        self.assertIsInstance(x.get_layout(), ir.FlexibleLayout)

    @parametrize("duplicate", [False, True])
    def test_snapshot_metadata_ignores_other_choice_constraints(self, duplicate):
        choice, x = self.make_choice()
        template = choice.template
        template.input_reorder = [1, 0]
        if duplicate:
            template.input_nodes[1] = template.input_nodes[0]
            template.original_input_nodes[1] = x
        self.graph.buffer_layout_constraints[x.get_name()] = ir.FixedLayout(
            torch.device("cpu"), torch.float32, [8, 17], [48, 1]
        )

        def render(kernel, **kwargs):
            return kernel.def_kernel(
                inputs=template.input_nodes,
                outputs=[template.output_node],
                size_args=[],
                names_str="x, w, y",
                input_reorder=template.input_reorder,
            )

        with (
            patch.object(template, "render", side_effect=render),
            patch(
                "torch._inductor.codegen.rocm.rocm_template.ROCmBenchmarkRequest"
            ) as request,
        ):
            template.generate(kBatch=1)
        metadata = request.call_args.kwargs["input_tensor_meta"]
        names = ["x"] if duplicate else ["w", "x"]
        self.assertEqual([meta.name for meta in metadata], names)
        self.assertEqual(metadata[-1].strides, [32, 1])
        self.assertEqual(metadata[-1].to_tensor().stride(), (32, 1))
        self.assertIsInstance(x.get_layout(), ir.FlexibleLayout)

    @parametrize("broadcast_weight", [False, True])
    def test_batched_gemm_uses_actual_batch_strides(self, broadcast_weight):
        device = torch.device("cpu")
        x = ir.Buffer(
            name="x", layout=ir.FlexibleLayout(device, torch.float32, [2, 8, 17])
        )
        w_strides = [0 if broadcast_weight else 136, 8, 1]
        w = ir.Buffer(
            name="w",
            layout=ir.FixedLayout(device, torch.float32, [2, 17, 8], w_strides),
        )
        layout = ir.FixedLayout(device, torch.float32, [2, 8, 8])
        template = CKGemmTemplate([x, w], layout, alpha=1, beta=0)
        batch_strides = (256, 0 if broadcast_weight else 136, 64)
        self.assertEqual(template.size_args()[-3:], batch_strides)
        op = Mock(
            c_shuffle_block_transfer_scalar_per_vector_n_per_block=(1,),
            c_elementwise_op="PassThrough",
        )
        kernel = Mock()
        with patch.object(template, "emit_ck_instance", return_value=("", "Op")):
            code = template.render(kernel, op)
        self.assertIn("BatchStrideA,", code)
        self.assertNotIn("M * K, // batch_stride_A", code)
        self.assertEqual(
            kernel.def_kernel.call_args.kwargs["size_args"][-3:],
            ["int32_t BatchStrideA", "int32_t BatchStrideB", "int32_t BatchStrideC"],
        )


if __name__ == "__main__":
    run_tests()
