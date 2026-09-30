# Owner(s): ["module: inductor"]
import math
import unittest
from unittest import mock
from unittest.mock import MagicMock

import torch
from torch._inductor import config
from torch._inductor.codegen.subgraph import (
    MultiKernelFusionPlanChoice,
    SubgraphChoiceCaller,
)
from torch._inductor.ir import Buffer, FixedLayout, FlexibleLayout
from torch._inductor.kernel.decompose_k import (
    BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS,
    decomposeK as blackwell_decomposeK,
    lower_blackwell_decompose_k_partial,
)
from torch._inductor.lowering import lowerings, register_lowering
from torch._inductor.scheduler import Scheduler
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_cuda import SM100OrLater
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CPU, HAS_GPU


def decomposeK(a, b, kPartitions):
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    B = k // kPartitions
    a_reshaped = torch.permute(a.reshape(m, B, kPartitions), (1, 0, 2))
    b_reshaped = b.reshape(B, kPartitions, n)
    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)
    result_fp32 = result.to(torch.float32)
    reduced_buf = torch.sum(result_fp32, 0)
    return reduced_buf.to(a.dtype)


BLACKWELL_K_SPLIT = 8


# This test-only op exposes the internal partial-BMM lowering so tests can
# force a specific Triton schedule without depending on whole-plan autotuning.
@torch.library.custom_op(
    "inductor_test::blackwell_decompose_k_partial", mutates_args={}
)
def blackwell_decompose_k_partial(
    a: torch.Tensor, b: torch.Tensor, two_ctas: bool
) -> torch.Tensor:
    m, k = a.shape
    n = b.shape[1]
    m_pad = math.ceil(m / 128) * 128
    block_k = 64 if two_ctas else 128
    k_part = math.ceil(math.ceil(k / BLACKWELL_K_SPLIT) / block_k) * block_k
    out = torch.zeros(
        (BLACKWELL_K_SPLIT, m_pad, n), device=a.device, dtype=torch.float32
    )
    for split in range(BLACKWELL_K_SPLIT):
        begin = split * k_part
        end = min(begin + k_part, k)
        if begin < end:
            out[split, :m] = torch.mm(
                a[:, begin:end], b[begin:end], out_dtype=torch.float32
            )
    return out.view(BLACKWELL_K_SPLIT * m_pad, n)


@blackwell_decompose_k_partial.register_fake
def _(a: torch.Tensor, b: torch.Tensor, two_ctas: bool) -> torch.Tensor:
    del two_ctas
    m_pad = math.ceil(a.shape[0] / 128) * 128
    return a.new_empty((BLACKWELL_K_SPLIT * m_pad, b.shape[1]), dtype=torch.float32)


class TestSubgraphChoice(TestCase):
    def setUp(self):
        super().setUp()

    def _create_buffer(self, name, shape, dtype):
        return Buffer(
            name=name,
            layout=FixedLayout(torch.device(f"{GPU_TYPE}:0"), dtype=dtype, size=shape),
        )

    def test_speculative_inline_rolls_back_graph_registrations(self):
        caller = object.__new__(SubgraphChoiceCaller)
        caller.gm = MagicMock()
        caller.input_nodes = []
        caller.name = "candidate"

        old_operation = MagicMock(operation_name="op0")
        old_buffer = MagicMock(name="buf0")
        graph = MagicMock()
        graph.operations = [old_operation]
        graph.buffers = [old_buffer]
        graph.name_to_op = {"op0": old_operation}
        graph.name_to_buffer = {"buf0": old_buffer}
        graph.env = {"old": object()}
        graph.removed_buffers = {"removed_buffer"}
        graph.removed_operations = {"removed_operation"}

        new_operation = MagicMock(operation_name="op1")
        new_buffer = MagicMock(name="buf1")

        def inline(*args, **kwargs):
            graph.operations.append(new_operation)
            graph.buffers.append(new_buffer)
            graph.name_to_op["op1"] = new_operation
            graph.name_to_buffer["buf1"] = new_buffer
            graph.env["candidate"] = object()
            graph.removed_buffers.add("candidate_buffer")
            graph.removed_operations.add("candidate_operation")
            return "output"

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.codegen.subgraph.inline_subgraph_to_ir_nodes",
                side_effect=inline,
            ),
        ):
            with caller.speculative_inline() as plan:
                self.assertEqual(plan.output, "output")
                self.assertEqual(plan.operations, (new_operation,))
                self.assertEqual(plan.buffers, (new_buffer,))

        self.assertEqual(graph.operations, [old_operation])
        self.assertEqual(graph.buffers, [old_buffer])
        self.assertEqual(graph.name_to_op, {"op0": old_operation})
        self.assertEqual(graph.name_to_buffer, {"buf0": old_buffer})
        self.assertEqual(set(graph.env), {"old"})
        self.assertEqual(graph.removed_buffers, {"removed_buffer"})
        self.assertEqual(graph.removed_operations, {"removed_operation"})
        self.assertIsNone(new_operation.operation_name)
        self.assertIsNone(new_buffer.name)

    def test_speculative_inline_can_commit_graph_registrations(self):
        caller = object.__new__(SubgraphChoiceCaller)
        caller.gm = MagicMock()
        caller.input_nodes = []
        caller.name = "candidate"

        graph = MagicMock()
        graph.operations = []
        graph.buffers = []
        graph.name_to_op = {}
        graph.name_to_buffer = {}
        graph.env = {}
        graph.removed_buffers = set()
        graph.removed_operations = set()
        new_operation = MagicMock(operation_name="op0")
        new_buffer = MagicMock(name="buf0")

        def inline(*args, **kwargs):
            graph.operations.append(new_operation)
            graph.buffers.append(new_buffer)
            graph.name_to_op["op0"] = new_operation
            graph.name_to_buffer["buf0"] = new_buffer
            return "output"

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.codegen.subgraph.inline_subgraph_to_ir_nodes",
                side_effect=inline,
            ),
        ):
            with caller.speculative_inline() as plan:
                plan.commit()

        self.assertEqual(graph.operations, [new_operation])
        self.assertEqual(graph.buffers, [new_buffer])
        self.assertEqual(graph.name_to_op, {"op0": new_operation})
        self.assertEqual(graph.name_to_buffer, {"buf0": new_buffer})

    def test_speculative_inline_rolls_back_after_error(self):
        caller = object.__new__(SubgraphChoiceCaller)
        caller.gm = MagicMock()
        caller.input_nodes = []
        caller.name = "candidate"

        graph = MagicMock()
        graph.operations = []
        graph.buffers = []
        graph.name_to_op = {}
        graph.name_to_buffer = {}
        graph.env = {"old": object()}
        graph.removed_buffers = set()
        graph.removed_operations = set()
        new_operation = MagicMock(operation_name="op0")
        new_buffer = MagicMock(name="buf0")

        def inline(*args, **kwargs):
            graph.operations.append(new_operation)
            graph.buffers.append(new_buffer)
            graph.name_to_op["op0"] = new_operation
            graph.name_to_buffer["buf0"] = new_buffer
            return "output"

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.codegen.subgraph.inline_subgraph_to_ir_nodes",
                side_effect=inline,
            ),
            self.assertRaisesRegex(RuntimeError, "reject candidate"),
        ):
            with caller.speculative_inline():
                raise RuntimeError("reject candidate")

        self.assertEqual(graph.operations, [])
        self.assertEqual(graph.buffers, [])
        self.assertEqual(graph.name_to_op, {})
        self.assertEqual(graph.name_to_buffer, {})
        self.assertEqual(set(graph.env), {"old"})

    def test_benchmark_fused_nodes_preserves_flat_backend_contract(self):
        scheduler = object.__new__(Scheduler)
        backend = MagicMock()
        backend.benchmark_fused_nodes.return_value = (1.25, "flat.py")
        scheduler.get_backend = MagicMock(return_value=backend)
        node = MagicMock()
        node.get_device.return_value = torch.device("cuda")

        result = scheduler.benchmark_fused_nodes([node])

        self.assertEqual(result, (1.25, "flat.py"))
        backend.benchmark_fused_nodes.assert_called_once_with([node])

    def test_multi_kernel_fusion_plan_choice_owns_ordered_groups(self):
        caller = object.__new__(MultiKernelFusionPlanChoice)
        caller.gm = MagicMock()
        caller.input_nodes = []
        caller.name = "candidate"
        first_operation = MagicMock(operation_name="op0")
        second_operation = MagicMock(operation_name="op1")
        first_buffer = MagicMock(name="buf0")
        second_buffer = MagicMock(name="buf1")
        caller.group_builder = lambda plan: (
            (plan.operations[0],),
            (plan.operations[1],),
        )

        graph = MagicMock()
        graph.operations = []
        graph.buffers = []
        graph.name_to_op = {}
        graph.name_to_buffer = {}
        graph.env = {}
        graph.removed_buffers = set()
        graph.removed_operations = set()

        def inline(*args, **kwargs):
            graph.operations.extend((first_operation, second_operation))
            graph.buffers.extend((first_buffer, second_buffer))
            graph.name_to_op.update(op0=first_operation, op1=second_operation)
            graph.name_to_buffer.update(buf0=first_buffer, buf1=second_buffer)
            return "output"

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.codegen.subgraph.inline_subgraph_to_ir_nodes",
                side_effect=inline,
            ),
        ):
            with caller.speculative_fusion_plan() as plan:
                self.assertEqual(
                    plan.operation_groups,
                    ((first_operation,), (second_operation,)),
                )

        self.assertEqual(graph.operations, [])
        self.assertEqual(graph.buffers, [])

    def test_multi_kernel_fusion_plan_benchmark_restores_scheduler_state(self):
        scheduler = object.__new__(Scheduler)
        scheduler.current_device = torch.device("cpu")
        original_name_to_buf = {"original": MagicMock()}
        scheduler.name_to_buf = original_name_to_buf

        workspace = MagicMock()
        workspace.get_name.return_value = "workspace"
        output = MagicMock()
        output.get_name.return_value = "output"
        workspace_read = MagicMock()
        workspace_read.name = "workspace"

        first_node = MagicMock()
        first_node.get_outputs.return_value = [workspace]
        first_node.read_writes.reads = set()
        second_node = MagicMock()
        second_node.get_outputs.return_value = [output]
        second_node.read_writes.reads = {workspace_read}

        first_operation = MagicMock()
        second_operation = MagicMock()
        plan = MagicMock()
        plan.buffers = []
        plan.operation_groups = ((first_operation,), (second_operation,))
        plan_output = MagicMock()
        plan_output.get_name.return_value = "output"
        plan.output = plan_output
        context = MagicMock()
        context.__enter__.return_value = plan

        choice = MagicMock()
        choice.input_nodes = []
        choice.speculative_fusion_plan.return_value = context
        choice._benchmark_callable.side_effect = lambda fn, *args: (fn(), 0.75)[1]
        choice.annotations = {}

        scheduler.create_scheduler_node = MagicMock(
            side_effect=(first_node, second_node)
        )

        def benchmark_group(group):
            scheduler.current_device = torch.device("cuda")
            return 1.0, f"group-{len(group)}.py"

        scheduler.benchmark_fused_nodes = MagicMock(side_effect=benchmark_group)
        first_module = MagicMock()
        first_module.benchmark_artifact_kind = "triton"
        first_module.arg_names = ("workspace",)
        first_module.get_args.return_value = (torch.empty(1),)
        second_module = MagicMock()
        second_module.benchmark_artifact_kind = "triton"
        second_module.arg_names = ("workspace", "output")
        second_module.get_args.return_value = (torch.empty(1), torch.empty(1))
        graph = MagicMock()
        graph.scheduler = scheduler
        graph.current_device = torch.device("cpu")

        with (
            V.set_graph_handler(graph),
            mock.patch.object(
                torch._inductor.scheduler.PyCodeCache,
                "load_by_key_path",
                side_effect=(first_module, second_module),
            ),
        ):
            result = scheduler.benchmark_multi_kernel_fusion_plan(
                choice, [], MagicMock()
            )

        self.assertEqual(result, 0.75)
        self.assertEqual(graph.current_device, torch.device("cpu"))
        self.assertEqual(scheduler.current_device, torch.device("cpu"))
        self.assertIs(scheduler.name_to_buf, original_name_to_buf)
        self.assertEqual(
            choice.annotations["fusion_plan_workspace_names"], ("workspace",)
        )
        self.assertEqual(
            choice.annotations["fusion_plan_complete_artifacts"],
            ("group-1.py", "group-1.py"),
        )
        self.assertEqual(
            choice.annotations["fusion_plan_allocation_names"],
            ("workspace", "output"),
        )
        self.assertEqual(scheduler.benchmark_fused_nodes.call_count, 2)
        first_module.call.assert_called_once()
        second_module.call.assert_called_once()
        self.assertIs(
            first_module.call.call_args.args[0][0],
            second_module.call.call_args.args[0][0],
        )
        context.__exit__.assert_called_once_with(None, None, None)

    def test_subgraph_decompose_k(self):
        from torch._inductor.kernel.decompose_k import DecomposeKSubgraphTemplate
        from torch._inductor.kernel.mm import aten_mm
        from torch._inductor.kernel.mm_common import mm_args

        mat1_shape, mat2_shape = (32, 4096), (4096, 32)

        @torch.library.custom_op("mylib::matmul_decompose", mutates_args={})
        def matmul_decompose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            return a @ b

        @register_lowering(torch.ops.mylib.matmul_decompose)
        def _(a, b):
            _, _, _, layout, mat1, mat2 = mm_args(a, b)

            choices = [aten_mm.bind((mat1, mat2), layout)]

            kPartitions = 256

            decompose_k_subgraph_template = DecomposeKSubgraphTemplate()

            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                k_split=kPartitions,
                input_nodes=(mat1, mat2),
                layout=layout,
            )

            # Test benchmarking against aten
            autotune_select_algorithm("test_subgraph_choice", choices, [a, b], layout)

            # Only return decomposeK case for codegen
            choices = [choices[1]]
            node, _ = autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )
            return node

        a_in = torch.randn(
            mat1_shape, dtype=torch.float16, device=torch.device(f"{GPU_TYPE}:0")
        )
        b_in = torch.randn(
            mat2_shape, dtype=torch.float16, device=torch.device(f"{GPU_TYPE}:0")
        )

        def func(mat1, mat2):
            return torch.ops.mylib.matmul_decompose(mat1, mat2)

        compiled_func = torch.compile(func, mode="max-autotune", dynamic=False)

        res = compiled_func(a_in, b_in)

        # Check same results of compiled result and regular torch.mm
        torch.testing.assert_close(res, a_in @ b_in, atol=1e-1, rtol=1e-1)

    def test_subgraph_freeze_layout(self):
        from torch._inductor.kernel.decompose_k import DecomposeKSubgraphTemplate
        from torch._inductor.kernel.mm_common import mm_args

        M, N, K = (4, 128, 14240)
        a_in = torch.randn(
            (M, K), dtype=torch.bfloat16, device=torch.device(f"{GPU_TYPE}:0")
        )
        b_in = torch.randn(
            (K, N), dtype=torch.bfloat16, device=torch.device(f"{GPU_TYPE}:0")
        )

        @torch.library.custom_op("mylib::matmul_decompose_padding", mutates_args={})
        def matmul_decompose(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
            return a @ b

        @matmul_decompose.register_fake
        def _(a, b):
            return a @ b

        @register_lowering(torch.ops.mylib.matmul_decompose_padding)
        def _(a, b):
            _, _, _, layout, mat1, mat2 = mm_args(a, b)
            mat1_layout = mat1.layout
            if not isinstance(mat1_layout, FlexibleLayout):
                raise AssertionError
            mat1_stride = mat1_layout.stride

            choices = []

            kPartitions = 2

            decompose_k_subgraph_template = DecomposeKSubgraphTemplate()

            decompose_k_subgraph_template.maybe_append_choice(
                choices,
                k_split=kPartitions,
                input_nodes=(mat1, mat2),
                layout=layout,
            )

            choice = choices[0]
            if not isinstance(mat1.layout, FixedLayout):
                raise AssertionError

            # Creating the subgraph choice should have frozen the layout
            # We ensure padding so the stride should differ
            if mat1.layout.stride == mat1_stride:
                raise AssertionError

            for example_stride, layout_stride in zip(
                choice.example_inputs[0].stride(), mat1.layout.stride
            ):
                # Example inputs should have same stride as current layout
                if example_stride != layout_stride:
                    raise AssertionError

            node, _ = autotune_select_algorithm(
                "test_subgraph_choice", choices, [a, b], layout
            )
            return node

        def func(mat1, mat2):
            return torch.ops.mylib.matmul_decompose_padding((mat1 + 1.0), mat2)

        with mock.patch("torch._inductor.ir.V.get_current_node") as get_node_mock:
            node_mock = MagicMock()
            node_mock.meta = {"dislike_padding": False}
            get_node_mock.return_value = node_mock

            compiled_func = torch.compile(func, mode="max-autotune", dynamic=False)

            compiled_func(a_in, b_in)


@unittest.skipUnless(
    HAS_GPU and SM100OrLater,
    "requires NVIDIA SM100+",
)
class TestBlackwellDecomposeKSubgraphChoice(TestCase):
    def _run_forced_triton_plan(
        self, two_ctas: bool, *, use_meta_ws: bool = True, m: int = 256
    ) -> None:
        config_index = 1 if two_ctas else 0
        partial_config = BLACKWELL_DECOMPOSE_K_PARTIAL_CONFIGS[config_index]
        effective_two_ctas = use_meta_ws and two_ctas

        def lowering(a, b, two_ctas_arg):
            if bool(two_ctas_arg) != two_ctas:
                raise AssertionError("unexpected 2CTA specialization")
            m, k = map(int, a.get_size())
            m_tiles = math.ceil(m / partial_config.block_m)
            if effective_two_ctas:
                m_tiles = math.ceil(m_tiles / 2) * 2
            m_pad = m_tiles * partial_config.block_m
            k_part = (
                math.ceil(math.ceil(k / BLACKWELL_K_SPLIT) / partial_config.block_k)
                * partial_config.block_k
            )
            return lower_blackwell_decompose_k_partial(
                a,
                b,
                BLACKWELL_K_SPLIT,
                config_index,
                m_pad,
                k_part,
            )

        k, n = 8193, 128
        a = torch.randn(k, m, device=GPU_TYPE, dtype=torch.bfloat16).T
        b = torch.randn(k, n, device=GPU_TYPE, dtype=torch.bfloat16)

        def fn(x, y):
            partial = blackwell_decompose_k_partial(x, y, two_ctas)
            return partial.view(BLACKWELL_K_SPLIT, m, n).sum(0).to(torch.bfloat16)

        with (
            mock.patch.dict(
                lowerings,
                {
                    torch.ops.inductor_test.blackwell_decompose_k_partial.default: lowering
                },
            ),
            mock.patch(
                "torch._inductor.kernel.decompose_k.meta_ws_enabled",
                return_value=use_meta_ws,
            ),
            mock.patch("torch._inductor.kernel.decompose_k.USE_META_WS", use_meta_ws),
            config.patch(
                compile_threads=1,
                **{"triton.enable_template_tma_store": True},
            ),
        ):
            actual, codes = run_and_get_code(torch.compile(fn, fullgraph=True), a, b)

        torch.testing.assert_close(actual, a @ b, atol=16.0, rtol=1e-1)
        source = "\n".join(codes)
        self.assertIn("make_tensor_descriptor", source)
        self.assertIn(f"BATCH_SIZE : tl.constexpr = {BLACKWELL_K_SPLIT}", source)
        self.assertEqual("USE_META_WS : tl.constexpr = True" in source, use_meta_ws)
        self.assertEqual("FLATTEN : tl.constexpr = True" in source, not use_meta_ws)
        self.assertEqual("TWO_CTAS : tl.constexpr = True" in source, effective_two_ctas)

    def test_forced_triton_1cta(self):
        self._run_forced_triton_plan(False)

    def test_forced_triton_2cta(self):
        self._run_forced_triton_plan(True)

    def test_forced_triton_2cta_config_without_meta_ws(self):
        # One M tile distinguishes the effective 1CTA geometry (M_PAD=128)
        # from the 2CTA cluster geometry (M_PAD=256).
        self._run_forced_triton_plan(True, use_meta_ws=False, m=128)

    def _run_backend_selection(
        self, outer_backends: str, nested_backends: str | None
    ) -> str:
        m, k, n = 256, 131072, 128
        a = torch.randn(k, m, device=GPU_TYPE, dtype=torch.bfloat16).T
        b = torch.randn(k, n, device=GPU_TYPE, dtype=torch.bfloat16)
        patch = {
            "max_autotune_gemm": True,
            "max_autotune_gemm_backends": outer_backends,
            "compile_threads": 1,
            "assume_aligned_inputs": True,
            "triton.enable_template_tma_store": True,
            "triton.enable_persistent_tma_matmul": True,
            "triton.enable_blackwell_decompose_k": True,
            "triton.num_decompose_k_splits": 4,
            "triton.disallow_failing_autotune_kernels_TESTING_ONLY": True,
        }
        if nested_backends is not None:
            patch["triton.decompose_k_bmm_backends"] = nested_backends

        with config.patch(patch):
            actual, codes = run_and_get_code(
                torch.compile(lambda x, y: x @ y, fullgraph=True), a, b
            )

        torch.testing.assert_close(actual, a @ b, atol=16.0, rtol=1e-1)
        return "\n".join(codes)

    def test_outer_aten_only_excludes_decompose_k(self):
        source = self._run_backend_selection("ATEN", "ATEN,TRITON")
        self.assertNotIn("_split_aten", source)
        self.assertNotIn("_split_triton_config_", source)

    def test_default_nested_backend_is_aten(self):
        source = self._run_backend_selection("TRITON", None)
        self.assertIn("_split_aten", source)
        self.assertNotIn("_split_triton_config_", source)

    def test_nested_triton_backend(self):
        source = self._run_backend_selection("TRITON", "triton")
        self.assertNotIn("_split_aten", source)
        self.assertIn("_split_triton_config_", source)

    def test_mixed_backend_plan_enumeration(self):
        source = self._run_backend_selection("TRITON", "aten, TRITON")
        self.assertIn("_split_aten", source)
        self.assertIn("_split_triton_config_", source)

    def test_complete_plan_forced_triton_codegen(self):
        m, k, n = 256, 8193, 128
        a = torch.randn(k, m, device=GPU_TYPE, dtype=torch.bfloat16).T
        b = torch.randn(k, n, device=GPU_TYPE, dtype=torch.bfloat16)
        decompose_k = torch._dynamo.dont_skip_tracing(blackwell_decomposeK)
        with config.patch(
            compile_threads=1,
            **{
                "triton.enable_template_tma_store": True,
                "triton.enable_persistent_tma_matmul": True,
            },
        ):
            actual, codes = run_and_get_code(
                torch.compile(
                    lambda x, y: decompose_k(x, y, 33, "triton", 0),
                    fullgraph=True,
                ),
                a,
                b,
            )
        torch.testing.assert_close(actual, a @ b, atol=16.0, rtol=1e-1)
        source = "\n".join(codes)
        self.assertIn("blackwell_decompose_k_partial", source)
        self.assertIn("BATCH_SIZE : tl.constexpr = 33", source)
        self.assertNotIn("extern_kernels.bmm_dtype", source)


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU:
        run_tests()
