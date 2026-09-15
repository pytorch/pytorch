# Owner(s): ["module: inductor"]
from unittest import mock
from unittest.mock import MagicMock

import torch
from torch._inductor.codegen.subgraph import (
    MultiKernelFusionPlanChoice,
    SubgraphChoiceCaller,
)
from torch._inductor.ir import Buffer, FixedLayout, FlexibleLayout
from torch._inductor.lowering import register_lowering
from torch._inductor.scheduler import Scheduler
from torch._inductor.select_algorithm import autotune_select_algorithm
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
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

            decompose_k_subgraph_template = (
                torch._inductor.kernel.mm.DecomposeKSugraphTemplate()
            )

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

            decompose_k_subgraph_template = (
                torch._inductor.kernel.mm.DecomposeKSugraphTemplate()
            )

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


if __name__ == "__main__":
    # Set env to make it work in CI.
    if HAS_GPU and HAS_CPU:
        run_tests()
