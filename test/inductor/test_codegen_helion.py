# Owner(s): ["module: inductor"]
from __future__ import annotations

import ast
import math
import unittest
from unittest.mock import Mock, patch

import sympy

import torch
from torch._inductor import config, ir
from torch._inductor.codegen.helion import HelionKernel, HelionValue
from torch._inductor.codegen.triton import TritonScheduling
from torch._inductor.decomposition import select_decomp_table
from torch._inductor.graph import GraphLowering
from torch._inductor.sizevars import SizeVarAllocator
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_device_type import (
    dtypes,
    instantiate_device_type_tests,
)
from torch.testing._internal.common_utils import (
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON
from torch.utils._sympy.functions import FloorDiv, ModularIndexing


class CodegenComplete(Exception):
    pass


@unittest.skipUnless(HAS_CUDA_AND_TRITON, "requires CUDA scheduling and Triton")
@instantiate_parametrized_tests
class TestHelionCodegen(TestCase):
    hw_classification = HardwareClassification.CUDA

    def codegen(
        self,
        fn,
        shapes,
        *,
        strides=None,
        dtypes=None,
        helion_code=True,
        schedules=None,
        decompose=False,
        complete=False,
    ):
        fake_mode = FakeTensorMode(allow_fallback_kernels=False)
        with fake_mode:
            inputs = [
                torch.empty(shape, device="cuda", dtype=dtype)
                if strides is None
                else torch.empty_strided(shape, stride, device="cuda", dtype=dtype)
                for shape, stride, dtype in zip(
                    shapes,
                    strides or [None] * len(shapes),
                    dtypes or [torch.bfloat16] * len(shapes),
                )
            ]
            gm = make_fx(
                fn,
                decomposition_table=select_decomp_table() if decompose else None,
                tracing_mode="fake",
            )(*inputs)
        debug = Mock()
        with (
            config.patch(
                {
                    "triton.native_matmul": True,
                    "trace.helion_code": helion_code,
                    "benchmark_fusion": False,
                    "max_autotune": False,
                    "max_autotune_gemm": False,
                    "graph_partition": False,
                }
            ),
            V.set_fake_mode(fake_mode),
            V.set_debug_handler(debug),
            patch.object(
                TritonScheduling, "create_kernel_choices", side_effect=CodegenComplete
            ),
        ):
            graph = GraphLowering(gm, example_inputs=inputs)
            with V.set_graph_handler(graph), V.set_extern_kernel_nodes([]):
                graph.run(*inputs)
                codegen_schedule = TritonScheduling.codegen_node_schedule

                def preview_native(scheduling, features):
                    if schedules is not None:
                        # Cache classification while native_matmul is patched on.
                        features.is_native_matmul()
                        schedules.append(features)
                    if features.is_native_matmul():
                        try:
                            return codegen_schedule(scheduling, features)
                        except CodegenComplete:
                            if not complete:
                                raise
                    for node in features.scheduler_nodes():
                        node.mark_run()

                with (
                    patch.object(
                        TritonScheduling, "codegen_node_schedule", preview_native
                    ),
                ):
                    try:
                        graph.codegen()
                    except CodegenComplete:
                        pass
        sources = [call.args[0] for call in debug.helion_code.call_args_list]
        for source in sources:
            compile(ast.parse(source), "<generated_helion>", "exec")
            self.assertNotIn(".view(-1)", source)
            self.assertNotIn(".reshape(-1)", source)
        return sources

    @parametrize("dtype", [torch.float16, torch.bfloat16])
    def test_mm(self, dtype):
        (source,) = self.codegen(
            lambda a, b: a @ b, [(40, 72), (72, 136)], dtypes=[dtype, dtype]
        )
        self.assertIn("A[tile_y, tile_r]", source)
        self.assertIn("B[tile_r, tile_x]", source)
        self.assertIn("acc = torch.addmm(acc, tmp1, tmp3)", source)
        self.assertIn("out0[tile_y, tile_x] = acc.to(", source)
        self.assertIn("helion_op0(a_1, b_1, buf0)", source)
        self.assertNotIn("reinterpret_tensor", source)

    @parametrize(
        "axes",
        [
            ("z",),
            ("y",),
            ("r0_",),
            ("x",),
            ("y", "x"),
            ("y", "r0_"),
            ("r0_", "x"),
            ("z", "y", "r0_"),
            ("z", "r0_", "x"),
            ("z", "y", "r0_", "x"),
        ],
    )
    def test_index_expr_axes(self, axes):
        from types import SimpleNamespace

        features = Mock()
        features.scheduler_nodes.return_value = ()
        features.op_counts.return_value = {"dot": 0}
        tiling = {
            axis: sympy.Integer(size)
            for axis, size in zip(("z", "y", "x", "r0_"), (2, 3, 5, 4))
        }
        with V.set_graph_handler(
            Mock(helion_matmul_inputs={}, sizevars=SizeVarAllocator())
        ):
            kernel = HelionKernel(features, tiling)
            expression = sum(
                (i + 1) * kernel.symbols[axis] for i, axis in enumerate(axes)
            )
            value = kernel.index_expr(expression, torch.int64)
        self.assertEqual(value.depending_indices, axes)
        scope = {"torch": torch}
        vectors = []
        for axis in axes:
            vector = torch.arange(int(tiling[axis]))
            vectors.append(vector)
            scope[kernel.tiles[axis]] = SimpleNamespace(index=vector)
        exec(kernel.reduction_body.getvalue(), scope)
        coordinates = torch.meshgrid(*vectors, indexing="ij")
        expected = sum((i + 1) * coord for i, coord in enumerate(coordinates))
        self.assertEqual(scope[value.name], expected)

    def test_index_expr_mm_epilogue(self):
        (source,) = self.codegen(
            lambda a, b: a @ b
            + torch.arange(40 * 136, device=a.device).reshape(40, 136),
            [(40, 72), (72, 136)],
            decompose=True,
        )
        self.assertIn("tile_y.index[:, None]", source)
        self.assertIn("tile_x.index[None, :]", source)

    def test_index_expr_bmm_epilogue(self):
        (source,) = self.codegen(
            lambda a, b: torch.bmm(a, b)
            + torch.arange(3 * 40 * 136, device=a.device).reshape(3, 40, 136),
            [(3, 40, 72), (3, 72, 136)],
            decompose=True,
        )
        self.assertIn("tile_z.index[:, None, None]", source)
        self.assertIn("tile_y.index[None, :, None]", source)
        self.assertIn("tile_x.index[None, None, :]", source)

    @parametrize("batched", [False, True])
    @parametrize(
        "axes", [("y", "x"), ("y", "r0_"), ("r0_", "x"), ("y",), ("x",), ("r0_",)]
    )
    def test_pointwise_axis_order(self, batched, axes):
        tiling = dict(zip(("y", "x", "r0_"), map(sympy.Integer, (3, 5, 4))))
        if batched:
            tiling = {"z": sympy.Integer(2), **tiling}
            axes = ("z", *axes)
        features = Mock()
        features.scheduler_nodes.return_value = ()
        features.op_counts.return_value = {}
        with V.set_graph_handler(Mock(helion_matmul_inputs={})):
            kernel = HelionKernel(features, tiling)
        scope = {"torch": torch, "zero": torch.zeros((), dtype=torch.int64)}
        scope.update({kernel.tiles[axis]: int(size) for axis, size in tiling.items()})
        value = HelionValue("zero", ())
        vectors = {axis: torch.arange(int(tiling[axis])) for axis in axes}
        for axis in reversed(axes):
            name = f"v_{axis}"
            scope[name] = vectors[axis]
            value = kernel._default("add", (value, HelionValue(name, (axis,))), {})
        self.assertEqual(value.depending_indices, axes)
        source = kernel.reduction_body.getvalue()
        self.assertNotIn(".permute(", source)
        exec(source, scope)
        expected = sum(torch.meshgrid(*vectors.values(), indexing="ij"))
        self.assertEqual(scope[value.name], expected)

    def test_bias_relu(self):
        schedules = []
        (source,) = self.codegen(
            lambda a, b, bias: torch.relu(a @ b + bias),
            [(40, 72), (72, 136), (136,)],
            schedules=schedules,
        )
        self.assertEqual(len(schedules), 1)
        self.assertGreater(len(tuple(schedules[0].scheduler_nodes())), 1)
        self.assertIn("in0[tile_x].to(torch.float32)", source)
        self.assertIn("tmp5 = acc + tmp4", source)
        self.assertIn("tmp6 = torch.relu(tmp5)", source)
        self.assertNotIn("hl.load(buf", source)

    @parametrize(
        "op",
        [
            torch.erf,
            torch.tanh,
            torch.neg,
            torch.sigmoid,
            torch.exp,
            torch.expm1,
            torch.abs,
            torch.log,
        ],
    )
    def test_unary_epilogue(self, op):
        (source,) = self.codegen(lambda a, b: op(a @ b), [(40, 72), (72, 136)])
        self.assertIn(f"torch.{op.__name__}(", source)

    def test_logical_not_epilogue(self):
        (source,) = self.codegen(
            lambda a, b: torch.logical_and(torch.logical_not(a @ b), a @ b > 0),
            [(40, 72), (72, 136)],
        )
        self.assertIn("torch.logical_not(", source)
        self.assertIn("torch.logical_and(", source)

    def test_bitwise_and_epilogue(self):
        def fn(a, b):
            index = torch.arange(136, device=a.device)
            return a @ b + torch.bitwise_and(index, 7)

        (source,) = self.codegen(fn, [(40, 72), (72, 136)], decompose=True)
        self.assertIn(" & ", source)

    def test_masked_epilogue_load(self):
        def fn(a, b, first, second):
            return a @ b + torch.cat((first, second))

        (source,) = self.codegen(
            fn, [(40, 72), (72, 136), (64,), (72,)], decompose=True
        )
        self.assertIn("hl.load(", source)
        self.assertIn("extra_mask=", source)
        self.assertIn("torch.full_like(", source)
        self.assertIn("torch.where(", source)

    def test_where_epilogue_broadcast(self):
        def fn(a, b, bias):
            result = a @ b
            return torch.where(result <= bias, result, bias)

        (source,) = self.codegen(fn, [(40, 72), (72, 136), (136,)])
        self.assertIn(" <= ", source)
        self.assertIn("torch.where(", source)
        self.assertNotIn(".reshape(", source)

    @parametrize("block,count", [(3, 5), (8, 7), (64, 12)])
    def test_index_expr_modulo_cancellation(self, block, count):
        features = Mock()
        features.scheduler_nodes.return_value = ()
        features.op_counts.return_value = {}
        tiling = dict(
            zip(("y", "x", "r0_"), map(sympy.Integer, (17, block * count, 16)))
        )
        with V.set_graph_handler(
            Mock(helion_matmul_inputs={}, sizevars=SizeVarAllocator())
        ):
            kernel = HelionKernel(features, tiling)
            x, y = kernel.symbols["x"], kernel.symbols["y"]
            index = ModularIndexing(x, 1, block) + block * ModularIndexing(
                FloorDiv(x, block) + count * ModularIndexing(y, 1, 4), 1, count
            )
            value = kernel.index_expr(index, torch.int64)
        self.assertEqual(value.depending_indices, ("x",))
        source = kernel.reduction_body.getvalue()
        self.assertNotIn("tile_y", source)
        self.assertNotIn("%", source)
        self.assertNotIn("//", source)

    @parametrize("op", [sympy.Min, sympy.Max])
    def test_index_expr_tensor_min_max(self, op):
        features = Mock()
        features.scheduler_nodes.return_value = ()
        features.op_counts.return_value = {}
        tiling = dict(zip(("y", "x", "r0_"), map(sympy.Integer, (40, 136, 72))))
        with V.set_graph_handler(
            Mock(helion_matmul_inputs={}, sizevars=SizeVarAllocator())
        ):
            kernel = HelionKernel(features, tiling)
            value = kernel.index_expr(
                op(0, kernel.symbols["y"] - kernel.symbols["x"]), torch.int64
            )
        source = kernel.reduction_body.getvalue()
        self.assertEqual(value.depending_indices, ("y", "x"))
        name = "minimum" if op is sympy.Min else "maximum"
        self.assertIn(f"torch.{name}(", source)
        self.assertNotIn(f"{op.__name__.lower()}(", source)

    def test_explicit_index_broadcast_rank(self):
        (source,) = self.codegen(
            lambda a, b, bias: (a @ b).reshape(5, 8, 136) + bias.transpose(1, 2),
            [(40, 72), (72, 136), (5, 136, 8)],
            decompose=True,
        )
        self.assertIn("tile_x.index.reshape([1, tile_x])", source)

    @parametrize("row_bias", [False, True])
    def test_bmm_broadcast_singleton_view(self, row_bias):
        shape = (2, 1, 40, 1 if row_bias else 136)
        (source,) = self.codegen(
            lambda a, b, bias: torch.bmm(a, b).reshape(2, 3, 40, 136) + bias,
            [(6, 40, 72), (6, 72, 136), shape],
            decompose=True,
        )
        self.assertIn("reinterpret_tensor(bias_1,", source)
        indices = [
            n.slice
            for n in ast.walk(ast.parse(source))
            if isinstance(n, ast.Subscript)
            and isinstance(n.value, ast.Name)
            and n.value.id == "in0"
        ]
        for index in indices:
            coordinates = index.elts if isinstance(index, ast.Tuple) else [index]
            self.assertEqual(len(coordinates), 2 if row_bias else 3)
            self.assertFalse(
                any(isinstance(c, ast.Constant) and c.value == 0 for c in coordinates)
            )

    @config.patch(comprehensive_padding=False)
    def test_bmm_affine_permuted_output_view(self):
        def fn(a, b):
            result = torch.bmm(a, b).reshape(2, 3, 40, 136)
            permuted = result.permute(3, 0, 1, 2).unsqueeze(2)
            return permuted.clone(memory_format=torch.contiguous_format)

        (source,) = self.codegen(fn, [(6, 40, 72), (6, 72, 136)], decompose=True)
        self.assertIn("out0[tile_z, tile_y, tile_x]", source)
        self.assertIn("(6, 40, 136), (40, 1, 240), 0)", source)
        self.assertNotIn("%", source)
        self.assertNotIn("//", source)

    @parametrize("axis", ["row", "column", "matrix"])
    def test_addmm_epilogue_axis_order(self, axis):
        shapes = {"row": (40, 1), "column": (136,), "matrix": (40, 136)}
        (source,) = self.codegen(
            lambda bias, a, b: torch.addmm(bias, a, b, alpha=1.25, beta=0.75),
            [shapes[axis], (40, 72), (72, 136)],
            decompose=True,
        )
        self.assertNotIn(".permute(", source)
        if axis == "row":
            self.assertIn(".reshape([tile_y, 1])", source)
        else:
            self.assertNotIn(".reshape(", source)

    @parametrize("batched", [False, True])
    @parametrize("op", [torch.add, torch.sub, torch.mul, torch.maximum])
    def test_bias_first_epilogue_axis_order(self, batched, op):
        def fn(a, b, bias):
            result = torch.bmm(a, b) if batched else a @ b
            return op(bias, result)

        shapes = [(40, 72), (72, 136)]
        if batched:
            shapes = [(3, *shape) for shape in shapes]
        (source,) = self.codegen(fn, [*shapes, (40, 136)])
        self.assertNotIn(".permute(", source)
        self.assertNotIn(".reshape(", source)

    @parametrize("axis", ["row", "column", "matrix"])
    def test_epilogue_broadcast(self, axis):
        shapes = {"row": (40,), "column": (136,), "matrix": (40, 136)}

        def fn(a, b, bias):
            return a @ b + (bias[:, None] if axis == "row" else bias)

        (source,) = self.codegen(fn, [(40, 72), (72, 136), shapes[axis]])
        coordinates = {"row": "tile_y", "column": "tile_x", "matrix": "tile_y, tile_x"}
        self.assertIn(f"in0[{coordinates[axis]}]", source)

    def test_nd_metadata_view(self):
        schedules = []
        (source,) = self.codegen(
            lambda a, b: a.reshape(33, 72) @ b,
            [(33, 9, 8), (72, 136)],
            schedules=schedules,
        )
        self.assertEqual(len(schedules), 1)
        self.assertIn("reinterpret_tensor(a_1, (33, 72), (72, 1), 0)", source)
        self.assertIn("A[tile_y, tile_r]", source)
        self.assertNotIn("//", source)
        self.assertNotIn("%", source)

    def test_nd_view_transpose_offset(self):
        (source,) = self.codegen(
            lambda a, bt: a[1:].reshape(33, 72) @ bt.T,
            [(34, 9, 8), (136, 72)],
        )
        self.assertIn("reinterpret_tensor(a_1, (33, 72), (72, 1), 72)", source)
        self.assertIn("reinterpret_tensor(bt_1, (72, 136), (1, 72), 0)", source)
        self.assertIn("B[tile_r, tile_x]", source)
        self.assertNotIn("//", source)
        self.assertNotIn("%", source)

    def test_caller_metadata_view(self):
        (source,) = self.codegen(
            lambda a, b: a[1:].reshape(33, 72) @ b,
            [(34, 9, 8), (72, 136)],
        )
        caller = next(
            node
            for node in ast.parse(source).body
            if isinstance(node, ast.FunctionDef) and node.name.startswith("call_")
        )
        kernel_name = caller.name.removeprefix("call_")
        kernel = Mock()
        namespace = {
            "torch": torch,
            "reinterpret_tensor": torch._C._dynamo.guards._reinterpret_tensor,
            kernel_name: kernel,
        }
        exec(
            compile(ast.Module(body=[caller], type_ignores=[]), "<caller>", "exec"),
            namespace,
        )
        a = torch.empty(34, 9, 8, dtype=torch.bfloat16)
        b = torch.empty(72, 136, dtype=torch.bfloat16)
        out = torch.empty(33, 136, dtype=torch.bfloat16)
        namespace[caller.name](a, b, out)
        lhs, rhs, result = kernel.call_args.args
        self.assertEqual(lhs.shape, (33, 72))
        self.assertEqual(lhs.stride(), (72, 1))
        self.assertEqual(lhs.storage_offset(), 72)
        self.assertEqual(lhs.untyped_storage()._cdata, a.untyped_storage()._cdata)
        self.assertIs(rhs, b)
        self.assertIs(result, out)

    @parametrize("batched", [False, True])
    @parametrize("layout", ["contiguous", "strided", "permuted", "higher_rank"])
    @config.patch(comprehensive_padding=False)
    def test_folded_epilogue_inputs_outputs(self, batched, layout):
        split = (2, 5, 4) if layout == "higher_rank" else (5, 8)
        matrix_shape = (*split, 136)
        matrix_stride = list(ir.FlexibleLayout.contiguous_strides(matrix_shape))
        if layout == "strided":
            matrix_stride = [8 * 144, 144, 1]
        elif layout == "permuted":
            matrix_shape = (8, 5, 136)
            matrix_stride = [136, 8 * 136, 1]

        def fn(a, b, matrix, row):
            result = torch.bmm(a, b) if batched else a @ b
            shape = (3, *split, 136) if batched else (*split, 136)
            if layout == "permuted":
                matrix = matrix.transpose(-3, -2)
            return (result.reshape(shape) + matrix) * row

        shapes = [(40, 72), (72, 136), matrix_shape, (*split, 1)]
        row_stride = list(ir.FlexibleLayout.contiguous_strides(shapes[-1]))
        strides = [[72, 1], [136, 1], matrix_stride, row_stride]
        if batched:
            shapes = [(3, *shape) for shape in shapes]
            strides = [
                [math.prod(shape[1:]), *stride]
                for shape, stride in zip(shapes, strides)
            ]
            if layout == "strided":
                strides[2][0] = 40 * 144
        (source,) = self.codegen(fn, shapes, strides=strides, decompose=True)
        axes = "tile_z, tile_y, tile_x" if batched else "tile_y, tile_x"
        row_axes = "tile_z, tile_y" if batched else "tile_y"
        self.assertIn(f"in0[{axes}]", source)
        self.assertIn(f"in1[{row_axes}]", source)
        self.assertIn(f"out0[{axes}]", source)
        self.assertIn("reinterpret_tensor(matrix_1,", source)
        self.assertIn("reinterpret_tensor(row_1,", source)
        self.assertIn("reinterpret_tensor(buf", source)
        self.assertNotIn("tile_y.index", source)
        self.assertNotIn("tile_x.index", source)

    def test_unfoldable_epilogue_input(self):
        with self.assertRaisesRegex(
            NotImplementedError, "Cannot recover Helion tensor indices"
        ):
            self.codegen(
                lambda a, b, bias: (a @ b).reshape(5, 8, 136) + bias,
                [(40, 72), (72, 136), (5, 8, 136)],
                strides=[(72, 1), (136, 1), (1200, 144, 1)],
                decompose=True,
            )

    def test_same_storage_two_operand_views(self):
        (source,) = self.codegen(lambda a: a @ a.T, [(40, 72)])
        self.assertIn("A[tile_y, tile_r]", source)
        self.assertIn("B[tile_r, tile_x]", source)
        self.assertIn("reinterpret_tensor(a_1, (72, 40), (1, 72), 0)", source)
        self.assertEqual(source.count("a_1: torch.Tensor"), 1)

    def test_strided_input(self):
        (source,) = self.codegen(
            lambda a, b: a @ b,
            [(40, 72), (72, 136)],
            strides=[(80, 1), (1, 80)],
        )
        self.assertIn("A[tile_y, tile_r]", source)
        self.assertIn("B[tile_r, tile_x]", source)
        self.assertNotIn("reinterpret_tensor", source)

    @parametrize("case", ["compute", "repeat", "gather", "reshape_copy"])
    def test_materialized_operand(self, case):
        if case == "compute":

            def fn(a, b):
                return (a.reshape(33, 72) + 1) @ (b - 2)

            shapes = [(33, 9, 8), (72, 136)]
            dtypes = None
        elif case == "repeat":

            def fn(a, b):
                return a.repeat(1, 9) @ b

            shapes = [(33, 8), (72, 136)]
            dtypes = None
        elif case == "gather":

            def fn(a, indices, b):
                return a[indices] @ b

            shapes = [(80, 72), (33,), (72, 136)]
            dtypes = [torch.bfloat16, torch.int64, torch.bfloat16]
        else:

            def fn(a, b):
                return a.permute(1, 0, 2).reshape(33, 64) @ b

            shapes = [(8, 33, 8), (64, 136)]
            dtypes = None
        schedules = []
        (source,) = self.codegen(fn, shapes, dtypes=dtypes, schedules=schedules)
        self.assertGreater(len(schedules), 1)
        native = schedules[-1]
        self.assertTrue(native.is_native_matmul())
        self.assertEqual(len(tuple(native.scheduler_nodes())), 1)
        self.assertIn("A[tile_y, tile_r]", source)
        self.assertIn("B[tile_r, tile_x]", source)
        self.assertIn("buf0: torch.Tensor", source)
        self.assertNotIn("a_1: torch.Tensor", source)
        self.assertNotIn("//", source)
        self.assertNotIn("%", source)
        self.assertNotIn("hl.full(", source)

    def test_producer_internal_fusion(self):
        schedules = []
        self.codegen(
            lambda a, b: ((a + 1) * 2) @ b,
            [(40, 72), (72, 136)],
            schedules=schedules,
        )
        self.assertEqual(len(schedules), 2)
        producer, native = schedules
        self.assertGreater(producer.op_counts().get("add", 0), 0)
        self.assertGreater(producer.op_counts().get("mul", 0), 0)
        self.assertEqual(len(tuple(native.scheduler_nodes())), 1)

    def test_triton_prologue_unchanged(self):
        schedules = []
        sources = self.codegen(
            lambda a, b: (a + 1) @ b,
            [(40, 72), (72, 136)],
            helion_code=False,
            schedules=schedules,
        )
        self.assertEqual(sources, [])
        self.assertEqual(len(schedules), 1)
        self.assertGreater(schedules[0].op_counts().get("add", 0), 0)

    def test_pointwise_not_selected(self):
        self.assertEqual(self.codegen(lambda a: a + 1, [(40, 72)]), [])

    @parametrize("epilogue", [False, True])
    def test_horizontal_mm(self, epilogue):
        def fn(a, b, c, d):
            first, second = a @ b, c @ d
            # Shared A makes this a horizontal candidate under the usual scheduler rules.
            return (
                torch.relu(first - second + first * second)
                if epilogue
                else (first, a @ d)
            )

        schedules = []
        (source,) = self.codegen(fn, [(40, 72), (72, 136)] * 2, schedules=schedules)
        self.assertEqual(len(schedules), 1)
        self.assertEqual(source.count(" = torch.addmm("), 2)
        self.assertIn("acc1 = hl.zeros(", source)
        self.assertIn("B1[tile_r, tile_x]", source)
        if epilogue:
            self.assertIn("A1[tile_y, tile_r]", source)
            self.assertIn("acc - acc1", source)
            self.assertIn("acc * acc1", source)
            self.assertIn("torch.relu(", source)

    def test_horizontal_then_epilogue_chain(self):
        (source,) = self.codegen(
            lambda a, b, c, d, e, f: torch.relu(a @ b + c @ d + e @ f),
            [(40, 72), (72, 136)] * 3,
        )
        self.assertEqual(source.count(" = torch.addmm("), 3)
        self.assertIn("acc2 = hl.zeros(", source)
        self.assertIn("torch.relu(", source)

    @parametrize("batched", [False, True])
    @parametrize("shared_lhs", [False, True])
    def test_horizontal_shared_load_cse(self, batched, shared_lhs):
        def fn(a, b, c):
            if batched:
                return (
                    torch.bmm(a, b) + torch.bmm(a, c)
                    if shared_lhs
                    else torch.bmm(a, c) + torch.bmm(b, c)
                )
            return a @ b + a @ c if shared_lhs else a @ c + b @ c

        lhs, rhs = (40, 72), (72, 136)
        shapes = [lhs, rhs, rhs] if shared_lhs else [lhs, lhs, rhs]
        if batched:
            shapes = [(3, *shape) for shape in shapes]
        (source,) = self.codegen(fn, shapes)
        op = "baddbmm" if batched else "addmm"
        self.assertEqual(source.count(f" = torch.{op}("), 2)
        indices = "tile_y, tile_r" if shared_lhs else "tile_r, tile_x"
        if batched:
            indices = "tile_z, " + indices
        argument = "A" if shared_lhs else "B"
        self.assertEqual(source.count(f"{argument}[{indices}]"), 1)

    def test_horizontal_distinct_operand_views(self):
        (source,) = self.codegen(
            lambda a, b: a[:-1] @ b + a[1:] @ b,
            [(41, 72), (72, 136)],
        )
        self.assertEqual(source.count(" = torch.addmm("), 2)
        self.assertIn("A[tile_y, tile_r]", source)
        self.assertIn("A1[tile_y, tile_r]", source)
        self.assertIn("reinterpret_tensor(a_1, (40, 72), (72, 1), 72)", source)

    def test_cse_scopes_and_atomic_invalidation(self):
        features = Mock()
        features.scheduler_nodes.return_value = []
        features.op_counts.return_value = {}
        kernel = HelionKernel(
            features,
            {
                axis: sympy.Integer(size)
                for axis, size in zip(("y", "x", "r0_"), (40, 136, 72))
            },
        )
        reduction_value = kernel.constant(1, torch.float32)
        kernel.body = kernel.epilogue
        epilogue_value = kernel.constant(1, torch.float32)
        self.assertNotEqual(reduction_value.name, epilogue_value.name)
        self.assertIs(kernel.constant(1, torch.float32), epilogue_value)

        graph = Mock(sizevars=SizeVarAllocator())
        target = ir.InputBuffer(
            name="buf0",
            layout=ir.FixedLayout(
                torch.device("cuda"), torch.float32, [40, 136], [136, 1]
            ),
        )
        with V.set_graph_handler(graph):
            mutation = ir.Buffer(
                name="mutation", layout=ir.MutationLayoutSHOULDREMOVE(target)
            )
            graph.get_buffer.side_effect = {
                "buf0": target,
                "mutation": mutation,
            }.__getitem__
            index = 136 * kernel.symbols["y"] + kernel.symbols["x"]
            before = kernel.load("buf0", index)
            self.assertIs(kernel.load("buf0", index), before)
            kernel.store("mutation", index, epilogue_value, mode="atomic_add")
            self.assertNotEqual(kernel.load("buf0", index).name, before.name)
        kernel.body = kernel.reduction_body
        self.assertIs(kernel.constant(1, torch.float32), reduction_value)

    @parametrize("horizontal", [False, True])
    def test_bmm_epilogue(self, horizontal):
        def fn(a, b, c, d, bias):
            first = torch.bmm(a, b)
            result = first - torch.bmm(c, d) if horizontal else first
            return torch.relu(result) + first * first + bias

        schedules = []
        (source,) = self.codegen(
            fn,
            [(3, 40, 72), (3, 72, 136)] * 2 + [(136,)],
            schedules=schedules,
        )
        self.assertEqual(len(schedules), 1)
        self.assertIn("for tile_z, tile_y, tile_x in hl.tile([3, 40, 136]):", source)
        self.assertIn("A[tile_z, tile_y, tile_r]", source)
        self.assertIn("B[tile_z, tile_r, tile_x]", source)
        self.assertIn("out0[tile_z, tile_y, tile_x]", source)
        self.assertEqual(source.count(" = torch.baddbmm("), 2 if horizontal else 1)

    def test_bmm_operand_view(self):
        (source,) = self.codegen(
            lambda a, bt: torch.bmm(a.reshape(3, 40, 72), bt.permute(0, 2, 1)),
            [(3, 40, 9, 8), (3, 136, 72)],
        )
        self.assertIn("reinterpret_tensor(a_1, (3, 40, 72), (2880, 72, 1), 0)", source)
        self.assertIn(
            "reinterpret_tensor(bt_1, (3, 72, 136), (9792, 1, 72), 0)", source
        )
        self.assertNotIn("//", source)
        self.assertNotIn("%", source)

    def test_fp32(self):
        (source,) = self.codegen(
            lambda a, b: a @ b,
            [(40, 72), (72, 136)],
            dtypes=[torch.float32, torch.float32],
        )
        self.assertIn("acc = torch.addmm(acc, tmp0, tmp1)", source)

    @parametrize("batched", [False, True])
    def test_atomic_add_epilogue(self, batched):
        def fn(a, b, indices, out):
            value = torch.bmm(a, b) if batched else a @ b
            out.index_add_(0, indices, value)
            return out

        shapes = (
            [(3, 40, 72), (3, 72, 136), (3,), (7, 40, 136)]
            if batched
            else [(40, 72), (72, 136), (40,), (80, 136)]
        )
        (source,) = self.codegen(
            fn,
            shapes,
            dtypes=[torch.float32, torch.float32, torch.int64, torch.float32],
            decompose=True,
            complete=True,
        )
        self.assertIn("hl.atomic_add(", source)
        self.assertIn("out_1", source)
        self.assertIn("acc", source)

    def test_gather_mm_broadcast_epilogue(self):
        # The normal memory heuristic can reject this fusion after input realization.
        # Bypass only that heuristic to check the legal fused schedule's codegen.
        with patch(
            "torch._inductor.scheduler.Scheduler.can_fusion_increase_peak_memory",
            return_value=False,
        ):
            (source,) = self.codegen(
                lambda x, y, z, w: x[z] @ y + w + 3,
                [(128, 128)] * 4,
                dtypes=[torch.float32, torch.float32, torch.int64, torch.float32],
                decompose=True,
                complete=True,
            )
        self.assertIn("torch.addmm(", source)
        self.assertIn("in0[", source)
        self.assertIn("3.0", source)
        self.assertIn("% 128", source)

    def test_scalar_tensor_epilogue(self):
        (source,) = self.codegen(
            lambda a, b, bias: a @ b + bias,
            [(40, 72), (72, 136), ()],
        )
        self.assertIn("in0[()]", source)

    def test_padded_scatter_rejected(self):
        def fn(a, b, indices, out):
            out.index_add_(0, indices, a @ b)
            return out

        with self.assertRaisesRegex(
            NotImplementedError, "Cannot recover Helion tensor indices"
        ):
            self.codegen(
                fn,
                [(40, 72), (72, 136), (40,), (80, 136)],
                strides=[(72, 1), (136, 1), (1,), (144, 1)],
                dtypes=[torch.float32, torch.float32, torch.int64, torch.float32],
                decompose=True,
                complete=True,
            )

    def test_upcast_is_required(self):
        with config.patch("triton.codegen_upcast_to_fp32", False):
            with self.assertRaisesRegex(NotImplementedError, "FP32 compute upcasting"):
                self.codegen(lambda a, b: a @ b, [(40, 72), (72, 136)])


class TestHelionIndexExpressions(TestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @parametrize("op", [sympy.Min, sympy.Max])
    @dtypes(torch.int64, torch.float32)
    def test_integer_min_max_before_cast(self, device, dtype, op):
        from types import SimpleNamespace

        limit = 2**25 + 1
        integers = [limit - 2, limit - 1, limit, limit + 1, limit + 2]
        features = Mock()
        features.scheduler_nodes.return_value = ()
        features.op_counts.return_value = {}
        tiling = dict(zip(("y", "x", "r0_"), map(sympy.Integer, (1, limit + 3, 1))))
        graph = Mock(helion_matmul_inputs={}, sizevars=SizeVarAllocator())
        with V.set_graph_handler(graph):
            kernel = HelionKernel(features, tiling)
            x = kernel.symbols["x"]
            value = kernel.index_expr(op(x, limit) - x, dtype)
        scope = {
            "torch": torch,
            "tile_x": SimpleNamespace(index=torch.tensor(integers, device=device)),
            "hl": SimpleNamespace(
                full=lambda shape, value, dtype: torch.full(
                    shape, value, dtype=dtype, device=device
                )
            ),
        }
        exec(kernel.reduction_body.getvalue(), scope)
        operation = min if op is sympy.Min else max
        expected = torch.tensor(
            [operation(n, limit) - n for n in integers], device=device, dtype=dtype
        )
        self.assertEqual(scope[value.name], expected)


instantiate_device_type_tests(TestHelionIndexExpressions, globals())


if __name__ == "__main__":
    run_tests()
