# Owner(s): ["module: inductor"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import inspect
import unittest
from collections.abc import Callable
from unittest import mock

import torch
from torch._higher_order_ops.flydsl_kernel_wrap import (
    flydsl_kernel_wrapper_functional,
    flydsl_kernel_wrapper_mutation,
    flydsl_launcher_side_table,
    invoke_flydsl_launcher,
    restore_flydsl_launcher_arguments,
    split_flydsl_launcher_arguments,
    TraceableFlyDSLLauncher,
)
from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available
from torch._library.utils import get_layout_constraint_tag
from torch.testing._internal.common_utils import TestCase


HAS_FLYDSL = runtime_available()
if HAS_FLYDSL:
    import flydsl.compiler as flyc
    import flydsl.expr as fx
    from flydsl.compiler.jit_argument import PointerJitArg
    from flydsl.expr.typing import Stream


if HAS_FLYDSL:

    @flyc.jit
    def _launcher(out: fx.Tensor, inp: fx.Tensor, rows: fx.Int32):
        pass

else:
    _launcher = None


class _EagerLauncher:
    def __init__(self) -> None:
        self.func = self.launch

    def launch(
        self,
        out: torch.Tensor,
        workspace: torch.Tensor,
        inp: torch.Tensor,
    ) -> None:
        out.copy_(inp)
        workspace.fill_(7)

    def __call__(
        self,
        out: torch.Tensor,
        workspace: torch.Tensor,
        inp: torch.Tensor,
    ) -> None:
        self.launch(out, workspace, inp)


@unittest.skipUnless(HAS_FLYDSL, "FlyDSL is not available")
class FlyDSLCaptureTest(TestCase):
    def setUp(self):
        flydsl_launcher_side_table.reset_table()
        torch._dynamo.reset()

    def test_export_captures_explicit_launcher(self):
        captured_launcher = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )

        class Model(torch.nn.Module):
            def forward(self, inp):
                out = torch.empty_like(inp)
                captured_launcher(out=out, inp=inp, rows=inp.numel())
                return out

        exported = torch.export.export(Model(), (torch.randn(8),))

        nodes = exported.graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_mutation,
        )
        self.assertEqual(1, len(nodes))
        self.assertEqual((0,), nodes[0].kwargs["mutated_arg_indices"])

    def test_mutations_must_be_explicit(self):
        with self.assertRaisesRegex(TypeError, "mutates_args"):
            torch.library.wrap_flydsl(_launcher)

    def test_repeated_wrap_reuses_registration(self):
        first = torch.library.wrap_flydsl(_launcher, mutates_args={"out"})
        second = torch.library.wrap_flydsl(_launcher, mutates_args={"out"})
        different_mutations = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out", "inp"},
        )

        self.assertEqual(first.launcher_idx, second.launcher_idx)
        self.assertNotEqual(first.launcher_idx, different_mutations.launcher_idx)

    def test_capture_rejects_unknown_mutated_argument(self):
        with self.assertRaisesRegex(ValueError, "not launcher parameters"):
            torch.library.wrap_flydsl(
                _launcher,
                mutates_args={"missing"},
            )

    def test_wrap_rejects_non_jit_function(self):
        with self.assertRaisesRegex(RuntimeError, "annotated with flydsl.compiler.jit"):
            torch.library.wrap_flydsl(lambda: None, mutates_args=())

    def test_wrap_reports_missing_optional_runtime(self):
        with (
            mock.patch(
                "torch._inductor.codegen.flydsl.flydsl_utils.runtime_available",
                return_value=False,
            ) as available,
            self.assertRaisesRegex(RuntimeError, "optional `flydsl` runtime"),
        ):
            torch.library.wrap_flydsl(object(), mutates_args=())

        available.assert_called_once_with()

    def test_wrap_rejects_explicit_stream(self):
        @flyc.jit
        def launcher(out: fx.Tensor, stream: Stream):
            pass

        with self.assertRaisesRegex(TypeError, "current device stream"):
            torch.library.wrap_flydsl(launcher, mutates_args={"out"})

    def test_split_launcher_arguments_preserves_parameter_kinds(self):
        def launcher(out, /, inp, *, rows):
            pass

        positional, keyword = split_flydsl_launcher_arguments(
            inspect.signature(launcher),
            ("out", "inp", 8),
        )

        self.assertEqual(("out", "inp"), positional)
        self.assertEqual({"rows": 8}, keyword)

    def test_split_launcher_arguments_rejects_variadic_parameters(self):
        @flyc.jit
        def launcher(out: fx.Tensor, *inputs: fx.Tensor):
            pass

        with self.assertRaisesRegex(TypeError, "variadic parameters cannot be wrapped"):
            torch.library.wrap_flydsl(launcher, mutates_args={"out"})

    def test_export_captures_dynamic_dimension(self):
        captured_launcher = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )

        class Model(torch.nn.Module):
            def forward(self, inp):
                out = torch.empty_like(inp)
                captured_launcher(out, inp, inp.numel())
                return out

        batch = torch.export.Dim("batch", min=1, max=32)
        exported = torch.export.export(
            Model(),
            (torch.randn(4, 8),),
            dynamic_shapes=({0: batch},),
        )

        nodes = exported.graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_mutation,
        )
        self.assertEqual(1, len(nodes))
        self.assertTrue(exported.range_constraints)
        rows = nodes[0].kwargs["args"][2]
        self.assertIsInstance(rows, torch.fx.Node)
        self.assertIsInstance(rows.meta["val"], torch.SymInt)

    def test_export_keeps_compile_time_arguments_out_of_fx(self):
        callback = lambda value: value  # noqa: E731

        @flyc.jit
        def launcher(
            out: fx.Tensor,
            inp: fx.Tensor,
            transform: fx.Constexpr[Callable],
            element_type: type[fx.Float32],
            block_dim: fx.Constexpr[int],
        ):
            pass

        captured = torch.library.wrap_flydsl(launcher, mutates_args={"out"})

        class Model(torch.nn.Module):
            def forward(self, inp):
                out = torch.empty_like(inp)
                captured(out, inp, callback, fx.Float32, 256)
                return out

        exported = torch.export.export(Model(), (torch.randn(8),))
        (node,) = exported.graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_mutation,
        )

        runtime_args = node.kwargs["args"]
        self.assertEqual((None, None, None), runtime_args[2:])
        restored = restore_flydsl_launcher_arguments(
            runtime_args,
            node.kwargs["call_spec_idx"],
        )
        self.assertIs(callback, restored[2])
        self.assertIs(fx.Float32, restored[3])
        self.assertEqual(256, restored[4])

    def test_constexpr_call_specs_use_flydsl_value_identity(self):
        @flyc.jit
        def launcher(out: fx.Tensor, config: fx.Constexpr[tuple]):
            pass

        captured = torch.library.wrap_flydsl(launcher, mutates_args={"out"})

        class Model(torch.nn.Module):
            def forward(self, inp):
                first = torch.empty_like(inp)
                second = torch.empty_like(inp)
                captured(first, (True,))
                captured(second, (1,))
                return first, second

        exported = torch.export.export(Model(), (torch.randn(8),))
        nodes = exported.graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_mutation,
        )

        self.assertEqual(2, len(nodes))
        self.assertEqual(
            [(True,), (1,)],
            [
                restore_flydsl_launcher_arguments(
                    node.kwargs["args"],
                    node.kwargs["call_spec_idx"],
                )[1]
                for node in nodes
            ],
        )
        self.assertNotEqual(
            nodes[0].kwargs["call_spec_idx"],
            nodes[1].kwargs["call_spec_idx"],
        )

    def test_wrap_rejects_preconstructed_runtime_jit_arguments(self):
        captured = torch.library.wrap_flydsl(_launcher, mutates_args={"out"})
        out = torch.empty(4)
        inp = torch.empty(4)

        for value in (fx.Int32(4), object.__new__(PointerJitArg)):
            with self.assertRaisesRegex(TypeError, "graphable PyTorch values"):
                captured(out, inp, value)

    def test_wrap_rejects_non_tensor_mutation(self):
        @flyc.jit
        def launcher(value: fx.Int32):
            pass

        with self.assertRaisesRegex(TypeError, "flydsl.expr.Tensor annotation"):
            torch.library.wrap_flydsl(launcher, mutates_args={"value"})

    def test_wrap_rejects_unannotated_mutation(self):
        @flyc.jit
        def launcher(value):
            pass

        with self.assertRaisesRegex(TypeError, "flydsl.expr.Tensor annotation"):
            torch.library.wrap_flydsl(launcher, mutates_args={"value"})

    @unittest.skipUnless(torch.cuda.is_available(), "requires a GPU")
    def test_eager_wrap_rejects_non_default_stream(self):
        registration = TraceableFlyDSLLauncher(
            _EagerLauncher(),
            (0, 1),
        )
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({})
        out = torch.empty(4, device="cuda")
        workspace = torch.empty(4, device="cuda")
        inp = torch.empty(4, device="cuda")

        with (
            torch.cuda.stream(torch.cuda.Stream()),
            self.assertRaisesRegex(RuntimeError, "default device stream"),
        ):
            flydsl_kernel_wrapper_mutation(
                registration.launcher_idx,
                call_spec_idx,
                (out, workspace, inp),
                (0, 1),
            )

    def test_export_registers_and_deduplicates_compile_time_call_specs(self):
        @flyc.jit
        def launcher(
            out: fx.Tensor,
            inp: fx.Tensor,
            block_dim: fx.Constexpr[int],
        ):
            pass

        captured = torch.library.wrap_flydsl(launcher, mutates_args={"out"})

        class Model(torch.nn.Module):
            def forward(self, inp):
                first = torch.empty_like(inp)
                second = torch.empty_like(inp)
                third = torch.empty_like(inp)
                captured(first, inp, 64)
                captured(second, inp, 128)
                captured(third, inp, 64)
                return first, second, third

        exported = torch.export.export(Model(), (torch.randn(8),))
        nodes = exported.graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_mutation,
        )

        self.assertEqual(3, len(nodes))
        self.assertEqual(
            [64, 128, 64],
            [
                restore_flydsl_launcher_arguments(
                    node.kwargs["args"],
                    node.kwargs["call_spec_idx"],
                )[2]
                for node in nodes
            ],
        )
        self.assertEqual(
            nodes[0].kwargs["call_spec_idx"],
            nodes[2].kwargs["call_spec_idx"],
        )
        self.assertNotEqual(
            nodes[0].kwargs["call_spec_idx"],
            nodes[1].kwargs["call_spec_idx"],
        )

    def test_export_captures_bound_jit_method_without_self_operand(self):
        class LauncherOwner:
            @flyc.jit
            def launch(self, out: fx.Tensor, inp: fx.Tensor, rows: fx.Int32):
                pass

        owner = LauncherOwner()
        captured = torch.library.wrap_flydsl(owner.launch, mutates_args={"out"})

        class Model(torch.nn.Module):
            def forward(self, inp):
                out = torch.empty_like(inp)
                captured(out, inp, inp.numel())
                return out

        exported = torch.export.export(Model(), (torch.randn(8),))
        (node,) = exported.graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_mutation,
        )

        registration = flydsl_launcher_side_table.get_registration(
            node.kwargs["launcher_idx"]
        )
        self.assertIs(owner, registration.bound_self)
        self.assertEqual(3, len(node.kwargs["args"]))
        with mock.patch.object(
            type(registration.launcher),
            "__call__",
            autospec=True,
        ) as launch:
            invoke_flydsl_launcher(registration, ("out", "inp", 8))
        launch.assert_called_once_with(
            registration.launcher,
            owner,
            "out",
            "inp",
            8,
        )

    def test_hop_reports_tensor_subclass_once(self):
        class TensorSubclass(torch.Tensor):
            @classmethod
            def __torch_dispatch__(cls, func, types, args=(), kwargs=None):
                return NotImplemented

        tensor = TensorSubclass._make_subclass(
            TensorSubclass,
            torch.ones(4),
            False,
        )
        self.assertTrue(torch._C._dispatch_keys(tensor).has("Python"))

        overloaded = flydsl_kernel_wrapper_mutation._get_overloaded_args(
            (),
            {"args": (tensor,)},
        )

        self.assertEqual(1, len(overloaded))
        self.assertIs(tensor, overloaded[0])

    def test_functional_wrapper_clones_only_requested_outputs(self):
        registration = TraceableFlyDSLLauncher(
            _EagerLauncher(),
            (0, 1),
        )
        out = torch.empty(4)
        workspace = torch.zeros(4)
        inp = torch.arange(4, dtype=torch.float32)
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({})

        actual_out, actual_workspace = flydsl_kernel_wrapper_functional(
            registration.launcher_idx,
            call_spec_idx,
            (out, workspace, inp),
            (0, 1),
            (1,),
        )

        torch.testing.assert_close(out, inp)
        self.assertEqual(actual_out.data_ptr(), out.data_ptr())
        torch.testing.assert_close(workspace, torch.zeros_like(workspace))
        torch.testing.assert_close(actual_workspace, torch.full_like(workspace, 7))
        self.assertNotEqual(actual_workspace.data_ptr(), workspace.data_ptr())

    def test_functional_wrapper_rejects_identical_aliased_arguments(self):
        registration = TraceableFlyDSLLauncher(
            _EagerLauncher(),
            (0, 1),
        )
        tensor = torch.zeros(4)
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({})

        with self.assertRaisesRegex(RuntimeError, "aliased launcher arguments"):
            flydsl_kernel_wrapper_functional(
                registration.launcher_idx,
                call_spec_idx,
                (tensor, tensor, tensor),
                (0, 1),
                (0, 1),
            )

    def test_functional_wrapper_rejects_view_alias(self):
        registration = TraceableFlyDSLLauncher(
            _EagerLauncher(),
            (0, 1),
        )
        base = torch.zeros(8)
        out = base[:4]
        workspace = torch.zeros(4)
        inp = base[2:6]
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({})

        with self.assertRaisesRegex(RuntimeError, "aliased launcher arguments"):
            flydsl_kernel_wrapper_functional(
                registration.launcher_idx,
                call_spec_idx,
                (out, workspace, inp),
                (0, 1),
                (0,),
            )

    def test_flydsl_op_is_opaque_to_symbolic_trace(self):
        captured_launcher = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )

        @torch.library.flydsl_op(
            "test_flydsl_capture::symbolic_trace",
            mutates_args=(),
        )
        def flydsl_add(inp: torch.Tensor) -> torch.Tensor:
            if inp.shape[1] != 8:
                raise ValueError("expected width eight")
            out = torch.empty_like(inp)
            captured_launcher(out=out, inp=inp, rows=inp.numel())
            return out

        class Model(torch.nn.Module):
            def forward(self, inp):
                return flydsl_add(inp)

        traced = torch.fx.symbolic_trace(Model())

        nodes = traced.graph.find_nodes(
            op="call_function",
            target=torch.ops.test_flydsl_capture.symbolic_trace.default,
        )
        self.assertEqual(1, len(nodes))

    def test_flydsl_op_preserves_exact_strides_by_default(self):
        captured_launcher = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )

        @torch.library.flydsl_op(
            "test_flydsl_capture::exact_strides",
            mutates_args=(),
        )
        def flydsl_add(inp: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(inp)
            captured_launcher(out=out, inp=inp, rows=inp.numel())
            return out

        self.assertEqual(
            torch._C.Tag.needs_exact_strides,
            get_layout_constraint_tag(
                torch.ops.test_flydsl_capture.exact_strides.default
            ),
        )

        example = torch.randn(8, 4).t()
        self.assertEqual((1, 4), example.stride())

        class Model(torch.nn.Module):
            def forward(self, inp):
                return flydsl_add(inp)

        exported = torch.export.export(Model(), (example,))
        custom_op_node = exported.graph_module.graph.find_nodes(
            op="call_function",
            target=torch.ops.test_flydsl_capture.exact_strides.default,
        )[0]
        input_node = custom_op_node.args[0]
        self.assertIsInstance(input_node, torch.fx.Node)
        self.assertEqual(example.stride(), input_node.meta["val"].stride())

    def test_flydsl_op_decomposes_for_export(self):
        captured_launcher = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )

        @torch.library.flydsl_op(
            "test_flydsl_capture::export",
            mutates_args=(),
        )
        def flydsl_add(inp: torch.Tensor) -> torch.Tensor:
            if inp.shape[1] != 8:
                raise ValueError("expected width eight")
            out = torch.empty_like(inp)
            captured_launcher(out=out, inp=inp, rows=inp.numel())
            return out

        class Model(torch.nn.Module):
            def forward(self, inp):
                return flydsl_add(inp)

        batch = torch.export.Dim("batch", min=1, max=32)
        exported = torch.export.export(
            Model(),
            (torch.randn(4, 8),),
            dynamic_shapes=({0: batch},),
        )

        custom_op_nodes = exported.graph_module.graph.find_nodes(
            op="call_function",
            target=torch.ops.test_flydsl_capture.export.default,
        )
        self.assertEqual(1, len(custom_op_nodes))

        preserved = exported.run_decompositions(
            decomp_table={},
            decompose_custom_triton_ops=True,
        )
        self.assertEqual(
            1,
            len(
                preserved.graph_module.graph.find_nodes(
                    op="call_function",
                    target=torch.ops.test_flydsl_capture.export.default,
                )
            ),
        )
        self.assertEqual(
            [],
            preserved.graph_module.graph.find_nodes(
                op="call_function",
                target=flydsl_kernel_wrapper_functional,
            ),
        )

        with torch._functorch.config.patch(decompose_custom_flydsl_ops=False):
            decomposed = exported.run_decompositions(
                decomp_table={},
                decompose_custom_triton_ops=False,
                decompose_custom_flydsl_ops=True,
            )
        flydsl_nodes = decomposed.graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_functional,
        )
        self.assertEqual(1, len(flydsl_nodes))

    def test_flydsl_op_decomposes_for_torch_compile_by_default(self):
        from torch._dynamo.testing import AotEagerAndRecordGraphs

        captured = torch.library.wrap_flydsl(_launcher, mutates_args={"out"})

        @torch.library.flydsl_op(
            "test_flydsl_capture::compile_default",
            mutates_args=(),
        )
        def flydsl_copy(inp: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(inp)
            captured(out, inp, inp.numel())
            return out

        class Model(torch.nn.Module):
            def forward(self, inp):
                return flydsl_copy(inp)

        backend = AotEagerAndRecordGraphs()
        inp = torch.randn(8)

        def invoke(_, args):
            args[0].copy_(args[1])

        with mock.patch(
            "torch._higher_order_ops.flydsl_kernel_wrap.invoke_flydsl_launcher",
            side_effect=invoke,
        ):
            actual = torch.compile(
                Model(),
                backend=backend,
                fullgraph=True,
            )(inp)

        torch.testing.assert_close(actual, inp)
        self.assertEqual(1, len(backend.fw_graphs))
        graph = backend.fw_graphs[0].graph
        self.assertEqual(
            1,
            len(
                graph.find_nodes(
                    op="call_function",
                    target=flydsl_kernel_wrapper_functional,
                )
            ),
        )
        self.assertEqual(
            [],
            graph.find_nodes(
                op="call_function",
                target=torch.ops.test_flydsl_capture.compile_default.default,
            ),
        )

    def test_flydsl_op_preserved_in_joint_export(self):
        from torch.export.experimental import _export_forward_backward

        captured = torch.library.wrap_flydsl(_launcher, mutates_args={"out"})

        @torch.library.flydsl_op(
            "test_flydsl_capture::joint_export",
            mutates_args=(),
        )
        def flydsl_copy(inp: torch.Tensor) -> torch.Tensor:
            out = torch.empty_like(inp)
            captured(out, inp, inp.numel())
            return out

        def backward(ctx, grad):
            return grad

        flydsl_copy.register_autograd(backward)

        class Model(torch.nn.Module):
            def forward(self, inp):
                return flydsl_copy(inp).sum()

        exported = torch.export.export(
            Model(),
            (torch.randn(8, requires_grad=True),),
        )
        joint = _export_forward_backward(exported)

        self.assertEqual(
            1,
            len(
                joint.graph_module.graph.find_nodes(
                    op="call_function",
                    target=torch.ops.test_flydsl_capture.joint_export.default,
                )
            ),
        )
        self.assertEqual(
            [],
            joint.graph_module.graph.find_nodes(
                op="call_function",
                target=flydsl_kernel_wrapper_functional,
            ),
        )

        decomposed = joint.run_decompositions(
            decomp_table={},
            decompose_custom_flydsl_ops=True,
        )
        self.assertEqual(
            [],
            decomposed.graph_module.graph.find_nodes(
                op="call_function",
                target=torch.ops.test_flydsl_capture.joint_export.default,
            ),
        )
        self.assertEqual(
            1,
            len(
                decomposed.graph_module.graph.find_nodes(
                    op="call_function",
                    target=flydsl_kernel_wrapper_functional,
                )
            ),
        )

    def test_flydsl_op_preserves_mutation_when_decomposed(self):
        captured_launcher = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )

        @torch.library.flydsl_op(
            "test_flydsl_capture::mutation",
            mutates_args={"out"},
        )
        def flydsl_copy(out: torch.Tensor, inp: torch.Tensor) -> None:
            captured_launcher(out=out, inp=inp, rows=inp.numel())

        class Model(torch.nn.Module):
            def forward(self, inp):
                out = torch.empty_like(inp)
                flydsl_copy(out, inp)
                return out

        exported = torch.export.export(Model(), (torch.randn(8),))
        decomposed = exported.run_decompositions(decompose_custom_flydsl_ops=True)

        nodes = decomposed.graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_functional,
        )
        self.assertEqual(1, len(nodes))
        self.assertEqual((0,), nodes[0].kwargs["mutated_arg_indices"])

    def test_flydsl_op_decomposes_multiple_launchers_and_aten(self):
        captured_launcher = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )

        @torch.library.flydsl_op(
            "test_flydsl_capture::composed",
            mutates_args=(),
        )
        def composed(inp: torch.Tensor) -> torch.Tensor:
            intermediate = torch.empty_like(inp)
            captured_launcher(
                out=intermediate,
                inp=inp,
                rows=inp.numel(),
            )
            shifted = intermediate + 1
            out = torch.empty_like(inp)
            captured_launcher(
                out=out,
                inp=shifted,
                rows=shifted.numel(),
            )
            return out

        class Model(torch.nn.Module):
            def forward(self, inp):
                return composed(inp)

        batch = torch.export.Dim("batch", min=1, max=32)
        exported = torch.export.export(
            Model(),
            (torch.randn(4, 8),),
            dynamic_shapes=({0: batch},),
        )
        decomposed = exported.run_decompositions(decompose_custom_flydsl_ops=True)

        flydsl_nodes = decomposed.graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_functional,
        )
        aten_nodes = decomposed.graph_module.graph.find_nodes(
            op="call_function",
            target=torch.ops.aten.add.Tensor,
        )
        self.assertEqual(2, len(flydsl_nodes))
        self.assertEqual(1, len(aten_nodes))
        for node in flydsl_nodes:
            rows = node.kwargs["args"][2]
            self.assertIsInstance(rows, torch.fx.Node)
            self.assertIsInstance(rows.meta["val"], torch.SymInt)


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
