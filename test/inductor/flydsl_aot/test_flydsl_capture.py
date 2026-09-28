# Owner(s): ["module: inductor"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import inspect
import io
import unittest
from collections.abc import Callable
from dataclasses import replace
from typing import cast
from unittest import mock

import torch
from torch._higher_order_ops.flydsl_kernel_wrap import (
    _register_flydsl_call_spec,
    flydsl_kernel_wrapper_functional,
    flydsl_kernel_wrapper_mutation,
    flydsl_launcher_side_table,
    invoke_flydsl_launcher,
    restore_flydsl_launcher_arguments,
    split_flydsl_launcher_arguments,
    TraceableFlyDSLLauncher,
)
from torch._inductor.codecache import BypassFxGraphCache, CacheabilityValidator
from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available
from torch._library.utils import get_layout_constraint_tag
from torch.export.graph_signature import OutputKind
from torch.fx.passes.canonicalize import (
    _canonical_node_key,
    _is_safe_to_reorder,
    canonicalize_graph,
)
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


requires_flydsl = unittest.skipUnless(HAS_FLYDSL, "FlyDSL is not available")


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


class _ConstexprEagerLauncher:
    def __init__(self) -> None:
        self.func = self.launch

    def launch(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        increment: int,
    ) -> None:
        out.copy_(inp + increment)

    def __call__(
        self,
        out: torch.Tensor,
        inp: torch.Tensor,
        increment: int,
    ) -> None:
        self.launch(out, inp, increment)


class FlyDSLCaptureTest(TestCase):
    def setUp(self):
        super().setUp()
        flydsl_launcher_side_table.reset_table()
        torch._dynamo.reset()

    @requires_flydsl
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

    @requires_flydsl
    def test_repeated_wrap_reuses_registration(self):
        first = torch.library.wrap_flydsl(_launcher, mutates_args={"out"})
        second = torch.library.wrap_flydsl(_launcher, mutates_args={"out"})
        different_mutations = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out", "inp"},
        )

        self.assertEqual(first.launcher_idx, second.launcher_idx)
        self.assertNotEqual(first.launcher_idx, different_mutations.launcher_idx)

    @requires_flydsl
    def test_wrap_accepts_single_mutated_argument_name(self):
        captured = torch.library.wrap_flydsl(_launcher, mutates_args="out")

        registration = flydsl_launcher_side_table.get_registration(
            captured.launcher_idx
        )
        self.assertEqual((0,), registration.mutated_arg_indices)

    @requires_flydsl
    def test_capture_rejects_unknown_mutated_argument(self):
        with self.assertRaisesRegex(ValueError, "not launcher parameters"):
            torch.library.wrap_flydsl(
                _launcher,
                mutates_args={"missing"},
            )

    @requires_flydsl
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

    @requires_flydsl
    def test_wrap_rejects_required_stream(self):
        @flyc.jit
        def launcher(out: fx.Tensor, stream: Stream):
            pass

        with self.assertRaisesRegex(TypeError, "required Stream parameters"):
            torch.library.wrap_flydsl(launcher, mutates_args={"out"})

    @requires_flydsl
    def test_wrap_omits_defaulted_stream(self):
        @flyc.jit
        def launcher(
            out: fx.Tensor,
            inp: fx.Tensor,
            stream: Stream = fx.Stream(None),
        ):
            pass

        captured = torch.library.wrap_flydsl(launcher, mutates_args={"out"})
        registration = flydsl_launcher_side_table.get_registration(
            captured.launcher_idx
        )

        self.assertEqual(("out", "inp"), tuple(registration.signature.parameters))
        self.assertIsNotNone(registration.stream_parameter)
        stream_parameter = cast(
            tuple[int, inspect.Parameter], registration.stream_parameter
        )
        self.assertEqual(2, stream_parameter[0])
        self.assertEqual("stream", stream_parameter[1].name)
        with self.assertRaisesRegex(TypeError, "unexpected keyword argument 'stream'"):
            captured("out", "inp", stream=fx.Stream(None))

    @requires_flydsl
    def test_invoke_restores_defaulted_stream_in_original_position(self):
        @flyc.jit
        def launcher(
            out: fx.Tensor,
            stream: Stream = fx.Stream(None),
            inp: fx.Tensor = None,
        ):
            pass

        captured = torch.library.wrap_flydsl(launcher, mutates_args={"out"})
        registration = flydsl_launcher_side_table.get_registration(
            captured.launcher_idx
        )
        calls = []

        def record(*args, **kwargs):
            calls.append((args, kwargs))

        invoke_flydsl_launcher(
            replace(registration, launcher=record),
            ("OUT", "INP"),
        )

        self.assertIsNotNone(registration.stream_parameter)
        stream_parameter = cast(
            tuple[int, inspect.Parameter], registration.stream_parameter
        )
        self.assertEqual(
            [(("OUT", stream_parameter[1].default, "INP"), {})],
            calls,
        )

    @requires_flydsl
    def test_wrap_rejects_multiple_defaulted_streams(self):
        @flyc.jit
        def launcher(
            out: fx.Tensor,
            first: Stream = fx.Stream(None),
            second: Stream = fx.Stream(None),
        ):
            pass

        with self.assertRaisesRegex(TypeError, "at most one Stream parameter"):
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

    @requires_flydsl
    def test_split_launcher_arguments_rejects_variadic_parameters(self):
        @flyc.jit
        def launcher(out: fx.Tensor, *inputs: fx.Tensor):
            pass

        with self.assertRaisesRegex(TypeError, "variadic parameters cannot be wrapped"):
            torch.library.wrap_flydsl(launcher, mutates_args={"out"})

    @requires_flydsl
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

    @requires_flydsl
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

    @requires_flydsl
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

    def test_constexpr_call_specs_require_value_signature(self):
        with self.assertRaisesRegex(AssertionError, "value-signature callback"):
            _register_flydsl_call_spec({0: (True,)}, (0,), None)

    def test_compile_specializes_dynamic_constexpr(self):
        from torch._dynamo.decorators import assume_constant_result

        captured = TraceableFlyDSLLauncher(
            _ConstexprEagerLauncher(),
            (0,),
            compile_time_arg_indices=frozenset({2}),
            constexpr_arg_indices=frozenset({2}),
            constexpr_value_signature=lambda value: value,
        )
        assume_constant_result(_register_flydsl_call_spec)
        torch._dynamo.allow_in_graph(flydsl_kernel_wrapper_mutation)

        def fn(inp):
            out = torch.empty_like(inp)
            captured(out, inp, inp.size(0))
            return out

        compiled = torch.compile(fn, backend="eager", fullgraph=True, dynamic=True)
        for size in (4, 7):
            inp = torch.arange(size, dtype=torch.float32)
            torch.testing.assert_close(compiled(inp), inp + size)

    @requires_flydsl
    def test_wrap_rejects_preconstructed_runtime_jit_arguments(self):
        captured = torch.library.wrap_flydsl(_launcher, mutates_args={"out"})
        out = torch.empty(4)
        inp = torch.empty(4)

        for value in (fx.Int32(4), object.__new__(PointerJitArg)):
            with self.assertRaisesRegex(TypeError, "graphable PyTorch values"):
                captured(out, inp, value)

    @requires_flydsl
    def test_wrap_rejects_non_tensor_mutation(self):
        @flyc.jit
        def launcher(value: fx.Int32):
            pass

        with self.assertRaisesRegex(TypeError, "flydsl.expr.Tensor annotation"):
            torch.library.wrap_flydsl(launcher, mutates_args={"value"})

    @requires_flydsl
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

    @requires_flydsl
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

    @requires_flydsl
    def test_wrap_rejects_bound_jit_method_without_self_receiver(self):
        class LauncherOwner:
            @flyc.jit
            def launch(owner, out: fx.Tensor, inp: fx.Tensor, rows: fx.Int32):
                pass

        with self.assertRaisesRegex(TypeError, "name their receiver 'self'"):
            torch.library.wrap_flydsl(
                LauncherOwner().launch,
                mutates_args={"out"},
            )

    @requires_flydsl
    def test_plain_launcher_parameter_named_self_is_not_a_receiver(self):
        @flyc.jit
        def launcher(self: fx.Tensor, inp: fx.Tensor):
            pass

        captured = torch.library.wrap_flydsl(launcher, mutates_args={"self"})
        registration = flydsl_launcher_side_table.get_registration(
            captured.launcher_idx
        )

        self.assertIsNone(registration.bound_self)
        self.assertEqual(("self", "inp"), tuple(registration.signature.parameters))
        self.assertEqual((0,), registration.mutated_arg_indices)

    @requires_flydsl
    def test_invoke_bound_jit_method_reattaches_receiver(self):
        class LauncherOwner:
            @flyc.jit
            def launch(
                self,
                out: fx.Tensor,
                inp: fx.Tensor,
                *,
                rows: fx.Int32,
            ):
                pass

        owner = LauncherOwner()
        captured = torch.library.wrap_flydsl(
            owner.launch,
            mutates_args={"out"},
        )
        registration = flydsl_launcher_side_table.get_registration(
            captured.launcher_idx
        )
        calls = []

        def record(*args, **kwargs):
            calls.append((args, kwargs))

        invoke_flydsl_launcher(
            replace(registration, launcher=record),
            ("OUT", "INP", 8),
        )

        self.assertIs(owner, registration.bound_self)
        self.assertEqual(
            ("out", "inp", "rows"), tuple(registration.signature.parameters)
        )
        self.assertEqual((0,), registration.mutated_arg_indices)
        self.assertEqual([((owner, "OUT", "INP"), {"rows": 8})], calls)

    def test_mutation_hop_is_impure(self):
        graph = torch.fx.Graph()
        out = graph.placeholder("out")
        mutation = graph.call_function(
            flydsl_kernel_wrapper_mutation,
            kwargs={
                "launcher_idx": 0,
                "call_spec_idx": 0,
                "args": (out,),
                "mutated_arg_indices": (0,),
            },
        )
        graph.output(out)

        self.assertTrue(mutation.is_impure())

    def test_dce_preserves_mutation_hop(self):
        graph = torch.fx.Graph()
        out = graph.placeholder("out")
        graph.call_function(
            flydsl_kernel_wrapper_mutation,
            kwargs={
                "launcher_idx": 0,
                "call_spec_idx": 0,
                "args": (out,),
                "mutated_arg_indices": (0,),
            },
        )
        graph.output(out)

        graph.eliminate_dead_code()

        self.assertEqual(
            1,
            len(
                graph.find_nodes(
                    op="call_function",
                    target=flydsl_kernel_wrapper_mutation,
                )
            ),
        )

    def test_canonicalization_preserves_mutation_before_read(self):
        graph = torch.fx.Graph()
        out = graph.placeholder("out")
        graph.call_function(
            flydsl_kernel_wrapper_mutation,
            kwargs={
                "launcher_idx": 0,
                "call_spec_idx": 0,
                "args": (out,),
                "mutated_arg_indices": (0,),
            },
        )
        result = graph.call_function(torch.ops.aten.add.Tensor, (out, 1))
        graph.output(result)

        def canonical_key(node, canonical_idx):
            if node.op == "placeholder":
                return (0, node.name)
            return _canonical_node_key(node, canonical_idx)

        canonicalize_graph(graph, canonical_key, _is_safe_to_reorder)

        call_targets = [
            node.target for node in graph.nodes if node.op == "call_function"
        ]
        self.assertEqual(
            [flydsl_kernel_wrapper_mutation, torch.ops.aten.add.Tensor],
            call_targets,
        )

    def test_flydsl_hops_bypass_persistent_fx_graph_cache(self):
        for target, kwargs in (
            (
                flydsl_kernel_wrapper_mutation,
                {
                    "launcher_idx": 0,
                    "call_spec_idx": 0,
                    "args": (),
                    "mutated_arg_indices": (),
                },
            ),
            (
                flydsl_kernel_wrapper_functional,
                {
                    "launcher_idx": 0,
                    "call_spec_idx": 0,
                    "args": (),
                    "mutated_arg_indices": (),
                    "tensors_to_clone": (),
                },
            ),
        ):
            with self.subTest(target=target):
                graph = torch.fx.Graph()
                result = graph.call_function(target, kwargs=kwargs)
                graph.output(result)
                graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

                with self.assertRaisesRegex(
                    BypassFxGraphCache,
                    f"Can't cache HigherOrderOperator: {target.name()}",
                ):
                    CacheabilityValidator(
                        graph_module,
                        require_shape_env=False,
                    ).validate()

    @requires_flydsl
    def test_export_serde_rejects_process_local_indices(self):
        @flyc.jit
        def launcher(out: fx.Tensor, inp: fx.Tensor):
            pass

        captured = torch.library.wrap_flydsl(launcher, mutates_args={"out"})

        class Model(torch.nn.Module):
            def forward(self, inp):
                out = torch.empty_like(inp)
                captured(out, inp)
                return out

        mutation = torch.export.export(Model(), (torch.randn(8),))
        functional = mutation.run_decompositions()

        for name, exported in (("mutation", mutation), ("functional", functional)):
            with self.subTest(name=name):
                with self.assertRaisesRegex(RuntimeError, "process-local"):
                    torch.export.save(exported, io.BytesIO())

                legacy_artifact = io.BytesIO()
                with mock.patch(
                    "torch._export.serde.serialize._is_flydsl_kernel_wrapper",
                    return_value=False,
                ):
                    torch.export.save(exported, legacy_artifact)
                legacy_artifact.seek(0)
                with self.assertRaisesRegex(RuntimeError, "error when deserializing"):
                    torch.export.load(legacy_artifact)

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

    @requires_flydsl
    def test_run_decompositions_functionalizes_mutation(self):
        captured = torch.library.wrap_flydsl(_launcher, mutates_args="out")

        class Model(torch.nn.Module):
            def forward(self, out, inp):
                captured(out, inp, inp.numel())
                return out

        exported = torch.export.export(
            Model(),
            (torch.empty(8), torch.randn(8)),
        ).run_decompositions()

        self.assertEqual(
            1,
            len(
                exported.graph_module.graph.find_nodes(
                    op="call_function",
                    target=flydsl_kernel_wrapper_functional,
                )
            ),
        )
        self.assertEqual(
            0,
            len(
                exported.graph_module.graph.find_nodes(
                    op="call_function",
                    target=flydsl_kernel_wrapper_mutation,
                )
            ),
        )
        self.assertTrue(
            any(
                output.kind is OutputKind.USER_INPUT_MUTATION and output.target == "out"
                for output in exported.graph_signature.output_specs
            )
        )

    @requires_flydsl
    def test_run_decompositions_rejects_aliased_arguments(self):
        captured = torch.library.wrap_flydsl(_launcher, mutates_args="out")

        class Model(torch.nn.Module):
            def forward(self, inp):
                captured(inp, inp, inp.numel())
                return inp

        exported = torch.export.export(Model(), (torch.randn(8),))
        with self.assertRaisesRegex(RuntimeError, "aliased launcher arguments"):
            exported.run_decompositions()

    def test_aot_eager_functionalizes_mutation(self):
        registration = TraceableFlyDSLLauncher(_EagerLauncher(), (0, 1))
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({})
        torch._dynamo.allow_in_graph(flydsl_kernel_wrapper_mutation)

        def fn(out, workspace, inp):
            flydsl_kernel_wrapper_mutation(
                registration.launcher_idx,
                call_spec_idx,
                (out, workspace, inp),
                (0, 1),
            )
            return out + workspace

        out = torch.zeros(4)
        workspace = torch.zeros(4)
        inp = torch.arange(4, dtype=torch.float32)
        actual = torch.compile(fn, backend="aot_eager", fullgraph=True)(
            out,
            workspace,
            inp,
        )

        torch.testing.assert_close(actual, inp + 7)
        torch.testing.assert_close(out, inp)
        torch.testing.assert_close(workspace, torch.full_like(workspace, 7))

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

    def test_functional_wrapper_clones_offset_view_safely(self):
        registration = TraceableFlyDSLLauncher(_EagerLauncher(), (0, 1))
        base = torch.full((8,), -1.0)
        out = base[2:6]
        workspace = torch.zeros(4)
        inp = torch.arange(4, dtype=torch.float32)
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({})

        actual_out, _ = flydsl_kernel_wrapper_functional(
            registration.launcher_idx,
            call_spec_idx,
            (out, workspace, inp),
            (0, 1),
            (0,),
        )

        torch.testing.assert_close(actual_out, inp)
        torch.testing.assert_close(base, torch.full_like(base, -1))
        self.assertEqual(0, actual_out.storage_offset())

    def test_functional_wrapper_accepts_disjoint_views(self):
        registration = TraceableFlyDSLLauncher(_EagerLauncher(), (0, 1))
        base = torch.zeros(8)
        out = base[:4]
        workspace = base[4:]
        inp = torch.arange(4, dtype=torch.float32)
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({})

        actual_out, actual_workspace = flydsl_kernel_wrapper_functional(
            registration.launcher_idx,
            call_spec_idx,
            (out, workspace, inp),
            (0, 1),
            (0, 1),
        )

        torch.testing.assert_close(actual_out, inp)
        torch.testing.assert_close(actual_workspace, torch.full_like(workspace, 7))
        torch.testing.assert_close(base, torch.zeros_like(base))

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

    def test_side_table_does_not_reuse_indices_after_reset(self):
        first = TraceableFlyDSLLauncher(_EagerLauncher(), (0, 1))
        first_call_spec = flydsl_launcher_side_table.add_call_spec({0: "first"})

        flydsl_launcher_side_table.reset_table()
        second = TraceableFlyDSLLauncher(_EagerLauncher(), (0, 1))
        second_call_spec = flydsl_launcher_side_table.add_call_spec({0: "second"})

        self.assertNotEqual(first.launcher_idx, second.launcher_idx)
        self.assertNotEqual(first_call_spec, second_call_spec)
        with self.assertRaisesRegex(AssertionError, "was not registered"):
            flydsl_launcher_side_table.get_registration(first.launcher_idx)
        with self.assertRaisesRegex(AssertionError, "was not registered"):
            flydsl_launcher_side_table.get_call_spec(first_call_spec)

    @requires_flydsl
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

    @requires_flydsl
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

    @requires_flydsl
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

    @requires_flydsl
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

    @requires_flydsl
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

    @requires_flydsl
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

    @requires_flydsl
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
