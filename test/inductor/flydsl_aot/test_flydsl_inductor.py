# Owner(s): ["module: inductor"]
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.

import tempfile
import unittest
from types import SimpleNamespace
from unittest import mock

import torch
from torch._higher_order_ops.flydsl_kernel_wrap import (
    flydsl_kernel_wrapper_functional,
    flydsl_kernel_wrapper_mutation,
    flydsl_launcher_side_table,
    FlyDSLPythonLauncher,
    TraceableFlyDSLLauncher,
)
from torch._inductor import config, ir
from torch._inductor.codecache import (
    BypassFxGraphCache,
    CacheabilityValidator,
    ROCmCodeCache,
)
from torch._inductor.codegen.cpp_wrapper_gpu import CppWrapperGpu
from torch._inductor.codegen.flydsl.flydsl_aot import (
    compile_launcher,
    define_aot_kernel,
    FlyDSLAOTArtifact,
    generate_aot_kernel_call,
)
from torch._inductor.codegen.flydsl.flydsl_utils import runtime_available
from torch._inductor.codegen.flydsl.user_defined_kernel import (
    decompose_functional_wrapper,
    lower_flydsl_kernel,
)
from torch._inductor.fx_passes.post_grad import _has_flydsl_kernel_wrapper
from torch._inductor.fx_passes.reinplace import reinplace_inplaceable_ops_core
from torch._inductor.utils import IndentedBuffer
from torch._inductor.virtualized import V
from torch._subclasses.fake_tensor import FakeTensorMode
from torch.fx.experimental.proxy_tensor import make_fx
from torch.testing._internal.common_utils import TestCase


HAS_FLYDSL = runtime_available()
if HAS_FLYDSL:
    import flydsl.compiler as flyc
    import flydsl.expr as fx

    @flyc.jit
    def _launcher(out: fx.Tensor, inp: fx.Tensor, *, rows: fx.Int32):
        pass

else:
    _launcher = None


@unittest.skipUnless(HAS_FLYDSL, "FlyDSL is not available")
class FlyDSLInductorTest(TestCase):
    def setUp(self):
        flydsl_launcher_side_table.reset_table()

    def test_lowering_preserves_arguments_and_mutations(self):
        captured = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )
        launcher_idx = captured.launcher_idx
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({})
        out = mock.create_autospec(ir.TensorBox, instance=True)
        inp = mock.create_autospec(ir.TensorBox, instance=True)
        for tensor in (out, inp):
            tensor.get_size.return_value = (8,)
            tensor.get_stride.return_value = (1,)
            tensor.get_dtype.return_value = torch.float32
        graph = mock.Mock()
        graph.sizevars.optimization_hints.side_effect = lambda values: tuple(
            int(value) for value in values
        )

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.codegen.flydsl.user_defined_kernel.UserDefinedFlyDSLKernel"
            ) as kernel,
        ):
            lower_flydsl_kernel(
                launcher_idx=launcher_idx,
                call_spec_idx=call_spec_idx,
                args=(out, inp, 8),
                mutated_arg_indices=(0,),
            )

        kernel.assert_called_once_with(
            launcher_idx=launcher_idx,
            call_spec_idx=call_spec_idx,
            kernel_args=(out, inp, 8),
            mutated_arg_indices=(0,),
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

    def test_aot_kernel_rejects_cpp_only_packaging(self):
        captured = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )
        wrapper = SimpleNamespace(user_defined_kernel_cache={})
        graph = SimpleNamespace(aot_mode=True)

        with (
            V.set_graph_handler(graph),
            config.patch({"aot_inductor.package_cpp_only": True}),
            mock.patch(
                "torch._inductor.codegen.flydsl.flydsl_aot.compile_launcher"
            ) as compile_launcher_mock,
            self.assertRaisesRegex(NotImplementedError, "package_cpp_only"),
        ):
            define_aot_kernel(
                wrapper,
                captured.launcher_idx,
                flydsl_launcher_side_table.add_call_spec({}),
                (),
            )

        compile_launcher_mock.assert_not_called()

    def test_post_grad_decomposes_functional_wrapper(self):
        captured = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )
        launcher_idx = captured.launcher_idx
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({})
        fake_mode = FakeTensorMode()
        graph = torch.fx.Graph()
        out = graph.placeholder("out")
        inp = graph.placeholder("inp")
        out.meta["val"] = fake_mode.from_tensor(torch.empty(8))
        inp.meta["val"] = fake_mode.from_tensor(torch.empty(8))
        functional = graph.call_function(
            flydsl_kernel_wrapper_functional,
            kwargs={
                "launcher_idx": launcher_idx,
                "call_spec_idx": call_spec_idx,
                "args": (out, inp, 8),
                "mutated_arg_indices": (0,),
                "tensors_to_clone": (0,),
            },
        )
        functional.meta["val"] = (fake_mode.from_tensor(torch.empty(8)),)
        graph.output(functional)
        graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

        decompose_functional_wrapper(graph_module.graph)
        graph_module.graph.lint()

        self.assertEqual(
            [],
            graph_module.graph.find_nodes(
                op="call_function",
                target=flydsl_kernel_wrapper_functional,
            ),
        )
        self.assertEqual(
            1,
            len(
                graph_module.graph.find_nodes(
                    op="call_function",
                    target=flydsl_kernel_wrapper_mutation,
                )
            ),
        )

    def test_post_grad_skips_graphs_without_flydsl(self):
        graph = torch.fx.Graph()
        inp = graph.placeholder("inp")
        graph.output(inp)
        graph_module = torch.fx.GraphModule(torch.nn.Module(), graph)

        self.assertFalse(_has_flydsl_kernel_wrapper(graph_module.graph))
        with mock.patch(
            "torch._inductor.codegen.flydsl.user_defined_kernel.PatternMatcherPass",
            side_effect=AssertionError("unneeded FlyDSL pattern pass"),
        ):
            decompose_functional_wrapper(graph_module.graph)

    def test_post_grad_guard_finds_nested_flydsl(self):
        child_graph = torch.fx.Graph()
        child_graph.call_function(
            flydsl_kernel_wrapper_functional,
            kwargs={
                "launcher_idx": 0,
                "call_spec_idx": 0,
                "args": (),
                "mutated_arg_indices": (),
                "tensors_to_clone": (),
            },
        )
        child_graph.output(None)
        child = torch.fx.GraphModule(torch.nn.Module(), child_graph)
        root = torch.nn.Module()
        root.add_module("child", child)
        graph = torch.fx.Graph()
        child_attr = graph.get_attr("child")
        graph.output(child_attr)
        graph_module = torch.fx.GraphModule(root, graph)

        self.assertTrue(_has_flydsl_kernel_wrapper(graph_module.graph))

    def test_python_launcher_restores_compile_time_arguments(self):
        class EagerLauncher:
            def __init__(self) -> None:
                self.func = self.launch

            def launch(self, out, factor, inp) -> None:
                out.copy_(inp * factor)

            def __call__(self, out, factor, inp) -> None:
                self.launch(out, factor, inp)

        captured = TraceableFlyDSLLauncher(
            EagerLauncher(),
            (0,),
        )
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({1: 3})
        launcher = FlyDSLPythonLauncher(captured.launcher_idx, call_spec_idx)
        inp = torch.arange(4, dtype=torch.float32)
        out = torch.empty_like(inp)

        launcher.run(out, inp)

        torch.testing.assert_close(out, inp * 3)

    def _functional_graph(self, *, keep_original: bool) -> torch.fx.GraphModule:
        captured = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({})

        def f(inp):
            out = torch.empty_like(inp)
            (result,) = flydsl_kernel_wrapper_functional(
                captured.launcher_idx,
                call_spec_idx,
                (out, inp, inp.numel()),
                (0,),
                (0,),
            )
            return (result, out) if keep_original else result

        return make_fx(f, tracing_mode="fake")(torch.randn(8))

    def test_reinplace_removes_clone_for_fresh_output(self):
        graph_module = self._functional_graph(keep_original=False)

        reinplace_inplaceable_ops_core(graph_module.graph)
        graph_module.graph.lint()

        (functional,) = graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_functional,
        )
        self.assertEqual((), functional.kwargs["tensors_to_clone"])

        decompose_functional_wrapper(graph_module.graph)
        graph_module.graph.lint()
        self.assertEqual(
            [],
            graph_module.graph.find_nodes(
                op="call_function",
                target=torch.ops.aten.clone.default,
            ),
        )

    def test_reinplace_keeps_clone_for_live_original(self):
        graph_module = self._functional_graph(keep_original=True)

        reinplace_inplaceable_ops_core(graph_module.graph)
        graph_module.graph.lint()

        (functional,) = graph_module.graph.find_nodes(
            op="call_function",
            target=flydsl_kernel_wrapper_functional,
        )
        self.assertEqual((0,), functional.kwargs["tensors_to_clone"])

    def test_inductor_compiler_exports_flydsl_object(self):
        captured = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )
        registration = flydsl_launcher_side_table.get_registration(
            captured.launcher_idx
        )
        compiled = mock.Mock()
        compiled.export_to_c.return_value = {
            "object_file_path": "/cache/launcher.o",
            "symbol": "_mlir_flydsl_launcher_test",
            "runtime_libraries": ["/runtime/libfly.so"],
            "abi": [{"kind": "stream", "arg_index": None}],
            "module_init_symbol": "flydsl_launcher_test__init",
            "module_load_symbol": "flydsl_launcher_test__load",
        }
        with (
            tempfile.TemporaryDirectory() as output_dir,
            mock.patch(
                "torch._inductor.codegen.flydsl.aot_compile.compile_aot",
                return_value=compiled,
            ) as compile_aot,
            mock.patch(
                "torch._inductor.codegen.flydsl.flydsl_aot.cache_dir",
                return_value=output_dir,
            ),
        ):
            artifact = compile_launcher(
                registration.launcher,
                ("out", "inp", 8),
                signature=registration.signature,
            )

        compile_aot.assert_called_once_with(_launcher, "out", "inp", rows=8)
        compiled.export_to_c.assert_called_once()
        self.assertEqual("_mlir_flydsl_launcher_test", artifact.symbol)
        self.assertEqual("flydsl_launcher_test__load", artifact.module_load_symbol)

    def test_inductor_compiler_exports_bound_flydsl_object(self):
        class Owner:
            @flyc.jit
            def launcher(
                self,
                out: fx.Tensor,
                block_size: fx.Constexpr[int],
                *,
                rows: fx.Int32,
            ):
                pass

        owner = Owner()
        captured = torch.library.wrap_flydsl(
            owner.launcher,
            mutates_args={"out"},
        )
        registration = flydsl_launcher_side_table.get_registration(
            captured.launcher_idx
        )
        compiled = mock.Mock()
        compiled.export_to_c.return_value = {
            "object_file_path": "/cache/launcher.o",
            "symbol": "_mlir_flydsl_bound_launcher_test",
            "runtime_libraries": [],
            "abi": [{"kind": "stream", "arg_index": None}],
            "module_init_symbol": "flydsl_bound_launcher_test__init",
            "module_load_symbol": "flydsl_bound_launcher_test__load",
        }
        with (
            tempfile.TemporaryDirectory() as output_dir,
            mock.patch(
                "torch._inductor.codegen.flydsl.aot_compile.compile_aot",
                return_value=compiled,
            ) as compile_aot,
            mock.patch(
                "torch._inductor.codegen.flydsl.flydsl_aot.cache_dir",
                return_value=output_dir,
            ),
        ):
            artifact = compile_launcher(
                registration.launcher,
                ("out", 256, 8),
                signature=registration.signature,
                bound_self=registration.bound_self,
            )

        compile_aot.assert_called_once_with(
            Owner.launcher,
            owner,
            "out",
            256,
            rows=8,
        )
        self.assertEqual("_mlir_flydsl_bound_launcher_test", artifact.symbol)

    def test_aot_kernel_registers_runtime_libraries_separately(self):
        captured = torch.library.wrap_flydsl(
            _launcher,
            mutates_args={"out"},
        )
        artifact = FlyDSLAOTArtifact(
            object_file_path="/cache/launcher.o",
            symbol="_mlir_flydsl_launcher_test",
            runtime_libraries=("/runtime/libfly.so",),
            abi=(),
            module_init_symbol="flydsl_launcher_test__init",
            module_load_symbol="flydsl_launcher_test__load",
        )
        wrapper = SimpleNamespace(
            additional_files=[],
            external_kernel_libs=set(),
            header=IndentedBuffer(),
            user_defined_kernel_cache={},
        )
        graph = SimpleNamespace(aot_mode=True)

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.codegen.flydsl.flydsl_aot.compile_launcher",
                return_value=artifact,
            ),
            mock.patch.object(ROCmCodeCache, "aot_kernels_o", []),
        ):
            result = define_aot_kernel(
                wrapper,
                captured.launcher_idx,
                flydsl_launcher_side_table.add_call_spec({}),
                (),
            )

            self.assertIs(artifact, result)
            self.assertEqual([artifact.object_file_path], ROCmCodeCache.aot_kernels_o)
            self.assertEqual(
                {
                    "-L/runtime",
                    "-l:libfly.so",
                    "-Wl,-rpath,$ORIGIN",
                },
                wrapper.external_kernel_libs,
            )
            self.assertEqual(
                list(artifact.runtime_libraries),
                wrapper.additional_files,
            )
            generated_header = wrapper.header.getvalue()
            self.assertIn("cuModuleUnload", generated_header)
            self.assertIn("cudaGetDevice", generated_header)
            self.assertIn("hipModuleUnload", generated_header)
            self.assertIn("hipGetDevice", generated_header)
            self.assertIn("cannot be shared across GPU devices", generated_header)

    def test_aot_kernel_restores_call_spec_and_bound_self(self):
        class Owner:
            @flyc.jit
            def launcher(
                self,
                out: fx.Tensor,
                block_size: fx.Constexpr[int],
                *,
                rows: fx.Int32,
            ):
                pass

        owner = Owner()
        captured = torch.library.wrap_flydsl(
            owner.launcher,
            mutates_args={"out"},
        )
        registration = flydsl_launcher_side_table.get_registration(
            captured.launcher_idx
        )
        call_spec_idx = flydsl_launcher_side_table.add_call_spec({1: 256})
        other_call_spec_idx = flydsl_launcher_side_table.add_call_spec({1: 512})
        artifact = FlyDSLAOTArtifact(
            object_file_path="/cache/launcher.o",
            symbol="_mlir_flydsl_launcher_test",
            runtime_libraries=(),
            abi=(),
            module_init_symbol="flydsl_launcher_test__init",
            module_load_symbol="flydsl_launcher_test__load",
        )
        wrapper = SimpleNamespace(
            additional_files=[],
            external_kernel_libs=set(),
            header=IndentedBuffer(),
            user_defined_kernel_cache={},
        )
        graph = SimpleNamespace(aot_mode=True)

        with (
            V.set_graph_handler(graph),
            mock.patch(
                "torch._inductor.codegen.flydsl.flydsl_aot.compile_launcher",
                return_value=artifact,
            ) as compile_launcher_mock,
            mock.patch.object(ROCmCodeCache, "aot_kernels_o", []),
        ):
            define_aot_kernel(
                wrapper,
                captured.launcher_idx,
                call_spec_idx,
                ("out", None, 8),
            )
            define_aot_kernel(
                wrapper,
                captured.launcher_idx,
                other_call_spec_idx,
                ("out", None, 8),
            )

        compile_launcher_mock.assert_has_calls(
            [
                mock.call(
                    Owner.launcher,
                    ("out", 256, 8),
                    signature=registration.signature,
                    bound_self=owner,
                ),
                mock.call(
                    Owner.launcher,
                    ("out", 512, 8),
                    signature=registration.signature,
                    bound_self=owner,
                ),
            ],
        )
        self.assertEqual(2, compile_launcher_mock.call_count)
        self.assertEqual(2, len(wrapper.user_defined_kernel_cache))

    def test_cpp_emitter_loads_module_and_packs_runtime_stream(self):
        wrapper = object.__new__(CppWrapperGpu)
        wrapper.header = IndentedBuffer()
        wrapper.lines = []
        wrapper.write_get_raw_stream = mock.Mock(return_value="stream")
        graph = mock.Mock()
        graph.aot_mode = True
        graph.name = "graph"
        graph.sizevars.simplify.side_effect = lambda value: value
        artifact = FlyDSLAOTArtifact(
            object_file_path="/cache/launcher.o",
            symbol="_mlir_flydsl_launcher_test",
            runtime_libraries=("/runtime/libfly.so",),
            abi=(
                {
                    "kind": "scalar",
                    "arg_index": 0,
                    "ctype": "int32",
                },
                {
                    "kind": "stream",
                    "arg_index": None,
                    "ctype": "pointer",
                },
            ),
            module_init_symbol="flydsl_launcher_test__init",
            module_load_symbol="flydsl_launcher_test__load",
        )

        with V.set_graph_handler(graph):
            generate_aot_kernel_call(
                wrapper,
                artifact,
                (8,),
                device=torch.device("cuda", 0),
                current_stream_idx=0,
            )
        generated = "\n".join(wrapper.lines)
        self.assertIn("__ensure_module_loaded", generated)
        self.assertIn("int32_t flydsl_0_0", generated)
        self.assertIn("reinterpret_cast<void*>(stream)", generated)
        self.assertIn(
            "_mlir_flydsl_launcher_test(flydsl_packed_0)",
            generated,
        )
        wrapper.write_get_raw_stream.assert_called_once_with(0, "graph")

    def test_cpp_emitter_packs_float_bits_and_pointer(self):
        wrapper = object.__new__(CppWrapperGpu)
        wrapper.header = IndentedBuffer()
        wrapper.lines = []
        wrapper.write_get_raw_stream = mock.Mock(return_value="stream")
        graph = mock.Mock()
        graph.aot_mode = True
        graph.name = "graph"
        graph.sizevars.simplify.side_effect = lambda value: value
        artifact = FlyDSLAOTArtifact(
            object_file_path="/cache/launcher.o",
            symbol="_mlir_flydsl_launcher_test",
            runtime_libraries=(),
            abi=(
                {
                    "kind": "scalar",
                    "arg_index": 0,
                    "ctype": "uint16",
                    "encoding": "float16_bits",
                },
                {
                    "kind": "scalar",
                    "arg_index": 1,
                    "ctype": "uint16",
                    "encoding": "bfloat16_bits",
                },
                {
                    "kind": "pointer",
                    "arg_index": 2,
                    "ctype": "pointer",
                },
            ),
            module_init_symbol="flydsl_launcher_test__init",
            module_load_symbol="flydsl_launcher_test__load",
        )

        with V.set_graph_handler(graph):
            generate_aot_kernel_call(
                wrapper,
                artifact,
                (1.5, 2.25, 4096),
                device=torch.device("cuda", 0),
                current_stream_idx=0,
            )

        generated = "\n".join(wrapper.lines)
        self.assertIn("fp16_ieee_from_fp32_value", generated)
        self.assertIn("bits_from_f32", generated)
        self.assertIn("reinterpret_cast<void*>(4096)", generated)
        self.assertIn("#include <c10/util/Half.h>", wrapper.header.getvalue())
        self.assertIn("#include <c10/util/BFloat16.h>", wrapper.header.getvalue())

    def test_cpp_emitter_uses_scheduler_auxiliary_stream(self):
        wrapper = object.__new__(CppWrapperGpu)
        wrapper.header = IndentedBuffer()
        wrapper.lines = []
        wrapper.write_get_raw_stream = mock.Mock(return_value="stream")
        graph = mock.Mock()
        graph.aot_mode = True
        graph.name = "graph"
        artifact = FlyDSLAOTArtifact(
            object_file_path="/cache/launcher.o",
            symbol="_mlir_flydsl_launcher_test",
            runtime_libraries=(),
            abi=({"kind": "stream", "arg_index": None},),
            module_init_symbol="flydsl_launcher_test__init",
            module_load_symbol="flydsl_launcher_test__load",
        )

        with V.set_graph_handler(graph):
            generate_aot_kernel_call(
                wrapper,
                artifact,
                (),
                device=torch.device("cuda", 0),
                current_stream_idx=1,
            )

        self.assertIn(
            "reinterpret_cast<void*>(stream1)",
            "\n".join(wrapper.lines),
        )
        wrapper.write_get_raw_stream.assert_not_called()

    def test_cpp_emitter_rejects_explicit_stream_argument(self):
        wrapper = object.__new__(CppWrapperGpu)
        wrapper.header = IndentedBuffer()
        wrapper.lines = []
        wrapper.write_get_raw_stream = mock.Mock(return_value="stream")
        graph = mock.Mock()
        graph.aot_mode = True
        graph.name = "graph"
        artifact = FlyDSLAOTArtifact(
            object_file_path="/cache/launcher.o",
            symbol="_mlir_flydsl_launcher_test",
            runtime_libraries=(),
            abi=({"kind": "stream", "arg_index": 0},),
            module_init_symbol="flydsl_launcher_test__init",
            module_load_symbol="flydsl_launcher_test__load",
        )

        with (
            V.set_graph_handler(graph),
            self.assertRaisesRegex(
                AssertionError,
                "explicit stream",
            ),
        ):
            generate_aot_kernel_call(
                wrapper,
                artifact,
                (1234,),
                device=torch.device("cuda", 0),
                current_stream_idx=0,
            )


if __name__ == "__main__":
    from torch.testing._internal.common_utils import run_tests

    run_tests()
