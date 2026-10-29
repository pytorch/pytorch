# Owner(s): ["module: inductor"]

import contextlib
import io
from unittest.mock import patch

import torch
from torch._inductor import config
from torch._inductor.loops_ir import (
    load_cpp_ir,
    LoopsIrUnsupportedError,
    lower_fx_to_cpp,
)
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx


class TestCppLoopsIr(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cpp = load_cpp_ir()

    def lower(self, function, *inputs):
        graph = make_fx(function)(*inputs)
        return lower_fx_to_cpp(graph, list(inputs))

    def test_permute_add_preserves_view_and_stores(self):
        program = self.lower(
            lambda x, y: x.permute(1, 0) + y, torch.randn(4, 8), torch.randn(8, 4)
        )
        dump = program.dump()
        self.assertIn("view_read", dump)
        self.assertIn("strides", dump)
        self.assertIn("store", dump)
        self.assertIn("32", dump)
        self.assertNotIn("layout", dump)
        self.assertNotIn("value_read", dump)
        self.assertEqual(dump, program.dump())

    def test_tiny_model_keeps_separate_materializations(self):
        dump = self.lower(
            lambda x, w: torch.relu((x * w).sum(dim=1)),
            torch.randn(2, 3),
            torch.randn(3),
        ).dump()
        self.assertIn("broadcast", dump)
        self.assertIn("reduction_body", dump)
        self.assertIn("post_body", dump)
        self.assertIn("accumulate", dump)
        self.assertIn("mul", dump)
        self.assertIn("relu", dump)
        self.assertEqual(dump.count("compute "), 3)
        self.assertEqual(dump.count("store "), 3)

    def test_keepdim_and_zero_extent(self):
        for shape in ((2, 3), (2, 0)):
            with self.subTest(shape=shape):
                dump = self.lower(
                    lambda x: x.sum(dim=1, keepdim=True), torch.randn(*shape)
                ).dump()
                self.assertIn("accumulate", dump)
                self.assertIn("store", dump)
                self.assertIn("output0", dump)

    def test_scalar_sum(self):
        dump = self.lower(lambda x: x.sum(), torch.randn(())).dump()
        self.assertIn("store", dump)
        self.assertNotIn("accumulate", dump)

    def test_empty_axes_are_reduce_all(self):
        x = torch.randn(2, 3)
        self.assertEqual(x.sum(dim=[]), x.sum())
        dump = self.lower(lambda x: x.sum(dim=[]), x).dump()
        self.assertIn("r0", dump)
        self.assertIn("r1", dump)
        self.assertIn("accumulate", dump)

    def test_negative_and_multiple_reduction_axes(self):
        dump = self.lower(lambda x: x.sum(dim=(-1, -2)), torch.randn(2, 3, 4)).dump()
        self.assertIn("r0", dump)
        self.assertIn("r1", dump)
        self.assertIn("accumulate", dump)

    def test_returned_view_and_noncontiguous_input(self):
        dump = self.lower(lambda x: x.permute(1, 0), torch.randn(4, 8).t()).dump()
        self.assertIn("view", dump)
        self.assertIn("output0", dump)
        self.assertNotIn("compute ", dump)

    def test_input_pointer_is_relative_to_first_tensor_element(self):
        x = torch.randn(4, 8)[1:3]
        self.assertGreater(x.storage_offset(), 0)
        program = self.lower(lambda x: x.permute(1, 0), x)
        self.assertEqual(program.input_descriptors[0].offset, 0)
        self.assertEqual(program.inputs[0].elements, 16)
        self.assertIn("view", program.dump())

    def test_nested_alias_and_expand(self):
        dump = self.lower(
            lambda x: x.permute(1, 0).expand(5, 3, 2), torch.randn(2, 3)
        ).dump()
        self.assertGreaterEqual(dump.count("view "), 2)
        self.assertIn("output0", dump)

    def test_matrix_multiply_is_explicit_opaque_call(self):
        dump = self.lower(
            lambda x, w: x @ w, torch.randn(2, 3), torch.randn(3, 4)
        ).dump()
        self.assertIn("extern", dump)
        self.assertIn("aten.mm.default", dump)
        self.assertIn("unknown", dump)
        self.assertIn("tensor_argument lhs", dump)
        self.assertIn("tensor_argument rhs", dump)
        self.assertIn("tensor_argument result", dump)
        self.assertNotIn("compute ", dump)

    def test_extern_repeated_operands_and_intermediate_descriptors(self):
        dump = self.lower(lambda x: (x * 2) @ (x * 2), torch.randn(2, 2)).dump()
        self.assertIn("tensor_argument lhs", dump)
        self.assertIn("sizes=[2, 2]", dump)
        repeated = self.lower(lambda x: x @ x, torch.randn(2, 2)).dump()
        self.assertIn("inputs=[x_1, x_1]", repeated)

    def test_extern_requires_tensor_descriptors(self):
        cpp = self.cpp
        program = cpp.Program("incomplete_extern")
        program.add_input(cpp.Buffer("x", "f32", 4))
        program.add_extern_call(
            cpp.ExternCall(
                "call",
                "aten.mm.default",
                ["x"],
                [cpp.Buffer("out", "f32", 4)],
                [],
                "unknown",
                False,
            )
        )
        with self.assertRaisesRegex(ValueError, "descriptor"):
            program.verify()

    def test_view_offsets_compose_relative_to_source_pointer(self):
        cpp = self.cpp
        program = cpp.Program("aliases")
        program.add_input(cpp.Buffer("x", "f32", 8))
        program.add_view(cpp.View("v0", "x", [4], [1], 2))
        program.add_view(cpp.View("v1", "v0", [2], [1], 1))
        program.add_output(cpp.Descriptor("out", "v1", [2], [1], 0))
        self.assertIn("offset=1", program.dump())
        invalid = cpp.Program("invalid_alias")
        invalid.add_input(cpp.Buffer("x", "f32", 8))
        invalid.add_view(cpp.View("v0", "x", [4], [1], 2))
        invalid.add_view(cpp.View("v1", "v0", [2], [1], 5))
        with self.assertRaisesRegex(ValueError, "capacity"):
            invalid.verify()

    def _pointwise_program(self, name, numerics="strict_eager"):
        cpp = self.cpp
        program = cpp.Program(name, numerics)
        program.add_input(cpp.Buffer("x", "f32", 4))
        compute = cpp.Compute(
            "op",
            ["x"],
            [cpp.Buffer("out", "f32", 4)],
            [cpp.Range("p0", 0, 4)],
            [],
            [],
        )
        compute.add_post_instruction(
            cpp.Instruction.load(0, "x", cpp.Index.variable("p0"))
        )
        compute.add_post_instruction(cpp.Instruction.relu(1, 0))
        compute.add_post_instruction(
            cpp.Instruction.store("out", cpp.Index.variable("p0"), 1)
        )
        program.add_compute(compute)
        program.add_output(cpp.Descriptor("output", "out", [4], [1], 0))
        return program

    def test_program_numerics_defaults_to_strict(self):
        program = self.cpp.Program("default_numerics")
        self.assertEqual(program.numerics, "strict_eager")
        program.verify()
        legacy = self.cpp.Program("legacy")
        self.assertEqual(legacy.numerics, "strict_eager")

    def test_program_numerics_invalid_rejected(self):
        with self.assertRaisesRegex(ValueError, "numerics"):
            self.cpp.Program("bad", "bogus")

    def test_program_numerics_propagates_through_passes_and_lowering(self):
        for numerics in ("strict_eager", "inductor"):
            with self.subTest(numerics=numerics):
                program = self._pointwise_program("pipe", numerics)
                program.verify()
                self.assertEqual(program.numerics, numerics)
                for name in self.cpp.pass_names():
                    passed = self.cpp.run_pass(name, program)
                    self.assertEqual(passed.numerics, numerics)
                optimized = self.cpp.optimize_program(program)
                self.assertEqual(optimized.numerics, numerics)
                kernels = self.cpp.lower_to_kernels(optimized)
                self.assertEqual(kernels.numerics, numerics)
                for node in kernels.nodes:
                    if isinstance(node, self.cpp.KernelFunction):
                        self.assertEqual(node.numerics, numerics)
                fused = self.cpp.fuse_kernels(kernels)
                self.assertEqual(fused.numerics, numerics)
                for node in fused.nodes:
                    if isinstance(node, self.cpp.KernelFunction):
                        self.assertEqual(node.numerics, numerics)
                tiled = self.cpp.lower_to_tiled(fused, self.cpp.TileOptions(256, 4))
                for function in tiled.functions:
                    self.assertEqual(function.numerics, numerics)

    def test_tiled_flags_defaults_and_strict_rejection(self):
        options = self.cpp.TileOptions()
        self.assertEqual(options.block_size, 256)
        self.assertEqual(options.num_warps, 4)
        self.assertFalse(options.enable_fp_fusion)
        self.assertTrue(options.disable_ftz)
        program = self._pointwise_program("flags")
        kernels = self.cpp.fuse_kernels(self.cpp.lower_to_kernels(program))
        tiled = self.cpp.lower_to_tiled(kernels, self.cpp.TileOptions(256, 4))
        function = tiled.functions[0]
        self.assertFalse(function.enable_fp_fusion)
        self.assertTrue(function.disable_ftz)
        self.assertEqual(function.numerics, "strict_eager")
        with self.assertRaisesRegex(ValueError, "fp_fusion"):
            self.cpp.lower_to_tiled(kernels, self.cpp.TileOptions(256, 4, True, True))
        with self.assertRaisesRegex(ValueError, "disable_ftz"):
            self.cpp.lower_to_tiled(kernels, self.cpp.TileOptions(256, 4, False, False))
        with self.assertRaisesRegex(ValueError, "fp_fusion"):
            self.cpp.TiledFunction("bad", [], 1, 256, 4, True, True, "strict_eager")
        with self.assertRaisesRegex(ValueError, "disable_ftz"):
            self.cpp.TiledFunction("bad", [], 1, 256, 4, False, False, "strict_eager")

    def test_tiled_inductor_policy_permits_effective_flags(self):
        program = self._pointwise_program("ind_flags", "inductor")
        kernels = self.cpp.fuse_kernels(self.cpp.lower_to_kernels(program))
        options = self.cpp.TileOptions(128, 2, True, False)
        self.assertTrue(options.enable_fp_fusion)
        self.assertFalse(options.disable_ftz)
        tiled = self.cpp.lower_to_tiled(kernels, options)
        function = tiled.functions[0]
        self.assertTrue(function.enable_fp_fusion)
        self.assertFalse(function.disable_ftz)
        self.assertEqual(function.numerics, "inductor")
        self.assertEqual(function.block_size, 128)
        self.assertEqual(function.num_warps, 2)
        tiled.verify()

    def test_kernel_mixed_policy_rejected(self):
        with self.assertRaisesRegex(ValueError, "numerics"):
            self.cpp.KernelProgram("bad", "bogus")
        with self.assertRaisesRegex(ValueError, "numerics"):
            self.cpp.KernelFunction("bad", [], "bogus")
        program = self.cpp.KernelProgram("mixed", "strict_eager")
        foreign = self.cpp.KernelFunction("foreign", [], "inductor")
        with self.assertRaisesRegex(ValueError, "numerics"):
            program.add_function(foreign)

    def test_callarg_layout_and_memory_format_round_trip(self):
        cpp = self.cpp
        for name in (
            "strided",
            "sparse_coo",
            "sparse_csr",
            "sparse_csc",
            "sparse_bsr",
            "sparse_bsc",
            "_mkldnn",
            "mkldnn",
            "jagged",
        ):
            with self.subTest(layout=name):
                arg = cpp.CallArg.layout(name)
                self.assertEqual(arg.kind_name, "layout")
                self.assertEqual(arg.text, name)
        for name in (
            "contiguous_format",
            "preserve_format",
            "channels_last",
            "channels_last_3d",
        ):
            with self.subTest(memory_format=name):
                arg = cpp.CallArg.memory_format(name)
                self.assertEqual(arg.kind_name, "memory_format")
                self.assertEqual(arg.text, name)
        with self.assertRaisesRegex(ValueError, "layout"):
            cpp.CallArg.layout("bogus_layout")
        with self.assertRaisesRegex(ValueError, "memory"):
            cpp.CallArg.memory_format("bogus_format")
        program = cpp.Program("full_layout")
        program.add_input(cpp.Buffer("x", "f32", 4))
        arguments = [
            cpp.CallArg.tensor("x"),
            cpp.CallArg.layout("strided"),
            cpp.CallArg.memory_format("contiguous_format"),
        ]
        keywords = [cpp.CallKeyword("layout", cpp.CallArg.layout("strided"))]
        program.add_extern_call(
            cpp.ExternCall.structured(
                "call",
                "aten.full.default",
                arguments,
                keywords,
                [cpp.Buffer("out", "f32", 4)],
                [
                    cpp.Descriptor("arg0", "x", [4], [1], 0),
                    cpp.Descriptor("result", "out", [4], [1], 0),
                ],
            )
        )
        program.add_output(cpp.Descriptor("output", "out", [4], [1], 0))
        program.verify()
        dump = program.dump()
        self.assertIn("layout(strided)", dump)
        self.assertIn("memory_format(contiguous_format)", dump)

    def test_callarg_nested_layout(self):
        cpp = self.cpp
        nested = cpp.CallArg.list(
            [
                cpp.CallArg.layout("strided"),
                cpp.CallArg.memory_format("preserve_format"),
            ]
        )
        program = cpp.Program("nested_layout")
        program.add_input(cpp.Buffer("x", "f32", 4))
        program.add_extern_call(
            cpp.ExternCall.structured(
                "call",
                "aten.clone.default",
                [cpp.CallArg.tensor("x"), nested],
                [],
                [cpp.Buffer("out", "f32", 4)],
                [
                    cpp.Descriptor("arg0", "x", [4], [1], 0),
                    cpp.Descriptor("result", "out", [4], [1], 0),
                ],
            )
        )
        program.add_output(cpp.Descriptor("output", "out", [4], [1], 0))
        program.verify()
        self.assertIn("layout(strided)", program.dump())

    def test_nonfinite_float_mask_constant_round_trip(self):
        cpp = self.cpp
        inf = cpp.CallArg.floating(float("inf"))
        self.assertEqual(inf.kind_name, "float")
        self.assertEqual(inf.float_value, float("inf"))
        neg_inf = cpp.CallArg.floating(float("-inf"))
        self.assertEqual(neg_inf.float_value, float("-inf"))
        with self.assertRaisesRegex(ValueError, "NaN"):
            cpp.CallArg.floating(float("nan"))
        program = cpp.Program("inf_mask")
        program.add_input(cpp.Buffer("x", "f32", 4))
        program.add_extern_call(
            cpp.ExternCall.structured(
                "call",
                "aten.full.default",
                [cpp.CallArg.tensor("x"), cpp.CallArg.floating(float("inf"))],
                [],
                [cpp.Buffer("out", "f32", 4)],
                [
                    cpp.Descriptor("arg0", "x", [4], [1], 0),
                    cpp.Descriptor("result", "out", [4], [1], 0),
                ],
            )
        )
        program.add_output(cpp.Descriptor("output", "out", [4], [1], 0))
        program.verify()
        self.assertIn("inf", program.dump())

    def test_non_f32_boundary_views_accepted(self):
        cpp = self.cpp
        for dtype in ("i64", "bool"):
            with self.subTest(dtype=dtype):
                program = cpp.Program(f"{dtype}_view")
                program.add_input(cpp.Buffer("idx", dtype, 8))
                program.add_view(cpp.View("v0", "idx", [4, 2], [2, 1], 0))
                program.add_output(cpp.Descriptor("out", "v0", [4, 2], [2, 1], 0))
                program.verify()
                self.assertIn("view v0", program.dump())
        program = cpp.Program("i64_extern")
        program.add_input(cpp.Buffer("idx", "i64", 4))
        program.add_extern_call(
            cpp.ExternCall.structured(
                "call",
                "aten.clone.default",
                [cpp.CallArg.tensor("idx")],
                [],
                [cpp.Buffer("out", "i64", 4)],
                [
                    cpp.Descriptor("arg0", "idx", [4], [1], 0),
                    cpp.Descriptor("result", "out", [4], [1], 0),
                ],
            )
        )
        program.add_output(cpp.Descriptor("output", "out", [4], [1], 0))
        program.verify()
        self.assertNotIn("compute ", program.dump())

    def test_unsupported_operator_is_contextual(self):
        with self.assertRaisesRegex(LoopsIrUnsupportedError, "FX node.*sin"):
            self.lower(lambda x: x.sin(), torch.randn(3))

    def test_unsupported_dtype(self):
        with self.assertRaisesRegex(LoopsIrUnsupportedError, "f32"):
            self.lower(lambda x: x + 1, torch.ones(3, dtype=torch.float64))

    def test_symbolic_input_is_not_silently_specialized(self):
        from torch._subclasses.fake_tensor import FakeTensorMode
        from torch.fx.experimental.symbolic_shapes import ShapeEnv

        x = torch.randn(3)
        mode = FakeTensorMode(shape_env=ShapeEnv())
        fake_x = mode.from_tensor(x, static_shapes=False)
        graph = make_fx(lambda x: x + 1)(x)
        with self.assertRaisesRegex(LoopsIrUnsupportedError, "symbolic"):
            lower_fx_to_cpp(graph, [fake_x])

    def test_mutation_is_not_an_opaque_pure_fallback(self):
        with self.assertRaisesRegex(LoopsIrUnsupportedError, "unsupported operation"):
            self.lower(lambda x: x.add_(1), torch.randn(3))

    def test_multi_accumulator_builder_and_explicit_stores(self):
        cpp = self.cpp
        p = cpp.Program("two_sums")
        p.add_input(cpp.Buffer("x", "f32", 6))
        compute = cpp.Compute(
            "reduce",
            ["x"],
            [cpp.Buffer("s", "f32", 2), cpp.Buffer("q", "f32", 2)],
            [cpp.Range("p0", 0, 2)],
            [cpp.Range("r0", 0, 3)],
            [
                cpp.Accumulator("sum", "f32", 0.0, "add"),
                cpp.Accumulator("sumsq", "f32", 0.0, "add"),
            ],
        )
        point = cpp.Index.variable("p0")
        offset = cpp.Index.binary(
            "add",
            cpp.Index.binary("mul", point, cpp.Index.constant(3)),
            cpp.Index.variable("r0"),
        )
        compute.add_reduction_instruction(cpp.Instruction.load(0, "x", offset))
        compute.add_reduction_instruction(cpp.Instruction.binary(1, "mul", 0, 0))
        compute.add_reduction_instruction(cpp.Instruction.accumulate([0, 1]))
        compute.add_post_instruction(cpp.Instruction.read_accumulator(0, 0))
        compute.add_post_instruction(cpp.Instruction.read_accumulator(1, 1))
        compute.add_post_instruction(cpp.Instruction.store("s", point, 0))
        compute.add_post_instruction(cpp.Instruction.store("q", point, 1))
        p.add_compute(compute)
        p.add_output(cpp.Descriptor("output0", "s", [2], [1], 0))
        p.add_output(cpp.Descriptor("output1", "q", [2], [1], 0))
        dump = p.dump()
        self.assertIn("sumsq", dump)
        self.assertEqual(dump.count("store "), 2)
        self.assertNotIn("yield", dump)

    def invalid_compute(self, instruction):
        cpp = self.cpp
        p = cpp.Program("invalid")
        p.add_input(cpp.Buffer("x", "f32", 2))
        compute = cpp.Compute(
            "op",
            ["x"],
            [cpp.Buffer("out", "f32", 2)],
            [cpp.Range("p0", 0, 2)],
            [],
            [],
        )
        compute.add_post_instruction(instruction)
        p.add_compute(compute)
        return p

    def test_verifier_rejects_unknown_ssa(self):
        p = self.invalid_compute(
            self.cpp.Instruction.store("out", self.cpp.Index.variable("p0"), 9)
        )
        with self.assertRaises(ValueError):
            p.dump()

    def test_verifier_rejects_out_of_scope_index(self):
        p = self.invalid_compute(
            self.cpp.Instruction.load(0, "x", self.cpp.Index.variable("r0"))
        )
        with self.assertRaises(ValueError):
            p.verify()

    def test_verifier_rejects_out_of_bounds_constant(self):
        p = self.invalid_compute(
            self.cpp.Instruction.load(0, "x", self.cpp.Index.constant(2))
        )
        with self.assertRaises(ValueError):
            p.verify()

    def test_verifier_rejects_index_overflow(self):
        cpp = self.cpp
        offset = cpp.Index.binary(
            "add", cpp.Index.constant(2**63 - 1), cpp.Index.constant(1)
        )
        program = self.invalid_compute(cpp.Instruction.load(0, "x", offset))
        with self.assertRaisesRegex(ValueError, "overflow"):
            program.verify()

    def test_verifier_rejects_constant_outside_f32(self):
        with self.assertRaisesRegex(ValueError, "constant"):
            program = self.invalid_compute(self.cpp.Instruction.constant(0, 1e100))
            program.verify()

    def test_verifier_rejects_post_body_reduction_variable(self):
        cpp = self.cpp
        program = cpp.Program("bad_phase_scope")
        program.add_input(cpp.Buffer("x", "f32", 2))
        compute = cpp.Compute(
            "sum",
            ["x"],
            [cpp.Buffer("out", "f32", 1)],
            [],
            [cpp.Range("r0", 0, 2)],
            [cpp.Accumulator("sum", "f32", 0, "add")],
        )
        compute.add_reduction_instruction(
            cpp.Instruction.load(0, "x", cpp.Index.variable("r0"))
        )
        compute.add_reduction_instruction(cpp.Instruction.accumulate([0]))
        compute.add_post_instruction(cpp.Instruction.read_accumulator(0, 0))
        compute.add_post_instruction(
            cpp.Instruction.store("out", cpp.Index.variable("r0"), 0)
        )
        program.add_compute(compute)
        with self.assertRaisesRegex(ValueError, "out of scope"):
            program.verify()

    def test_verifier_rejects_post_body_accumulate(self):
        p = self.invalid_compute(self.cpp.Instruction.accumulate([0]))
        with self.assertRaises(ValueError):
            p.verify()

    def test_verifier_rejects_wrong_accumulator_arity(self):
        cpp = self.cpp
        p = cpp.Program("invalid_accumulators")
        p.add_input(cpp.Buffer("x", "f32", 2))
        compute = cpp.Compute(
            "op",
            ["x"],
            [cpp.Buffer("out", "f32", 1)],
            [],
            [cpp.Range("r0", 0, 2)],
            [cpp.Accumulator("sum", "f32", 0, "add")],
        )
        compute.add_reduction_instruction(cpp.Instruction.accumulate([]))
        p.add_compute(compute)
        with self.assertRaises(ValueError):
            p.verify()

    def test_real_inductor_calls_cpp_printer_without_changing_execution(self):
        def model(x, w):
            return torch.relu((x * w).sum(dim=1))

        inputs = (torch.randn(2, 3), torch.randn(3))
        output = io.StringIO()
        with (
            config.patch(
                {
                    "experimental_cpp_loops_ir_dump": True,
                    "fx_graph_cache": False,
                }
            ),
            contextlib.redirect_stdout(output),
        ):
            actual = torch.compile(model, backend="inductor", fullgraph=True)(*inputs)
        self.assertEqual(actual, model(*inputs))
        dump = output.getvalue()
        self.assertIn("program fx_program", dump)
        self.assertIn("accumulate", dump)
        self.assertIn("store", dump)

    def test_default_off_does_not_load_extension(self):
        with (
            config.patch(
                {
                    "experimental_cpp_loops_ir_dump": False,
                    "fx_graph_cache": False,
                }
            ),
            patch(
                "torch._inductor.loops_ir.load_cpp_ir",
                side_effect=AssertionError("unexpected extension load"),
            ),
        ):
            x = torch.randn(3)
            self.assertEqual(torch.compile(lambda x: x * 2, fullgraph=True)(x), x * 2)


if __name__ == "__main__":
    run_tests()
