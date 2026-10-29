# Owner(s): ["module: inductor"]

import torch
from torch._inductor.loops_ir import load_cpp_ir, lower_fx_to_cpp
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx


class TestLoopsIrPasses(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cpp = load_cpp_ir()

    def lower(self, function, *inputs):
        return lower_fx_to_cpp(make_fx(function)(*inputs), list(inputs))

    def computes(self, program):
        return [n for n in program.nodes if isinstance(n, self.cpp.Compute)]

    def test_view_inlining_and_idempotence(self):
        before = self.lower(
            lambda x, y: x.permute(1, 0) + y,
            torch.randn(4, 8),
            torch.randn(8, 4),
        )
        self.assertIn("view_read", before.dump())
        after = self.cpp.optimize_program(before)
        after.verify()
        self.assertNotIn("view_read", after.dump())
        self.assertNotIn("view ", after.dump())
        self.assertEqual(after.dump(), self.cpp.optimize_program(after).dump())

    def test_local_load_cse_does_not_remove_multiply(self):
        before = self.lower(lambda x: x * x, torch.randn(5))
        self.assertEqual(before.dump().count("load "), 2)
        after = self.cpp.optimize_program(before)
        self.assertEqual(after.dump().count("load "), 1)
        self.assertIn(" = mul ", after.dump())

    def test_fp_neutral_identities_are_not_rewritten(self):
        for function, operation in (
            (lambda x: x + 0.0, "add"),
            (lambda x: x * 0.0, "mul"),
            (lambda x: x * 1.0, "mul"),
        ):
            with self.subTest(operation=operation):
                after = self.cpp.optimize_program(self.lower(function, torch.randn(5)))
                self.assertIn(f" = {operation} ", after.dump())

    def test_unit_pointwise_ranges_are_substituted(self):
        after = self.cpp.optimize_program(
            self.lower(lambda x: x + 1, torch.randn(1, 5, 1))
        )
        compute = self.computes(after)[0]
        self.assertEqual(len(compute.pointwise_ranges), 1)
        self.assertEqual(compute.pointwise_ranges[0].stop, 5)
        self.assertEqual(after.outputs[0].sizes, [1, 5, 1])
        after.verify()

    def test_dead_compute_pruning(self):
        def function(x):
            _unused = x * 2
            return x + 1

        before = self.lower(function, torch.randn(5))
        self.assertEqual(len(self.computes(before)), 2)
        after = self.cpp.optimize_program(before)
        self.assertEqual(len(self.computes(after)), 1)
        self.assertNotIn(" = mul ", after.dump())

    def test_returned_view_remains_live(self):
        after = self.cpp.optimize_program(
            self.lower(lambda x: (x.permute(1, 0), x + 1), torch.randn(2, 3))
        )
        self.assertEqual(len(after.outputs), 2)
        self.assertTrue(any(isinstance(n, self.cpp.View) for n in after.nodes))
        after.verify()

    def test_vertical_pointwise_fusion(self):
        program = self.cpp.optimize_program(
            self.lower(lambda x: torch.relu((x + 1) * 2), torch.randn(5))
        )
        kernels = self.cpp.lower_to_kernels(program)
        self.assertEqual(len(kernels.nodes), 3)
        fused = self.cpp.fuse_kernels(kernels)
        fused.verify()
        self.assertEqual(len(fused.nodes), 1)
        self.assertEqual(len(fused.nodes[0].regions), 1)
        dump = fused.dump()
        self.assertIn(" = add ", dump)
        self.assertIn(" = mul ", dump)
        self.assertIn(" = relu ", dump)
        self.assertEqual(dump, self.cpp.fuse_kernels(fused).dump())

    def test_exported_intermediate_is_preserved(self):
        def function(x):
            intermediate = x + 1
            return intermediate, intermediate * 2

        program = self.cpp.optimize_program(self.lower(function, torch.randn(5)))
        fused = self.cpp.fuse_kernels(self.cpp.lower_to_kernels(program))
        fused.verify()
        names = {
            b.name
            for node in fused.nodes
            if isinstance(node, self.cpp.KernelFunction)
            for region in node.regions
            for b in region.results
        }
        self.assertEqual(len(fused.outputs), 2)
        self.assertTrue(all(d.buffer in names for d in fused.outputs))

    def test_extern_barrier_and_dependencies(self):
        program = self.cpp.optimize_program(
            self.lower(
                lambda x, w: ((x + 1) @ w) + 2,
                torch.randn(2, 2),
                torch.randn(2, 2),
            )
        )
        kernels = self.cpp.fuse_kernels(self.cpp.lower_to_kernels(program))
        kernels.verify()
        self.assertEqual(len(kernels.nodes), 3)
        self.assertIsInstance(kernels.nodes[1], self.cpp.ExternCall)
        self.assertIn(0, kernels.dependencies[1])
        self.assertIn(1, kernels.dependencies[2])

    def test_constant_cse_uses_exact_values(self):
        cpp = self.cpp
        program = cpp.Program("distinct_constants")
        compute = cpp.Compute("op", [], [cpp.Buffer("out", "f32", 1)], [], [], [])
        compute.add_post_instruction(cpp.Instruction.constant(0, 0.10000001))
        compute.add_post_instruction(cpp.Instruction.constant(1, 0.10000004))
        compute.add_post_instruction(cpp.Instruction.binary(2, "add", 0, 1))
        compute.add_post_instruction(
            cpp.Instruction.store("out", cpp.Index.constant(0), 2)
        )
        program.add_compute(compute)
        program.add_output(cpp.Descriptor("output", "out", [1], [1], 0))
        after = cpp.optimize_program(program)
        self.assertEqual(after.dump().count(" = constant "), 2)
        self.assertIn(" = add ", after.dump())

    def test_overlapping_store_map_is_not_fused(self):
        cpp = self.cpp
        program = cpp.Program("overlapping_stores")
        program.add_input(cpp.Buffer("x", "f32", 2))
        index = cpp.Index.variable("p0")
        ranges = [cpp.Range("p0", 0, 2)]
        producer = cpp.Compute(
            "producer", ["x"], [cpp.Buffer("tmp", "f32", 2)], ranges, [], []
        )
        producer.add_post_instruction(cpp.Instruction.load(0, "x", index))
        producer.add_post_instruction(
            cpp.Instruction.store("tmp", cpp.Index.constant(0), 0)
        )
        consumer = cpp.Compute(
            "consumer", ["tmp"], [cpp.Buffer("out", "f32", 2)], ranges, [], []
        )
        consumer.add_post_instruction(
            cpp.Instruction.load(0, "tmp", cpp.Index.constant(0))
        )
        consumer.add_post_instruction(cpp.Instruction.relu(1, 0))
        consumer.add_post_instruction(cpp.Instruction.store("out", index, 1))
        program.add_compute(producer)
        program.add_compute(consumer)
        program.add_output(cpp.Descriptor("output", "out", [2], [1], 0))
        fused = cpp.fuse_kernels(cpp.lower_to_kernels(program))
        self.assertEqual(len(fused.nodes), 2)

    def test_cse_does_not_parse_pretty_printed_indices(self):
        cpp = self.cpp
        program = cpp.Program("structural_indices")
        program.add_input(cpp.Buffer("input", "f32", 11))
        variable = cpp.Index.variable("10")
        compute = cpp.Compute(
            "op",
            ["input"],
            [cpp.Buffer("out", "f32", 11)],
            [cpp.Range("10", 0, 11)],
            [],
            [],
        )
        compute.add_post_instruction(cpp.Instruction.load(0, "input", variable))
        compute.add_post_instruction(
            cpp.Instruction.load(1, "input", cpp.Index.constant(10))
        )
        compute.add_post_instruction(cpp.Instruction.binary(2, "add", 0, 1))
        compute.add_post_instruction(cpp.Instruction.store("out", variable, 2))
        program.add_compute(compute)
        program.add_output(cpp.Descriptor("output", "out", [11], [1], 0))
        after = cpp.optimize_program(program)
        self.assertEqual(after.dump().count(" = load "), 2)

    def test_view_of_computed_result_preserves_exported_storage(self):
        def function(x):
            intermediate = x + 1
            return intermediate.permute(1, 0), intermediate * 2

        program = self.cpp.optimize_program(self.lower(function, torch.randn(2, 3)))
        kernels = self.cpp.fuse_kernels(self.cpp.lower_to_kernels(program))
        kernels.verify()
        view = next(v for v in kernels.views if v.name == kernels.outputs[0].buffer)
        buffers = {
            b.name
            for n in kernels.nodes
            if isinstance(n, self.cpp.KernelFunction)
            for region in n.regions
            for b in region.results
        }
        self.assertIn(view.source, buffers)
        self.assertEqual(len(kernels.outputs), 2)

    def test_disjoint_extern_is_still_a_dependency_barrier(self):
        program = self.cpp.optimize_program(
            self.lower(
                lambda x, w: (x + 1, w @ w, x * 2),
                torch.randn(2, 2),
                torch.randn(2, 2),
            )
        )
        kernels = self.cpp.fuse_kernels(self.cpp.lower_to_kernels(program))
        self.assertEqual(len(kernels.nodes), 3)
        self.assertIn(0, kernels.dependencies[1])
        self.assertIn(1, kernels.dependencies[2])

    def test_tiled_contiguous_offsets_use_flat_lane(self):
        program = self.cpp.optimize_program(
            self.lower(lambda x: torch.relu(x), torch.randn(2, 3, 4))
        )
        kernels = self.cpp.fuse_kernels(self.cpp.lower_to_kernels(program))
        tiled = self.cpp.lower_to_tiled(kernels, self.cpp.TileOptions(256, 4))
        ops = [i.op for b in tiled.functions[0].blocks for i in b.instructions]
        self.assertFalse(any("floordiv" in op or "mod" in op for op in ops))
        tiled.verify()

    def test_tiled_noncontiguous_loads_retain_coordinate_decode(self):
        program = self.cpp.optimize_program(
            self.lower(
                lambda x: x * 2,
                torch.randn(4, 6).transpose(0, 1),
            )
        )
        tiled = self.cpp.lower_to_tiled(
            self.cpp.lower_to_kernels(program), self.cpp.TileOptions(256, 4)
        )
        ops = [i.op for b in tiled.functions[0].blocks for i in b.instructions]
        self.assertTrue(any("floordiv" in op or "mod" in op for op in ops))
        tiled.verify()

    def test_numerics_propagates_through_all_passes(self):
        cpp = self.cpp
        program = cpp.Program("pipe", "inductor")
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
        for name in cpp.pass_names():
            with self.subTest(pass_name=name):
                passed = cpp.run_pass(name, program)
                self.assertEqual(passed.numerics, "inductor")
                passed.verify()
        dumps = cpp.optimize_program_dumps(program)
        self.assertEqual(
            set(dumps.keys()),
            {
                "view_inline",
                "unit_range_removal",
                "index_simplify",
                "local_cse_dce",
                "dead_prune",
            },
        )
        optimized = cpp.optimize_program(program)
        self.assertEqual(optimized.numerics, "inductor")
        kernels = cpp.lower_to_kernels(optimized)
        self.assertEqual(kernels.numerics, "inductor")
        fused = cpp.fuse_kernels(kernels)
        self.assertEqual(fused.numerics, "inductor")
        tiled = cpp.lower_to_tiled(fused, cpp.TileOptions(256, 4, True, False))
        self.assertEqual(tiled.functions[0].numerics, "inductor")
        self.assertTrue(tiled.functions[0].enable_fp_fusion)
        self.assertFalse(tiled.functions[0].disable_ftz)
        tiled.verify()

    def test_tiled_program_numerics_guards(self):
        cpp = self.cpp
        program = cpp.TiledProgram()
        self.assertEqual(program.numerics, "strict_eager")
        program.verify()
        with self.assertRaisesRegex(ValueError, "numerics"):
            cpp.TiledProgram("bogus")
        # Build real functions via lowering (hand-built functions have no
        # blocks and fail the entry/return check before numerics is reached).
        strict_src = cpp.Program("s_src")
        strict_src.add_input(cpp.Buffer("x", "f32", 4))
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
        strict_src.add_compute(compute)
        strict_src.add_output(cpp.Descriptor("output", "out", [4], [1], 0))
        strict_tiled = cpp.lower_to_tiled(
            cpp.fuse_kernels(cpp.lower_to_kernels(strict_src)),
            cpp.TileOptions(256, 4),
        )
        strict_fn = strict_tiled.functions[0]
        program.add_function(strict_fn)
        program.verify()
        inductor_src = cpp.Program("i_src", "inductor")
        inductor_src.add_input(cpp.Buffer("x", "f32", 4))
        compute2 = cpp.Compute(
            "op",
            ["x"],
            [cpp.Buffer("out", "f32", 4)],
            [cpp.Range("p0", 0, 4)],
            [],
            [],
        )
        compute2.add_post_instruction(
            cpp.Instruction.load(0, "x", cpp.Index.variable("p0"))
        )
        compute2.add_post_instruction(cpp.Instruction.relu(1, 0))
        compute2.add_post_instruction(
            cpp.Instruction.store("out", cpp.Index.variable("p0"), 1)
        )
        inductor_src.add_compute(compute2)
        inductor_src.add_output(cpp.Descriptor("output", "out", [4], [1], 0))
        inductor_tiled = cpp.lower_to_tiled(
            cpp.fuse_kernels(cpp.lower_to_kernels(inductor_src)),
            cpp.TileOptions(256, 4, True, False),
        )
        inductor_fn = inductor_tiled.functions[0]
        with self.assertRaisesRegex(ValueError, "numerics"):
            program.add_function(inductor_fn)
        inductor_program = cpp.TiledProgram("inductor")
        inductor_program.add_function(inductor_fn)
        inductor_program.verify()
        self.assertIn("inductor", inductor_program.dump())


if __name__ == "__main__":
    run_tests()
