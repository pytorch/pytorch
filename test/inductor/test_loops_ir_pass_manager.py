# Owner(s): ["module: inductor"]

import struct

import torch
from torch._inductor.loops_ir import load_cpp_ir, lower_fx_to_cpp
from torch._inductor.test_case import run_tests, TestCase
from torch.fx.experimental.proxy_tensor import make_fx


class TestLoopsIrPassManager(TestCase):
    """Formal typed pass API: registry, descriptors, pipeline, trace.

    CPU-only. The extension is built via ``load_cpp_ir``.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        cls.cpp = load_cpp_ir()

    def lower(self, function, *inputs):
        return lower_fx_to_cpp(make_fx(function)(*inputs), list(inputs))

    def pointwise_program(self, numerics="strict_eager"):
        cpp = self.cpp
        program = cpp.Program("pipe", numerics)
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

    def extern_program(self, scalar=1.0, name="ext"):
        """One extern call (mm) between two pointwise computes."""
        cpp = self.cpp
        program = cpp.Program(name)
        program.add_input(cpp.Buffer("a", "f32", 4))
        program.add_input(cpp.Buffer("b", "f32", 4))
        program.add_input(cpp.Buffer("m1", "f32", 4))
        program.add_input(cpp.Buffer("m2", "f32", 4))
        pre = cpp.Compute(
            "pre",
            ["a", "b"],
            [cpp.Buffer("e", "f32", 4)],
            [cpp.Range("p0", 0, 4)],
            [],
            [],
        )
        pre.add_post_instruction(cpp.Instruction.load(0, "a", cpp.Index.variable("p0")))
        pre.add_post_instruction(cpp.Instruction.load(1, "b", cpp.Index.variable("p0")))
        pre.add_post_instruction(cpp.Instruction.binary(2, "add", 0, 1))
        pre.add_post_instruction(
            cpp.Instruction.store("e", cpp.Index.variable("p0"), 2)
        )
        program.add_compute(pre)
        program.add_extern_call(
            cpp.ExternCall(
                "mm0",
                "aten.mm.default",
                ["m1", "m2"],
                [cpp.Buffer("w", "f32", 4)],
                [scalar],
                "unknown",
                True,
                [
                    cpp.Descriptor("lhs", "m1", [2, 2], [2, 1], 0),
                    cpp.Descriptor("rhs", "m2", [2, 2], [2, 1], 0),
                    cpp.Descriptor("result", "w", [2, 2], [2, 1], 0),
                ],
            )
        )
        post = cpp.Compute(
            "post",
            ["e"],
            [cpp.Buffer("out", "f32", 4)],
            [cpp.Range("p0", 0, 4)],
            [],
            [],
        )
        post.add_post_instruction(
            cpp.Instruction.load(0, "e", cpp.Index.variable("p0"))
        )
        post.add_post_instruction(cpp.Instruction.relu(1, 0))
        post.add_post_instruction(
            cpp.Instruction.store("out", cpp.Index.variable("p0"), 1)
        )
        program.add_compute(post)
        program.add_output(cpp.Descriptor("output", "out", [4], [1], 0))
        return program

    # -- registry ------------------------------------------------------

    def test_pipeline_pass_names_all_eight_in_order(self):
        self.assertEqual(
            self.cpp.pipeline_pass_names(),
            [
                "view_inline",
                "unit_range_removal",
                "index_simplify",
                "local_cse_dce",
                "dead_prune",
                "lower_to_kernels",
                "fuse_kernels",
                "lower_to_tiled",
            ],
        )

    def test_legacy_pass_names_preserved(self):
        self.assertEqual(
            self.cpp.pass_names(),
            [
                "view_inline",
                "unit_range_removal",
                "index_simplify",
                "local_cse_dce",
                "dead_prune",
            ],
        )

    def test_pass_descriptors_structured(self):
        descriptors = {d["name"]: d for d in self.cpp.pass_descriptors()}
        self.assertEqual(
            sorted(descriptors),
            sorted(self.cpp.pipeline_pass_names()),
        )
        loops = {
            "view_inline",
            "unit_range_removal",
            "index_simplify",
            "local_cse_dce",
            "dead_prune",
        }
        for name, entry in descriptors.items():
            with self.subTest(pass_name=name):
                self.assertTrue(entry["version"])
                self.assertIn(entry["input_ir"], ("loops_ir", "kernel_ir"))
                self.assertIn(entry["output_ir"], ("loops_ir", "kernel_ir", "tiled_ir"))
                self.assertTrue(entry["strict_safe"])
                self.assertTrue(entry["capability"])
                # No catch-all IR: typed stage pair per pass.
                if name in loops:
                    self.assertEqual(entry["input_ir"], "loops_ir")
                    self.assertEqual(entry["output_ir"], "loops_ir")
                # Invalidated and preserved never overlap.
                self.assertEqual(
                    set(entry["invalidated_analyses"])
                    & set(entry["preserved_analyses"]),
                    set(),
                )
        self.assertEqual(descriptors["lower_to_kernels"]["input_ir"], "loops_ir")
        self.assertEqual(descriptors["lower_to_kernels"]["output_ir"], "kernel_ir")
        self.assertEqual(descriptors["fuse_kernels"]["input_ir"], "kernel_ir")
        self.assertEqual(descriptors["fuse_kernels"]["output_ir"], "kernel_ir")
        self.assertEqual(descriptors["lower_to_tiled"]["input_ir"], "kernel_ir")
        self.assertEqual(descriptors["lower_to_tiled"]["output_ir"], "tiled_ir")
        self.assertTrue(descriptors["lower_to_tiled"]["consults_tile_options"])
        # Pure-local transforms honestly declare empty requirements.
        self.assertEqual(descriptors["index_simplify"]["required_analyses"], [])
        self.assertEqual(descriptors["local_cse_dce"]["required_analyses"], [])
        self.assertEqual(descriptors["unit_range_removal"]["required_analyses"], [])
        # Fusion genuinely requires the dense proof + dependency/alias facts.
        self.assertIn(
            "dense_injective", descriptors["fuse_kernels"]["required_analyses"]
        )
        # Every listed kind is a known analysis kind.
        known = {
            "alias_storage_roots",
            "index_intervals",
            "liveness",
            "dependencies",
            "effects",
            "dense_injective",
        }
        for entry in descriptors.values():
            self.assertTrue(
                set(entry["required_analyses"]) <= known,
                entry["name"],
            )
            self.assertTrue(
                set(entry["invalidated_analyses"]) <= known,
                entry["name"],
            )

    def test_unknown_pass_rejected(self):
        program = self.pointwise_program()
        with self.assertRaisesRegex(ValueError, "unknown pass"):
            self.cpp.run_pass("not_a_pass", program)

    def test_no_metadata_only_registration_api(self):
        # Registration without a bound executor is unsupported: there is no
        # registration API at all (review F7).
        self.assertFalse(hasattr(self.cpp, "register_external_pass"))
        self.assertEqual(len(self.cpp.pass_descriptors()), 8)

    def test_registry_self_test(self):
        # Bounded C++ helper exercising duplicate / missing-executor /
        # unknown / stage-mismatch validation without dummy passes.
        self.assertTrue(self.cpp._registry_self_test())

    def test_cse_length_prefixed_buffer_names(self):
        # Regression: raw buffer-name concatenation in CSE keys collides
        # when a buffer name spans the ':' delimiter. Load(buf="w",
        # off=Var 22-char name) and Load(buf="w:V(22", off=mul(x,y))
        # share the naive key "load:w:V(22:B(mul,V(1:x),V(1:y))" but
        # address different buffers; both loads must survive.
        cpp = self.cpp
        # Variable name chosen so its V-encoding spans the ':' delimiter:
        # encoding("w", Var(name)) and encoding("w:V(L", mul(x,y)) share
        # the naive key but address different buffers.
        var_name = "B(mul,V(1:x),V(1:y)"
        tag = f"V({len(var_name)}"
        other_buffer = f"w:{tag}"
        program = cpp.Program("cse_delimiter")
        program.add_input(cpp.Buffer("w", "f32", 2))
        program.add_input(cpp.Buffer(other_buffer, "f32", 2))
        ranges = [
            cpp.Range(var_name, 0, 2),
            cpp.Range("x", 0, 2),
            cpp.Range("y", 0, 2),
        ]
        compute = cpp.Compute(
            "op",
            ["w", other_buffer],
            [cpp.Buffer("out", "f32", 1)],
            ranges,
            [],
            [],
        )
        compute.add_post_instruction(
            cpp.Instruction.load(0, "w", cpp.Index.variable(var_name))
        )
        mul = cpp.Index.binary("mul", cpp.Index.variable("x"), cpp.Index.variable("y"))
        compute.add_post_instruction(cpp.Instruction.load(1, other_buffer, mul))
        compute.add_post_instruction(cpp.Instruction.binary(2, "add", 0, 1))
        compute.add_post_instruction(
            cpp.Instruction.store("out", cpp.Index.constant(0), 2)
        )
        program.add_compute(compute)
        program.add_output(cpp.Descriptor("output", "out", [1], [1], 0))
        after = cpp.optimize_program(program)
        after.verify()
        self.assertEqual(after.dump().count(" = load "), 2)

    def test_tiled_fingerprint_uses_canonical_key(self):
        # The lower_to_tiled trace fingerprint must change with TileOptions
        # (block size is in the canonical key), proving it is not a dump
        # hash that could miss typed fields.
        cpp = self.cpp
        program = self.pointwise_program()
        first = cpp.run_pipeline(program, tile_options=cpp.TileOptions(256, 4))
        second = cpp.run_pipeline(program, tile_options=cpp.TileOptions(128, 4))
        entries_first = {e.name: e for e in first.trace}
        entries_second = {e.name: e for e in second.trace}
        self.assertNotEqual(
            entries_first["lower_to_tiled"].output_fingerprint,
            entries_second["lower_to_tiled"].output_fingerprint,
        )

    def test_prove_dense_registered_once(self):
        # A single registration: calling twice works identically (no dup).
        compute = self.pointwise_program().nodes[0]
        first = self.cpp.prove_dense_injective_store(compute, "out")
        second = self.cpp.prove_dense_injective_store(compute, "out")
        self.assertEqual(first, second)
        self.assertTrue(first)

    def test_run_pass_typed_stage_mismatch_rejected(self):
        program = self.pointwise_program()
        kernels = self.cpp.lower_to_kernels(self.cpp.optimize_program(program))
        # run_pass only serves Loops->Loops micro-passes; a KernelProgram
        # argument is a typed stage mismatch (pybind type error).
        with self.assertRaises(TypeError):
            self.cpp.run_pass("view_inline", kernels)

    def test_kernels_before_fusion_is_typed_kernel_program(self):
        result = self.cpp.run_pipeline(
            self.pointwise_program(), tile_options=self.cpp.TileOptions(256, 4)
        )
        self.assertIsInstance(result.kernels_before_fusion, self.cpp.KernelProgram)
        result.kernels_before_fusion.verify()

    # -- pipeline results ----------------------------------------------

    def test_run_pipeline_matches_manual_chain(self):
        program = self.lower(lambda x: torch.relu((x + 1) * 2), torch.randn(5))
        cpp = self.cpp
        optimized = cpp.optimize_program(program)
        kernels = cpp.lower_to_kernels(optimized)
        fused = cpp.fuse_kernels(kernels)
        tiled = cpp.lower_to_tiled(fused, cpp.TileOptions(256, 4))
        result = cpp.run_pipeline(program, tile_options=cpp.TileOptions(256, 4))
        self.assertEqual(result.optimized_program.dump(), optimized.dump())
        self.assertEqual(result.kernels_before_fusion.dump(), kernels.dump())
        self.assertEqual(result.kernel_program.dump(), fused.dump())
        self.assertEqual(result.tiled_program.dump(), tiled.dump())

    def test_all_eight_public_wrappers_go_through_registry(self):
        # Every public free function returns the registry result: running
        # the wrapper twice (fresh contexts) is dump-identical, and the
        # full pipeline matches the manual chain stage by stage.
        cpp = self.cpp
        program = self.lower(lambda x: torch.relu((x + 1) * 2), torch.randn(5))
        for name in cpp.pass_names():
            with self.subTest(pass_name=name):
                first = cpp.run_pass(name, program)
                second = cpp.run_pass(name, program)
                self.assertEqual(first.dump(), second.dump())
                first.verify()
        kernels = cpp.lower_to_kernels(cpp.optimize_program(program))
        self.assertEqual(
            cpp.fuse_kernels(kernels).dump(),
            cpp.fuse_kernels(kernels).dump(),
        )
        self.assertEqual(
            cpp.lower_to_tiled(kernels, cpp.TileOptions(256, 4)).dump(),
            cpp.lower_to_tiled(kernels, cpp.TileOptions(256, 4)).dump(),
        )

    def test_lower_to_tiled_rejects_reduction_before_body(self):
        # Reduction computes are rejected (must be eager externs) with a
        # legality error, not tiled.
        cpp = self.cpp
        program = cpp.Program("reduction_probe")
        program.add_input(cpp.Buffer("x", "f32", 4))
        compute = cpp.Compute(
            "red",
            ["x"],
            [cpp.Buffer("out", "f32", 1)],
            [],
            [cpp.Range("r0", 0, 4)],
            [cpp.Accumulator("sum", "f32", 0.0, "add")],
        )
        compute.add_reduction_instruction(
            cpp.Instruction.load(0, "x", cpp.Index.variable("r0"))
        )
        compute.add_reduction_instruction(cpp.Instruction.accumulate([0]))
        compute.add_post_instruction(cpp.Instruction.read_accumulator(0, 0))
        compute.add_post_instruction(
            cpp.Instruction.store("out", cpp.Index.constant(0), 0)
        )
        program.add_compute(compute)
        program.add_output(cpp.Descriptor("output", "out", [1], [1], 0))
        program.verify()
        with self.assertRaisesRegex(ValueError, "reduction"):
            cpp.lower_to_kernels(program)

    def test_run_pipeline_trace_order_and_fields(self):
        program = self.pointwise_program()
        result = self.cpp.run_pipeline(
            program, tile_options=self.cpp.TileOptions(256, 4)
        )
        trace = list(result.trace)
        self.assertEqual([e.name for e in trace], self.cpp.pipeline_pass_names())
        for entry in trace:
            with self.subTest(pass_name=entry.name):
                self.assertTrue(entry.version)
                self.assertIn(entry.input_ir, ("loops_ir", "kernel_ir"))
                self.assertIn(entry.output_ir, ("loops_ir", "kernel_ir", "tiled_ir"))
                self.assertEqual(entry.numerics, "strict_eager")
                self.assertTrue(entry.input_fingerprint)
                self.assertTrue(entry.output_fingerprint)
                self.assertTrue(entry.dump)
        # Fingerprint chaining: each stage output feeds the next input.
        for index in range(len(trace) - 1):
            first, second = trace[index], trace[index + 1]
            if first.output_ir == second.input_ir:
                self.assertEqual(first.output_fingerprint, second.input_fingerprint)

    def test_run_pipeline_trace_invalidations_match_descriptors(self):
        program = self.pointwise_program()
        result = self.cpp.run_pipeline(
            program, tile_options=self.cpp.TileOptions(256, 4)
        )
        descriptors = {d["name"]: d for d in self.cpp.pass_descriptors()}
        for entry in result.trace:
            with self.subTest(pass_name=entry.name):
                self.assertEqual(
                    sorted(entry.analyses_invalidated),
                    sorted(descriptors[entry.name]["invalidated_analyses"]),
                )

    def test_run_pipeline_fingerprint_changes_across_transform(self):
        cpp = self.cpp
        program = cpp.Program("rewrite_probe")
        program.add_input(cpp.Buffer("x", "f32", 1))
        compute = cpp.Compute("op", ["x"], [cpp.Buffer("out", "f32", 1)], [], [], [])
        # load x[0 + 0] carries foldable integer arithmetic for index_simplify.
        offset = cpp.Index.binary("add", cpp.Index.constant(0), cpp.Index.constant(0))
        compute.add_post_instruction(cpp.Instruction.load(0, "x", offset))
        compute.add_post_instruction(
            cpp.Instruction.store("out", cpp.Index.constant(0), 0)
        )
        program.add_compute(compute)
        program.add_output(cpp.Descriptor("output", "out", [1], [1], 0))
        result = cpp.run_pipeline(program, tile_options=cpp.TileOptions(256, 4))
        entries = {e.name: e for e in result.trace}
        self.assertNotEqual(
            entries["index_simplify"].input_fingerprint,
            entries["index_simplify"].output_fingerprint,
        )

    def test_run_pipeline_capture_dumps_flag(self):
        program = self.pointwise_program()
        without = self.cpp.run_pipeline(
            program,
            tile_options=self.cpp.TileOptions(256, 4),
            capture_dumps=False,
        )
        self.assertTrue(all(e.dump == "" for e in without.trace))
        # Fingerprints are always recorded, even without dumps.
        self.assertTrue(
            all(e.input_fingerprint and e.output_fingerprint for e in without.trace)
        )

    def test_run_pipeline_numerics_propagate(self):
        for numerics in ("strict_eager", "inductor"):
            with self.subTest(numerics=numerics):
                program = self.pointwise_program(numerics)
                result = self.cpp.run_pipeline(
                    program, tile_options=self.cpp.TileOptions(256, 4)
                )
                self.assertEqual(result.optimized_program.numerics, numerics)
                self.assertEqual(result.kernels_before_fusion.numerics, numerics)
                self.assertEqual(result.kernel_program.numerics, numerics)
                for entry in result.trace:
                    self.assertEqual(entry.numerics, numerics)

    def test_run_pipeline_tile_flag_guards(self):
        cpp = self.cpp
        kernels = cpp.fuse_kernels(cpp.lower_to_kernels(self.pointwise_program()))
        program = self.pointwise_program()
        with self.assertRaisesRegex(ValueError, "fp_fusion"):
            cpp.run_pipeline(program, tile_options=cpp.TileOptions(256, 4, True, True))
        with self.assertRaisesRegex(ValueError, "disable_ftz"):
            cpp.run_pipeline(
                program, tile_options=cpp.TileOptions(256, 4, False, False)
            )
        tiled = cpp.run_pipeline(
            self.pointwise_program("inductor"),
            tile_options=cpp.TileOptions(128, 2, True, False),
        ).tiled_program
        self.assertTrue(tiled.functions[0].enable_fp_fusion)
        self.assertFalse(tiled.functions[0].disable_ftz)
        self.assertEqual(tiled.functions[0].block_size, 128)
        self.assertEqual(kernels.numerics, "strict_eager")

    # -- structural cache identity (review F1/F2) ------------------------

    def test_structural_identity_views_change_pipeline_key(self):
        # Two programs with identical pretty dumps except view metadata
        # must produce different trace fingerprints (views are cache keys).
        cpp = self.cpp

        def program_with_offset(offset):
            program = cpp.Program("view_key")
            program.add_input(cpp.Buffer("x", "f32", 5))
            program.add_view(cpp.View("v", "x", [4], [1], offset))
            compute = cpp.Compute(
                "op",
                ["v"],
                [cpp.Buffer("out", "f32", 4)],
                [cpp.Range("p0", 0, 4)],
                [],
                [],
            )
            compute.add_post_instruction(
                cpp.Instruction.view_read(0, "v", [cpp.Index.variable("p0")])
            )
            compute.add_post_instruction(cpp.Instruction.relu(1, 0))
            compute.add_post_instruction(
                cpp.Instruction.store("out", cpp.Index.variable("p0"), 1)
            )
            program.add_compute(compute)
            program.add_output(cpp.Descriptor("output", "out", [4], [1], 0))
            return program

        first = cpp.run_pipeline(
            program_with_offset(0), tile_options=cpp.TileOptions(256, 4)
        )
        second = cpp.run_pipeline(
            program_with_offset(1), tile_options=cpp.TileOptions(256, 4)
        )
        self.assertNotEqual(
            first.trace[0].input_fingerprint, second.trace[0].input_fingerprint
        )

    def test_structural_identity_extern_scalar_bits(self):
        # 1.0 vs 1.0+1ulp extern scalar args are bit-distinct cache keys
        # (pretty text would collide after f32 truncation).
        cpp = self.cpp
        one = 1.0
        one_plus_ulp = struct.unpack(
            "<d", struct.pack("<Q", struct.unpack("<Q", struct.pack("<d", 1.0))[0] + 1)
        )[0]
        # Exact (not assertNotEqual: torch's assertEqual is approximate).
        self.assertTrue(one != one_plus_ulp)
        first = cpp.run_pipeline(
            self.extern_program(scalar=one), tile_options=cpp.TileOptions(256, 4)
        )
        second = cpp.run_pipeline(
            self.extern_program(scalar=one_plus_ulp),
            tile_options=cpp.TileOptions(256, 4),
        )
        self.assertNotEqual(
            first.trace[0].input_fingerprint, second.trace[0].input_fingerprint
        )
        # Both still verify and tile (externs stay eager).
        first.tiled_program.verify()
        second.tiled_program.verify()

    def test_effects_barrier_survives_pipeline(self):
        # The extern barrier node is present in every stage (never fused
        # away or reordered across).
        cpp = self.cpp
        result = cpp.run_pipeline(
            self.extern_program(), tile_options=cpp.TileOptions(256, 4)
        )
        self.assertIn("extern", result.optimized_program.dump())
        kernels = list(result.kernel_program.nodes)
        self.assertEqual(len(kernels), 3)
        self.assertIsInstance(kernels[1], cpp.ExternCall)
        self.assertEqual(len(result.tiled_program.functions), 2)

    # -- genuine analysis computation (review F3/F4) ---------------------

    def test_trace_computed_kinds_are_genuine(self):
        # view_inline genuinely computes alias facts; dead_prune genuinely
        # computes liveness + effects; pure-local passes compute nothing.
        result = self.cpp.run_pipeline(
            self.pointwise_program(), tile_options=self.cpp.TileOptions(256, 4)
        )
        entries = {e.name: e for e in result.trace}
        self.assertIn("alias_storage_roots", entries["view_inline"].analyses_computed)
        self.assertIn("liveness", entries["dead_prune"].analyses_computed)
        self.assertIn("effects", entries["dead_prune"].analyses_computed)
        self.assertEqual(entries["index_simplify"].analyses_computed, [])
        self.assertEqual(entries["local_cse_dce"].analyses_computed, [])
        self.assertEqual(entries["unit_range_removal"].analyses_computed, [])

    def test_fuse_kernels_computes_dense_proof_in_trace(self):
        # The fusion stage genuinely computes the dense/injective proof
        # for the fusible candidate (not a placeholder).
        program = self.lower(lambda x: torch.relu((x + 1) * 2), torch.randn(5))
        result = self.cpp.run_pipeline(
            program, tile_options=self.cpp.TileOptions(256, 4)
        )
        entries = {e.name: e for e in result.trace}
        self.assertIn("dense_injective", entries["fuse_kernels"].analyses_computed)
        # Fusion actually happened (3 kernels -> 1).
        self.assertEqual(len(result.kernel_program.nodes), 1)

    def test_fuse_kernels_no_dense_without_candidates(self):
        # Extern-only programs have no generated computes: no dense proof
        # is (or should be) computed; effects/dependencies still are.
        cpp = self.cpp
        program = cpp.Program("extern_only")
        program.add_input(cpp.Buffer("m1", "f32", 4))
        program.add_input(cpp.Buffer("m2", "f32", 4))
        program.add_extern_call(
            cpp.ExternCall(
                "mm0",
                "aten.mm.default",
                ["m1", "m2"],
                [cpp.Buffer("w", "f32", 4)],
                [],
                "unknown",
                True,
                [
                    cpp.Descriptor("lhs", "m1", [2, 2], [2, 1], 0),
                    cpp.Descriptor("rhs", "m2", [2, 2], [2, 1], 0),
                    cpp.Descriptor("result", "w", [2, 2], [2, 1], 0),
                ],
            )
        )
        program.add_output(cpp.Descriptor("output", "w", [2, 2], [2, 1], 0))
        result = cpp.run_pipeline(program, tile_options=cpp.TileOptions(256, 4))
        entries = {e.name: e for e in result.trace}
        self.assertNotIn("dense_injective", entries["fuse_kernels"].analyses_computed)
        self.assertEqual(len(result.tiled_program.functions), 0)

    # -- preservation through the formal API ----------------------------

    def test_old_abi_uses_formal_internals(self):
        program = self.lower(lambda x: x * x, torch.randn(5))
        cpp = self.cpp
        self.assertEqual(program.dump().count("load "), 2)
        after = cpp.optimize_program(program)
        self.assertEqual(after.dump().count("load "), 1)
        self.assertIn(" = mul ", after.dump())
        # Idempotent through the registry executor.
        self.assertEqual(after.dump(), cpp.optimize_program(after).dump())

    def test_special_fp_preserved_through_pipeline(self):
        cpp = self.cpp
        for function, operation in (
            (lambda x: x + 0.0, "add"),
            (lambda x: x * 0.0, "mul"),
            (lambda x: x * 1.0, "mul"),
        ):
            with self.subTest(operation=operation):
                program = self.lower(function, torch.randn(5))
                result = cpp.run_pipeline(program, tile_options=cpp.TileOptions(256, 4))
                result.optimized_program.verify()
                result.tiled_program.verify()
                self.assertIn(f" = {operation} ", result.optimized_program.dump())

    def test_incorrect_preservation_rejected(self):
        # A program that violates the nonempty-domain store rule fails
        # verification both directly and through the pipeline.
        cpp = self.cpp
        program = cpp.Program("missing_store")
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
        # No store: invalid program.
        program.add_compute(compute)
        program.add_output(cpp.Descriptor("output", "out", [4], [1], 0))
        with self.assertRaises(ValueError):
            program.verify()
        with self.assertRaises(ValueError):
            cpp.run_pipeline(program, tile_options=cpp.TileOptions(256, 4))


if __name__ == "__main__":
    run_tests()
