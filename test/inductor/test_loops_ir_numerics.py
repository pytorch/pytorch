# Owner(s): ["module: inductor"]

import unittest
from types import SimpleNamespace

import torch
from torch._inductor.loops_ir_backend import loops_ir_backend
from torch._inductor.test_case import run_tests, TestCase


def _fake_tiled_function(
    *,
    numerics="strict_eager",
    enable_fp_fusion=False,
    disable_ftz=True,
    name="test_kernel",
    block_size=256,
    num_warps=4,
    grid_size=1,
):
    """Minimal TiledFunction-shaped namespace for source-level emitter tests.

    No C++ build or GPU needed: only the properties emit_triton reads
    (name/parameters/blocks/grid_size/block_size/num_warps/enable_fp_fusion/
    disable_ftz/numerics) are provided. Blocks carry one small but complete
    program (pid/arange/offsets/mask/load/fadd/relu/store) so policy branches
    for arithmetic and ReLU are exercised in emitted source.
    """
    pid = SimpleNamespace(op="program_id", id=0, operands=[], dtype="i64", shape=[])
    lanes = SimpleNamespace(
        op="arange", id=1, operands=[], dtype="i64", shape=[block_size]
    )
    stride = SimpleNamespace(
        op="const_int", id=2, operands=[], dtype="i64", shape=[], int_value=1
    )
    flat = SimpleNamespace(
        op="add", id=3, operands=[0, 1], dtype="i64", shape=[block_size]
    )
    total = SimpleNamespace(
        op="const_int",
        id=4,
        operands=[],
        dtype="i64",
        shape=[],
        int_value=block_size,
    )
    mask = SimpleNamespace(
        op="cmp_lt", id=5, operands=[3, 4], dtype="bool", shape=[block_size]
    )
    loaded = SimpleNamespace(
        op="load",
        id=6,
        operands=[3, 5],
        dtype="f32",
        shape=[block_size],
        buffer="in_buf",
    )
    one = SimpleNamespace(
        op="const_float",
        id=7,
        operands=[],
        dtype="f32",
        shape=[],
        float_value=1.25,
    )
    one_vec = SimpleNamespace(
        op="broadcast", id=8, operands=[7], dtype="f32", shape=[block_size]
    )
    added = SimpleNamespace(
        op="fadd", id=9, operands=[6, 8], dtype="f32", shape=[block_size]
    )
    activated = SimpleNamespace(
        op="relu", id=10, operands=[9], dtype="f32", shape=[block_size]
    )
    stored = SimpleNamespace(
        op="store",
        id=-1,
        operands=[10, 3, 5],
        dtype="f32",
        shape=[],
        buffer="out_buf",
    )
    # const_int 'stride' (v2) intentionally keeps the emitter honest about
    # unused SSA: dominance-ordered straight-line blocks allow defined-but-
    # unused values (no use-before-def, no duplicates).
    _ = stride
    entry = SimpleNamespace(
        name="entry",
        instructions=[
            pid,
            lanes,
            stride,
            flat,
            total,
            mask,
            loaded,
            one,
            one_vec,
            added,
            activated,
            stored,
        ],
        terminator="return",
    )
    in_buf = SimpleNamespace(name="in_buf", dtype="f32")
    out_buf = SimpleNamespace(name="out_buf", dtype="f32")
    return SimpleNamespace(
        name=name,
        parameters=[in_buf, out_buf],
        blocks=[entry],
        grid_size=grid_size,
        block_size=block_size,
        num_warps=num_warps,
        enable_fp_fusion=enable_fp_fusion,
        disable_ftz=disable_ftz,
        numerics=numerics,
    )


@unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
class TestLoopsIrNumerics(TestCase):
    def assert_eager_bits(self, function, value):
        compiled = torch.compile(
            function, backend=loops_ir_backend, fullgraph=True, dynamic=False
        )
        with torch.inference_mode():
            expected = function(value)
            actual = compiled(value)
            repeated = compiled(value)
        self.assertTrue(
            torch.equal(
                repeated.contiguous().view(torch.int32),
                actual.contiguous().view(torch.int32),
            )
        )
        self.assertEqual(actual.dtype, expected.dtype)
        self.assertEqual(actual.shape, expected.shape)
        expected_bits = expected.contiguous().view(torch.int32)
        actual_bits = actual.contiguous().view(torch.int32)
        equal = torch.equal(expected_bits, actual_bits)
        if not equal:
            indices = (expected_bits != actual_bits).nonzero().flatten()[:8]
            details = [
                (int(i), int(expected_bits[i]), int(actual_bits[i]))
                for i in indices.cpu()
            ]
            self.fail(f"eager/compiled bits differ: {details}")

    def test_special_values_are_not_algebraically_elided(self):
        bits = [
            0x00000000,
            0x80000000,
            0x00000001,
            0x80000001,
            0x007FFFFF,
            0x807FFFFF,
            0x00800000,
            0x80800000,
            0x3F800000,
            0xBF800000,
            0x7F800000,
            0xFF800000,
            0x7FC00001,
            0xFFC00001,
            0x7F800001,
            0xFF800001,
            0x7FFFFFFF,
        ]
        value = (
            torch.tensor(bits, dtype=torch.int64)
            .to(torch.int32)
            .view(torch.float32)
            .cuda()
        )
        for name, function in (
            ("mul_one", lambda x: x * 1.0),
            ("add_negative_zero", lambda x: x + (-0.0)),
            ("add_positive_zero", lambda x: x + 0.0),
            ("mul_zero", lambda x: x * 0.0),
            ("relu", lambda x: torch.relu(x)),
        ):
            with self.subTest(operation=name):
                self.assert_eager_bits(function, value)

    def test_random_input_bits(self):
        generator = torch.Generator().manual_seed(923)
        value = (
            torch.randint(
                -(2**31), 2**31, (16384,), dtype=torch.int32, generator=generator
            )
            .view(torch.float32)
            .cuda()
        )
        for name, function in (
            ("mul_one", lambda x: x * 1.0),
            ("add_negative_zero", lambda x: x + (-0.0)),
            ("add_positive_zero", lambda x: x + 0.0),
            ("mul_rounding", lambda x: x * 1.0000001192092896),
            ("dependent_chain", lambda x: (x * 1.0000001192092896) + x),
            ("relu", lambda x: torch.relu(x)),
        ):
            with self.subTest(operation=name):
                self.assert_eager_bits(function, value)


class TestLoopsIrNumericsEmission(TestCase):
    """Source-level numerics-policy tests (no GPU, no C++ build).

    These pin the emitter contract per policy using a fake TiledFunction:
    strict keeps bit-exact PTX arithmetic and the signed-zero/NaN-sensitive
    ReLU; inductor emits regular typed Triton arithmetic/ReLU matching the
    stock convention and passes function flags through triton_meta without
    forcing fast math. Unknown policies and cross-policy flag violations
    raise.
    """

    def test_strict_source_keeps_bit_exact_ptx_and_sensitive_relu(self):
        from torch._inductor.codegen.loops_ir_triton import emit_triton

        src = emit_triton(_fake_tiled_function(numerics="strict_eager"))
        # Strict arithmetic: one pinned-RN PTX instruction, never TL +/*.
        self.assertIn("add.rn.f32", src)
        self.assertIn("tl.inline_asm_elementwise", src)
        self.assertNotIn("tl.maximum", src)
        # Strict ReLU: signed-zero/NaN-sensitive compare/select lane.
        self.assertIn("tl.where(", src)
        self.assertIn("< 0.0", src)
        # Strict flags recorded verbatim in the header.
        self.assertIn("enable_fp_fusion=False", src)
        self.assertIn("disable_ftz=True", src)
        self.assertIn("numerics='strict_eager'", src)

    def test_strict_source_has_no_fma_or_fast_math(self):
        from torch._inductor.codegen.loops_ir_triton import emit_triton

        src = emit_triton(_fake_tiled_function(numerics="strict_eager"))
        self.assertNotIn("tl.fma", src)
        self.assertNotIn("fma.", src)
        self.assertNotIn("tl.exp", src)
        self.assertNotIn("libdevice", src)
        self.assertNotIn("tl.sigmoid", src)

    def test_inductor_source_matches_stock_convention(self):
        from torch._inductor.codegen.loops_ir_triton import emit_triton

        src = emit_triton(
            _fake_tiled_function(
                numerics="inductor",
                enable_fp_fusion=True,
                disable_ftz=False,
            )
        )
        # Regular typed Triton arithmetic (no PTX pinning), stock ReLU form:
        # tl.maximum against an int32 zero with NaN propagation -- exactly
        # what stock TritonOverrides.relu/maximum emit for fused ReLU.
        self.assertNotIn("inline_asm_elementwise", src)
        self.assertNotIn("add.rn.f32", src)
        self.assertIn("tl.maximum(", src)
        self.assertIn("tl.PropagateNan.ALL", src)
        self.assertIn("tl.int32", src)
        # Effective flags pass through; fast math is never forced.
        self.assertIn("enable_fp_fusion=True", src)
        self.assertIn("disable_ftz=False", src)
        self.assertIn("numerics='inductor'", src)
        self.assertNotIn("use_fast_math", src)
        self.assertNotIn("tl.exp", src)
        self.assertNotIn("libdevice", src)

    def test_inductor_triton_meta_passes_function_flags_through(self):
        from torch._inductor.codegen.loops_ir_triton import emit_triton

        device = torch.device("cuda", 0)
        for enable_fp_fusion, disable_ftz in ((False, True), (True, False)):
            with self.subTest(
                enable_fp_fusion=enable_fp_fusion, disable_ftz=disable_ftz
            ):
                src = emit_triton(
                    _fake_tiled_function(
                        numerics="inductor",
                        enable_fp_fusion=enable_fp_fusion,
                        disable_ftz=disable_ftz,
                    ),
                    device=device,
                )
                # Stock flag spelling in triton_meta (raw Triton has no
                # disable_ftz kwarg; the mapping lives in the runtime).
                self.assertIn(f"'enable_fp_fusion': {enable_fp_fusion}", src)
                self.assertIn(f"'disable_ftz': {disable_ftz}", src)

    def test_strict_triton_meta_pins_strict_flags(self):
        from torch._inductor.codegen.loops_ir_triton import emit_triton

        src = emit_triton(
            _fake_tiled_function(numerics="strict_eager"),
            device=torch.device("cuda", 0),
        )
        self.assertIn("'enable_fp_fusion': False", src)
        self.assertIn("'disable_ftz': True", src)

    def test_unknown_numerics_policy_rejected(self):
        from torch._inductor.codegen.loops_ir_triton import emit_triton

        with self.assertRaisesRegex(ValueError, "unknown numerics policy"):
            emit_triton(_fake_tiled_function(numerics="fast_math"))

    def test_strict_flag_deviation_rejected(self):
        from torch._inductor.codegen.loops_ir_triton import emit_triton

        with self.assertRaisesRegex(ValueError, "enable_fp_fusion"):
            emit_triton(
                _fake_tiled_function(numerics="strict_eager", enable_fp_fusion=True)
            )
        with self.assertRaisesRegex(ValueError, "disable_ftz"):
            emit_triton(
                _fake_tiled_function(numerics="strict_eager", disable_ftz=False)
            )

    def test_render_frozen_layout_and_memory_format_literals(self):
        from torch._inductor.codegen.loops_ir_wrapper import render_frozen_arg

        for value, literal in (
            (torch.strided, "torch.strided"),
            (torch.contiguous_format, "torch.contiguous_format"),
            (torch.preserve_format, "torch.preserve_format"),
            (torch.channels_last, "torch.channels_last"),
            (torch.channels_last_3d, "torch.channels_last_3d"),
        ):
            with self.subTest(literal=literal):
                rendered = render_frozen_arg(value, {})
                self.assertEqual(rendered, literal)
                # The literal must evaluate back to the same singleton.
                self.assertIs(eval(rendered, {"torch": torch}), value)

    def test_render_frozen_arg_rejects_unknown_tokens(self):
        from torch._inductor.codegen.loops_ir_wrapper import render_frozen_arg

        # Plain strings never render as attribute access (no repr execution).
        self.assertEqual(
            render_frozen_arg("contiguous_format", {}), "'contiguous_format'"
        )
        # Unknown dtypes/layout-likes raise instead of emitting code.
        with self.assertRaisesRegex(ValueError, "unsupported"):
            render_frozen_arg(torch.float8_e4m3fn, {})
        with self.assertRaisesRegex(ValueError, "unsupported"):
            render_frozen_arg(SimpleNamespace(text="arbitrary_code"), {})


if __name__ == "__main__":
    run_tests()
