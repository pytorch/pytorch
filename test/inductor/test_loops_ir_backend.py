# Owner(s): ["module: inductor"]
"""GPU execution tests for the strict LoopsIR inference backend.

Hard constraint: raw-byte bitwise equality with same-device eager for
supported deterministic f32 ops (finite, signed zeros, NaN payloads,
infinities, subnormals, FMA-sensitive chains). PTX must contain no
``fma.f32`` and no ``.ftz``. Only an explicitly-reported third-party
Inductor baseline may use assert_close; these tests use byte equality.

Requires CUDA (H100 in the dev environment). Skipped without CUDA or
without the C++ extension sources (other workers own them; the extension
hash check may fail until all sources land -- tests fail loudly rather
than passing vacuously).
"""

import unittest

import torch

from torch._inductor.loops_ir import load_cpp_ir, LoopsIrUnsupportedError
from torch._inductor.test_case import run_tests, TestCase


def _require_cuda() -> None:
    if not torch.cuda.is_available():
        raise unittest.SkipTest("CUDA is required")


def _raw_bytes(tensor: torch.Tensor) -> bytes:
    return (
        tensor.detach()
        .contiguous()
        .reshape(-1)
        .view(torch.uint8)
        .cpu()
        .numpy()
        .tobytes()
    )


def _assert_bitwise_equal(actual: torch.Tensor, expected: torch.Tensor) -> None:
    if actual.shape != expected.shape:
        raise AssertionError((actual.shape, expected.shape))
    if actual.stride() != expected.stride():
        raise AssertionError((actual.stride(), expected.stride()))
    if actual.dtype != expected.dtype:
        raise AssertionError((actual.dtype, expected.dtype))
    a, b = _raw_bytes(actual), _raw_bytes(expected)
    if a != b:
        raise AssertionError(
            f"byte mismatch: len {len(a)} vs {len(b)}; first differing byte at "
            f"{next((i for i, (x, y) in enumerate(zip(a, b)) if x != y), None)}"
        )


def _compile(fn, *inputs, **flags):
    return torch.compile(
        fn,
        backend="inductor",
        options={"experimental_cpp_ir": True, "triton.cudagraphs": False},
        fullgraph=True,
        dynamic=False,
    )(*inputs)


class TestLoopsIrBackend(TestCase):
    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        _require_cuda()
        cls.cpp = load_cpp_ir()

    def test_pointwise_chain_bitwise(self):
        _require_cuda()
        torch.manual_seed(0)
        x = torch.randn(64, 64, device="cuda", dtype=torch.float32)
        y = torch.randn(64, 64, device="cuda", dtype=torch.float32)
        fn = lambda a, b: torch.relu(a * b + b)  # noqa: E731
        with torch.inference_mode():
            expected = fn(x, y)
            actual = _compile(fn, x, y)
        _assert_bitwise_equal(actual, expected)

    def test_multidimensional_spans_use_cpp_store_coverage_proof(self):
        from torch._inductor.loops_ir_backend import (
            _build_frozen_plan,
            compile_fx_entry,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(2, 3, 5, device="cuda")

        def function(value):
            return torch.relu(value + 1.0)

        gm = make_fx(function)(x)
        plan = _build_frozen_plan(gm, [x])
        self.assertTrue(plan.gen_spans)
        self.assertTrue(all(plan.span_full_coverage))
        compiled = compile_fx_entry(gm, [x])
        with torch.inference_mode():
            _assert_bitwise_equal(compiled([x]), function(x))

    def test_private_update_rebinds_existing_view_aliases(self):
        from torch._inductor.loops_ir_backend import compile_fx_entry
        from torch.fx.experimental.proxy_tensor import make_fx

        def function(x):
            intermediate = x + 1.0
            alias = intermediate.permute(1, 0)
            intermediate.relu_()
            return intermediate, alias

        x = torch.randn(2, 3, device="cuda")
        gm = make_fx(function)(x)
        compiled = compile_fx_entry(gm, [x])
        with torch.inference_mode():
            actual = compiled([x])
            expected = function(x)
        for result, reference in zip(actual, expected):
            _assert_bitwise_equal(result, reference)
        self.assertEqual(
            actual[0].untyped_storage().data_ptr(),
            actual[1].untyped_storage().data_ptr(),
        )

    def test_escaped_private_update_is_rejected(self):
        from torch._inductor.loops_ir import lower_fx_to_cpp
        from torch.fx.experimental.proxy_tensor import make_fx

        def function(x):
            intermediate = x + 1.0
            alias = torch.ops.aten.alias.default(intermediate)
            intermediate.relu_()
            return alias

        x = torch.randn(8, device="cuda")
        gm = make_fx(function)(x)
        with self.assertRaisesRegex(LoopsIrUnsupportedError, "private temporary"):
            lower_fx_to_cpp(gm, [x], executable=True)

    def test_single_element_extern_list_is_not_a_tensor(self):
        from torch._inductor.loops_ir_backend import compile_fx_entry
        from torch.fx.experimental.proxy_tensor import make_fx

        def function(x):
            return torch.ops.aten.split.Tensor(x, 100)[0] * 2.0

        x = torch.randn(8, device="cuda")
        compiled = compile_fx_entry(make_fx(function)(x), [x])
        with torch.inference_mode():
            actual = compiled([x])
            expected = function(x)
        _assert_bitwise_equal(actual, expected)

    def test_cached_launches_keep_prior_outputs(self):
        from torch._inductor.loops_ir_backend import loops_ir_backend

        def function(x):
            return torch.relu(x * 2.0 + 1.0)

        compiled = torch.compile(
            function, backend=loops_ir_backend, fullgraph=True, dynamic=False
        )
        inputs = [torch.randn(16, device="cuda") for _ in range(2)]
        with torch.inference_mode():
            outputs = [compiled(value) for value in inputs]
            for actual, value in zip(outputs, inputs):
                _assert_bitwise_equal(actual, function(value))
        self.assertNotEqual(outputs[0].data_ptr(), outputs[1].data_ptr())

    def test_special_values_bitwise(self):
        _require_cuda()
        vals = torch.tensor(
            [
                0.0,
                -0.0,
                1.0,
                -1.0,
                float("inf"),
                float("-inf"),
                1.1754944e-38,  # min normal
                1.4e-45,  # min subnormal
                1e-30,
                -1e-30,
                3.4028235e38,
                -3.4028235e38,
            ],
            device="cuda",
            dtype=torch.float32,
        )
        nan_payload = torch.tensor(
            [0x7FC00001, 0x7FC00002, 0x7F800001], dtype=torch.int32, device="cuda"
        ).view(torch.float32)
        x = torch.cat([vals, nan_payload])
        fn = lambda a: torch.relu(a * 1.5 + 0.25)  # noqa: E731
        with torch.inference_mode():
            expected = fn(x)
            actual = _compile(fn, x)
        _assert_bitwise_equal(actual, expected)

    def test_fma_sensitive_chain_bitwise(self):
        _require_cuda()
        # a*b + c must not contract into FMA: rounding boundary preserved.
        torch.manual_seed(7)
        a = torch.randn(1024, device="cuda", dtype=torch.float32)
        b = torch.randn(1024, device="cuda", dtype=torch.float32)
        c = torch.randn(1024, device="cuda", dtype=torch.float32)
        fn = lambda x, y, z: (x * y) + z  # noqa: E731
        with torch.inference_mode():
            expected = fn(a, b, c)
            actual = _compile(fn, a, b, c)
        _assert_bitwise_equal(actual, expected)

    def test_broadcast_noncontig_offset(self):
        _require_cuda()
        base = torch.randn(8, 16, device="cuda", dtype=torch.float32)
        x = base[1:7, ::2]  # non-contiguous slice, nonzero storage offset
        self.assertGreater(x.storage_offset(), 0)
        w = torch.randn(6, 8, device="cuda", dtype=torch.float32)
        fn = lambda a, b: a * b + 1.0  # noqa: E731
        with torch.inference_mode():
            expected = fn(x, w)
            actual = _compile(fn, x, w)
        self.assertEqual(tuple(actual.stride()), tuple(expected.stride()))
        _assert_bitwise_equal(actual, expected)

    def test_scalar_and_empty(self):
        _require_cuda()
        x = torch.randn(0, 8, device="cuda", dtype=torch.float32)
        fn = lambda a: a * 2.0 + 1.0  # noqa: E731
        with torch.inference_mode():
            expected = fn(x)
            actual = _compile(fn, x)
        self.assertEqual(actual.numel(), 0)
        self.assertEqual(tuple(actual.shape), tuple(expected.shape))

        s = torch.randn((), device="cuda", dtype=torch.float32)
        fn2 = lambda a: a * 3.0  # noqa: E731
        with torch.inference_mode():
            _assert_bitwise_equal(_compile(fn2, s), fn2(s))

    def test_extern_sum_keepdim(self):
        _require_cuda()
        torch.manual_seed(1)
        x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
        for dims, keepdim in [((1,), False), ((1,), True), ((0, 1), False)]:
            with self.subTest(dims=dims, keepdim=keepdim):
                fn = lambda a: a.sum(dim=dims, keepdim=keepdim)  # noqa: E731
                with torch.inference_mode():
                    expected = fn(x)
                    actual = _compile(fn, x)
                self.assertEqual(tuple(actual.shape), tuple(expected.shape))
                self.assertEqual(tuple(actual.stride()), tuple(expected.stride()))
                self.assertEqual(actual.dtype, expected.dtype)
                _assert_bitwise_equal(actual, expected)

    def test_extern_max_and_any_mixed_dtypes(self):
        _require_cuda()
        torch.manual_seed(2)
        x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
        fn_max = lambda a: torch.max(a, dim=1)  # noqa: E731
        with torch.inference_mode():
            expected_vals, expected_idx = fn_max(x)
            actual_vals, actual_idx = _compile(fn_max, x)
        _assert_bitwise_equal(actual_vals, expected_vals)
        # indices are i64 opaque boundary values: exact metadata + values.
        self.assertEqual(actual_idx.dtype, torch.int64)
        self.assertEqual(tuple(actual_idx.shape), tuple(expected_idx.shape))
        self.assertTrue(torch.equal(actual_idx, expected_idx))

        fn_any = lambda a: torch.any(a > 0, dim=1)  # noqa: E731
        with torch.inference_mode():
            expected_b = fn_any(x)
            actual_b = _compile(fn_any, x)
        self.assertEqual(actual_b.dtype, torch.bool)
        self.assertTrue(torch.equal(actual_b, expected_b))

    def test_custom_op_deterministic_exact_calls(self):
        _require_cuda()
        calls: list[tuple[tuple, dict]] = []

        @torch.library.custom_op("loops_ir_test_ns::det_scale", mutates_args=())
        def det_scale(x: torch.Tensor, factor: float) -> torch.Tensor:
            return x * factor

        @det_scale.register_fake
        def _(x, factor):
            return torch.empty_like(x)

        orig = det_scale

        def counting(*args, **kwargs):
            calls.append((args, kwargs))
            return orig(*args, **kwargs)

        torch.library.opcheck(det_scale, (torch.randn(4, device="cuda"), 2.0))
        x = torch.randn(32, device="cuda", dtype=torch.float32)
        fn = lambda a: counting(a, 2.0) + 1.0  # noqa: E731
        with torch.inference_mode():
            expected = x * 2.0 + 1.0
            actual = _compile(fn, x)
        _assert_bitwise_equal(actual, expected)
        self.assertGreaterEqual(len(calls), 1)

    def test_multioutput_custom_op(self):
        _require_cuda()

        @torch.library.custom_op("loops_ir_test_ns::two_out", mutates_args=())
        def two_out(x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
            return x + 1, x * 2

        @two_out.register_fake
        def _(x):
            return torch.empty_like(x), torch.empty_like(x)

        x = torch.randn(16, device="cuda", dtype=torch.float32)
        fn = lambda a: two_out(a)[0] + two_out(a)[1]  # noqa: E731
        with torch.inference_mode():
            expected = fn(x)
            actual = _compile(fn, x)
        _assert_bitwise_equal(actual, expected)

    def test_alpha_fma_fallback_exact(self):
        _require_cuda()
        # alpha != 1 must lower to exact aten.add extern, not mul+add.
        from torch._inductor import loops_ir as lir

        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(8, device="cuda", dtype=torch.float32)
        y = torch.randn(8, device="cuda", dtype=torch.float32)
        gm = make_fx(lambda a, b: torch.ops.aten.add.Tensor(a, b, alpha=2.0))(x, y)
        program = lir.lower_fx_to_cpp(gm, [x, y], executable=True)
        dump = program.dump()
        self.assertIn("aten.add.Tensor", dump)
        self.assertNotIn("compute ", dump)
        torch.manual_seed(3)
        with torch.inference_mode():
            _assert_bitwise_equal(
                _compile(lambda a, b: torch.ops.aten.add.Tensor(a, b, alpha=2.0), x, y),
                torch.ops.aten.add.Tensor(x, y, alpha=2.0),
            )

    def test_output_alias_and_exported_temp(self):
        _require_cuda()
        x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
        # Returned view alias.
        fn_view = lambda a: a.permute(1, 0)  # noqa: E731
        with torch.inference_mode():
            expected = fn_view(x)
            actual = _compile(fn_view, x)
        self.assertEqual(tuple(actual.stride()), tuple(expected.stride()))
        _assert_bitwise_equal(actual.contiguous(), expected.contiguous())

    def test_no_fusion_across_extern(self):
        _require_cuda()
        from torch._inductor.loops_ir_backend import (
            LoopsIrArtifacts,
            compile_fx_loops_ir,
        )
        from torch.fx.experimental.proxy_tensor import make_fx

        torch.manual_seed(5)
        x = torch.randn(4, 8, device="cuda", dtype=torch.float32)
        gm = make_fx(lambda a: torch.relu(a.sum(dim=1, keepdim=True) + 0 * a[:, :1]))(x)
        artifacts = LoopsIrArtifacts()
        from torch._inductor.loops_ir_backend import _Collector

        collect = _Collector(artifacts=artifacts, ir=True)
        run = compile_fx_loops_ir(gm, [x], collector=collect)
        with torch.inference_mode():
            run([x])
        # One generated kernel region max (pointwise side), extern present.
        self.assertGreaterEqual(artifacts.num_extern_calls, 1, artifacts.kernel_dump)

    def test_ptx_has_no_fma_or_ftz(self):
        _require_cuda()
        from torch._inductor import loops_ir as lir
        from torch._inductor.codegen.loops_ir_triton import emit_triton
        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(256, device="cuda", dtype=torch.float32)
        y = torch.randn(256, device="cuda", dtype=torch.float32)
        gm = make_fx(lambda a, b: torch.relu(a * b + b))(x, y)
        program = lir.lower_fx_to_cpp(gm, [x, y], executable=True)
        kernels = self.cpp.fuse_kernels(
            self.cpp.lower_to_kernels(self.cpp.optimize_program(program))
        )
        tiled = self.cpp.lower_to_tiled(kernels, self.cpp.TileOptions(256, 4))
        self.assertGreaterEqual(len(tiled.functions), 1)
        for fn in tiled.functions:
            src = emit_triton(fn)
            self.assertIn(f"def {fn.name}(", src)
            self.assertIn("@triton.jit", src)
            self.assertNotIn("tl.fma", src)
            self.assertNotIn("tl.maximum", src)
            # PTX check via compiled kernel when available.
            from torch._inductor.codecache import PyCodeCache

            mod = PyCodeCache.load(src)
            jit_fn = getattr(mod, fn.name)
            # Launch once to materialize PTX, then inspect.
            a = torch.empty(256, dtype=torch.float32, device="cuda")
            compiled_kernel = jit_fn[(int(fn.grid_size), 1, 1)](
                x,
                y,
                a,
                num_warps=int(fn.num_warps),
                enable_fp_fusion=False,
                enable_reflect_ftz=False,
            )
            ptx_text = compiled_kernel.asm["ptx"]
            self.assertTrue(ptx_text, "compiled kernel must expose actual PTX")
            for line in ptx_text.splitlines():
                code = line.split("//")[0].split(";")[0]
                self.assertNotIn("fma.", code)
                # Integer mad is fine; FP FTZ changes subnormal semantics.
                if ".ftz" in code and "f32" in code:
                    self.fail(f"unexpected FTZ FP instruction: {line.strip()}")

    def test_rng_rejected(self):
        _require_cuda()
        from torch._inductor import loops_ir as lir
        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(8, device="cuda", dtype=torch.float32)
        gm = make_fx(lambda a: a + torch.rand_like(a))(x)
        with self.assertRaises(LoopsIrUnsupportedError):
            lir.lower_fx_to_cpp(gm, [x], executable=True)

    def test_extern_result_dtype_mismatch_rejected(self):
        @torch.library.custom_op("loops_ir_test_ns::wrong_dtype", mutates_args=())
        def wrong_dtype(x: torch.Tensor) -> torch.Tensor:
            return x.to(torch.float64)

        @wrong_dtype.register_fake
        def _(x):
            return torch.empty_like(x)

        x = torch.randn(8, device="cuda", dtype=torch.float32)
        with (
            torch.inference_mode(),
            self.assertRaisesRegex(
                LoopsIrUnsupportedError, "eager result layout mismatch"
            ),
        ):
            _compile(lambda a: wrong_dtype(a) + 1.0, x)

    def test_mutation_rejected(self):
        _require_cuda()
        from torch._inductor import loops_ir as lir
        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(8, device="cuda", dtype=torch.float32)
        gm = make_fx(lambda a: a.add_(1))(x)
        with self.assertRaises(LoopsIrUnsupportedError):
            lir.lower_fx_to_cpp(gm, [x], executable=True)

    def test_structured_extern_args_preserved(self):
        _require_cuda()
        from torch._inductor import loops_ir as lir
        from torch.fx.experimental.proxy_tensor import make_fx

        x = torch.randn(2, 3, device="cuda", dtype=torch.float32)
        gm = make_fx(lambda a: a.sum(dim=[1], keepdim=True))(x)
        program = lir.lower_fx_to_cpp(gm, [x], executable=True)
        dump = program.dump()
        self.assertIn("aten.sum.dim_IntList", dump)
        self.assertIn("arguments", dump)

    def test_eager_capture_fidelity_note(self):
        _require_cuda()
        # AOT may decompose composites; fidelity of the capture boundary must
        # be checked against raw eager, not assumed. This test pins the
        # comparison harness (exact bytes) for a decomposed candidate.
        torch.manual_seed(9)
        x = torch.randn(32, device="cuda", dtype=torch.float32)
        fn = lambda a: torch.softmax(a, dim=0)  # noqa: E731 (may decompose under AOT)
        with torch.inference_mode():
            expected = fn(x)
            try:
                actual = _compile(fn, x)
            except LoopsIrUnsupportedError:
                self.skipTest("softmax not in strict subset (extern path)")
                return
        _assert_bitwise_equal(actual, expected)

    # ------------------------------------------------------------------
    # Numerical policy + typed extern metadata (CPU-runnable except where
    # noted; GPU execution covered by test_full_layout_and_clone below).
    # ------------------------------------------------------------------

    def test_numerics_policy_reader_modes(self):
        from torch._inductor import config as inductor_config
        from torch._inductor.fx_passes.cpp_ir_strict import (
            cpp_ir_enabled,
            cpp_ir_numerics,
            cpp_ir_strict_enabled,
        )

        with inductor_config.patch({"experimental_cpp_ir": False}):
            self.assertFalse(cpp_ir_enabled())
            self.assertFalse(cpp_ir_strict_enabled())
            # Off-lane: option is neither read nor validated (bogus ignored).
            with inductor_config.patch({"experimental_cpp_ir_numerics": "bogus"}):
                self.assertEqual(cpp_ir_numerics(), "strict_eager")
        with inductor_config.patch({"experimental_cpp_ir": True}):
            self.assertTrue(cpp_ir_enabled())
            self.assertTrue(cpp_ir_strict_enabled())
            self.assertEqual(cpp_ir_numerics(), "strict_eager")
        with inductor_config.patch(
            {
                "experimental_cpp_ir": True,
                "experimental_cpp_ir_numerics": "inductor",
            }
        ):
            self.assertTrue(cpp_ir_enabled())
            self.assertFalse(cpp_ir_strict_enabled())
            self.assertEqual(cpp_ir_numerics(), "inductor")
        with inductor_config.patch(
            {
                "experimental_cpp_ir": True,
                "experimental_cpp_ir_numerics": "bogus",
            }
        ):
            with self.assertRaisesRegex(ValueError, "unknown"):
                cpp_ir_numerics()

    def test_decomp_split_capability_vs_strict(self):
        from torch._inductor import config as inductor_config
        from torch._inductor.decomposition import select_decomp_table

        capability = (
            torch.ops.aten.max_pool2d_with_indices.default,
            torch.ops.aten.max_pool3d_with_indices.default,
            torch.ops.aten.mm.default,
            torch.ops.aten.bmm.default,
            torch.ops.aten.addmm.default,
        )
        strict_numerical = (
            torch.ops.aten.gelu.default,
            torch.ops.aten.softmax.int,
            torch.ops.aten._softmax.default,
            torch.ops.aten.var.dim,
            torch.ops.aten.std.correction,
            torch.ops.aten.layer_norm.default,
            torch.ops.aten.native_layer_norm.default,
            torch.ops.aten.group_norm.default,
        )
        with inductor_config.patch({"experimental_cpp_ir": False}):
            stock = select_decomp_table()
            for op in (*capability, *strict_numerical):
                self.assertIn(op, stock)
        with inductor_config.patch(
            {
                "experimental_cpp_ir": True,
                "experimental_cpp_ir_numerics": "strict_eager",
            }
        ):
            strict = select_decomp_table()
            for op in (*capability, *strict_numerical):
                self.assertNotIn(op, strict)
        with inductor_config.patch(
            {
                "experimental_cpp_ir": True,
                "experimental_cpp_ir_numerics": "inductor",
            }
        ):
            inductor = select_decomp_table()
            # Capability exclusions hold in both modes ...
            for op in capability:
                self.assertNotIn(op, inductor)
            # ... while strict numerical rewrites are restored where the
            # lane executes them (generated Triton or native extern).
            for op in strict_numerical:
                self.assertIn(op, inductor)

    def test_capability_gates_block_both_modes(self):
        from torch._inductor import config as inductor_config
        from torch._inductor.fx_passes.cpp_ir_strict import (
            allow_online_softmax_rewrite,
            allow_quack_symmetric_mm,
        )

        with inductor_config.patch({"experimental_cpp_ir": False}):
            self.assertTrue(allow_online_softmax_rewrite())
            self.assertTrue(allow_quack_symmetric_mm())
        for numerics in ("strict_eager", "inductor"):
            with (
                self.subTest(numerics=numerics),
                inductor_config.patch(
                    {
                        "experimental_cpp_ir": True,
                        "experimental_cpp_ir_numerics": numerics,
                    }
                ),
            ):
                # Backend-private online-softmax prim and template-only QUACK
                # stay blocked until an executable native fallback is proven.
                self.assertFalse(allow_online_softmax_rewrite())
                self.assertFalse(allow_quack_symmetric_mm())

    def test_frozen_plan_threads_effective_policy(self):
        from torch._inductor import config as inductor_config
        from torch._inductor.loops_ir_backend import _build_frozen_plan
        from torch.fx.experimental.proxy_tensor import make_fx

        _require_cuda()
        x = torch.randn(8, device="cuda")
        with inductor_config.patch(
            {
                "experimental_cpp_ir": True,
                "experimental_cpp_ir_numerics": "strict_eager",
            }
        ):
            plan = _build_frozen_plan(make_fx(lambda a: a * 2.0 + 1.0)(x), [x])
        self.assertEqual(plan.numerics, "strict_eager")
        self.assertFalse(plan.enable_fp_fusion)
        self.assertTrue(plan.disable_ftz)
        with inductor_config.patch(
            {
                "experimental_cpp_ir": True,
                "experimental_cpp_ir_numerics": "inductor",
                "emulate_precision_casts": True,
                "eager_numerics.disable_ftz": True,
            }
        ):
            plan = _build_frozen_plan(make_fx(lambda a: a * 2.0 + 1.0)(x), [x])
        self.assertEqual(plan.numerics, "inductor")
        self.assertFalse(plan.enable_fp_fusion)
        self.assertTrue(plan.disable_ftz)
        with inductor_config.patch(
            {
                "experimental_cpp_ir": True,
                "experimental_cpp_ir_numerics": "inductor",
                "emulate_precision_casts": False,
                "eager_numerics.disable_ftz": False,
            }
        ):
            plan = _build_frozen_plan(make_fx(lambda a: a * 2.0 + 1.0)(x), [x])
        self.assertEqual(plan.numerics, "inductor")
        self.assertTrue(plan.enable_fp_fusion)
        self.assertFalse(plan.disable_ftz)

    def test_layout_memory_format_round_trip(self):
        from types import SimpleNamespace

        from torch._inductor.codegen.loops_ir_wrapper import render_frozen_arg
        from torch._inductor.loops_ir import _to_call_arg, load_cpp_ir
        from torch._inductor.loops_ir_backend import _freeze_call_arg

        cpp = load_cpp_ir()
        for value, kind, text, expression in (
            (torch.strided, "layout", "strided", "torch.strided"),
            (
                torch.contiguous_format,
                "memory_format",
                "contiguous_format",
                "torch.contiguous_format",
            ),
            (
                torch.preserve_format,
                "memory_format",
                "preserve_format",
                "torch.preserve_format",
            ),
        ):
            with self.subTest(kind=kind, text=text):
                arg = _to_call_arg(cpp, value, {})
                self.assertEqual(arg.kind_name, kind)
                self.assertEqual(arg.text, text)
                frozen = _freeze_call_arg(SimpleNamespace(kind_name=kind, text=text))
                self.assertIs(frozen, value)
                self.assertEqual(render_frozen_arg(frozen, {}), expression)
        with self.assertRaisesRegex(Exception, "unsupported|invalid"):
            _freeze_call_arg(
                SimpleNamespace(kind_name="layout", text="arbitrary_code()")
            )

    def test_full_layout_and_clone_memory_format_execute(self):
        _require_cuda()
        from torch._inductor.loops_ir_backend import compile_fx_entry
        from torch.fx.experimental.proxy_tensor import make_fx

        def function(value):
            filled = torch.full(
                value.shape,
                0.25,
                dtype=value.dtype,
                layout=torch.strided,
                device=value.device,
            )
            cloned = filled.clone(memory_format=torch.contiguous_format)
            return value * cloned + filled

        value = torch.randn(4, 16, device="cuda")
        gm = make_fx(function)(value)
        run = compile_fx_entry(gm, [value])
        with torch.inference_mode():
            _assert_bitwise_equal(run([value]), function(value))

    def test_non_float_boundary_views_preserve_aliases(self):
        _require_cuda()
        from torch._inductor.loops_ir_backend import compile_fx_entry
        from torch.fx.experimental.proxy_tensor import make_fx

        indices = torch.arange(16, device="cuda", dtype=torch.int64)[1:7:2].view(1, 3)
        mask = torch.tensor([[True], [False], [True]], device="cuda")
        value = torch.randn(4, 3, device="cuda")

        def function(index, keep, data):
            return index.expand(4, 3), keep.transpose(0, 1).expand(4, 3), data + 1.0

        gm = make_fx(function)(indices, mask, value)
        run = compile_fx_entry(gm, [indices, mask, value])
        with torch.inference_mode():
            actual = run([indices, mask, value])
            expected = function(indices, mask, value)
        for result, reference in zip(actual, expected):
            _assert_bitwise_equal(result, reference)
        self.assertEqual(
            actual[0].untyped_storage().data_ptr(),
            indices.untyped_storage().data_ptr(),
        )
        self.assertEqual(
            actual[1].untyped_storage().data_ptr(),
            mask.untyped_storage().data_ptr(),
        )

    def test_nonfinite_static_mask_inf_executes_nan_rejected(self):
        _require_cuda()
        from torch._inductor import loops_ir as lir
        from torch._inductor.loops_ir_backend import compile_fx_entry
        from torch.fx.experimental.proxy_tensor import make_fx

        value = torch.randn(4, device="cuda")
        gm = make_fx(lambda a: a + float("inf"))(value)
        program = lir.lower_fx_to_cpp(gm, [value], executable=True)
        self.assertIn("aten.add", program.dump())
        run = compile_fx_entry(gm, [value])
        with torch.inference_mode():
            _assert_bitwise_equal(run([value]), value + float("inf"))
        gm_nan = make_fx(lambda a: a + float("nan"))(value)
        with self.assertRaises(LoopsIrUnsupportedError):
            lir.lower_fx_to_cpp(gm_nan, [value], executable=True)

    def test_bert_norm_keeps_native_boundaries(self):
        _require_cuda()
        from torch._inductor import loops_ir as lir
        from torch._inductor.loops_ir_backend import compile_fx_entry
        from torch.fx.experimental.proxy_tensor import make_fx

        value = torch.randn(8, 32, device="cuda")
        weight = torch.randn(32, device="cuda")
        bias = torch.randn(32, device="cuda")

        def function(data, scale, offset):
            # ManualBERT pattern: sample std (N-1) with eps OUTSIDE the sqrt;
            # not equivalent to biased-variance LayerNorm -- must not be
            # rewritten. mean/std/GELU/softmax all stay native externs.
            mean = data.mean(-1, keepdim=True)
            std = data.std(-1, keepdim=True)
            normalized = scale * (data - mean) / (std + 1e-6) + offset
            return (
                normalized,
                torch.nn.functional.gelu(data),
                torch.nn.functional.softmax(data, dim=-1),
            )

        gm = make_fx(function)(value, weight, bias)
        dump = lir.lower_fx_to_cpp(gm, [value, weight, bias], executable=True).dump()
        # mean/std/GELU/softmax all stay native extern boundaries (softmax may
        # appear decomposed as aten._softmax -- still a native extern, never
        # generated pointwise or a layernorm synthesis).
        for marker in ("aten.mean", "aten.gelu", "aten._softmax"):
            self.assertIn(marker, dump)
        self.assertNotIn("layer_norm", dump)
        run = compile_fx_entry(gm, [value, weight, bias])
        with torch.inference_mode():
            actual = run([value, weight, bias])
            expected = function(value, weight, bias)
        for result, reference in zip(actual, expected):
            _assert_bitwise_equal(result, reference)


if __name__ == "__main__":
    run_tests()
