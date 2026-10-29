# Owner(s): ["module: inductor"]
"""Measured LoopsIR op autotuning: CPU tests with mocked timer/compilers.

CUDA-gated tests below are accuracy-only with a mocked benchmark (no real
GPU timing yet).
"""

import dataclasses
import hashlib
import unittest

import torch
from torch._inductor.codegen import loops_ir_op_autotune as at
from torch.testing._internal.common_utils import run_tests, TestCase


def _h(s: str) -> str:
    return hashlib.sha256(s.encode()).hexdigest()


def _spec(sizes=(4,), strides=(1,), dtype="float32", device="cpu", offset=0):
    return at.TensorSpec(
        dtype=dtype,
        sizes=tuple(sizes),
        strides=tuple(strides),
        offset=offset,
        device=device,
    )


def _req(policy="inductor", **kw):
    fence = at.NumericsFence(policy=policy)
    s = _spec()
    base = dict(op_name="add", numerics=fence, inputs=(s, s), output=s)
    base.update(kw)
    return at.OpTuneRequest(**base)


def _desc(
    cid,
    kind="generated",
    inductor_only=False,
    cfg="block=32",
    key=None,
    requires_zeroed_output=False,
):
    return at.CandidateDescriptor(
        candidate_id=cid,
        option_id="triton_tiled_gemm" if kind == "generated" else "native_aten_gemm",
        option_version=1,
        kind=kind,
        config=cfg,
        cache_key=key or f"key-{cid}",
        source_hash=_h(f"src-{cid}-{cfg}"),
        inductor_only=inductor_only,
        requires_zeroed_output=requires_zeroed_output,
    )


def _add(inputs, out):
    return inputs[0] + inputs[1]


def _ref_add(inputs):
    return inputs[0] + inputs[1]


def _bench_seq(values):
    it = iter(values)

    def bench(zero_arg):
        zero_arg()  # exercise the callable; timing is mocked
        return float(next(it))

    return bench


class FakeCache:
    """In-memory JSON cache mimicking LocalAutotuneCache serde."""

    def __init__(self):
        self.d: dict[str, object] = {}
        self.puts = 0

    def get(self, key):
        import copy

        v = self.d.get(key)
        return copy.deepcopy(v)

    def put(self, key, value):
        import copy

        self.d[key] = copy.deepcopy(value)
        self.puts += 1


class TestLoopsIrOpAutotune(TestCase):
    def test_fastest_generated_wins(self):
        req = _req()
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}
        res = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([1.0, 5.0]),
            use_cache=False,
        )
        self.assertTrue(res.autotuned and res.measured and not res.cache_hit)
        self.assertEqual(res.winner.candidate_id, "a_gen")
        self.assertEqual(res.winner.origin, "cold")
        self.assertEqual(res.measurements_ms, {"a_gen": 1.0, "b_native": 5.0})

    def test_fastest_native_wins(self):
        req = _req()
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}
        res = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([5.0, 1.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "b_native")
        self.assertEqual(res.winner.kind, "native")

    def test_strict_vetoes_inductor_only(self):
        req = _req(policy="strict_eager")
        cands = [_desc("a_fast", inductor_only=True), _desc("b_native", kind="native")]
        factories = {"a_fast": lambda: _add, "b_native": lambda: _add}
        res = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([2.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "b_native")
        self.assertNotIn("a_fast", res.measurements_ms)

    def test_wrong_values_excluded(self):
        req = _req()
        cands = [_desc("a_bad"), _desc("b_native", kind="native")]

        def bad(inputs, out):
            return torch.zeros_like(inputs[0]) + 12345.0

        factories = {"a_bad": lambda: bad, "b_native": lambda: _add}
        res = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([7.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "b_native")

    def test_wrong_dtype_shape_stride_excluded(self):
        req = _req()
        cands = [
            _desc("a_bad"),
            _desc("b_bad2"),
            _desc("c_bad3"),
            _desc("d_native", kind="native"),
        ]

        def bad_dtype(inputs, out):
            return (inputs[0] + inputs[1]).to(torch.float64)

        def bad_shape(inputs, out):
            return torch.zeros(8, dtype=torch.float32)

        def bad_stride(inputs, out):
            base = torch.randn(7, dtype=torch.float32)
            return torch.as_strided(base, (4,), (2,))

        factories = {
            "a_bad": lambda: bad_dtype,
            "b_bad2": lambda: bad_shape,
            "c_bad3": lambda: bad_stride,
            "d_native": lambda: _add,
        }
        res = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([3.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "d_native")
        self.assertEqual(list(res.measurements_ms), ["d_native"])

    def test_compile_failure_falls_back(self):
        req = _req()
        cands = [_desc("a_broken"), _desc("b_native", kind="native")]

        def boom():
            raise RuntimeError("compile boom")

        factories = {"a_broken": boom, "b_native": lambda: _add}
        res = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([4.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "b_native")

    def test_timing_failure_falls_back(self):
        req = _req()
        cands = [_desc("a_flaky"), _desc("b_native", kind="native")]
        factories = {"a_flaky": lambda: _add, "b_native": lambda: _add}
        calls = {"n": 0}

        def bench(zero_arg):
            calls["n"] += 1
            zero_arg()
            if calls["n"] == 1:
                raise RuntimeError("timer boom")
            return 2.0

        res = at.autotune_op(
            req, cands, factories, _ref_add, benchmark_fn=bench, use_cache=False
        )
        self.assertEqual(res.winner.candidate_id, "b_native")

    def test_nonfinite_timing_skipped(self):
        req = _req()
        cands = [_desc("a_inf"), _desc("b_native", kind="native")]
        factories = {"a_inf": lambda: _add, "b_native": lambda: _add}
        res = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([float("inf"), 2.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "b_native")
        self.assertNotIn("a_inf", res.measurements_ms)

    def test_nan_inf_semantics(self):
        s = _spec()
        req = _req(inputs=(s, s), output=s)

        def ref(inputs):
            return torch.tensor([float("nan"), 1.0, float("inf"), 2.0])

        def bad_nan(inputs, out):
            return torch.tensor([0.0, 1.0, float("inf"), 2.0])

        def bad_inf_sign(inputs, out):
            return torch.tensor([float("nan"), 1.0, float("-inf"), 2.0])

        def good(inputs, out):
            return torch.tensor([float("nan"), 1.0, float("inf"), 2.0])

        cands = [_desc("a_bad"), _desc("b_bad2"), _desc("c_native", kind="native")]
        factories = {
            "a_bad": lambda: bad_nan,
            "b_bad2": lambda: bad_inf_sign,
            "c_native": lambda: good,
        }
        res = at.autotune_op(
            req, cands, factories, ref, benchmark_fn=_bench_seq([1.0]), use_cache=False
        )
        self.assertEqual(res.winner.candidate_id, "c_native")

    def test_strict_bytes_exact(self):
        s = _spec()
        req = _req(policy="strict_eager", inputs=(s, s), output=s)

        def ref(inputs):
            return torch.tensor([1.0, 2.0, 3.0, 4.0])

        def near(inputs, out):
            return torch.tensor([1.0 + 1e-7, 2.0, 3.0, 4.0])

        cands = [_desc("a_near"), _desc("b_native", kind="native")]
        factories = {
            "a_near": lambda: near,
            "b_native": lambda: lambda inputs, out: ref(inputs),
        }
        res = at.autotune_op(
            req, cands, factories, ref, benchmark_fn=_bench_seq([1.0]), use_cache=False
        )
        self.assertEqual(res.winner.candidate_id, "b_native")

    def test_all_fail_raises(self):
        req = _req()
        cands = [_desc("a_bad"), _desc("b_native", kind="native")]

        def bad(inputs, out):
            return torch.zeros_like(inputs[0]) + 1.0

        def ref(inputs):
            return torch.ones(4) * 999.0 + inputs[0] * 0

        # Both runners return zeros; reference returns 999s -> both fail.
        def zero(inputs, out):
            return torch.zeros(4)

        factories = {"a_bad": lambda: zero, "b_native": lambda: bad}
        with self.assertRaises(at.NoValidCandidateError):
            at.autotune_op(
                req,
                cands,
                factories,
                ref,
                benchmark_fn=_bench_seq([1.0, 1.0]),
                use_cache=False,
            )

    def test_reference_failure_raises(self):
        req = _req()
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}

        def ref(inputs):
            raise RuntimeError("ref boom")

        with self.assertRaises(at.NoValidCandidateError):
            at.autotune_op(
                req,
                cands,
                factories,
                ref,
                benchmark_fn=_bench_seq([1.0, 1.0]),
                use_cache=False,
            )

    def test_deterministic_tie(self):
        req = _req()
        cands = [_desc("b_gen"), _desc("a_gen"), _desc("c_native", kind="native")]
        factories = {c.candidate_id: (lambda: _add) for c in cands}
        res = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([1.0, 1.0, 1.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "a_gen")

    def test_singleton_not_measured(self):
        req = _req()
        cands = [_desc("only_native", kind="native")]
        factories = {"only_native": lambda: _add}
        calls = {"n": 0}

        def bench(zero_arg):
            calls["n"] += 1
            return 1.0

        res = at.autotune_op(
            req, cands, factories, _ref_add, benchmark_fn=bench, use_cache=False
        )
        self.assertFalse(res.autotuned)
        self.assertFalse(res.measured)
        self.assertFalse(res.cache_hit)
        self.assertEqual(res.winner.origin, "singleton")
        self.assertEqual(calls["n"], 0)
        self.assertEqual(res.measurements_ms, {})

    def test_cache_hit(self):
        req = _req()
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}
        cache = FakeCache()
        r1 = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([1.0, 9.0]),
            cache=cache,
        )
        self.assertFalse(r1.cache_hit)
        self.assertEqual(r1.winner.origin, "cold")

        def boom(zero_arg):
            raise AssertionError("must not remeasure on cache hit")

        r2 = at.autotune_op(
            req, cands, factories, _ref_add, benchmark_fn=boom, cache=cache
        )
        self.assertTrue(r2.cache_hit)
        self.assertEqual(r2.winner.origin, "cache")
        self.assertFalse(r2.autotuned)
        self.assertEqual(r2.winner.candidate_id, r1.winner.candidate_id)
        self.assertEqual(r2.winner.measured_ms, r1.winner.measured_ms)

    def test_cache_mismatch_remeasures(self):
        req = _req()
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}
        cache = FakeCache()
        at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([1.0, 9.0]),
            cache=cache,
        )
        # Same ids but different config -> manifest hash differs -> cold.
        cands2 = [_desc("a_gen", cfg="block=64"), _desc("b_native", kind="native")]
        factories2 = {"a_gen": lambda: _add, "b_native": lambda: _add}
        r2 = at.autotune_op(
            req,
            cands2,
            factories2,
            _ref_add,
            benchmark_fn=_bench_seq([8.0, 1.0]),
            cache=cache,
        )
        self.assertFalse(r2.cache_hit)
        self.assertEqual(r2.winner.origin, "cold")
        self.assertEqual(r2.winner.candidate_id, "b_native")

    def test_cache_corruption_fails_closed(self):
        req = _req()
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}
        cache = FakeCache()
        key = at.cache_key_hex(req, cands, {"type": "cpu", "index": -1})
        import os

        # Simulate a corrupt entry at the resolved path.
        from torch._inductor.runtime.cache_dir_utils import cache_dir

        path = os.path.join(cache_dir(), "loops_ir_op_autotune", f"{key}.json")
        cache.put(path, {"version": 999, "garbage": True})
        r = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([1.0, 9.0]),
            cache=cache,
        )
        self.assertFalse(r.cache_hit)
        self.assertEqual(r.winner.origin, "cold")

    def test_rng_restored_on_success_and_error(self):
        req = _req()
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}
        torch.manual_seed(0)
        before = torch.clone(torch.random.get_rng_state())
        at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([1.0, 2.0]),
            use_cache=False,
        )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

        def bad(inputs, out):
            return torch.zeros_like(inputs[0]) + 5.0

        def ref(inputs):
            return torch.ones(4) * 7.0

        torch.manual_seed(0)
        before = torch.clone(torch.random.get_rng_state())
        with self.assertRaises(at.NoValidCandidateError):
            at.autotune_op(
                req,
                cands,
                {"a_gen": lambda: bad, "b_native": lambda: bad},
                ref,
                benchmark_fn=_bench_seq([1.0, 1.0]),
                use_cache=False,
            )
        self.assertTrue(torch.equal(before, torch.random.get_rng_state()))

    def test_no_user_mutation_or_grad(self):
        req = _req()
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        user = torch.randn(4, requires_grad=True)
        snap = user.detach().clone()
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}
        at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([1.0, 2.0]),
            use_cache=False,
        )
        self.assertTrue(torch.equal(user.detach(), snap))
        self.assertIsNone(user.grad)

    def test_manifest_validation(self):
        req = _req()
        with self.assertRaises(ValueError):
            at.autotune_op(
                req, [], {}, _ref_add, benchmark_fn=_bench_seq([]), use_cache=False
            )
        with self.assertRaises(ValueError):
            at.autotune_op(
                req,
                [_desc("a"), _desc("a")],
                {"a": lambda: _add},
                _ref_add,
                benchmark_fn=_bench_seq([]),
            )
        with self.assertRaises(ValueError):  # no native baseline
            at.autotune_op(
                req,
                [_desc("a")],
                {"a": lambda: _add},
                _ref_add,
                benchmark_fn=_bench_seq([]),
                use_cache=False,
            )
        with self.assertRaises(ValueError):  # missing factory
            at.autotune_op(
                req,
                [_desc("a_gen"), _desc("b_native", kind="native")],
                {"a_gen": lambda: _add},
                _ref_add,
                benchmark_fn=_bench_seq([]),
            )

    def test_cache_key_sensitive(self):
        req = _req()
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        k1 = at.cache_key_hex(req, cands, {"type": "cpu", "index": -1})
        k2 = at.cache_key_hex(
            req,
            [_desc("a_gen", cfg="block=64"), _desc("b_native", kind="native")],
            {"type": "cpu", "index": -1},
        )
        k3 = at.cache_key_hex(
            req,
            cands,
            {
                "type": "cuda",
                "index": 0,
                "major": 9,
                "minor": 0,
                "multi_processor_count": 1,
                "total_memory": 1,
            },
        )
        self.assertNotEqual(k1, k2)
        self.assertNotEqual(k1, k3)
        self.assertEqual(len(k1), 64)

    def test_winner_record_json_roundtrip_has_no_callbacks(self):
        req = _req()
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}
        res = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([1.0, 2.0]),
            use_cache=False,
        )
        blob = res.winner.to_json()
        back = at.WinnerRecord.from_json(blob)
        self.assertEqual(back, res.winner)
        for v in blob.values():
            self.assertNotIsInstance(v, torch.Tensor)

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_accuracy_only_mocked_benchmark(self):
        s = _spec(device="cuda:0")
        req = _req(inputs=(s, s), output=s)
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]

        def add(inputs, out):
            return inputs[0] + inputs[1]

        def ref(inputs):
            return inputs[0] + inputs[1]

        factories = {"a_gen": lambda: add, "b_native": lambda: add}
        res = at.autotune_op(
            req,
            cands,
            factories,
            ref,
            benchmark_fn=_bench_seq([1.0, 2.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "a_gen")

    # ---- Review findings (isolated-layout copies, fences, hashes) ----

    def test_nondense_inputs_keep_layout_and_values(self):
        # Non-dense sizes/strides plus a nonzero offset: runners must see
        # isolated tensors with the exact spec layout and same values.
        spec = _spec(sizes=(2, 3), strides=(6, 2), offset=4)
        out_spec = _spec(sizes=(2, 3), strides=(6, 2), offset=4)
        req = _req(inputs=(spec, spec), output=out_spec)
        seen: dict[str, list[torch.Tensor]] = {}

        def ref(inputs):
            seen.setdefault("ref", []).append(inputs[0].clone())
            self.assertEqual(tuple(inputs[0].shape), (2, 3))
            self.assertEqual(tuple(inputs[0].stride()), (6, 2))
            self.assertEqual(inputs[0].storage_offset(), 4)
            out = at._fresh_tensor(out_spec)
            out.copy_(inputs[0] + inputs[1])
            return out

        def cand(inputs, out):
            seen.setdefault("cand", []).append(inputs[0].clone())
            self.assertEqual(tuple(inputs[0].shape), (2, 3))
            self.assertEqual(tuple(inputs[0].stride()), (6, 2))
            self.assertEqual(inputs[0].storage_offset(), 4)
            self.assertEqual(tuple(out.shape), (2, 3))
            self.assertEqual(tuple(out.stride()), (6, 2))
            self.assertEqual(out.storage_offset(), 4)
            got = at._fresh_tensor(out_spec)
            got.copy_(inputs[0] + inputs[1])
            return got

        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: cand, "b_native": lambda: cand}
        res = at.autotune_op(
            req,
            cands,
            factories,
            ref,
            benchmark_fn=_bench_seq([1.0, 2.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "a_gen")
        # Same logical values flowed to reference and candidate.
        self.assertTrue(torch.equal(seen["cand"][0], seen["ref"][0]))

    def test_strided_offset_miscompile_rejected(self):
        # A candidate that densifies/ignores strides (the old .clone() bug)
        # must fail validation against the strided reference.
        spec = _spec(sizes=(2, 3), strides=(6, 2), offset=0)
        out_spec = _spec(sizes=(2, 3), strides=(6, 2), offset=0)
        req = _req(inputs=(spec, spec), output=out_spec)

        def ref(inputs):
            out = at._fresh_tensor(out_spec)
            out.copy_(inputs[0] + inputs[1])
            return out

        def miscompile(inputs, out):
            # Densifies: contiguous (2,3) result; strides mismatch.
            return (inputs[0].contiguous() + inputs[1].contiguous()).clone()

        def native(inputs, out):
            got = at._fresh_tensor(out_spec)
            got.copy_(inputs[0] + inputs[1])
            return got

        cands = [_desc("a_bad"), _desc("b_native", kind="native")]
        factories = {"a_bad": lambda: miscompile, "b_native": lambda: native}
        res = at.autotune_op(
            req, cands, factories, ref, benchmark_fn=_bench_seq([3.0]), use_cache=False
        )
        self.assertEqual(res.winner.candidate_id, "b_native")
        self.assertEqual(list(res.measurements_ms), ["b_native"])

    def test_offset_int_bool_tensors(self):
        for dtype in ("int32", "int64", "bool"):
            spec = _spec(sizes=(4,), strides=(1,), dtype=dtype, offset=3)
            req = _req(
                inputs=(spec,), output=_spec(sizes=(4,), strides=(1,), dtype=dtype)
            )
            cands = [_desc(f"a_{dtype}"), _desc(f"b_{dtype}", kind="native")]

            def ident(inputs, out):
                return inputs[0].clone()

            def ref(inputs):
                return inputs[0].clone()

            factories = {c.candidate_id: (lambda: ident) for c in cands}
            res = at.autotune_op(
                req,
                cands,
                factories,
                ref,
                benchmark_fn=_bench_seq([1.0, 2.0]),
                use_cache=False,
            )
            self.assertEqual(res.winner.candidate_id, f"a_{dtype}")

    def test_inf_with_finite_sign_difference_passes(self):
        # Unrelated Inf present + a within-tolerance finite -0.0/+0.0 sign
        # difference: the Inf-sign gate must not veto the finite lane.
        s = _spec()
        req = _req(inputs=(s, s), output=s, atol=1e-6, rtol=1e-5)

        def ref(inputs):
            return torch.tensor([float("inf"), 0.0, 1.0, 2.0])

        def cand(inputs, out):
            return torch.tensor([float("inf"), -0.0, 1.0, 2.0])

        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: cand, "b_native": lambda: cand}
        res = at.autotune_op(
            req,
            cands,
            factories,
            ref,
            benchmark_fn=_bench_seq([1.0, 2.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "a_gen")

    def test_complex_rejected_fail_closed(self):
        spec = _spec(dtype="complex64")
        req = _req(inputs=(spec, spec), output=spec)
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]

        def cc(inputs, out):
            return inputs[0] + inputs[1]

        factories = {"a_gen": lambda: cc, "b_native": lambda: cc}
        with self.assertRaises(at.NoValidCandidateError):
            at.autotune_op(
                req,
                cands,
                factories,
                cc,
                benchmark_fn=_bench_seq([1.0, 1.0]),
                use_cache=False,
            )

    def test_singleton_accuracy_failure_raises_no_valid(self):
        req = _req()
        cands = [_desc("only_bad", kind="native")]

        def bad(inputs, out):
            return torch.zeros_like(inputs[0]) + 1.0

        with self.assertRaises(at.NoValidCandidateError):
            at.autotune_op(
                req,
                cands,
                {"only_bad": lambda: bad},
                _ref_add,
                benchmark_fn=_bench_seq([1.0]),
                use_cache=False,
            )

    def test_singleton_reference_mismatch_raises(self):
        req = _req()
        cands = [_desc("only_bad", kind="native")]

        def ref(inputs):
            return torch.zeros(8)

        with self.assertRaises(at.NoValidCandidateError):
            at.autotune_op(
                req,
                cands,
                {"only_bad": lambda: _add},
                ref,
                benchmark_fn=_bench_seq([1.0]),
                use_cache=False,
            )

    def test_reference_stride_device_mismatch_raises(self):
        # Stride mismatch on a size>1 dim, and device mismatch, both fence.
        spec = _spec(sizes=(2, 3), strides=(6, 2))
        req = _req(inputs=(spec, spec), output=spec)
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}

        def ref_stride(inputs):
            return torch.as_strided(inputs[0] + inputs[1], (2, 3), (3, 1))

        with self.assertRaises(at.NoValidCandidateError):
            at.autotune_op(
                req,
                cands,
                factories,
                ref_stride,
                benchmark_fn=_bench_seq([1.0, 1.0]),
                use_cache=False,
            )

        # Device fence: declare cuda output on a cpu-only box.
        s_cuda = _spec(device="cuda:0")
        req_cuda = _req(inputs=(s_cuda, s_cuda), output=s_cuda)

        def ref_cpu(inputs):
            return (inputs[0] + inputs[1]).cpu()  # CPU result vs cuda spec

        with self.assertRaises(at.NoValidCandidateError):
            at.autotune_op(
                req_cuda,
                cands,
                factories,
                ref_cpu,
                benchmark_fn=_bench_seq([1.0, 1.0]),
                use_cache=False,
            )

    def test_reference_size1_stride_latitude(self):
        # Size-1 dims: reference stride latitude matches the candidate rule.
        spec = _spec(sizes=(1, 4), strides=(99, 1))
        req = _req(inputs=(spec, spec), output=_spec(sizes=(1, 4), strides=(4, 1)))

        def ref(inputs):
            return (inputs[0] + inputs[1]).as_strided((1, 4), (4, 1))

        cands = [_desc("a_gen"), _desc("b_native", kind="native")]

        def cand(inputs, out):
            return (inputs[0] + inputs[1]).as_strided((1, 4), (4, 1))

        factories = {"a_gen": lambda: cand, "b_native": lambda: cand}
        res = at.autotune_op(
            req,
            cands,
            factories,
            ref,
            benchmark_fn=_bench_seq([1.0, 2.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "a_gen")

    def test_negative_cached_timing_fails_closed(self):
        req = _req()
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}
        cache = FakeCache()
        r1 = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([1.0, 9.0]),
            cache=cache,
        )
        self.assertEqual(r1.winner.origin, "cold")
        # Poison the stored record with a negative timing.
        (path,) = list(cache.d.keys())
        blob = cache.d[path]
        blob["measured_ms"] = -5.0
        cache.d[path] = blob
        r2 = at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([9.0, 1.0]),
            cache=cache,
        )
        self.assertFalse(r2.cache_hit)
        self.assertEqual(r2.winner.origin, "cold")
        self.assertEqual(r2.winner.candidate_id, "b_native")

    def test_implementation_hash_covers_correctness_flags(self):
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        flipped = [
            dataclasses.replace(cands[0], inductor_only=not cands[0].inductor_only),
            cands[1],
        ]
        zeroed = [
            dataclasses.replace(cands[0], requires_zeroed_output=True),
            cands[1],
        ]
        kinded = [
            dataclasses.replace(cands[0], kind="native"),
            cands[1],
        ]
        h0 = at.implementation_hash_hex(cands)
        self.assertNotEqual(h0, at.implementation_hash_hex(flipped))
        self.assertNotEqual(h0, at.implementation_hash_hex(zeroed))
        self.assertNotEqual(h0, at.implementation_hash_hex(kinded))
        # ... and a flag flip defeats the cache even with identical ids.
        req = _req(policy="inductor")
        factories = {"a_gen": lambda: _add, "b_native": lambda: _add}
        cache = FakeCache()
        at.autotune_op(
            req,
            cands,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([1.0, 9.0]),
            cache=cache,
        )
        req_strict = _req(policy="strict_eager")
        flipped_strict = [
            dataclasses.replace(cands[0], inductor_only=True),
            cands[1],
        ]
        r = at.autotune_op(
            req_strict,
            flipped_strict,
            factories,
            _ref_add,
            benchmark_fn=_bench_seq([4.0]),
            cache=cache,
        )
        self.assertFalse(r.cache_hit)
        self.assertEqual(r.winner.candidate_id, "b_native")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_strided_offset_accuracy_only(self):
        spec = _spec(sizes=(2, 3), strides=(6, 2), offset=4, device="cuda:0")
        out = _spec(sizes=(2, 3), strides=(6, 2), offset=4, device="cuda:0")
        req = _req(inputs=(spec, spec), output=out)
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]

        def run(inputs, ro_out):
            got = at._fresh_tensor(out)
            got.copy_(inputs[0] + inputs[1])
            return got

        def ref(inputs):
            got = at._fresh_tensor(out)
            got.copy_(inputs[0] + inputs[1])
            return got

        factories = {"a_gen": lambda: run, "b_native": lambda: run}
        res = at.autotune_op(
            req,
            cands,
            factories,
            ref,
            benchmark_fn=_bench_seq([1.0, 2.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "a_gen")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_zeroed_output_accuracy_only(self):
        s = _spec(device="cuda:0")
        req = _req(inputs=(s,), output=s)
        cands = [
            _desc("a_zeroed", requires_zeroed_output=True),
            _desc("b_native", kind="native"),
        ]

        def accum(inputs, out):
            out += inputs[0]
            return None

        def ref(inputs):
            return inputs[0].clone()

        factories = {"a_zeroed": lambda: accum, "b_native": lambda: lambda i, o: ref(i)}
        res = at.autotune_op(
            req,
            cands,
            factories,
            ref,
            benchmark_fn=_bench_seq([1.0, 2.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "a_zeroed")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_int_offset_accuracy_only(self):
        spec = _spec(sizes=(4,), strides=(1,), dtype="int32", offset=2, device="cuda:0")
        out = _spec(sizes=(4,), strides=(1,), dtype="int32", device="cuda:0")
        req = _req(inputs=(spec,), output=out)
        cands = [_desc("a_gen"), _desc("b_native", kind="native")]

        def run(inputs, ro_out):
            return inputs[0].clone()

        def ref(inputs):
            return inputs[0].clone()

        factories = {"a_gen": lambda: run, "b_native": lambda: run}
        res = at.autotune_op(
            req,
            cands,
            factories,
            ref,
            benchmark_fn=_bench_seq([1.0, 2.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "a_gen")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_singleton_strided_validates(self):
        spec = _spec(sizes=(2, 3), strides=(6, 2), offset=4, device="cuda:0")
        req = _req(inputs=(spec, spec), output=spec)
        cands = [_desc("only_gen", kind="native")]

        def ref(inputs):
            got = at._fresh_tensor(spec)
            got.copy_(inputs[0] + inputs[1])
            return got

        factories = {"only_gen": lambda: lambda i, o: ref(i)}
        res = at.autotune_op(
            req, cands, factories, ref, benchmark_fn=_bench_seq([]), use_cache=False
        )
        self.assertFalse(res.autotuned)
        self.assertEqual(res.winner.origin, "singleton")

    @unittest.skipUnless(torch.cuda.is_available(), "requires CUDA")
    def test_cuda_inf_sign_mask_accuracy_only(self):
        s = _spec(device="cuda:0")
        req = _req(inputs=(s, s), output=s, atol=1e-6, rtol=1e-5)

        def ref(inputs):
            return torch.tensor([float("inf"), 0.0, 1.0, 2.0], device="cuda:0")

        def cand(inputs, out):
            return torch.tensor([float("inf"), -0.0, 1.0, 2.0], device="cuda:0")

        cands = [_desc("a_gen"), _desc("b_native", kind="native")]
        factories = {"a_gen": lambda: cand, "b_native": lambda: cand}
        res = at.autotune_op(
            req,
            cands,
            factories,
            ref,
            benchmark_fn=_bench_seq([1.0, 2.0]),
            use_cache=False,
        )
        self.assertEqual(res.winner.candidate_id, "a_gen")


if __name__ == "__main__":
    run_tests()
