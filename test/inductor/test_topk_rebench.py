# Owner(s): ["module: inductor"]
"""Tests for config.autotune_topk_rebench_*: one re-timed template/ATen pair,
a verdict applied at pick time and stored in the autotune-cache bucket, autotune
timings left as measured.

Mocked tests run on CPU (device faked as cuda); TestTopkRebenchGPU uses the real
benchmark requests in-process and in a TuningProcess, and an epilogue fusion
through MultiTemplateBuffer.
"""

import contextlib
import math
import pickle
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import torch
from torch._inductor import (
    autotune_process as ap,
    codecache,
    config,
    ir,
    select_algorithm as sa,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache, run_and_get_code
from torch.testing._internal.common_utils import HardwareClassification
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU, IS_BIG_GPU


A = sa.AlgorithmSelectorCache
LOGGER = "torch._inductor.select_algorithm"


class _Bm:
    def __init__(self, steady, fail=False, offset=0):
        self.steady, self.fail, self.cleaned = steady, fail, 0
        self.input_tensor_meta = [mock.Mock(offset=offset)]

    def make_run_fn(self, *args, out=None):
        if self.fail:
            raise RuntimeError("boom")
        return lambda: self.steady  # the fake rebench_batch reads the return value

    def cleanup_run_fn(self):
        self.cleaned += 1


def _mk(cls, name, steady, fail=False, offset=0, **attrs):
    c = cls.__new__(cls)
    c.name = name
    c.bmreq = _Bm(steady, fail, offset)
    c.hash_key = lambda n=name: f"hash:{n}"
    for k, v in attrs.items():
        setattr(c, k, v)
    return c


def tmpl(name, steady, **kw):
    return _mk(sa.TritonTemplateCaller, name, steady, **kw)


def aten(name, steady, **kw):
    kw.setdefault("tunable_dyn_dims_mask", None)
    return _mk(sa.ExternKernelCaller, name, steady, **kw)


class _Args:
    def get_benchmark_tensors(self, extern=False):
        return mock.Mock(unpack=lambda: ([], None))


def _fake_batch(fns, warmup_s, rounds, synchronize):
    return [[fn() for _ in range(rounds)] for fn in fns]


def run(timings, device="cuda", **cfg):
    cfg = {
        "autotune_topk_rebench_margin": 0.25,
        "autotune_topk_rebench_hysteresis": 0.05,
        "autotune_topk_rebench_warmup_s": 0.0,
        "autotune_topk_rebench_rounds": 3,
        "autotune_in_subproc": False,
        "autotune_cudagraph_benchmarking": False,
        **cfg,
    }
    layout = mock.Mock(device=torch.device(device))
    cpu = sa.get_interface_for_device("cpu")
    with contextlib.ExitStack() as st:
        st.enter_context(config.patch(cfg))
        st.enter_context(mock.patch.object(A, "get_inputs", return_value=_Args()))
        st.enter_context(
            mock.patch.object(ap, "rebench_batch", side_effect=_fake_batch)
        )
        st.enter_context(
            mock.patch.object(sa, "get_interface_for_device", return_value=cpu)
        )
        return A.rebench_template_vs_aten(timings, [], layout, None)


def pick(timings, verdict, **cfg):
    with config.patch({"autotune_topk_rebench_hysteresis": 0.05, **cfg}):
        return sa.topk_rebench_pick(sa.TopkTimings(timings, verdict))


class TestTopkRebench(TestCase):
    hw_classification = HardwareClassification.GENERIC

    def test_flip_to_aten_and_timings_untouched(self):
        w, a = tmpl("w", 2.0), aten("mm", 1.8)
        orig = {w: 1.5, a: 1.6}
        before = dict(orig)
        v = run(orig)
        self.assertEqual(orig, before)  # measured single-shot timings are not modified
        self.assertEqual(
            v,
            {
                "template": "hash:w",
                "aten": "hash:mm",
                "t_single": 1.5,
                "a_single": 1.6,
                "t_steady": 2.0,
                "a_steady": 1.8,
            },
        )
        choice, ms = pick(orig, v)
        self.assertIs(choice, a)
        self.assertAlmostEqual(ms, 1.8 * 1.5 / 2.0)  # ATen in the single-shot regime
        self.assertEqual((w.bmreq.cleaned, a.bmreq.cleaned), (1, 1))

    def test_template_confirmed(self):
        w, a = tmpl("w", 1.5), aten("mm", 1.8)
        t = {w: 1.4, a: 1.6}
        self.assertEqual(pick(t, run(t)), (w, 1.4))

    def test_hysteresis_at_pick_time(self):
        w, a = tmpl("w", 1.75), aten("mm", 1.8)  # 2.8% faster steady
        t = {w: 1.4, a: 1.6}
        v = run(t)
        self.assertIs(pick(t, v)[0], a)
        self.assertIs(pick(t, v, autotune_topk_rebench_hysteresis=0.0)[0], w)

    def test_no_trigger(self):
        w, a = tmpl("w", 9.0), aten("mm", 1.0)
        for t in ({w: 1.0, a: 2.0}, {w: 1.7, a: 1.6}):
            self.assertIsNone(run(t))
        self.assertIsNone(run({w: 1.0, other("dk"): 1.2}))  # no ATen

    def test_five_templates_ahead(self):
        # s014-like: only the single-shot winner is re-timed; the others keep their
        # measured timings and can never be picked over it (they are slower single-shot)
        ts = [tmpl(f"w{i}", s) for i, s in enumerate((1.20, 1.18, 1.22, 1.15, 1.25))]
        a = aten("mm", 1.05)
        t = {**dict(zip(ts, (0.83, 0.835, 0.86, 0.87, 0.93))), a: 1.0}
        v = run(t)
        self.assertEqual([c.bmreq.cleaned for c in ts], [1, 0, 0, 0, 0])
        self.assertIs(pick(t, v)[0], a)

    def test_stale_verdict_ignored(self):
        w, w2, a = tmpl("w", 2.0), tmpl("w2", 0), aten("mm", 1.8)
        v = run({w: 1.5, a: 1.6})
        self.assertIsNone(
            pick({w: 1.5, w2: 1.4, a: 1.6}, v)
        )  # winner is not the re-timed one
        self.assertIsNone(sa.topk_rebench_pick({w: 1.5, a: 1.6}))  # no verdict attached

    def test_gpu_only(self):
        w, a = tmpl("w", 2.0), aten("mm", 1.8)
        with mock.patch.object(ap, "rebench_batch", side_effect=AssertionError("ran")):
            self.assertIsNone(run({w: 1.5, a: 1.6}, device="cpu"))

    def _assert_fallback(self, reason, t, **cfg):
        n0 = A._topk_rebench_fallbacks[reason]
        with self.assertLogs(LOGGER, level="INFO") as cm:
            self.assertIsNone(run(t, **cfg))
        self.assertTrue(
            any(f"fallback ({reason})" in r.getMessage() for r in cm.records)
        )
        self.assertEqual(A._topk_rebench_fallbacks[reason], n0 + 1)

    def test_fallbacks_logged(self):
        a = aten("mm", 1.8)
        self._assert_fallback("RuntimeError", {tmpl("w", 2.0, fail=True): 1.5, a: 1.6})
        self._assert_fallback(
            "cudagraph", {tmpl("w", 2.0, _benchmark_with_cudagraphs=True): 1.5, a: 1.6}
        )
        self._assert_fallback(
            "cudagraph",
            {tmpl("w", 2.0): 1.5, a: 1.6},
            autotune_cudagraph_benchmarking=True,
            max_autotune=True,
        )
        # a choice without a benchmark request cannot be re-timed
        self._assert_fallback(
            "no_benchmark_request", {tmpl("w", 2.0, bmreq=None): 1.5, a: 1.6}
        )

    def test_subproc_skips_hint_override_and_offsets(self):
        w = tmpl("w", 2.0)
        with self.assertLogs(LOGGER, level="INFO") as cm:
            layout = mock.Mock(device=torch.device("cuda"))
            with config.patch(
                autotune_topk_rebench_margin=0.25,
                autotune_in_subproc=True,
                autotune_cudagraph_benchmarking=False,
            ):
                self.assertIsNone(
                    A.rebench_template_vs_aten(
                        {w: 1.5, aten("mm", 1.8): 1.6}, [], layout, None, 64
                    )
                )
        self.assertTrue(
            any("subproc_hint_override" in r.getMessage() for r in cm.records)
        )
        self._assert_fallback(
            "subproc_input_offset",
            {w: 1.5, aten("mm", 1.8, offset=128): 1.6},
            autotune_in_subproc=True,
        )

    def test_tunableop_mask_respected(self):
        w, a = tmpl("w", 2.0), aten("mm", 1.8, tunable_dyn_dims_mask=(1, 0, 0, 0))
        calls = []

        @contextlib.contextmanager
        def fake_mask(**kw):
            calls.append(kw)
            yield

        with mock.patch("torch.cuda.tunable.dynamic_dims_mask", fake_mask, create=True):
            v = run({w: 1.5, a: 1.6})
        self.assertEqual(v["a_steady"], 1.8)
        self.assertTrue(calls and all(c["M"] == 1 for c in calls))

    def test_debug_record(self):
        w, a = tmpl("w", 2.0), aten("mm", 1.8)
        with self.assertLogs(LOGGER, level="DEBUG") as cm:
            run({w: 1.5, a: 1.6})
        (rec,) = [
            r.args for r in cm.records if r.getMessage().startswith("topk rebench:")
        ]
        self.assertEqual((rec["winner_after"], len(rec["retimed"])), ("mm", 1))

    def test_subproc_runs_pair_in_tuning_pool(self):
        w, a = tmpl("w", 0), aten("mm", 0, tunable_dyn_dims_mask=(0, 1, 0, 0))
        jobs = []

        class Pool:
            def run(self, job, timeout):
                jobs.append((job, timeout))
                return [[1.8] * 3, [2.0] * 3]

        with (
            mock.patch.object(ap, "get_tuning_pool", return_value=Pool()),
            mock.patch.object(
                A,
                "get_inputs",
                side_effect=AssertionError("parent must not build inputs"),
            ),
        ):
            v = run(
                {w: 1.5, a: 1.6},
                autotune_in_subproc=True,
                max_autotune_subproc_result_timeout_seconds=7.0,
            )
        ((job, timeout),) = jobs
        self.assertIsInstance(job, ap.RebenchBatchRequest)
        self.assertEqual(job.requests, [a.bmreq, w.bmreq])
        self.assertEqual(
            (job.rounds, job.aten_tunable_mask, timeout), (3, (0, 1, 0, 0), 7.0)
        )
        self.assertEqual((v["a_steady"], v["t_steady"]), (1.8, 2.0))

    def test_subproc_failure_falls_back(self):
        pool = mock.Mock(
            run=mock.Mock(side_effect=RuntimeError("cudaErrorIllegalAddress"))
        )
        with mock.patch.object(ap, "get_tuning_pool", return_value=pool):
            self._assert_fallback(
                "RuntimeError",
                {tmpl("w", 0): 1.5, aten("mm", 0): 1.6},
                autotune_in_subproc=True,
            )

    def test_batch_request_pickles(self):
        meta = ap.TensorMeta(torch.device("cpu"), torch.float32, (4, 4), (4, 1), 0, "x")
        req = ap.ExternKernelCPUBenchmarkRequest(
            "mm", [meta, meta], meta, (), "extern_kernels.mm"
        )
        job = pickle.loads(
            pickle.dumps(ap.RebenchBatchRequest([req, req], 0.1, 2, (1, 0, 0, 0)))
        )
        self.assertEqual((len(job.requests), job.rounds), (2, 2))

    def test_rebench_batch_interleaves(self):
        order = []
        with mock.patch.object(
            ap.benchmarker, "benchmark_gpu", side_effect=lambda fn: fn() or len(order)
        ):
            s = ap.rebench_batch(
                [lambda: order.append("a"), lambda: order.append("t")],
                0.0,
                2,
                lambda: None,
            )
        self.assertEqual(order, ["a", "t"] * 2)
        self.assertEqual(s, [[1, 3], [2, 4]])

    def test_cache_key(self):
        with config.patch(autotune_topk_rebench_margin=0.0):
            off = sa.create_benchmark_cache_key("k", "cuda", False)
        with config.patch(autotune_topk_rebench_margin=0.25):
            on = sa.create_benchmark_cache_key("k", "cuda", False)
        self.assertNotEqual(off, on)
        self.assertIn("topk_rebench=0.25", on)
        self.assertFalse(hasattr(config, "autotune_topk_rebench_k"))

    # ---- verdict storage in the autotune-cache bucket ----
    def _cache(self, d, system="sysA"):
        sel = sa.AlgorithmSelectorCache()
        path = Path(d) / "cache" / system  # stands in for cache/<system hash>
        return sel, mock.patch.object(
            codecache.CacheBase, "get_local_cache_path", return_value=path
        )

    def _bench_and_store(self, sel, choices, v, name="mm", inputs="k", prec="highest"):
        # what PersistentCache.lookup writes on a miss, then the verdict sync
        with mock.patch.object(sa, "fp32_matmul_precision_key", return_value=prec):
            lc = sel.get_local_cache()  # merge like lookup() does
            lc.setdefault(name, {}).setdefault(inputs, {}).setdefault(prec, {}).update(
                {c.hash_key(): 1.0 for c in choices}
            )
            sel.update_local_cache(lc)
            v = dict(v, choices=sa._topk_choices_digest(choices))
            return sel._topk_sync_verdict(
                dict.fromkeys(choices, 1.0), choices, name, inputs, None, v, True
            )

    def _hit(self, sel, choices, name="mm", inputs="k", prec="highest"):
        with mock.patch.object(sa, "fp32_matmul_precision_key", return_value=prec):
            return sel._topk_sync_verdict(
                dict.fromkeys(choices, 1.0), choices, name, inputs, None, None, False
            )

    def test_verdict_reused_on_cache_hit(self):
        w, a = tmpl("w", 2.0), aten("mm", 1.8)
        v = run({w: 1.5, a: 1.6})
        with (
            tempfile.TemporaryDirectory() as d,
            config.patch(autotune_local_cache=True),
        ):
            sel, p = self._cache(d)
            with p:
                self.assertIsInstance(
                    self._bench_and_store(sel, [w, a], v), sa.TopkTimings
                )
                hit = self._hit(sel, [w, a])
                self.assertEqual(hit.topk_verdict["a_steady"], 1.8)
                bucket = sel.get_local_cache()["mm"]["k"]["highest"]
                self.assertIn(sa.TOPK_VERDICT_KEY, bucket)
                # re-benchmark without a trigger removes it
                with mock.patch.object(
                    sa, "fp32_matmul_precision_key", return_value="highest"
                ):
                    sel._topk_sync_verdict(
                        {w: 1.0, a: 1.0}, [w, a], "mm", "k", None, None, True
                    )
                self.assertNotIsInstance(self._hit(sel, [w, a]), sa.TopkTimings)

    def test_verdict_not_reused_across_system_precision_or_choices(self):
        w, a, w2 = tmpl("w", 2.0), aten("mm", 1.8), tmpl("w2", 0)
        v = run({w: 1.5, a: 1.6})
        with (
            tempfile.TemporaryDirectory() as d,
            config.patch(autotune_local_cache=True),
        ):
            sel, p = self._cache(d, "sysA")
            with p:
                self._bench_and_store(sel, [w, a], v)
                self.assertNotIsInstance(
                    self._hit(sel, [w, a], prec="high"), sa.TopkTimings
                )
                self.assertNotIsInstance(
                    self._hit(sel, [w, a, w2]), sa.TopkTimings
                )  # other choice set
            sel_b, p_b = self._cache(d, "sysB")  # another GPU / CUDA / Triton
            with p_b:
                self.assertNotIsInstance(self._hit(sel_b, [w, a]), sa.TopkTimings)

    def test_choice_sets_sharing_a_bucket_keep_their_verdicts(self):
        w, a, w2 = tmpl("w", 2.0), aten("mm", 1.8), tmpl("w2", 0)
        v = run({w: 1.5, a: 1.6})
        with (
            tempfile.TemporaryDirectory() as d,
            config.patch(autotune_local_cache=True),
        ):
            sel, p = self._cache(d)
            with p:
                self._bench_and_store(sel, [w, a], v)
                self._bench_and_store(sel, [w, a, w2], dict(v, a_steady=1.7))
                self.assertEqual(self._hit(sel, [w, a]).topk_verdict["a_steady"], 1.8)
                self.assertEqual(
                    self._hit(sel, [w, a, w2]).topk_verdict["a_steady"], 1.7
                )
                # set B re-benchmarks without a trigger: only B's verdict goes
                with mock.patch.object(
                    sa, "fp32_matmul_precision_key", return_value="highest"
                ):
                    sel._topk_sync_verdict({}, [w, a, w2], "mm", "k", None, None, True)
                self.assertNotIsInstance(self._hit(sel, [w, a, w2]), sa.TopkTimings)
                self.assertEqual(self._hit(sel, [w, a]).topk_verdict["a_steady"], 1.8)

    def test_verdict_through_real_persistent_cache_lookup(self):
        # end to end: the bucket path _topk_bucket_keys derives must be the one lookup() writes
        w, a = tmpl("w", 2.0), aten("mm", 1.8)
        v = run({w: 1.5, a: 1.6})
        for hint in (None, 64):
            with (
                tempfile.TemporaryDirectory() as d,
                config.patch(autotune_local_cache=True),
            ):
                sel, p = self._cache(d)
                with p:
                    t = sel.lookup(
                        [w, a],
                        "mm",
                        "inp",
                        lambda cs: {w: 1.5, a: 1.6},
                        hint_override=hint,
                    )
                    vv = dict(v, choices=sa._topk_choices_digest([w, a]))
                    sel._topk_sync_verdict(t, [w, a], "mm", "inp", hint, vv, True)
                    bucket = next(
                        iter(next(iter(sel.get_local_cache()["mm"].values())).values())
                    )
                    self.assertIn(w.hash_key(), bucket)
                    self.assertIn(
                        sa.TOPK_VERDICT_KEY, bucket
                    )  # same bucket as the timings
                    hit = sel.lookup([w, a], "mm", "inp", None, hint_override=hint)
                    self.assertEqual(hit, {w: 1.5, a: 1.6})
                    out = sel._topk_sync_verdict(
                        hit, [w, a], "mm", "inp", hint, None, False
                    )
                    self.assertIsInstance(out, sa.TopkTimings)
                    self.assertIs(sa.topk_rebench_pick(out)[0], a)

    def test_malformed_verdict_does_not_crash(self):
        w, a = tmpl("w", 2.0), aten("mm", 1.8)
        with (
            tempfile.TemporaryDirectory() as d,
            config.patch(autotune_local_cache=True),
        ):
            sel, p = self._cache(d)
            with (
                p,
                mock.patch.object(
                    sa, "fp32_matmul_precision_key", return_value="highest"
                ),
            ):
                dg = sa._topk_choices_digest([w, a])
                for store in ({dg: {"t_steady": 1}}, {dg: "x"}, ["not", "a", "dict"]):
                    sel.update_local_cache(
                        {"mm": {"k": {"highest": {sa.TOPK_VERDICT_KEY: store}}}}
                    )
                    with self.assertLogs(LOGGER, level="WARNING"):
                        self.assertNotIsInstance(self._hit(sel, [w, a]), sa.TopkTimings)

    def test_threshold_pick_survives(self):
        w, a = tmpl("w", 2.0), aten("mm", 1.8)
        t = {w: 1.5, a: 1.6}
        v = run(t)
        tt = sa.TopkTimings(t, v)
        with config.patch(autotune_topk_rebench_hysteresis=0.05):
            self.assertIs(sa.topk_apply_at_pick(w, tt), a)  # argmin: verdict applies
            fb = aten("fallback", 0)
            tt2 = sa.TopkTimings({**t, fb: 1.7}, v)
            self.assertIs(
                sa.topk_apply_at_pick(fb, tt2), fb
            )  # threshold already chose: untouched
            v_t = dict(v, t_steady=1.0)  # verdict prefers the template
            self.assertIs(
                sa.topk_apply_at_pick(fb, sa.TopkTimings({**t, fb: 1.7}, v_t)), fb
            )

    def test_autotune_log_shows_actual_pick(self):
        w, a = tmpl("w", 2.0), aten("mm", 1.8)
        t = {w: 1.5, a: 1.6}
        v = run(t)
        with config.patch(autotune_topk_rebench_hysteresis=0.05):
            line = sa.topk_autotune_log_line(sa.TopkTimings(t, v))
            self.assertIn("SELECTED (topk rebench): mm over single-shot best w", line)
            self.assertIsNone(
                sa.topk_autotune_log_line(sa.TopkTimings(t, dict(v, t_steady=1.0)))
            )
            self.assertIsNone(sa.topk_autotune_log_line(t))

    def test_multi_template_get_min_choice(self):
        w, a = tmpl("w", 2.0), aten("mm", 1.8)
        t = {w: 1.5, a: 1.6}
        v = run(t)
        stub = mock.Mock(choice_timings=lambda hint_override=None: sa.TopkTimings(t, v))
        with config.patch(autotune_topk_rebench_hysteresis=0.05):
            choice, ms = ir.MultiTemplateBuffer.get_min_choice(stub)
        self.assertIs(choice, a)
        self.assertAlmostEqual(ms, 1.8 * 1.5 / 2.0)
        stub = mock.Mock(choice_timings=lambda hint_override=None: t)
        self.assertEqual(ir.MultiTemplateBuffer.get_min_choice(stub), (w, 1.5))


class _Other(sa.ChoiceCaller):  # stands in for a decomposeK SubgraphChoiceCaller etc.
    pass


def other(name):
    return _mk(_Other, name, 0.0)


@unittest.skipUnless(HAS_GPU and IS_BIG_GPU, "requires a big GPU (Triton mm templates)")
class TestTopkRebenchGPU(TestCase):
    hw_classification = HardwareClassification.CUDA

    def _force_near_tie(self, forced):
        real_autotune = A.autotune

        def near_tie(self_, *args, **kwargs):
            t = real_autotune(self_, *args, **kwargs)
            atens = [
                c
                for c in t
                if isinstance(c, sa.ExternKernelCaller) and math.isfinite(t[c])
            ]
            tmpls = [
                c
                for c in t
                if isinstance(c, sa.TritonTemplateCaller) and math.isfinite(t[c])
            ]
            if atens and tmpls:
                a = min(atens, key=t.__getitem__)
                best = min(tmpls, key=t.__getitem__)
                t = dict(t)
                t[best] = t[a] * 0.99
                forced.append((best.name, dict(t)))
            return t

        return mock.patch.object(A, "autotune", near_tie)

    def _cfg(self, in_subproc=False, **kw):
        return {
            "max_autotune_gemm": True,
            "max_autotune_gemm_backends": "ATEN,TRITON",
            "autotune_in_subproc": in_subproc,
            "autotune_topk_rebench_margin": 0.25,
            "autotune_topk_rebench_warmup_s": 0.05,
            "autotune_topk_rebench_rounds": 2,
            "autotune_local_cache": False,
            "fx_graph_cache": False,
            **kw,
        }

    def _real_run_fns(self, in_subproc):
        forced, seen = [], []
        a = torch.randn(512, 1024, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(1024, 768, device=GPU_TYPE, dtype=torch.bfloat16)
        torch._dynamo.reset()
        sa.clear_feedback_savers()
        sa.add_feedback_saver(lambda timings, *args: seen.append(dict(timings)))
        try:
            with (
                fresh_cache(),
                config.patch(self._cfg(in_subproc, benchmark_template_fusion=False)),
                self._force_near_tie(forced),
            ):
                with self.assertLogs(LOGGER, level="DEBUG") as cm:
                    out = torch.compile(lambda x, y: x @ y, dynamic=False)(a, b)
        finally:
            sa.clear_feedback_savers()
        self.assertTrue(forced)
        rec = [r.args for r in cm.records if r.getMessage().startswith("topk rebench:")]
        self.assertEqual(len(rec), 1)
        ((name, single, steady),) = rec[0]["retimed"]
        self.assertEqual(name, forced[0][0])
        self.assertTrue(
            math.isfinite(steady) and steady > 0 and math.isfinite(rec[0]["aten"][2])
        )
        # feedback savers get the measured single-shot timings, not steady/biased ones
        self.assertTrue(seen)
        fb = {c.name: t for c, t in seen[-1].items()}
        ft = {c.name: v for c, v in forced[0][1].items()}
        self.assertEqual(fb[name], ft[name])
        self.assertEqual(fb[rec[0]["aten"][0]], ft[rec[0]["aten"][0]])
        torch.testing.assert_close(out, a @ b, atol=1e-2, rtol=1e-2)

    def test_real_run_fns(self):
        self._real_run_fns(in_subproc=False)

    def test_real_run_fns_subproc(self):
        def parent_call(self_):
            raise AssertionError("RebenchBatchRequest ran in the parent")

        with mock.patch.object(ap.RebenchBatchRequest, "__call__", parent_call):
            self._real_run_fns(in_subproc=True)

    def _epilogue(self, a_ratio, t_ratio, fused_vs_unfused=None, ms2=0.02):
        """mm + relu with benchmark_template_fusion. The re-timed steady values are
        realistic multiples of the measured single-shot times (a_ratio for ATen,
        t_ratio for the forced near-tie template). With fused_vs_unfused set, the
        fusion benchmarks are fixed: the epilogue (relu) kernel takes ms2 and every
        fused template takes expected_ms1 + ms2 + fused_vs_unfused, where
        expected_ms1 = a_steady * t_single / t_steady is computed here from the
        verdict, so the scheduler's decision flips if it compares against any other ms1."""
        from torch._inductor import scheduler as sched

        forced, picks, fused_bench = [], [], []
        a = torch.randn(512, 1024, device=GPU_TYPE, dtype=torch.bfloat16)
        b = torch.randn(1024, 768, device=GPU_TYPE, dtype=torch.bfloat16)
        torch._dynamo.reset()
        real_pick = sa.topk_rebench_pick

        def fixed(fns, warmup_s, rounds, synchronize):  # order: [aten, template]
            for fn in fns:
                fn()
            name, t = forced[-1]
            a_single = min(
                v for c, v in t.items() if isinstance(c, sa.ExternKernelCaller)
            )
            t_single = next(v for c, v in t.items() if c.name == name)
            return [[a_single * a_ratio] * rounds, [t_single * t_ratio] * rounds]

        def spy(timings):
            r = real_pick(timings)
            if r is not None:
                picks.append((r[0], r[1], dict(timings), timings.topk_verdict))
            return r

        def expected_ms1():
            v = picks[-1][3]
            return v["a_steady"] * v["t_single"] / v["t_steady"]

        def fixed_ms2(self_, nodes):
            return ms2, ""

        def fixed_fused(self_, module, device):
            ms = expected_ms1() + ms2 + fused_vs_unfused
            fused_bench.append(ms)
            return ms, ""

        with contextlib.ExitStack() as st:
            st.enter_context(fresh_cache())
            st.enter_context(config.patch(self._cfg(benchmark_template_fusion=True)))
            st.enter_context(self._force_near_tie(forced))
            st.enter_context(mock.patch.object(ap, "rebench_batch", fixed))
            st.enter_context(mock.patch.object(sa, "topk_rebench_pick", spy))
            if fused_vs_unfused is not None:
                st.enter_context(
                    mock.patch.object(
                        sched.Scheduler, "benchmark_fused_nodes", fixed_ms2
                    )
                )
                st.enter_context(
                    mock.patch.object(
                        sched.Scheduler, "benchmark_codegened_module", fixed_fused
                    )
                )
            out, (code,) = run_and_get_code(
                torch.compile(lambda x, y: torch.relu(x @ y), dynamic=False), a, b
            )
        self.assertTrue(forced)
        self.assertTrue(
            picks
        )  # MultiTemplateBuffer.get_min_choice consulted the verdict
        if fused_vs_unfused is not None:
            self.assertTrue(fused_bench)  # the fused candidate was actually benchmarked
        torch.testing.assert_close(out, torch.relu(a @ b), atol=1e-2, rtol=1e-2)
        return code, picks

    def _assert_aten_picks(self, picks):
        for choice, ms, timings, v in picks:
            self.assertIsInstance(choice, sa.ExternKernelCaller)
            self.assertEqual(choice.hash_key(), v["aten"])
            # ms1 is ATen's steady time translated into the single-shot regime
            self.assertAlmostEqual(
                ms, v["a_steady"] * v["t_single"] / v["t_steady"], places=12
            )

    def test_epilogue_fusion_multi_template(self):
        """mm + relu through MultiTemplateBuffer/benchmark_template_fusion: the
        scheduler compares the fused template against the verdict's translated ms1
        (deterministic: fusion benchmarks are fixed around ms1 + ms2)."""
        self.assertTrue(config.benchmark_template_fusion)
        # ATen verdict: steady ATen 1.15x its single-shot, template 1.30x
        # (template/ATen steady ~1.12 > 1 + hysteresis).
        delta = (
            0.1 * 0.009 * (1.30 - 1.15)
        )  # well under the ms1 gap to the raw a_steady
        code, picks = self._epilogue(1.15, 1.30, fused_vs_unfused=+delta)
        self._assert_aten_picks(picks)
        self.assertIn(
            "extern_kernels.mm", code
        )  # fused slower than ms1 + ms2: unfused ATen
        self.assertNotIn("triton_tem_fused", code)
        code, picks = self._epilogue(1.15, 1.30, fused_vs_unfused=-delta)
        self._assert_aten_picks(picks)
        self.assertNotIn("extern_kernels.mm", code)  # fused faster: fused template
        self.assertIn("triton_tem_fused", code)
        # template verdict: steady ATen 1.30x, template 1.10x -> template, raw single-shot ms1
        code, picks = self._epilogue(1.30, 1.10)
        for choice, ms, timings, v in picks:
            self.assertIsInstance(choice, sa.TritonTemplateCaller)
            self.assertEqual(ms, timings[choice])
        self.assertNotIn("extern_kernels.mm", code)


if __name__ == "__main__":
    run_tests()
