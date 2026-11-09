# Owner(s): ["module: inductor"]

import functools
import os
import subprocess
import sys
import types
import unittest.mock

import torch
import torch._dynamo
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.choices import InductorChoices
from torch._inductor.heuristics.template.triton import (
    _shared_memory_per_block,
    BaseHeuristicSingleton,
    CUDAConfigHeuristic,
    ROCmConfigHeuristic,
    ScatterPreaggConfig,
    ScatterPreaggTarget,
)
from torch._inductor.kernel import scatter_preagg
from torch._inductor.test_case import TestCase
from torch._inductor.utils import run_and_get_code
from torch._inductor.virtualized import V
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    HardwareClassification,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_CUDA_AND_TRITON


def _hot_indices(n, num_rows, device, hot_fraction=0.5, dtype=torch.int64):
    tail = torch.randint(0, num_rows, (n,), device=device, dtype=dtype)
    hot = torch.rand(n, device=device) < hot_fraction
    return torch.where(hot, torch.zeros_like(tail), tail)


_BF16_SCATTER_FNS = {
    "index_put": lambda x, i, v: torch.index_put(x, (i,), v, accumulate=True),
    "index_add": lambda x, i, v: x.index_add(0, i, v),
    "scatter_add": lambda x, i, v: x.scatter_add(0, i[:, None].expand_as(v), v),
    "scatter_reduce": lambda x, i, v: x.scatter_reduce(
        0, i[:, None].expand_as(v), v, "sum", include_self=False
    ),
    "scatter_reduce_add": lambda x, i, v: x.scatter(
        0, i[:, None].expand_as(v), v, reduce="add"
    ),
}


@unittest.skipIf(
    not (TEST_WITH_ROCM and HAS_CUDA_AND_TRITON), "scatter template is ROCm only"
)
class TestScatterPreagg(TestCase):
    hw_classification = HardwareClassification.CUDA

    def setUp(self):
        super().setUp()
        self.enterContext(config.patch(scatter_preagg_template=True))
        # Deterministic mode selects a different lowering; other suites can
        # leave it on.
        self.enterContext(DeterministicGuard(False))
        torch._dynamo.reset()
        torch.manual_seed(0)

    def _check(self, fn, *args, applied=True, atol=1e-3, rtol=1e-4):
        """Compare with eager and check whether the template applied; returns
        the generated code."""
        expect = fn(*[a.clone() for a in args])
        actual, codes = run_and_get_code(
            torch.compile(fn, fullgraph=True), *[a.clone() for a in args]
        )
        self.assertEqual(actual, expect, atol=atol, rtol=rtol)
        self.assertEqual(
            counters["inductor"]["scatter_preagg_applied"] > 0,
            applied,
            msg=f"counters={dict(counters['inductor'])}",
        )
        return "\n".join(codes)

    def test_embedding_dense_backward(self):
        n, v, d = 50_000, 24, 32
        idx = _hot_indices(n, v, GPU_TYPE, dtype=torch.int32)
        x = torch.randn(n, 224, device=GPU_TYPE, dtype=torch.bfloat16)

        def fn(i, a):
            return torch.ops.aten.embedding_dense_backward(
                a[:, 184 : 184 + d], i, v, -1, False
            )

        self._check(fn, idx, x, atol=5e-2, rtol=2e-2)

    def test_index_put_rows(self):
        x = torch.zeros(64, 32, device=GPU_TYPE)
        idx = _hot_indices(40_000, 64, GPU_TYPE)
        v = torch.randn(40_000, 32, device=GPU_TYPE)

        def fn(x, i, v):
            return x.index_put_((i,), v, accumulate=True)

        self._check(fn, x, idx, v)

    def test_multi_index_non_adjacent(self):
        x = torch.zeros(8, 16, 6, device=GPU_TYPE)
        i0 = torch.randint(0, 8, (20_000,), device=GPU_TYPE)
        i1 = torch.randint(0, 6, (20_000,), device=GPU_TYPE)
        v = torch.randn(20_000, 16, device=GPU_TYPE)

        def fn(x, i0, i1, v):
            x[i0, :, i1] += 0
            return torch.ops.aten.index_put(x, [i0, None, i1], v, True)

        self._check(fn, x, i0, i1, v)

    def test_fused_value_producer(self):
        x = torch.zeros(32, 64, device=GPU_TYPE)
        idx = _hot_indices(20_000, 32, GPU_TYPE)
        a = torch.randn(20_000, 64, device=GPU_TYPE)
        b = torch.randn(64, device=GPU_TYPE)

        def fn(x, i, a, b):
            return torch.index_put(x, (i,), (a * b).relu() + 1, accumulate=True)

        self._check(fn, x, idx, a, b)

    def test_index_add_dim1(self):
        x = torch.randn(16, 48, device=GPU_TYPE)
        idx = _hot_indices(30_000, 48, GPU_TYPE)
        src = torch.randn(16, 30_000, device=GPU_TYPE)

        def fn(x, i, s):
            return x.index_add(1, i, s)

        self._check(fn, x, idx, src)

    @config.patch(scatter_preagg_template_min_saved=-1.0)
    def test_wide_slices(self):
        for dups in [False, True]:
            for dtype in [torch.float32, torch.bfloat16]:
                with self.subTest(dups=dups, dtype=dtype):
                    torch._dynamo.reset()
                    counters.clear()
                    # Wide slices of blocks with and without duplicates.
                    cfg = ScatterPreaggConfig(
                        block_t=64, block_d=64, d_iters=4, num_stages=1, num_warps=4
                    )
                    x = torch.zeros(1024, 256, device=GPU_TYPE, dtype=dtype)
                    tok = torch.arange(65_536, device=GPU_TYPE)
                    idx = (tok // 2 if dups else tok) % 1024
                    v = torch.randn(65_536, 256, device=GPU_TYPE, dtype=dtype)

                    def fn(x, i, v):
                        return x.index_put_((i,), v, accumulate=True)

                    ref = fn(x.double(), idx, v.double())
                    with unittest.mock.patch.object(
                        InductorChoices,
                        "get_scatter_preagg_configs",
                        return_value=[cfg],
                    ):
                        out, (code,) = run_and_get_code(
                            torch.compile(fn, fullgraph=True), x.clone(), idx, v
                        )
                    scale = 1e-5 if dtype == torch.float32 else 1e-2
                    self.assertEqual(
                        out.double(), ref, atol=ref.abs().max().item() * scale, rtol=0
                    )
                    self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
                    self.assertIn("BLOCK_D : tl.constexpr = 64", code)

    def test_static_inner_shifted(self):
        for deterministic in [False, True]:
            with self.subTest(deterministic=deterministic):
                torch._dynamo.reset()
                counters.clear()
                # A static row size no inner block divides shifts its last block back
                # over part of the one before it, whose lanes it masks.
                cfg = ScatterPreaggConfig(
                    block_t=64, block_d=64, d_iters=4, num_stages=1, num_warps=4
                )
                x = torch.zeros(100, 1040, device=GPU_TYPE)
                idx = torch.randint(0, 100, (20_000,), device=GPU_TYPE)
                v = torch.randn(20_000, 1040, device=GPU_TYPE)

                def fn(x, i, v):
                    return x.index_add(0, i, v)

                ref = fn(x.double(), idx, v.double())
                self.enterContext(DeterministicGuard(deterministic))
                with unittest.mock.patch.object(
                    InductorChoices, "get_scatter_preagg_configs", return_value=[cfg]
                ):
                    out, codes = run_and_get_code(
                        torch.compile(fn, fullgraph=True), x.clone(), idx, v
                    )
                self.assertIn("INNER_SHIFT : tl.constexpr = True", "\n".join(codes))
                self.assertEqual(out.double(), ref, atol=1e-3, rtol=1e-4)
                self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)

    def test_exact_histogram(self):
        for rows in [512, 1 << 16]:
            with self.subTest(rows=rows):
                torch._dynamo.reset()
                counters.clear()
                # Destinations that fit the histogram get a bin each, rather than
                # hashed bins whose count linear counting would overestimate. The bins
                # scale with the block, which the default config sizes by CU count.
                cfg = ScatterPreaggConfig(
                    block_t=256, block_d=16, d_iters=1, num_stages=1, num_warps=2
                )
                x = torch.zeros(rows, 32, device=GPU_TYPE)
                idx = torch.randint(0, rows, (65_536,), device=GPU_TYPE)
                v = torch.randn(65_536, 32, device=GPU_TYPE)

                def fn(x, i, v):
                    return x.index_put_((i,), v, accumulate=True)

                ref = fn(x.double(), idx, v.double())
                with unittest.mock.patch.object(
                    InductorChoices, "get_scatter_preagg_configs", return_value=[cfg]
                ):
                    out, (code,) = run_and_get_code(
                        torch.compile(fn, fullgraph=True), x.clone(), idx, v
                    )
                self.assertEqual(out.double(), ref, atol=1e-3, rtol=1e-4)
                self.assertIn("ADAPTIVE : tl.constexpr = True", code)
                self.assertEqual(
                    "HIST_EXACT : tl.constexpr = True" in code, rows == 512
                )

    def test_gathered_values_config(self):
        # A message-passing scatter of gathered rows, on a config whose plain
        # path once miscompiled with buffer ops into a memory fault.
        cfg = ScatterPreaggConfig(
            block_t=512, block_d=16, d_iters=4, num_stages=1, num_warps=4
        )
        x = torch.zeros(65_536, 64, device=GPU_TYPE)
        h = torch.randn(65_536, 64, device=GPU_TYPE)
        src = torch.randint(0, 65_536, (524_288,), device=GPU_TYPE)
        dst = torch.randint(0, 65_536, (524_288,), device=GPU_TYPE)
        w = torch.rand(524_288, device=GPU_TYPE)

        def fn(x, dst, h, src, w):
            return x.index_add(0, dst, h[src] * w[:, None])

        with unittest.mock.patch.object(
            InductorChoices, "get_scatter_preagg_configs", return_value=[cfg]
        ):
            self._check(fn, x, dst, h, src, w, atol=1e-3)

    def test_scatter_add_expanded_index(self):
        x = torch.zeros(16, 32, device=GPU_TYPE)
        rows = _hot_indices(20_000, 16, GPU_TYPE)
        src = torch.randn(20_000, 32, device=GPU_TYPE)

        def fn(x, r, s):
            return x.scatter_add(0, r[:, None].expand(-1, 32), s)

        self._check(fn, x, rows, src)

    def test_scatter_reduce_sum(self):
        for include_self in [False, True]:
            with self.subTest(include_self=include_self):
                torch._dynamo.reset()
                counters.clear()
                x = torch.randn(16, 8, device=GPU_TYPE)
                idx = torch.randint(0, 16, (20_000, 8), device=GPU_TYPE)
                src = torch.randn(20_000, 8, device=GPU_TYPE)

                def fn(x, i, s):
                    return x.scatter_reduce(0, i, s, "sum", include_self=include_self)

                self._check(fn, x, idx, src)

    def test_bf16(self):
        for op in list(_BF16_SCATTER_FNS):
            with self.subTest(op=op):
                torch._dynamo.reset()
                counters.clear()
                fn = _BF16_SCATTER_FNS[op]
                x = torch.randn(24, 64, device=GPU_TYPE).bfloat16()
                idx = _hot_indices(50_000, 24, GPU_TYPE)
                v = torch.randn(50_000, 64, device=GPU_TYPE).bfloat16()

                # Accumulated in an fp32 copy, so each destination is rounded once.
                ref = fn(x.float(), idx, v.float())
                out = torch.compile(fn, fullgraph=True)(x.clone(), idx, v)
                self.assertEqual(out.dtype, torch.bfloat16)
                self.assertEqual(
                    out.float(), ref, atol=ref.abs().max().item() * 1e-2, rtol=0
                )
                self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
                self.assertEqual(counters["inductor"]["scatter_preagg_fp32_copy"], 1)

    def test_bf16_declined_falls_back(self):
        for reason in ["too_small", "fp32_copy_too_large"]:
            for op in ["index_put", "scatter_reduce"]:
                with self.subTest(reason=reason, op=op):
                    torch._dynamo.reset()
                    counters.clear()
                    # Where bf16 atomics need the eager fallback, a scatter the template
                    # declines for its size must still go to it.
                    fn = _BF16_SCATTER_FNS[op]
                    rows, n = (1000, 400) if reason == "too_small" else (100_000, 8192)
                    x = torch.randn(rows, 8, device=GPU_TYPE).bfloat16()
                    idx = torch.randint(0, rows, (n,), device=GPU_TYPE)
                    v = torch.randn(n, 8, device=GPU_TYPE).bfloat16()

                    code = self._check(
                        fn, x, idx, v, applied=False, atol=2e-2, rtol=2e-2
                    )
                    self.assertGreater(
                        counters["inductor"][f"scatter_preagg_skipped_{reason}"], 0
                    )
                    self.assertNotIn("tl.atomic_add(", code)
                    # Nothing set up for the template is left behind.
                    torch._dynamo.reset()
                    with config.patch(scatter_preagg_template=False):
                        _, (regular,) = run_and_get_code(
                            torch.compile(fn, fullgraph=True), x.clone(), idx, v
                        )
                    self.assertEqual(
                        code.count("@triton.jit"), regular.count("@triton.jit")
                    )

    @config.patch(scatter_preagg_template_min_numel=0)
    def test_fp16_copy_too_large_falls_back(self):
        # fp16 into few destinations stages in fp32 too, which declines_by_size
        # does not predict: the template checks again once it has the plan.
        x = torch.randn(64, 8, device=GPU_TYPE).half()
        idx = torch.randint(0, 64, (20,), device=GPU_TYPE)
        v = torch.randn(20, 8, device=GPU_TYPE).half()

        def fn(x, i, v):
            return x.index_add(0, i, v)

        self._check(fn, x, idx, v, applied=False, atol=1e-2, rtol=1e-2)
        self.assertEqual(
            counters["inductor"]["scatter_preagg_skipped_fp32_copy_too_large"], 1
        )

    def test_declined_emits_regular_lowering(self):
        for mode in [
            "deterministic",
            "inductor_deterministic",
            "bf16",
            "int8",
            "int16",
        ]:
            for op in ["index_put", "scatter_add", "scatter_reduce"]:
                with self.subTest(mode=mode, op=op):
                    torch._dynamo.reset()
                    counters.clear()
                    # A scatter the template declines must still be lowered: to the eager
                    # fallback where it bypassed that (deterministic algorithms, bf16, and
                    # 8/16-bit ints, which have no atomics), otherwise to atomics
                    # (TORCHINDUCTOR_DETERMINISTIC alone).
                    fn = _BF16_SCATTER_FNS[op]
                    dtype = {
                        "bf16": torch.bfloat16,
                        "int8": torch.int8,
                        "int16": torch.int16,
                    }.get(mode, torch.float32)
                    # Few writes per destination keep the eager bf16 fallback accurate.
                    make = torch.randn if dtype.is_floating_point else torch.ones
                    x = make(4096, 8, device=GPU_TYPE).to(dtype)
                    idx = torch.randint(0, 4096, (20_000,), device=GPU_TYPE)
                    v = make(20_000, 8, device=GPU_TYPE).to(dtype)
                    ref = fn(x.double(), idx, v.double())
                    with (
                        unittest.mock.patch.object(
                            scatter_preagg, "_try_template", return_value=False
                        ),
                        config.patch(deterministic=mode == "inductor_deterministic"),
                        DeterministicGuard(mode == "deterministic"),
                    ):
                        out, codes = run_and_get_code(
                            torch.compile(fn, fullgraph=True), x.clone(), idx, v
                        )
                    self.assertEqual(
                        out.double(), ref, atol=ref.abs().max().item() * 2e-2, rtol=0
                    )
                    code = "\n".join(codes)
                    if mode == "inductor_deterministic":
                        self.assertIn("tl.atomic_add(", code)
                    else:
                        self.assertNotIn("tl.atomic_add(", code)
                        self.assertIn(
                            "aten.index_put_"
                            if op == "index_put"
                            else "aten.scatter_reduce_.two(",
                            code,
                        )

    def test_skip_overwriting_scatter(self):
        for deterministic in [False, True]:
            for op in ["scatter", "index_put"]:
                with self.subTest(deterministic=deterministic, op=op):
                    torch._dynamo.reset()
                    counters.clear()
                    # Only accumulating scatters reach the template; deterministic mode
                    # also skips its size gate.
                    x = torch.randn(64, 32, device=GPU_TYPE)
                    idx = torch.randint(0, 64, (20_000, 32), device=GPU_TYPE)
                    # Equal values make the duplicate writes' order irrelevant.
                    v = torch.ones(20_000, 32, device=GPU_TYPE)

                    def fn(x, i, v):
                        if op == "scatter":
                            return x.scatter(0, i, v)
                        return torch.index_put(x, (i[:, 0],), v, accumulate=False)

                    with DeterministicGuard(deterministic):
                        self._check(fn, x, idx, v, applied=False)

    @config.patch(max_autotune=True)
    def test_autotune_index_inputs(self):
        # Autotuning randomizes index inputs only within the size each one
        # indexes.
        x = torch.zeros(4096, 64, device=GPU_TYPE)
        v = torch.randn(8192, 64, device=GPU_TYPE)
        idx = torch.randint(0, 1000, (8192,), device=GPU_TYPE)
        remap = torch.randint(0, 4096, (1000,), device=GPU_TYPE)
        generated = []
        index_gen_fns = scatter_preagg._index_gen_fns

        def record(fn, node):
            t = fn(node)
            generated.append(t.clone())
            return t

        def spy(*args):
            fns = index_gen_fns(*args)
            return {i: functools.partial(record, fn) for i, fn in fns.items()}

        with unittest.mock.patch.object(scatter_preagg, "_index_gen_fns", spy):
            self._check(lambda x, i, r, v: x.index_add(0, r[i], v), x, idx, remap, v)
        bounds = {idx.numel(): remap.numel(), remap.numel(): x.shape[0]}
        self.assertEqual({t.numel() for t in generated}, bounds.keys())
        for t in generated:
            self.assertGreaterEqual(t.min().item(), 0)
            self.assertLess(t.max().item(), bounds[t.numel()])
            self.assertGreater(t.unique().numel(), 100)

    @config.patch(max_autotune=True)
    def test_autotune_shared_index_bounds(self):
        for shared in ["key_and_value", "two_values"]:
            with self.subTest(shared=shared):
                torch._dynamo.reset()
                counters.clear()
                # An index addressing two sizes is randomized within the smaller,
                # whether it is the key and gathers values, or gathers two values.
                x = torch.zeros(100, 32, device=GPU_TYPE)
                idx = torch.randint(0, 100, (8192,), device=GPU_TYPE)
                key = torch.randint(0, 100, (8192,), device=GPU_TYPE)
                small = torch.randn(100, 32, device=GPU_TYPE)
                table = torch.randn(4096, 32, device=GPU_TYPE)
                generated = []
                index_gen_fns = scatter_preagg._index_gen_fns

                def record(fn, node):
                    t = fn(node)
                    generated.append(t.clone())
                    # Benchmark in bounds either way, so a wrong bound fails below
                    # rather than aborting the process.
                    return t % 100

                def spy(*args):
                    fns = index_gen_fns(*args)
                    return {i: functools.partial(record, fn) for i, fn in fns.items()}

                def fn(x, i, k, a, b):
                    if shared == "key_and_value":
                        return x.index_add(0, i, b[i])
                    return x.index_add(0, k, a[i] + b[i])

                with unittest.mock.patch.object(scatter_preagg, "_index_gen_fns", spy):
                    self._check(fn, x, idx, key, small, table)
                self.assertTrue(generated)
                for t in generated:
                    self.assertLess(t.max().item(), 100)

    @config.patch(max_autotune=True)
    def test_autotune_expanded_index(self):
        # An expanded index input overlaps itself in memory (cloning it, as
        # _check does, would not).
        x = torch.zeros(512, 64, device=GPU_TYPE)
        idx = torch.randint(0, 512, (16384,), device=GPU_TYPE)[:, None].expand(-1, 64)
        v = torch.randn(16384, 64, device=GPU_TYPE)

        def fn(x, i, v):
            return x.scatter_add(0, i, v)

        self.assertEqual(
            torch.compile(fn, fullgraph=True)(x, idx, v),
            fn(x, idx, v),
            atol=1e-3,
            rtol=1e-4,
        )
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)

    @config.patch(max_autotune=True)
    def test_autotune_computed_index(self):
        for index in ["offset", "float", "value_gather", "key_gather", "unchecked_key"]:
            with self.subTest(index=index):
                torch._dynamo.reset()
                counters.clear()
                # Benchmark inputs (zeros, or random floats) put an index computed from
                # them out of bounds. Such scatters take the default config
                # unbenchmarked, unless the index only addresses unchecked writes,
                # which the template masks.
                checked = index in ("offset", "float")
                x = torch.zeros(100, 32, device=GPU_TYPE)
                v = torch.randn(20_000, 32, device=GPU_TYPE)
                idx = torch.randint(
                    int(index == "offset"), 100, (20_000,), device=GPU_TYPE
                )
                if index == "float":
                    u = torch.randint(0, 100, (20_000,), device=GPU_TYPE).float()
                else:
                    u = torch.rand(20_000, device=GPU_TYPE)
                table = torch.randn(1000, 32, device=GPU_TYPE)
                remap = torch.randint(0, 100, (1000,), device=GPU_TYPE)

                def fn(x, idx, u, v, table, remap):
                    if index == "offset":
                        i = idx - 1
                    elif index == "float":
                        i = u.long()
                    elif index == "key_gather":
                        i = remap[(u * 1000).long()]
                    elif index == "unchecked_key":
                        i = (u * 100).long()
                    else:
                        i = idx
                        v = table[(u * 1000).long()]
                    return x.scatter_add(0, i[:, None].expand(-1, 32), v)

                # Benchmarking would hit the bounds assert or read out of bounds, and
                # abort the process, so only record whether max_autotune was requested.
                requested = []
                get_configs = InductorChoices.get_scatter_preagg_configs

                def configs(self, *args, max_autotune, **kwargs):
                    requested.append(max_autotune)
                    safe = max_autotune and index == "unchecked_key"
                    return get_configs(self, *args, max_autotune=safe, **kwargs)

                with (
                    unittest.mock.patch.object(
                        InductorChoices, "get_scatter_preagg_configs", configs
                    ),
                    config.patch(assert_indirect_indexing=checked),
                ):
                    self._check(fn, x, idx, u, v, table, remap)
                self.assertEqual(requested, [index == "unchecked_key"])

    def test_dynamic_tokens_no_recompile(self):
        for max_autotune in [False, True]:
            with self.subTest(max_autotune=max_autotune):
                torch._dynamo.reset()
                counters.clear()

                # Neither the token count in a value stride nor the guards of
                # unselected autotuning configs may specialize the graph.
                def fn(x, i, v_t):
                    return x.index_add(0, i, v_t.t())

                compiled = torch.compile(fn, fullgraph=True, dynamic=True)
                with config.patch(
                    max_autotune=max_autotune, scatter_preagg_template_min_numel=0
                ):
                    # Down to fewer tokens than a block.
                    for n in (100_000, 8208, 8193, 300, 100):
                        x = torch.zeros(1024, 64, device=GPU_TYPE)
                        idx = torch.randint(0, 1024, (n,), device=GPU_TYPE)
                        v_t = torch.randn(64, n, device=GPU_TYPE)
                        expect = fn(x.clone(), idx, v_t)
                        self.assertEqual(
                            compiled(x, idx, v_t), expect, atol=1e-3, rtol=1e-3
                        )
                self.assertEqual(counters["stats"]["unique_graphs"], 1)
                self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)

    @config.patch(cpp_wrapper=True)
    def test_cpp_wrapper(self):
        for deterministic in (False, True):
            with self.subTest(deterministic=deterministic):
                torch._dynamo.reset()
                counters.clear()
                x = torch.zeros(24, 32, device=GPU_TYPE)
                idx = _hot_indices(20_000, 24, GPU_TYPE)
                v = torch.randn(20_000, 32, device=GPU_TYPE)

                def fn(x, i, v):
                    return torch.index_put(x, (i,), v, accumulate=True)

                with DeterministicGuard(deterministic):
                    self._check(fn, x, idx, v)
                self.assertEqual(
                    counters["inductor"]["scatter_preagg_sorted"], int(deterministic)
                )

    def test_aoti(self):
        for deterministic in [False, True]:
            with self.subTest(deterministic=deterministic):
                torch._dynamo.reset()
                counters.clear()

                class Model(torch.nn.Module):
                    def forward(self, x, i, v):
                        return torch.index_put(x, (i,), v, accumulate=True)

                x = torch.zeros(24, 32, device=GPU_TYPE)
                idx = _hot_indices(20_000, 24, GPU_TYPE)
                v = torch.randn(20_000, 32, device=GPU_TYPE)
                with DeterministicGuard(deterministic):
                    path = torch._inductor.aoti_compile_and_package(
                        torch.export.export(Model(), (x, idx, v))
                    )
                out = torch._inductor.aoti_load_package(path)(x, idx, v)
                self.assertEqual(out, Model()(x, idx, v), atol=1e-3, rtol=1e-4)
                self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
                self.assertEqual(
                    counters["inductor"]["scatter_preagg_sorted"], int(deterministic)
                )

    def test_aoti_dynamic_rows(self):
        # AOTI never checks guards, so a model exported at 1000 rows must not
        # pick int32 sort keys from that hint: they overflow past 2**23 rows.
        class Model(torch.nn.Module):
            def forward(self, x, i, v):
                return torch.index_put(x, (i,), v, accumulate=True)

        x = torch.zeros(1000, 64, device=GPU_TYPE)
        idx = torch.randint(0, 1000, (50_000,), device=GPU_TYPE)
        v = torch.randn(50_000, 64, device=GPU_TYPE)
        rows, inner, tokens = (
            torch.export.Dim(name, max=1 << 24) for name in ("rows", "inner", "tokens")
        )
        dynamic_shapes = ({0: rows, 1: inner}, {0: tokens}, {0: tokens, 1: inner})
        # 256-token blocks put the int32 limit below the rows tested; the
        # default config's block depends on the CU count.
        cfg = ScatterPreaggConfig(
            block_t=256, block_d=16, d_iters=1, num_stages=1, num_warps=2
        )
        with unittest.mock.patch.object(
            InductorChoices, "get_scatter_preagg_configs", return_value=[cfg]
        ):
            path = torch._inductor.aoti_compile_and_package(
                torch.export.export(Model(), (x, idx, v), dynamic_shapes=dynamic_shapes)
            )
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
        n = 9_000_000
        x = torch.zeros(n, 2, device=GPU_TYPE)
        idx = n - 1 - torch.randint(0, 8, (50_000,), device=GPU_TYPE)
        v = torch.randn(50_000, 2, device=GPU_TYPE)
        out = torch._inductor.aoti_load_package(path)(x, idx, v)
        self.assertEqual(out, Model()(x, idx, v), atol=1e-3, rtol=1e-4)

    def test_dtypes(self):
        for dtype in [
            torch.float16,
            torch.bfloat16,
            torch.float64,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
        ]:
            with self.subTest(dtype=dtype):
                torch._dynamo.reset()
                counters.clear()
                x = torch.zeros(20, 16, device=GPU_TYPE, dtype=dtype)
                idx = _hot_indices(20_000, 20, GPU_TYPE)
                if dtype.is_floating_point:
                    v = torch.randn(20_000, 16, device=GPU_TYPE).to(dtype)
                else:
                    # Small integer outputs wrap around, as in eager.
                    low = 0 if dtype == torch.uint8 else -5
                    v = torch.randint(
                        low, 5, (20_000, 16), device=GPU_TYPE, dtype=dtype
                    )

                def fn(x, i, v):
                    return torch.index_put(x, (i,), v, accumulate=True)

                if dtype in (torch.int16, torch.int8, torch.uint8):
                    # No Triton atomics for these; plain Inductor fails to compile them.
                    self._check(fn, x, idx, v, atol=0, rtol=0)
                    self.assertEqual(counters["inductor"]["scatter_preagg_sorted"], 1)
                    continue

                if dtype in (torch.float16, torch.bfloat16):
                    # Few destinations, so fp16 is also accumulated in an fp32 copy and
                    # each destination is rounded once.
                    ref = fn(x.float(), idx, v.float())
                    out = torch.compile(fn, fullgraph=True)(x.clone(), idx, v)
                    self.assertEqual(
                        out.float(), ref, atol=ref.abs().max().item() * 1e-2, rtol=0
                    )
                    self.assertEqual(
                        counters["inductor"]["scatter_preagg_fp32_copy"], 1
                    )
                    continue
                self._check(fn, x, idx, v, atol=1e-6, rtol=1e-6)

    @config.patch(scatter_preagg_template_min_numel=0)
    def test_token_tails(self):
        for n in [1000, 4096, 4097, 50001]:
            with self.subTest(n=n):
                torch._dynamo.reset()
                counters.clear()
                x = torch.zeros(10, 32, device=GPU_TYPE)
                idx = torch.randint(0, 10, (n,), device=GPU_TYPE)
                v = torch.randn(n, 32, device=GPU_TYPE)

                def fn(x, i, v):
                    return torch.index_put(x, (i,), v, accumulate=True)

                self._check(fn, x, idx, v)

    def test_inner_sizes(self):
        for d in [1, 48, 37, 300]:
            with self.subTest(d=d):
                torch._dynamo.reset()
                counters.clear()
                x = torch.zeros(12, d, device=GPU_TYPE)
                idx = _hot_indices(30_000, 12, GPU_TYPE)
                v = torch.randn(30_000, d, device=GPU_TYPE)

                def fn(x, i, v):
                    return torch.index_put(x, (i,), v, accumulate=True)

                self._check(fn, x, idx, v)

    @config.patch(max_autotune=True)
    def test_max_autotune(self):
        # Several configs are benchmarked on example inputs; the in-place
        # output must only be accumulated into once, by the chosen kernel.
        x = torch.randn(24, 64, device=GPU_TYPE)
        idx = _hot_indices(50_000, 24, GPU_TYPE)
        v = torch.randn(50_000, 64, device=GPU_TYPE)

        def fn(x, i, v):
            return x.index_put_((i,), v, accumulate=True)

        self._check(fn, x, idx, v, atol=2e-3, rtol=1e-4)

    def _check_fp16(self, fn, x, idx, v, matrix=True):
        ref = fn(x.float(), idx, v.float())
        out, (code,) = run_and_get_code(
            torch.compile(fn, fullgraph=True), x.clone(), idx, v
        )
        out = out.float()
        self.assertEqual(out.isnan(), ref.isnan())
        self.assertEqual(out.isinf(), ref.isinf())
        finite = ref.isfinite()
        scale = ref[finite].abs().max().item()
        self.assertEqual(out[finite], ref[finite], atol=scale * 3e-2, rtol=0)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
        self.assertEqual(counters["inductor"]["scatter_preagg_fp32_copy"], 0)
        self.assertEqual("tl.dot(" in code, matrix)
        return code

    def test_fp16_matrix_preagg(self):
        for rows, d, n in [
            (20, 64, 50000),
            (4096, 128, 50000),
            (300, 100, 30000),
            (64, 320, 20000),
        ]:
            with self.subTest(rows=rows, d=d, n=n):
                torch._dynamo.reset()
                counters.clear()

                def fn(x, i, v):
                    return torch.index_put(x, (i,), v, accumulate=True)

                x = torch.randn(rows, d, device=GPU_TYPE).half()
                idx = _hot_indices(n, rows, GPU_TYPE)
                v = torch.randn(n, d, device=GPU_TYPE).half()
                self._check_fp16(fn, x, idx, v)

    def test_fp16_matrix_preagg_non_finite(self):
        for rows in [20, 4096]:
            with self.subTest(rows=rows):
                torch._dynamo.reset()
                counters.clear()

                # A block with inf or nan values must not spread nan (0 * inf) into
                # the other destinations of its product.
                def fn(x, i, v):
                    return torch.index_put(x, (i,), v, accumulate=True)

                x = torch.zeros(rows, 64, device=GPU_TYPE).half()
                idx = _hot_indices(50_000, rows, GPU_TYPE)
                v = torch.randn(50_000, 64, device=GPU_TYPE).half()
                v[torch.nonzero(idx == 3)[0], 5] = float("inf")
                v[torch.nonzero(idx == 7)[0], 9] = float("nan")
                self._check_fp16(fn, x, idx, v)

    def test_adaptive_forced_path(self):
        for min_saved, hot_fraction in [(-1.0, 0.5), (2.0, 0.0)]:
            with self.subTest(min_saved=min_saved, hot_fraction=hot_fraction):
                torch._dynamo.reset()
                counters.clear()

                # -1 always sorts; 2 sorts only blocks with a hot destination, of which
                # uniform keys have none. This covers both paths of each chunk.
                def fn(x, i, v):
                    return torch.index_put(x, (i,), v, accumulate=True)

                x = torch.zeros(4096, 64, device=GPU_TYPE)
                idx = _hot_indices(50_000, 4096, GPU_TYPE, hot_fraction=hot_fraction)
                v = torch.randn(50_000, 64, device=GPU_TYPE)
                with config.patch(scatter_preagg_template_min_saved=min_saved):
                    code = self._check(fn, x, idx, v, atol=1e-2, rtol=1e-3)
                self.assertIn("ADAPTIVE : tl.constexpr = True", code)
                self.assertIn(f"SORT_SAVED : tl.constexpr = {min_saved}", code)

    def test_adaptive_runs(self):
        for d in [1, 32]:
            with self.subTest(d=d):
                torch._dynamo.reset()
                counters.clear()
                # Runs of a few recurring keys, which the sort must not merge across
                # the runs between them, and a ragged last block whose shared tokens
                # must count once.
                idx = torch.randint(0, 4, (8192,), device=GPU_TYPE).repeat_interleave(8)
                idx = idx[:-37]
                shape = (4096, d) if d > 1 else (4096,)
                x = torch.zeros(shape, device=GPU_TYPE)
                v = torch.randn(idx.numel(), *shape[1:], device=GPU_TYPE)

                def fn(x, i, v):
                    return torch.index_put(x, (i,), v, accumulate=True)

                code = self._check(fn, x, idx, v, atol=1e-2, rtol=1e-3)
                # Single columns sort at half the saving.
                min_saved = config.scatter_preagg_template_min_saved / (
                    2 if d == 1 else 1
                )
                self.assertIn(f"SORT_SAVED : tl.constexpr = {min_saved}", code)

    def test_dynamic_shapes(self):
        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        compiled = torch.compile(fn, fullgraph=True)
        # Symbolic token and key counts; covers fewer tokens than a block and
        # an inner size no tile divides.
        for n, rows in [(100, 3), (20_000, 16), (30_001, 24)]:
            x = torch.zeros(rows, 72, device=GPU_TYPE)
            idx = torch.randint(0, rows, (n,), device=GPU_TYPE)
            v = torch.randn(n, 72, device=GPU_TYPE)
            expect = fn(x.clone(), idx, v)
            for t in (x, idx, v):
                torch._dynamo.mark_dynamic(t, 0)
            self.assertEqual(compiled(x, idx, v), expect, atol=1e-3, rtol=1e-3)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
        self.assertEqual(counters["stats"]["unique_graphs"], 1)

    def test_dynamic_inner_shifted(self):
        for dim in ["inner", "batch", "deterministic"]:
            with self.subTest(dim=dim):
                torch._dynamo.reset()
                counters.clear()

                # Aligned row sizes no tile divides shift their last inner block back
                # rather than recompiling per tile, including a batch dimension the
                # destination does not index, which is part of the inner size, and
                # in the sorted lowering.
                def fn(x, i, v):
                    return x.index_add(1 if dim == "batch" else 0, i, v)

                compiled = torch.compile(fn, fullgraph=True, dynamic=True)
                self.enterContext(DeterministicGuard(dim == "deterministic"))
                for size in (
                    (12, 8, 6, 10) if dim == "batch" else (2048, 1040, 1536, 1296)
                ):
                    if dim == "batch":
                        x = torch.zeros(size, 100, 64, device=GPU_TYPE)
                        v = torch.randn(size, 20_000, 64, device=GPU_TYPE)
                    else:
                        x = torch.zeros(100, size, device=GPU_TYPE)
                        v = torch.randn(20_000, size, device=GPU_TYPE)
                    idx = _hot_indices(20_000, 100, GPU_TYPE)
                    expect = fn(x.clone(), idx, v)
                    if size in (12, 2048):
                        out, (code,) = run_and_get_code(compiled, x, idx, v)
                        self.assertIn("INNER_SHIFT : tl.constexpr = True", code)
                    else:
                        out = compiled(x, idx, v)
                    self.assertEqual(out, expect, atol=1e-3, rtol=1e-3)
                self.assertEqual(counters["stats"]["unique_graphs"], 1)

    def test_deterministic_dynamic(self):
        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        with DeterministicGuard(True):
            # Token counts on both sides of the sort's int16 index range.
            for n, rows, d in [
                (1000, 5, 40),
                (40_000, 24, 72),
                (30_001, 24, 72),
                (20_000, 50, 3),
            ]:
                x = torch.zeros(rows, d, device=GPU_TYPE)
                idx = _hot_indices(n, rows, GPU_TYPE)
                v = torch.randn(n, d, device=GPU_TYPE)
                ref = fn(x.double(), idx, v.double())
                out1 = compiled(x.clone(), idx, v)
                out2 = compiled(x.clone(), idx, v)
                self.assertEqual(out1.double(), ref, atol=1e-4, rtol=1e-5)
                self.assertTrue(torch.equal(out1, out2))
        self.assertEqual(counters["inductor"]["scatter_preagg_sorted"], 1)
        self.assertEqual(counters["stats"]["unique_graphs"], 1)

    def test_materialized_keys_and_values(self):
        for deterministic in (False, True):
            with self.subTest(case="keys_read_output", deterministic=deterministic):
                torch._dynamo.reset()
                counters.clear()
                x = torch.randint(0, 16, (16, 32), device=GPU_TYPE).float()
                v = torch.randn(16 * 64, 32, device=GPU_TYPE)

                def keys_read_output(x, v):
                    i = x[:, 0].long().clamp(0, 15).repeat(64)
                    return x.index_put_((i,), v, accumulate=True)

                with DeterministicGuard(deterministic):
                    self._check(keys_read_output, x, v)
                self.assertEqual(
                    counters["inductor"]["scatter_preagg_materialized_keys"], 1
                )

        torch._dynamo.reset()
        counters.clear()
        x = torch.zeros(16, 32, device=GPU_TYPE)
        idx = _hot_indices(20_000, 16, GPU_TYPE)
        values = torch.randn(20_016, 32, device=GPU_TYPE)

        def unsupported_key(x, i, v):
            return torch.index_put(
                x, (torch.nn.functional.pad(i, (8, 8)),), v, accumulate=True
            )

        self._check(unsupported_key, x, idx, values)
        self.assertEqual(counters["inductor"]["scatter_preagg_materialized_keys"], 1)

        torch._dynamo.reset()
        counters.clear()
        x = torch.zeros(16, 40, device=GPU_TYPE)
        idx = torch.randint(0, 16, (20_000,), device=GPU_TYPE)
        values = torch.randn(20_000, 32, device=GPU_TYPE)

        def masked_value(x, i, v):
            return torch.index_put(
                x, (i,), torch.nn.functional.pad(v, (4, 4)), accumulate=True
            )

        self._check(masked_value, x, idx, values)
        self.assertEqual(counters["inductor"]["scatter_preagg_materialized_values"], 1)

    def test_output_view(self):
        def fn(i, v):
            base = v.new_zeros(64, 16)
            base[32:].t().index_add_(0, i, v)
            return base

        idx = _hot_indices(20_000, 16, GPU_TYPE)
        v = torch.randn(20_000, 32, device=GPU_TYPE)
        self._check(fn, idx, v, rtol=1e-3)

    def test_output_copies(self):
        for dtype in [torch.float32, torch.bfloat16]:
            with self.subTest(dtype=dtype):
                torch._dynamo.reset()
                counters.clear()
                # Many writes per element spread over copies of the output; rows no
                # token writes keep -0.0.
                n = 1 << 18
                x = torch.full((64, 32), -0.0, device=GPU_TYPE, dtype=dtype)
                idx = _hot_indices(n, 32, GPU_TYPE)
                v = torch.randn(n, 32, device=GPU_TYPE, dtype=dtype)

                def fn(x, i, v):
                    return torch.index_put(x, (i,), v, accumulate=True)

                tol = 1e-3 if dtype == torch.float32 else 0.05
                code = self._check(fn, x, idx, v, atol=tol * 100, rtol=tol)
                self.assertIn("COPIES : tl.constexpr = 8", code)
                self.assertEqual(counters["inductor"]["scatter_preagg_copies"], 1)
                self.assertEqual(
                    counters["inductor"]["scatter_preagg_fp32_copy"],
                    dtype == torch.bfloat16,
                )
                out = torch.compile(fn, fullgraph=True)(x, idx, v)
                self.assertTrue(out[32:].signbit().all())

                # Few writes per element, or too few tokens to serialize a hot
                # destination's atomics, do not pay for the copies.
                for rows, tokens in ((1 << 16, n), (64, 50_000)):
                    counters.clear()
                    torch._dynamo.reset()
                    x = torch.zeros(rows, 32, device=GPU_TYPE, dtype=dtype)
                    idx = _hot_indices(tokens, rows, GPU_TYPE)
                    self._check(fn, x, idx, v[:tokens], atol=tol * 100, rtol=tol)
                    self.assertEqual(counters["inductor"]["scatter_preagg_copies"], 0)

        torch._dynamo.reset()
        counters.clear()

        def dynamic_fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        compiled = torch.compile(dynamic_fn, fullgraph=True, dynamic=True)
        n = 1 << 18
        values = torch.randn(n, 32, device=GPU_TYPE)
        for rows in (64, 48):
            x = torch.zeros(rows, 32, device=GPU_TYPE)
            idx = _hot_indices(n, rows, GPU_TYPE)
            self.assertEqual(
                compiled(x, idx, values),
                dynamic_fn(x, idx, values),
                atol=0.1,
                rtol=1e-3,
            )
        self.assertEqual(counters["stats"]["unique_graphs"], 1)
        self.assertEqual(counters["inductor"]["scatter_preagg_copies"], 0)

        torch._dynamo.reset()
        counters.clear()
        x = torch.zeros(64, device=GPU_TYPE)
        idx = _hot_indices(1 << 14, 64, GPU_TYPE)
        values = torch.randn(1 << 14, device=GPU_TYPE)
        self._check(lambda x, i, v: x.index_add(0, i, v), x, idx, values, rtol=1e-3)
        self.assertEqual(counters["inductor"]["scatter_preagg_copies"], 1)

        torch._dynamo.reset()
        counters.clear()
        x = torch.zeros(8192, 32, device=GPU_TYPE)
        idx = _hot_indices(1 << 20, 8192, GPU_TYPE)
        values = torch.randn(1 << 20, 32, device=GPU_TYPE)
        code = self._check(dynamic_fn, x, idx, values, atol=0.1, rtol=1e-3)
        self.assertIn("COPIES : tl.constexpr = 4", code)

        torch._dynamo.reset()
        counters.clear()
        output_storage = scatter_preagg._output_storage
        calls = []

        def first_call_fails(out):
            calls.append(out)
            return None if len(calls) == 1 else output_storage(out)

        x = torch.zeros(16, 32, device=GPU_TYPE)
        idx = _hot_indices(20_000, 16, GPU_TYPE)
        values = torch.randn(20_000, 32, device=GPU_TYPE)
        with unittest.mock.patch.object(
            scatter_preagg, "_output_storage", first_call_fails
        ):
            self._check(
                lambda x, i, v: x.index_add_(0, i, v), x, idx, values, rtol=1e-3
            )
        self.assertEqual(counters["inductor"]["scatter_preagg_output_copy"], 1)

    @config.patch(
        comprehensive_padding=True,
        padding_alignment_bytes=128,
        padding_stride_threshold=1024,
    )
    def test_output_copies_padded_stride(self):
        # The copies of a 1700-element output are padded to a stride of 1728,
        # which the template must address them by. Small integer values make
        # every summation order exact.
        n = 1 << 18
        x = torch.randint(-2, 3, (17, 100), device=GPU_TYPE).float()
        idx = torch.randint(0, 17, (n,), device=GPU_TYPE)
        v = torch.randint(-1, 2, (n, 100), device=GPU_TYPE).float()

        def fn(x, i, v):
            return x.t().index_add(1, i, v.t()).t()

        code = self._check(fn, x, idx, v, atol=0, rtol=0)
        self.assertIn("COPY_NUMEL : tl.constexpr = 1728", code)
        self.assertEqual(counters["inductor"]["scatter_preagg_copies"], 1)

    def _check_deterministic(self, fn, *args, tol):
        ref = fn(*[a.double() if a.is_floating_point() else a.clone() for a in args])
        ref = ref.double()
        compiled = torch.compile(fn, fullgraph=True)
        with DeterministicGuard(True):
            out1, (code,) = run_and_get_code(compiled, *[a.clone() for a in args])
            out2 = compiled(*[a.clone() for a in args])
        self.assertEqual(out1.double(), ref, atol=ref.abs().max().item() * tol, rtol=0)
        self.assertTrue(torch.equal(out1, out2))
        self.assertEqual(counters["inductor"]["scatter_preagg_sorted"], 1)
        self.assertNotIn("tl.atomic_add(", code)

    def test_deterministic(self):
        for case in [
            "index_put",
            "index_add",
            "scatter_add",
            "one_key",
            "unique",
            "int32",
            "view",
        ]:
            with self.subTest(case=case):
                torch._dynamo.reset()
                counters.clear()
                n, rows, d = 30_000, 50, 24
                dtype = torch.int32 if case == "int32" else torch.float32
                x = torch.zeros(rows, d, device=GPU_TYPE, dtype=dtype)
                v = torch.randn(n, d, device=GPU_TYPE).to(dtype)
                idx = _hot_indices(n, rows, GPU_TYPE)
                fn = _BF16_SCATTER_FNS.get(case, _BF16_SCATTER_FNS["index_put"])
                if case == "one_key":
                    # One segment spanning every chunk.
                    idx = torch.zeros_like(idx)
                elif case == "unique":
                    x = torch.zeros(n, d, device=GPU_TYPE)
                    idx = torch.randperm(n, device=GPU_TYPE)
                elif case == "int32":
                    v = torch.randint(-5, 5, (n, d), device=GPU_TYPE, dtype=dtype)
                elif case == "view":
                    x = torch.zeros(rows, 3, device=GPU_TYPE)
                    v = torch.randn(n, 1, device=GPU_TYPE)

                    def fn(x, i, v):
                        return x[:, 1:2].index_add_(0, i, v)

                self._check_deterministic(fn, x, idx, v, tol=1e-5)

    def test_index_width_covers_tokens(self):
        # Broadcast indices into a small output can iterate past int32 tokens.
        n, rows = 50_000, 64
        x = torch.zeros(rows, rows, device=GPU_TYPE)
        i0 = torch.randint(0, rows, (n, 1), device=GPU_TYPE)
        i1 = torch.randint(0, rows, (1, n), device=GPU_TYPE)
        v = torch.ones((), device=GPU_TYPE)

        def fn(x, i0, i1, v):
            return torch.index_put(x, (i0, i1), v, accumulate=True)

        out, (code,) = run_and_get_code(torch.compile(fn), x, i0, i1, v)
        expect = torch.outer(
            torch.bincount(i0.view(-1), minlength=rows).double(),
            torch.bincount(i1.view(-1), minlength=rows).double(),
        )
        self.assertEqual(out.double(), expect, atol=0, rtol=1e-6)
        self.assertIn("INDEX_DTYPE : tl.constexpr = tl.int64", code)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)

    def test_broadcast_values_dynamic_index(self):
        # Values with the index's dynamic leading size, and values expanded to
        # it, compile once for every size.
        def fn(out, i, v):
            return out.index_put([i], v, accumulate=True)

        def fn_expanded(out, i, seed):
            v = seed.expand(i.shape[0], i.shape[1], -1)
            return out.index_put([i], v, accumulate=True)

        out = torch.zeros(512, 4, device=GPU_TYPE)
        seed = torch.randn(1, 1, 4, device=GPU_TYPE)
        for f, expanded in ((fn, False), (fn_expanded, True)):
            torch._dynamo.reset()
            counters.clear()
            compiled = torch.compile(f, fullgraph=True)
            for n in (16, 24):
                idx = torch.randint(0, 4, (n, 512), device=GPU_TYPE)
                torch._dynamo.mark_dynamic(idx, 0)
                if expanded:
                    v = seed
                else:
                    v = torch.randn(n, 512, 4, device=GPU_TYPE)
                    torch._dynamo.mark_dynamic(v, 0)
                self.assertEqual(
                    compiled(out.clone(), idx, v),
                    f(out.clone(), idx, v),
                    atol=1e-2,
                    rtol=1e-4,
                )
            self.assertEqual(counters["stats"]["unique_graphs"], 1)
            self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)

    @config.patch(scatter_preagg_template=False)
    def test_disabled(self):
        x = torch.zeros(24, 32, device=GPU_TYPE)
        idx = _hot_indices(20_000, 24, GPU_TYPE)
        v = torch.randn(20_000, 32, device=GPU_TYPE)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v, applied=False)


class TestScatterPreaggConfigs(TestCase):
    """Config heuristics for given device targets."""

    hw_classification = HardwareClassification.GENERIC

    def setUp(self):
        super().setUp()
        # Isolate the singleton and avoid querying a device.
        heuristics = "torch._inductor.heuristics.template.triton"
        self.enterContext(
            unittest.mock.patch.dict(BaseHeuristicSingleton._instances, clear=True)
        )
        self.enterContext(
            unittest.mock.patch(f"{heuristics}.get_backend_num_stages", return_value=2)
        )
        self.enterContext(
            unittest.mock.patch(f"{heuristics}.kpack_supported", return_value=False)
        )

    def _configs(self, target, *args, rocm=True, **kwargs):
        cls = ROCmConfigHeuristic if rocm else CUDAConfigHeuristic
        return cls().get_scatter_preagg_configs(*args, target=target, **kwargs)

    def _block(self, target, *args, **kwargs):
        (cfg,) = self._configs(target, *args, **kwargs)
        return cfg.block_t, cfg.block_d

    def test_configs(self):
        for rocm in [False, True]:
            with self.subTest(rocm=rocm):
                torch._dynamo.reset()
                counters.clear()
                target = ScatterPreaggTarget(64, 64 << 10, 80, fp16_matrix_preagg=True)
                block = functools.partial(self._block, target, rocm=rocm)
                default = self._configs(target, 50_000, 32, torch.float32, rocm=rocm)
                self.assertEqual(len(default), 1)
                self.assertEqual(default[0].num_warps, target.warps(128))
                self.assertEqual(block(50_000, 32, torch.float32), (256, 16))
                self.assertEqual(block(50_000, 128, torch.bfloat16), (128, 32))
                self.assertEqual(block(50_000, 512, torch.float32), (128, 32))
                # On ROCm, fp16 blocks take the matrix product.
                self.assertEqual(
                    block(50_000, 128, torch.float16), (64, 32) if rocm else (128, 32)
                )
                self.assertEqual(
                    block(50_000, 512, torch.float16), (64, 64) if rocm else (128, 32)
                )
                self.assertEqual(block(50_000, 16, torch.float16), (256, 16))
                strided = (50_000, 256, torch.float32)
                self.assertEqual(block(*strided, inner_contiguous=False), (512, 8))
                tuned = self._configs(
                    target, 1 << 20, 32, torch.float32, max_autotune=True, rocm=rocm
                )
                self.assertGreater(len(tuned), 1)
                self.assertEqual(
                    tuned[0],
                    self._configs(target, 1 << 20, 32, torch.float32, rocm=rocm)[0],
                )
                self.assertEqual(max(c.block_t * c.block_d for c in tuned), 8192)
                # Ragged inner sizes keep the shared-memory-safe tile.
                ragged = self._configs(
                    target, 1 << 20, 37, torch.float32, max_autotune=True, rocm=rocm
                )
                self.assertEqual(max(c.block_t * c.block_d for c in ragged), 4096)

    def test_rocm_occupancy_configs(self):
        target = ScatterPreaggTarget(64, 64 << 10, 80, fp16_matrix_preagg=True)
        # Narrow fp32 rows take 256 threads and a register cap; wide slices
        # 256 threads.
        (cfg,) = self._configs(target, 1 << 20, 4, torch.float32)
        self.assertEqual((cfg.num_warps, cfg.waves_per_eu), (target.warps(256), 4))
        (cfg,) = self._configs(target, 1 << 20, 1024, torch.float32)
        self.assertGreaterEqual(cfg.block_d * cfg.d_iters, 256)
        self.assertEqual(cfg.num_warps, target.warps(256))
        # Neither applies to other dtypes or strided rows.
        for args, kwargs in [
            ((1 << 20, 4, torch.bfloat16), {}),
            ((1 << 20, 4, torch.float32), {"inner_contiguous": False}),
        ]:
            (cfg,) = self._configs(target, *args, **kwargs)
            self.assertEqual(cfg.num_warps, target.warps(128))
            self.assertEqual(getattr(cfg, "waves_per_eu", 0), 0)
        # The matrix product takes one wavefront, and 64-wide slices past
        # 128-wide rows of cold duplicates.
        (cfg,) = self._configs(target, 50_000, 256, torch.float16, num_keys=1 << 20)
        self.assertEqual((cfg.block_d, cfg.num_warps), (64, 1))
        (cfg,) = self._configs(target, 50_000, 256, torch.float16, num_keys=64)
        self.assertEqual((cfg.block_d, cfg.num_warps), (32, 1))

    def test_target_configs(self):
        for warp_size, shared_mem_kb, num_sms, ragged_tile in [
            (64, 64, 80, 4096),
            (64, 160, 256, 8192),
            (32, 320, 128, 8192),
        ] + [(32, 64, 96, 4096)]:
            with self.subTest(
                warp_size=warp_size,
                shared_mem_kb=shared_mem_kb,
                num_sms=num_sms,
                ragged_tile=ragged_tile,
            ):
                torch._dynamo.reset()
                counters.clear()
                # Configs keep the lanes per program across warp sizes, scale the ragged
                # tile with shared memory and the small-grid split with the SM count.
                target = ScatterPreaggTarget(warp_size, shared_mem_kb << 10, num_sms)
                (cfg,) = self._configs(target, 50_000, 32, torch.float32)
                self.assertEqual(cfg.num_warps * warp_size, 128)
                tuned = self._configs(
                    target, 1 << 20, 37, torch.float32, max_autotune=True
                )
                self.assertEqual(max(c.block_t * c.block_d for c in tuned), ragged_tile)
                # Small scatters halve blocks until every SM has two programs: at
                # 256 tokens per SM, 128-token blocks.
                (cfg,) = self._configs(target, 256 * num_sms, 16, torch.float32)
                self.assertEqual((cfg.block_t, cfg.block_d), (128, 16))

    def test_shared_memory_per_block(self):
        # The opt-in limit where there is one; XPU reports local memory.
        props = types.SimpleNamespace(
            shared_memory_per_block_optin=160 << 10, shared_memory_per_block=64 << 10
        )
        self.assertEqual(_shared_memory_per_block(props), 160 << 10)
        del props.shared_memory_per_block_optin
        self.assertEqual(_shared_memory_per_block(props), 64 << 10)
        xpu = types.SimpleNamespace(local_mem_size=128 << 10)
        self.assertEqual(_shared_memory_per_block(xpu), 128 << 10)
        self.assertIsNone(_shared_memory_per_block(types.SimpleNamespace()))


@unittest.skipIf(not HAS_CUDA_AND_TRITON, "CUDA and ROCm targets")
class TestScatterPreaggTarget(TestCase):
    hw_classification = HardwareClassification.CUDA

    def test_target_from_device(self):
        device = torch.device("cuda", 0)
        props = torch.cuda.get_device_properties(device)
        target = V.choices.get_scatter_preagg_target(device)
        self.assertEqual(target.warp_size, props.warp_size)
        self.assertEqual(target.shared_mem_bytes, props.shared_memory_per_block_optin)
        self.assertEqual(target.num_sms, props.multi_processor_count)
        self.assertEqual(target.atomic_vec, 2 if TEST_WITH_ROCM else 0)
        self.assertEqual(target.plain_atomic_bytes, 4 if TEST_WITH_ROCM else 0)
        self.assertEqual(target.fp16_matrix_preagg, TEST_WITH_ROCM)


class TestScatterPreaggGating(TestCase):
    hw_classification = HardwareClassification.GENERIC

    @config.patch(scatter_preagg_template=True)
    def test_enabled_only_on_rocm_gpu(self):
        enabled = V.choices.should_use_scatter_preagg_template
        gpu = torch.device("cuda", 0)
        with unittest.mock.patch("torch._inductor.choices.IS_ROCM", True):
            self.assertTrue(enabled(gpu))
            self.assertFalse(enabled(torch.device("cpu")))
            self.assertFalse(enabled(torch.device("xpu", 0)))
            self.assertFalse(enabled(None))
            with config.patch(scatter_preagg_template=False):
                self.assertFalse(enabled(gpu))
        with unittest.mock.patch("torch._inductor.choices.IS_ROCM", False):
            self.assertFalse(enabled(gpu))

    def _reset_deprecation_warning(self, name):
        entry = config._config[name]
        self.addCleanup(
            setattr, entry, "_deprecation_warned", entry._deprecation_warned
        )
        entry._deprecation_warned = False

    def test_deprecated_partitioned_scatter_surface(self):
        self._reset_deprecation_warning("partitioned_scatter_enabled")
        with self.assertWarnsRegex(FutureWarning, "use that instead"):
            for value in (False, True):
                with config.patch(scatter_preagg_template=not value):
                    with config.patch(partitioned_scatter_enabled=value):
                        self.assertEqual(config.scatter_preagg_template, value)
                    with config.patch(scatter_preagg_template=value):
                        self.assertEqual(config.partitioned_scatter_enabled, value)
        for name in (
            "partitioned_scatter_min_partitions",
            "partitioned_scatter_max_partitions",
            "partitioned_scatter_min_index_size",
            "partitioned_scatter_min_contention_ratio",
            "partitioned_scatter_non_model_floor_bytes",
            "partitioned_scatter_force",
        ):
            self._reset_deprecation_warning(name)
            with self.assertWarnsRegex(FutureWarning, "has done nothing since 2.16"):
                value = getattr(config, name)
            self._reset_deprecation_warning(name)
            with self.assertWarnsRegex(FutureWarning, "has done nothing since 2.16"):
                with config.patch({name: value}):
                    pass

        script = "import torch._inductor.config as c; print(c.scatter_preagg_template)"
        # Only "1" enables, and the new name takes precedence.
        for legacy, new, expected in [
            (None, None, torch.version.hip is not None),
            (None, "1", True),
            ("0", None, False),
            ("1", None, True),
            ("true", None, False),
            ("0", "1", True),
            ("1", "0", False),
        ]:
            env_vars = {
                name: value
                for name, value in [
                    ("TORCHINDUCTOR_PARTITIONED_SCATTER_ENABLED", legacy),
                    ("TORCHINDUCTOR_SCATTER_PREAGG_TEMPLATE", new),
                ]
                if value is not None
            }
            out = self._run(script, **env_vars)
            self.assertEqual(out.stdout.strip(), str(expected))
            self.assertEqual(
                "TORCHINDUCTOR_PARTITIONED_SCATTER_ENABLED is deprecated since 2.16 "
                "(removal planned for 2.18) and sets scatter_preagg_template unless "
                "TORCHINDUCTOR_SCATTER_PREAGG_TEMPLATE is set" in out.stderr,
                legacy is not None,
            )
        out = self._run(script, TORCHINDUCTOR_PARTITIONED_SCATTER_FORCE="1")
        self.assertIn(
            "TORCHINDUCTOR_PARTITIONED_SCATTER_FORCE is deprecated since 2.16 "
            "(removal planned for 2.18) and does nothing",
            out.stderr,
        )

        message = "The 'partitioned_scatter' logging artifact is deprecated"
        out = self._run("import torch", TORCH_LOGS="partitioned_scatter")
        self.assertIn(message, out.stderr)
        self.assertIn("Use 'scatter_preagg' instead", out.stderr)
        out = self._run("import torch", TORCH_LOGS="scatter_preagg")
        self.assertNotIn(message, out.stderr)

    def _run(self, script, **env_vars):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("TORCHINDUCTOR_") and k != "TORCH_LOGS"
        }
        env.update(env_vars)
        return subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            text=True,
            capture_output=True,
            check=True,
        )


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
