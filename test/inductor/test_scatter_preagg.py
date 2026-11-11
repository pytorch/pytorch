# Owner(s): ["module: inductor"]

import functools
import math
import os
import subprocess
import sys
import types
import unittest.mock
import warnings

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
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_utils import (
    DeterministicGuard,
    HardwareClassification,
    instantiate_parametrized_tests,
    parametrize,
    TEST_WITH_ROCM,
)
from torch.testing._internal.inductor_utils import HAS_CUDA_AND_TRITON


def _hot_indices(n, num_rows, device, hot_fraction=0.5, dtype=torch.int64):
    tail = torch.randint(0, num_rows, (n,), device=device, dtype=dtype)
    hot = torch.rand(n, device=device) < hot_fraction
    return torch.where(hot, torch.zeros_like(tail), tail)


def _dtype_name(dtype):
    return str(dtype).removeprefix("torch.")


_SCATTER_FNS = {
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


def _cfg(block_t, block_d, d_iters, num_warps):
    return ScatterPreaggConfig(
        block_t=block_t,
        block_d=block_d,
        d_iters=d_iters,
        num_stages=1,
        num_warps=num_warps,
    )


def _with_configs(*cfgs):
    return unittest.mock.patch.object(
        InductorChoices, "get_scatter_preagg_configs", return_value=list(cfgs)
    )


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
        counters.clear()
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

    @parametrize("scale_grad_by_freq", [False, True])
    def test_embedding_dense_backward(self, device, scale_grad_by_freq):
        n, v, d = 50_000, 24, 32
        idx = _hot_indices(n, v, device, dtype=torch.int32)
        x = torch.randn(n, 224, device=device, dtype=torch.bfloat16)

        def fn(i, a):
            return torch.ops.aten.embedding_dense_backward(
                a[:, 184 : 184 + d], i, v, -1, scale_grad_by_freq
            )

        self._check(fn, idx, x, atol=5e-2, rtol=2e-2)

    def test_index_put_rows(self, device):
        x = torch.zeros(64, 32, device=device)
        idx = _hot_indices(40_000, 64, device)
        v = torch.randn(40_000, 32, device=device)

        def fn(x, i, v):
            return x.index_put_((i,), v, accumulate=True)

        self._check(fn, x, idx, v)

    def test_multi_index_non_adjacent(self, device):
        x = torch.zeros(8, 16, 6, device=device)
        i0 = torch.randint(0, 8, (20_000,), device=device)
        i1 = torch.randint(0, 6, (20_000,), device=device)
        v = torch.randn(20_000, 16, device=device)

        def fn(x, i0, i1, v):
            x[i0, :, i1] += 0
            return torch.ops.aten.index_put(x, [i0, None, i1], v, True)

        self._check(fn, x, i0, i1, v)

    def test_fused_value_producer(self, device):
        x = torch.zeros(32, 64, device=device)
        idx = _hot_indices(20_000, 32, device)
        a = torch.randn(20_000, 64, device=device)
        b = torch.randn(64, device=device)

        def fn(x, i, a, b):
            return torch.index_put(x, (i,), (a * b).relu() + 1, accumulate=True)

        self._check(fn, x, idx, a, b)

    def test_index_add_dim1(self, device):
        x = torch.randn(16, 48, device=device)
        idx = _hot_indices(30_000, 48, device)
        src = torch.randn(16, 30_000, device=device)

        def fn(x, i, s):
            return x.index_add(1, i, s)

        self._check(fn, x, idx, src)

    def test_index_add_alpha(self, device):
        x = torch.randn(16, 48, device=device)
        idx = _hot_indices(30_000, 16, device)
        src = torch.randn(30_000, 48, device=device)

        def fn(x, i, s):
            return x.index_add(0, i, s, alpha=2.5)

        self._check(fn, x, idx, src, atol=1e-2)

    def test_tiny_uint8_index_add_negative_alpha(self, device):
        x = torch.zeros(5, 5, device=device, dtype=torch.uint8)
        idx = torch.arange(5, device=device, dtype=torch.int32)
        src = torch.arange(25, device=device, dtype=torch.uint8).view(5, 5)

        def fn(x, i, s):
            return x.index_add(0, i, s, alpha=-1)

        self._check(fn, x, idx, src, applied=False, atol=0, rtol=0)
        self.assertEqual(counters["inductor"]["scatter_preagg_skipped_too_small"], 1)

    def test_inplace_graph_inputs(self, device):
        # Mutating graph inputs in place still reaches the template.
        b, n, rows, d = 2, 8192, 8, 4
        idx = torch.randint(0, 4, (n,), device=device)
        scatter_idx = idx.view(1, n, 1).expand(b, n, d)
        vals = torch.randn(b, n, d, device=device)

        def fn(index_out, scatter_out, idx, scatter_idx, vals):
            index_out.index_add_(1, idx, vals)
            scatter_out.scatter_add_(1, scatter_idx, vals)
            return index_out, scatter_out

        outs = (torch.zeros(b, rows, d, device=device),) * 2
        with config.patch(scatter_preagg_template_min_numel=0):
            self._check(fn, *outs, idx, scatter_idx, vals, atol=1e-2, rtol=1e-3)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 2)

    @config.patch(scatter_preagg_template_min_saved=-1.0)
    @parametrize("dups", [False, True])
    @parametrize("dtype", [torch.float32, torch.bfloat16], name_fn=_dtype_name)
    def test_wide_slices(self, device, dups, dtype):
        # Wide slices of blocks with and without duplicates.
        x = torch.zeros(1024, 256, device=device, dtype=dtype)
        tok = torch.arange(65_536, device=device)
        idx = (tok // 2 if dups else tok) % 1024
        v = torch.randn(65_536, 256, device=device, dtype=dtype)

        def fn(x, i, v):
            return x.index_put_((i,), v, accumulate=True)

        ref = fn(x.double(), idx, v.double())
        with _with_configs(_cfg(64, 64, 4, 4)):
            out, (code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), x.clone(), idx, v
            )
        scale = 1e-5 if dtype == torch.float32 else 1e-2
        self.assertEqual(out.double(), ref, atol=ref.abs().max().item() * scale, rtol=0)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
        self.assertIn("BLOCK_D : tl.constexpr = 64", code)

    @parametrize("deterministic", [False, True])
    def test_static_inner_shifted(self, device, deterministic):
        # A static row size no inner block divides shifts its last block back
        # over part of the one before it, whose lanes it masks.
        x = torch.zeros(100, 1040, device=device)
        idx = torch.randint(0, 100, (20_000,), device=device)
        v = torch.randn(20_000, 1040, device=device)

        def fn(x, i, v):
            return x.index_add(0, i, v)

        ref = fn(x.double(), idx, v.double())
        self.enterContext(DeterministicGuard(deterministic))
        with _with_configs(_cfg(64, 64, 4, 4)):
            out, codes = run_and_get_code(
                torch.compile(fn, fullgraph=True), x.clone(), idx, v
            )
        self.assertIn("INNER_SHIFT : tl.constexpr = True", "\n".join(codes))
        self.assertEqual(out.double(), ref, atol=1e-3, rtol=1e-4)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)

    @parametrize("rows", [512, 1 << 16])
    def test_exact_histogram(self, device, rows):
        # Destinations that fit the histogram get a bin each, rather than
        # hashed bins whose count linear counting would overestimate. The bins
        # scale with the block, which the default config sizes by CU count.
        x = torch.zeros(rows, 32, device=device)
        idx = torch.randint(0, rows, (65_536,), device=device)
        v = torch.randn(65_536, 32, device=device)

        def fn(x, i, v):
            return x.index_put_((i,), v, accumulate=True)

        ref = fn(x.double(), idx, v.double())
        with _with_configs(_cfg(256, 16, 1, 2)):
            out, (code,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), x.clone(), idx, v
            )
        self.assertEqual(out.double(), ref, atol=1e-3, rtol=1e-4)
        self.assertIn("ADAPTIVE : tl.constexpr = True", code)
        self.assertEqual("HIST_EXACT : tl.constexpr = True" in code, rows == 512)

    def test_gathered_values_config(self, device):
        # A message-passing scatter of gathered rows, on a config whose plain
        # path once miscompiled with buffer ops into a memory fault.
        x = torch.zeros(65_536, 64, device=device)
        h = torch.randn(65_536, 64, device=device)
        src = torch.randint(0, 65_536, (524_288,), device=device)
        dst = torch.randint(0, 65_536, (524_288,), device=device)
        w = torch.rand(524_288, device=device)

        def fn(x, dst, h, src, w):
            return x.index_add(0, dst, h[src] * w[:, None])

        with _with_configs(_cfg(512, 16, 4, 4)):
            self._check(fn, x, dst, h, src, w, atol=1e-3)

    @parametrize("dtype", [torch.float16, torch.float32], name_fn=_dtype_name)
    def test_plain_vec_narrower_than_block(self, device, dtype):
        # Single columns have 1-wide plain blocks, which two-element (fp16)
        # vector atomics must not be promised.
        x = torch.zeros(1024, device=device, dtype=dtype)
        idx = torch.randint(0, 1024, (50_000,), device=device)
        v = torch.randn(50_000, device=device, dtype=dtype)

        def fn(x, i, v):
            return x.index_add(0, i, v)

        code = self._check(fn, x, idx, v, atol=0.5, rtol=1e-2)
        self.assertIn("PLAIN_D : tl.constexpr = 1", code)
        self.assertIn(
            f"PLAIN_VEC : tl.constexpr = {0 if dtype == torch.float16 else 1}", code
        )

    def test_scatter_add_expanded_index(self, device):
        x = torch.zeros(16, 32, device=device)
        rows = _hot_indices(20_000, 16, device)
        src = torch.randn(20_000, 32, device=device)

        def fn(x, r, s):
            return x.scatter_add(0, r[:, None].expand(-1, 32), s)

        self._check(fn, x, rows, src)

    def test_scatter_add_int32_index(self, device):
        # A narrow index loads as int32 and widens, so it still reads as an index.
        x = torch.zeros(8, 4, device=device)
        idx = torch.randint(0, 4, (8192, 4), device=device, dtype=torch.int32)
        src = torch.randn(8192, 4, device=device)

        def fn(x, i, s):
            return x.scatter_add(0, i.long(), s)

        self._check(fn, x, idx, src)

    @parametrize("include_self", [False, True])
    def test_scatter_reduce_sum(self, device, include_self):
        x = torch.randn(16, 8, device=device)
        idx = torch.randint(0, 16, (20_000, 8), device=device)
        src = torch.randn(20_000, 8, device=device)

        def fn(x, i, s):
            return x.scatter_reduce(0, i, s, "sum", include_self=include_self)

        self._check(fn, x, idx, src)

    def test_scatter_reduce_sum_int32_index(self, device):
        x = torch.randn(16, 8, device=device)
        idx = torch.randint(0, 16, (20_000, 8), device=device, dtype=torch.int32)
        src = torch.randn(20_000, 8, device=device)

        def fn(x, i, s):
            return x.scatter_reduce(0, i.long(), s, "sum", include_self=False)

        self._check(fn, x, idx, src)

    @parametrize("op", list(_SCATTER_FNS))
    def test_bf16(self, device, op):
        fn = _SCATTER_FNS[op]
        x = torch.randn(24, 64, device=device).bfloat16()
        idx = _hot_indices(50_000, 24, device)
        v = torch.randn(50_000, 64, device=device).bfloat16()

        # Accumulated in an fp32 copy, so each destination is rounded once.
        ref = fn(x.float(), idx, v.float())
        out = torch.compile(fn, fullgraph=True)(x.clone(), idx, v)
        self.assertEqual(out.dtype, torch.bfloat16)
        self.assertEqual(out.float(), ref, atol=ref.abs().max().item() * 1e-2, rtol=0)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
        self.assertEqual(counters["inductor"]["scatter_preagg_fp32_copy"], 1)

    @parametrize("reason", ["too_small", "fp32_copy_too_large"])
    @parametrize("op", ["index_put", "scatter_reduce"])
    def test_bf16_declined_falls_back(self, device, reason, op):
        # A scatter the template declines for its size must still be lowered:
        # eager fallback where bf16 atomics are missing, atomics where they exist.
        fn = _SCATTER_FNS[op]
        rows, n = (1000, 400) if reason == "too_small" else (100_000, 8192)
        x = torch.randn(rows, 8, device=device).bfloat16()
        idx = torch.randint(0, rows, (n,), device=device)
        v = torch.randn(n, 8, device=device).bfloat16()

        code = self._check(fn, x, idx, v, applied=False, atol=2e-2, rtol=2e-2)
        # Recorded once, by whichever lowering declined.
        self.assertEqual(counters["inductor"][f"scatter_preagg_skipped_{reason}"], 1)
        self.assertNotIn("tl.atomic_add(", code)
        # Nothing set up for the template is left behind.
        torch._dynamo.reset()
        with config.patch(scatter_preagg_template=False):
            _, (regular,) = run_and_get_code(
                torch.compile(fn, fullgraph=True), x.clone(), idx, v
            )
        self.assertEqual(code.count("@triton.jit"), regular.count("@triton.jit"))

    @config.patch(scatter_preagg_template_min_numel=0)
    def test_fp16_copy_too_large_falls_back(self, device):
        # fp16 into few destinations stages in fp32 too, which declines_by_size
        # does not predict: the template checks again once it has the plan.
        x = torch.randn(64, 8, device=device).half()
        idx = torch.randint(0, 64, (20,), device=device)
        v = torch.randn(20, 8, device=device).half()

        def fn(x, i, v):
            return x.index_add(0, i, v)

        self._check(fn, x, idx, v, applied=False, atol=1e-2, rtol=1e-2)
        self.assertEqual(
            counters["inductor"]["scatter_preagg_skipped_fp32_copy_too_large"], 1
        )

    @parametrize(
        "mode", ["deterministic", "inductor_deterministic", "bf16", "int8", "int16"]
    )
    @parametrize("op", ["index_put", "scatter_add", "scatter_reduce"])
    def test_declined_emits_regular_lowering(self, device, mode, op):
        # A scatter the template declines must still be lowered: to the eager
        # fallback where it bypassed that (deterministic algorithms, bf16, and
        # 8/16-bit ints), otherwise to atomics (TORCHINDUCTOR_DETERMINISTIC).
        fn = _SCATTER_FNS[op]
        dtype = {"bf16": torch.bfloat16, "int8": torch.int8, "int16": torch.int16}.get(
            mode, torch.float32
        )
        # Few writes per destination keep the eager bf16 fallback accurate.
        make = torch.randn if dtype.is_floating_point else torch.ones
        x = make(4096, 8, device=device).to(dtype)
        idx = torch.randint(0, 4096, (20_000,), device=device)
        v = make(20_000, 8, device=device).to(dtype)
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
        self.assertEqual(out.double(), ref, atol=ref.abs().max().item() * 2e-2, rtol=0)
        code = "\n".join(codes)
        if mode == "inductor_deterministic":
            self.assertIn("tl.atomic_add(", code)
        else:
            self.assertNotIn("tl.atomic_add(", code)
            self.assertIn(
                "aten.index_put_" if op == "index_put" else "aten.scatter_reduce_.two(",
                code,
            )

    @parametrize("deterministic", [False, True])
    @parametrize("op", ["scatter", "index_put"])
    def test_skip_overwriting_scatter(self, device, deterministic, op):
        # Only accumulating scatters reach the template; deterministic mode
        # also skips its size gate.
        x = torch.randn(64, 32, device=device)
        idx = torch.randint(0, 64, (20_000, 32), device=device)
        # Equal values make the duplicate writes' order irrelevant.
        v = torch.ones(20_000, 32, device=device)

        def fn(x, i, v):
            if op == "scatter":
                return x.scatter(0, i, v)
            return torch.index_put(x, (i[:, 0],), v, accumulate=False)

        self.enterContext(DeterministicGuard(deterministic))
        self._check(fn, x, idx, v, applied=False)

    @config.patch(max_autotune=True)
    def test_autotune_index_inputs(self, device):
        # Autotuning randomizes index inputs only within the size each one
        # indexes.
        x = torch.zeros(4096, 64, device=device)
        v = torch.randn(8192, 64, device=device)
        idx = torch.randint(0, 1000, (8192,), device=device)
        remap = torch.randint(0, 4096, (1000,), device=device)
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
    @parametrize("shared", ["key_and_value", "two_values"])
    def test_autotune_shared_index_bounds(self, device, shared):
        # An index addressing two sizes is randomized within the smaller,
        # whether it is the key and gathers values, or gathers two values.
        x = torch.zeros(100, 32, device=device)
        idx = torch.randint(0, 100, (8192,), device=device)
        key = torch.randint(0, 100, (8192,), device=device)
        small = torch.randn(100, 32, device=device)
        table = torch.randn(4096, 32, device=device)
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
    def test_autotune_expanded_index(self, device):
        # An expanded index input overlaps itself in memory (cloning it, as
        # _check does, would not).
        x = torch.zeros(512, 64, device=device)
        idx = torch.randint(0, 512, (16384,), device=device)[:, None].expand(-1, 64)
        v = torch.randn(16384, 64, device=device)

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
    @parametrize(
        "index", ["offset", "float", "value_gather", "key_gather", "unchecked_key"]
    )
    def test_autotune_computed_index(self, device, index):
        # Benchmark inputs (zeros, or random floats) put an index computed from
        # them out of bounds. Such scatters take the default config
        # unbenchmarked, unless the index only addresses unchecked writes,
        # which the template masks.
        checked = index in ("offset", "float")
        x = torch.zeros(100, 32, device=device)
        v = torch.randn(20_000, 32, device=device)
        idx = torch.randint(int(index == "offset"), 100, (20_000,), device=device)
        if index == "float":
            u = torch.randint(0, 100, (20_000,), device=device).float()
        else:
            u = torch.rand(20_000, device=device)
        table = torch.randn(1000, 32, device=device)
        remap = torch.randint(0, 100, (1000,), device=device)

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

    @parametrize("max_autotune", [False, True])
    def test_dynamic_tokens_no_recompile(self, device, max_autotune):
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
                x = torch.zeros(1024, 64, device=device)
                idx = torch.randint(0, 1024, (n,), device=device)
                v_t = torch.randn(64, n, device=device)
                expect = fn(x.clone(), idx, v_t)
                self.assertEqual(compiled(x, idx, v_t), expect, atol=1e-3, rtol=1e-3)
        self.assertEqual(counters["stats"]["unique_graphs"], 1)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)

    @config.patch(cpp_wrapper=True)
    @parametrize("deterministic", [False, True])
    def test_cpp_wrapper(self, device, deterministic):
        x = torch.zeros(24, 32, device=device)
        idx = _hot_indices(20_000, 24, device)
        v = torch.randn(20_000, 32, device=device)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self.enterContext(DeterministicGuard(deterministic))
        self._check(fn, x, idx, v)
        self.assertEqual(
            counters["inductor"]["scatter_preagg_sorted"], int(deterministic)
        )

    @parametrize("deterministic", [False, True])
    def test_aoti(self, device, deterministic):
        class Model(torch.nn.Module):
            def forward(self, x, i, v):
                return torch.index_put(x, (i,), v, accumulate=True)

        x = torch.zeros(24, 32, device=device)
        idx = _hot_indices(20_000, 24, device)
        v = torch.randn(20_000, 32, device=device)
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

    def test_aoti_dynamic_rows(self, device):
        # AOTI never checks guards, so a model exported at 1000 rows must not
        # pick int32 sort keys from that hint: they overflow past 2**23 rows.
        class Model(torch.nn.Module):
            def forward(self, x, i, v):
                return torch.index_put(x, (i,), v, accumulate=True)

        x = torch.zeros(1000, 64, device=device)
        idx = torch.randint(0, 1000, (50_000,), device=device)
        v = torch.randn(50_000, 64, device=device)
        rows, inner, tokens = (
            torch.export.Dim(name, max=1 << 24) for name in ("rows", "inner", "tokens")
        )
        dynamic_shapes = ({0: rows, 1: inner}, {0: tokens}, {0: tokens, 1: inner})
        # 256-token blocks put the int32 limit below the rows tested; the
        # default config's block depends on the CU count.
        with _with_configs(_cfg(256, 16, 1, 2)):
            path = torch._inductor.aoti_compile_and_package(
                torch.export.export(Model(), (x, idx, v), dynamic_shapes=dynamic_shapes)
            )
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
        n = 9_000_000
        x = torch.zeros(n, 2, device=device)
        idx = n - 1 - torch.randint(0, 8, (50_000,), device=device)
        v = torch.randn(50_000, 2, device=device)
        out = torch._inductor.aoti_load_package(path)(x, idx, v)
        self.assertEqual(out, Model()(x, idx, v), atol=1e-3, rtol=1e-4)

    @parametrize(
        "dtype",
        [
            torch.float16,
            torch.bfloat16,
            torch.float64,
            torch.int64,
            torch.int32,
            torch.int16,
            torch.int8,
            torch.uint8,
        ],
        name_fn=_dtype_name,
    )
    def test_dtypes(self, device, dtype):
        x = torch.zeros(20, 16, device=device, dtype=dtype)
        idx = _hot_indices(20_000, 20, device)
        if dtype.is_floating_point:
            v = torch.randn(20_000, 16, device=device).to(dtype)
        else:
            # Small integer outputs wrap around, as in eager.
            low = 0 if dtype == torch.uint8 else -5
            v = torch.randint(low, 5, (20_000, 16), device=device, dtype=dtype)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        if dtype in (torch.int16, torch.int8, torch.uint8):
            # No Triton atomics for these; plain Inductor fails to compile them.
            self._check(fn, x, idx, v, atol=0, rtol=0)
            self.assertEqual(counters["inductor"]["scatter_preagg_sorted"], 1)
        elif dtype in (torch.float16, torch.bfloat16):
            # Few destinations, so fp16 is also accumulated in an fp32 copy and
            # each destination is rounded once.
            ref = fn(x.float(), idx, v.float())
            out = torch.compile(fn, fullgraph=True)(x.clone(), idx, v)
            self.assertEqual(
                out.float(), ref, atol=ref.abs().max().item() * 1e-2, rtol=0
            )
            self.assertEqual(counters["inductor"]["scatter_preagg_fp32_copy"], 1)
        else:
            self._check(fn, x, idx, v, atol=1e-6, rtol=1e-6)

    @config.patch(scatter_preagg_template_min_numel=0)
    @parametrize("n", [1000, 4096, 4097, 50001])
    def test_token_tails(self, device, n):
        x = torch.zeros(10, 32, device=device)
        idx = torch.randint(0, 10, (n,), device=device)
        v = torch.randn(n, 32, device=device)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v)

    @parametrize("d", [1, 48, 37, 300])
    def test_inner_sizes(self, device, d):
        x = torch.zeros(12, d, device=device)
        idx = _hot_indices(30_000, 12, device)
        v = torch.randn(30_000, d, device=device)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v)

    @config.patch(max_autotune=True)
    def test_max_autotune(self, device):
        # Several configs are benchmarked on example inputs; the in-place
        # output must only be accumulated into once, by the chosen kernel.
        x = torch.randn(24, 64, device=device)
        idx = _hot_indices(50_000, 24, device)
        v = torch.randn(50_000, 64, device=device)

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

    @parametrize(
        "shape",
        [(20, 64, 50000), (4096, 128, 50000), (300, 100, 30000), (64, 320, 20000)],
        name_fn=lambda shape: "x".join(map(str, shape)),
    )
    def test_fp16_matrix_preagg(self, device, shape):
        rows, d, n = shape

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        x = torch.randn(rows, d, device=device).half()
        idx = _hot_indices(n, rows, device)
        v = torch.randn(n, d, device=device).half()
        self._check_fp16(fn, x, idx, v)

    @parametrize("rows", [20, 4096])
    def test_fp16_matrix_preagg_non_finite(self, device, rows):
        # A block with inf or nan values must not spread nan (0 * inf) into
        # the other destinations of its product.
        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        x = torch.zeros(rows, 64, device=device).half()
        idx = _hot_indices(50_000, rows, device)
        v = torch.randn(50_000, 64, device=device).half()
        v[torch.nonzero(idx == 3)[0], 5] = float("inf")
        v[torch.nonzero(idx == 7)[0], 9] = float("nan")
        self._check_fp16(fn, x, idx, v)

    @parametrize(
        "min_saved,hot_fraction",
        [(-1.0, 0.5), (2.0, 0.0)],
        name_fn=lambda min_saved, hot_fraction: (
            "always_sorted" if min_saved < 0 else "hot_only"
        ),
    )
    def test_adaptive_forced_path(self, device, min_saved, hot_fraction):
        # -1 always sorts; 2 sorts only blocks with a hot destination, of which
        # uniform keys have none. This covers both paths of each chunk.
        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        x = torch.zeros(4096, 64, device=device)
        idx = _hot_indices(50_000, 4096, device, hot_fraction=hot_fraction)
        v = torch.randn(50_000, 64, device=device)
        with config.patch(scatter_preagg_template_min_saved=min_saved):
            code = self._check(fn, x, idx, v, atol=1e-2, rtol=1e-3)
        self.assertIn("ADAPTIVE : tl.constexpr = True", code)
        self.assertIn(f"SORT_SAVED : tl.constexpr = {min_saved}", code)

    @parametrize("d", [1, 32])
    def test_adaptive_runs(self, device, d):
        # Runs of a few recurring keys, which the sort must not merge across
        # the runs between them, and a ragged last block whose shared tokens
        # must count once.
        idx = torch.randint(0, 4, (8192,), device=device).repeat_interleave(8)
        idx = idx[:-37]
        shape = (4096, d) if d > 1 else (4096,)
        x = torch.zeros(shape, device=device)
        v = torch.randn(idx.numel(), *shape[1:], device=device)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        code = self._check(fn, x, idx, v, atol=1e-2, rtol=1e-3)
        # Single columns sort at half the saving.
        min_saved = config.scatter_preagg_template_min_saved / (2 if d == 1 else 1)
        self.assertIn(f"SORT_SAVED : tl.constexpr = {min_saved}", code)

    def test_dynamic_shapes(self, device):
        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        compiled = torch.compile(fn, fullgraph=True)
        # Symbolic token and key counts; covers fewer tokens than a block and
        # an inner size no tile divides.
        for n, rows in [(100, 3), (20_000, 16), (30_001, 24)]:
            x = torch.zeros(rows, 72, device=device)
            idx = torch.randint(0, rows, (n,), device=device)
            v = torch.randn(n, 72, device=device)
            expect = fn(x.clone(), idx, v)
            for t in (x, idx, v):
                torch._dynamo.mark_dynamic(t, 0)
            self.assertEqual(compiled(x, idx, v), expect, atol=1e-3, rtol=1e-3)
        self.assertEqual(counters["inductor"]["scatter_preagg_applied"], 1)
        self.assertEqual(counters["stats"]["unique_graphs"], 1)

    @parametrize("dim", ["inner", "batch", "deterministic"])
    def test_dynamic_inner_shifted(self, device, dim):
        # Aligned row sizes no tile divides shift their last inner block back
        # rather than recompiling per tile, including a batch dimension the
        # destination does not index, which is part of the inner size, and
        # in the sorted lowering.
        def fn(x, i, v):
            return x.index_add(1 if dim == "batch" else 0, i, v)

        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        self.enterContext(DeterministicGuard(dim == "deterministic"))
        for size in (12, 8, 6, 10) if dim == "batch" else (2048, 1040, 1536, 1296):
            if dim == "batch":
                x = torch.zeros(size, 100, 64, device=device)
                v = torch.randn(size, 20_000, 64, device=device)
            else:
                x = torch.zeros(100, size, device=device)
                v = torch.randn(20_000, size, device=device)
            idx = _hot_indices(20_000, 100, device)
            expect = fn(x.clone(), idx, v)
            if size in (12, 2048):
                out, (code,) = run_and_get_code(compiled, x, idx, v)
                self.assertIn("INNER_SHIFT : tl.constexpr = True", code)
            else:
                out = compiled(x, idx, v)
            self.assertEqual(out, expect, atol=1e-3, rtol=1e-3)
        self.assertEqual(counters["stats"]["unique_graphs"], 1)

    def test_deterministic_dynamic(self, device):
        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        self.enterContext(DeterministicGuard(True))
        # Token counts on both sides of the sort's int16 index range.
        for n, rows, d in [
            (1000, 5, 40),
            (40_000, 24, 72),
            (30_001, 24, 72),
            (20_000, 50, 3),
        ]:
            x = torch.zeros(rows, d, device=device)
            idx = _hot_indices(n, rows, device)
            v = torch.randn(n, d, device=device)
            ref = fn(x.double(), idx, v.double())
            out1 = compiled(x.clone(), idx, v)
            out2 = compiled(x.clone(), idx, v)
            self.assertEqual(out1.double(), ref, atol=1e-4, rtol=1e-5)
            self.assertTrue(torch.equal(out1, out2))
        self.assertEqual(counters["inductor"]["scatter_preagg_sorted"], 1)
        self.assertEqual(counters["stats"]["unique_graphs"], 1)

    @parametrize("deterministic", [False, True])
    def test_materialized_keys_read_output(self, device, deterministic):
        x = torch.randint(0, 16, (16, 32), device=device).float()
        v = torch.randn(16 * 64, 32, device=device)

        def fn(x, v):
            i = x[:, 0].long().clamp(0, 15).repeat(64)
            return x.index_put_((i,), v, accumulate=True)

        self.enterContext(DeterministicGuard(deterministic))
        self._check(fn, x, v)
        self.assertEqual(counters["inductor"]["scatter_preagg_materialized_keys"], 1)

    def test_materialized_unsupported_key(self, device):
        x = torch.zeros(16, 32, device=device)
        idx = _hot_indices(20_000, 16, device)
        values = torch.randn(20_016, 32, device=device)

        def fn(x, i, v):
            return torch.index_put(
                x, (torch.nn.functional.pad(i, (8, 8)),), v, accumulate=True
            )

        self._check(fn, x, idx, values)
        self.assertEqual(counters["inductor"]["scatter_preagg_materialized_keys"], 1)

    def test_materialized_unsupported_value(self, device):
        x = torch.zeros(16, 40, device=device)
        idx = torch.randint(0, 16, (20_000,), device=device)
        values = torch.randn(20_000, 32, device=device)

        def fn(x, i, v):
            return torch.index_put(
                x, (i,), torch.nn.functional.pad(v, (4, 4)), accumulate=True
            )

        self._check(fn, x, idx, values)
        self.assertEqual(counters["inductor"]["scatter_preagg_materialized_values"], 1)

    def test_output_view(self, device):
        def fn(i, v):
            base = v.new_zeros(64, 16)
            base[32:].t().index_add_(0, i, v)
            return base

        idx = _hot_indices(20_000, 16, device)
        v = torch.randn(20_000, 32, device=device)
        self._check(fn, idx, v, rtol=1e-3)

    def test_output_view_copy_on_unviewable_output(self, device):
        # An output that cannot be scattered through in place is scattered into
        # a copy, which is then copied back.
        output_storage = scatter_preagg._output_storage
        calls = []

        def first_call_fails(out, **kwargs):
            calls.append(out)
            return None if len(calls) == 1 else output_storage(out, **kwargs)

        x = torch.zeros(16, 32, device=device)
        idx = _hot_indices(20_000, 16, device)
        values = torch.randn(20_000, 32, device=device)
        with unittest.mock.patch.object(
            scatter_preagg, "_output_storage", first_call_fails
        ):
            self._check(
                lambda x, i, v: x.index_add_(0, i, v), x, idx, values, rtol=1e-3
            )
        self.assertEqual(counters["inductor"]["scatter_preagg_output_copy"], 1)

    @parametrize("dtype", [torch.float32, torch.bfloat16], name_fn=_dtype_name)
    def test_output_copies(self, device, dtype):
        # Many writes per element spread over copies of the output; rows no
        # token writes keep -0.0.
        n = 1 << 18
        x = torch.full((64, 32), -0.0, device=device, dtype=dtype)
        idx = _hot_indices(n, 32, device)
        v = torch.randn(n, 32, device=device, dtype=dtype)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        tol = 1e-3 if dtype == torch.float32 else 0.05
        code = self._check(fn, x, idx, v, atol=tol * 100, rtol=tol)
        self.assertIn("COPIES : tl.constexpr = 8", code)
        self.assertEqual(counters["inductor"]["scatter_preagg_copies"], 1)
        self.assertEqual(
            counters["inductor"]["scatter_preagg_fp32_copy"], dtype == torch.bfloat16
        )
        out = torch.compile(fn, fullgraph=True)(x, idx, v)
        self.assertTrue(out[32:].signbit().all())

    @parametrize("dtype", [torch.float32, torch.bfloat16], name_fn=_dtype_name)
    @parametrize(
        "rows,tokens",
        [(1 << 16, 1 << 18), (64, 50_000)],
        name_fn=lambda rows, tokens: "few_writes" if rows > 1024 else "few_tokens",
    )
    def test_no_output_copies(self, device, dtype, rows, tokens):
        # Few writes per element, or too few tokens to serialize a hot
        # destination's atomics, do not pay for the copies.
        x = torch.zeros(rows, 32, device=device, dtype=dtype)
        idx = _hot_indices(tokens, rows, device)
        v = torch.randn(tokens, 32, device=device, dtype=dtype)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        tol = 1e-3 if dtype == torch.float32 else 0.05
        self._check(fn, x, idx, v, atol=tol * 100, rtol=tol)
        self.assertEqual(counters["inductor"]["scatter_preagg_copies"], 0)

    def test_output_copies_dynamic_rows(self, device):
        # Symbolic sizes keep the output from being sized for copies.
        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        compiled = torch.compile(fn, fullgraph=True, dynamic=True)
        n = 1 << 18
        values = torch.randn(n, 32, device=device)
        for rows in (64, 48):
            x = torch.zeros(rows, 32, device=device)
            idx = _hot_indices(n, rows, device)
            self.assertEqual(
                compiled(x, idx, values), fn(x, idx, values), atol=0.1, rtol=1e-3
            )
        self.assertEqual(counters["stats"]["unique_graphs"], 1)
        self.assertEqual(counters["inductor"]["scatter_preagg_copies"], 0)

    def test_output_copies_single_column(self, device):
        # A single column's cheaper blocks pay for copies at fewer tokens.
        x = torch.zeros(64, device=device)
        idx = _hot_indices(1 << 14, 64, device)
        values = torch.randn(1 << 14, device=device)
        self._check(lambda x, i, v: x.index_add(0, i, v), x, idx, values, rtol=1e-3)
        self.assertEqual(counters["inductor"]["scatter_preagg_copies"], 1)

    def test_output_copies_scale_with_writes(self, device):
        x = torch.zeros(8192, 32, device=device)
        idx = _hot_indices(1 << 20, 8192, device)
        values = torch.randn(1 << 20, 32, device=device)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        code = self._check(fn, x, idx, values, atol=0.1, rtol=1e-3)
        self.assertIn("COPIES : tl.constexpr = 4", code)

    @config.patch(
        comprehensive_padding=True,
        padding_alignment_bytes=128,
        padding_stride_threshold=1024,
    )
    def test_output_copies_padded_stride(self, device):
        # The copies of a 1700-element output are padded to a stride of 1728,
        # which the template must address them by. Small integer values make
        # every summation order exact.
        n = 1 << 18
        x = torch.randint(-2, 3, (17, 100), device=device).float()
        idx = torch.randint(0, 17, (n,), device=device)
        v = torch.randint(-1, 2, (n, 100), device=device).float()

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

    @parametrize(
        "case",
        ["index_put", "index_add", "scatter_add", "one_key", "unique", "int32", "view"],
    )
    def test_deterministic(self, device, case):
        n, rows, d = 30_000, 50, 24
        dtype = torch.int32 if case == "int32" else torch.float32
        x = torch.zeros(rows, d, device=device, dtype=dtype)
        v = torch.randn(n, d, device=device).to(dtype)
        idx = _hot_indices(n, rows, device)
        fn = _SCATTER_FNS.get(case, _SCATTER_FNS["index_put"])
        if case == "one_key":
            # One segment spanning every chunk.
            idx = torch.zeros_like(idx)
        elif case == "unique":
            x = torch.zeros(n, d, device=device)
            idx = torch.randperm(n, device=device)
        elif case == "int32":
            v = torch.randint(-5, 5, (n, d), device=device, dtype=dtype)
        elif case == "view":
            x = torch.zeros(rows, 3, device=device)
            v = torch.randn(n, 1, device=device)

            def fn(x, i, v):
                return x[:, 1:2].index_add_(0, i, v)

        self._check_deterministic(fn, x, idx, v, tol=1e-5)

    def test_index_width_covers_tokens(self, device):
        # Broadcast indices into a small output can iterate past int32 tokens.
        n, rows = 50_000, 64
        x = torch.zeros(rows, rows, device=device)
        i0 = torch.randint(0, rows, (n, 1), device=device)
        i1 = torch.randint(0, rows, (1, n), device=device)
        v = torch.ones((), device=device)

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

    @parametrize("expanded", [False, True])
    def test_broadcast_values_dynamic_index(self, device, expanded):
        # Values with the index's dynamic leading size, and values expanded to
        # it, compile once for every size.
        def fn(out, i, v):
            return out.index_put([i], v, accumulate=True)

        def fn_expanded(out, i, seed):
            v = seed.expand(i.shape[0], i.shape[1], -1)
            return out.index_put([i], v, accumulate=True)

        f = fn_expanded if expanded else fn
        out = torch.zeros(512, 4, device=device)
        seed = torch.randn(1, 1, 4, device=device)
        compiled = torch.compile(f, fullgraph=True)
        for n in (16, 24):
            idx = torch.randint(0, 4, (n, 512), device=device)
            torch._dynamo.mark_dynamic(idx, 0)
            if expanded:
                v = seed
            else:
                v = torch.randn(n, 512, 4, device=device)
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
    def test_disabled(self, device):
        x = torch.zeros(24, 32, device=device)
        idx = _hot_indices(20_000, 24, device)
        v = torch.randn(20_000, 32, device=device)

        def fn(x, i, v):
            return torch.index_put(x, (i,), v, accumulate=True)

        self._check(fn, x, idx, v, applied=False)

    @parametrize("deterministic", [False, True])
    def test_epilogue_fusion_disabled(self, device, deterministic):
        # The template's own output is a placeholder; fusing a consumer onto it
        # would never generate the consumer's code.
        selected = []
        select = scatter_preagg.autotune_select_algorithm

        def spy(*args, **kwargs):
            result = select(*args, **kwargs)
            selected.append(result[0].data.data)
            return result

        x = torch.zeros(64, 32, device=device)
        idx = _hot_indices(20_000, 64, device)
        v = torch.randn(20_000, 32, device=device)
        self.enterContext(DeterministicGuard(deterministic))
        with unittest.mock.patch.object(
            scatter_preagg, "autotune_select_algorithm", spy
        ):
            self._check(
                lambda x, i, v: x.index_add(0, i, v).relu(), x, idx, v, atol=1e-3
            )
        self.assertEqual(len(selected), 2 if deterministic else 1)
        for template_buf in selected:
            self.assertIs(template_buf.allow_epilogue_fusion, False)

    @parametrize("deterministic", [False, True])
    def test_consumer_of_placeholder_size(self, device, deterministic):
        # The placeholder output holds (tokens + the largest block) * inner
        # elements; a pointwise consumer of the scattered output with exactly
        # that many must still see the scatter's result.
        block_t, rows, d = 256, 4096, 16
        x = torch.zeros(rows, d, device=device)
        idx = _hot_indices(rows - block_t, rows, device)
        v = torch.randn(rows - block_t, d, device=device)
        self.enterContext(DeterministicGuard(deterministic))
        with _with_configs(_cfg(block_t, 16, 1, 2)):
            self._check(
                lambda x, i, v: x.index_add(0, i, v).relu(), x, idx, v, atol=1e-3
            )


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

    def _assert_config_invariants(self, target, cfg, num_inner):
        self.assertEqual(cfg.block_t & (cfg.block_t - 1), 0)
        self.assertEqual(cfg.block_d & (cfg.block_d - 1), 0)
        self.assertGreaterEqual(cfg.block_t, 32)
        self.assertLessEqual(cfg.block_t, 1024)
        self.assertLessEqual(cfg.block_d, 1 << (num_inner - 1).bit_length())
        self.assertEqual((cfg.num_warps * target.warp_size) % target.warp_size, 0)
        if num_inner % cfg.block_d:
            # Ragged sort/scan tiles use about 16 bytes of LDS per element.
            self.assertLessEqual(
                cfg.block_t * cfg.block_d * 16, target.shared_mem_bytes
            )

    @parametrize("rocm", [False, True])
    def test_configs(self, rocm):
        target = ScatterPreaggTarget(64, 64 << 10, 80, fp16_matrix_preagg=True)
        default = self._configs(target, 50_000, 32, torch.float32, rocm=rocm)
        self.assertEqual(len(default), 1)
        cases = [
            (32, torch.float32, True),
            (128, torch.bfloat16, True),
            (512, torch.float32, True),
            (128, torch.float16, True),
            (512, torch.float16, True),
            (16, torch.float16, True),
            (256, torch.float32, False),
        ]
        for num_inner, dtype, contiguous in cases:
            (cfg,) = self._configs(
                target,
                50_000,
                num_inner,
                dtype,
                inner_contiguous=contiguous,
                rocm=rocm,
            )
            self._assert_config_invariants(target, cfg, num_inner)
            if rocm and target.matrix_preagg(dtype, num_inner, contiguous):
                self.assertGreaterEqual(cfg.block_t, target.matrix_min_block_t)
                self.assertLessEqual(cfg.block_t, target.matrix_max_block_t)
                self.assertGreaterEqual(cfg.block_d, target.matrix_min_block_d)
        tuned = self._configs(
            target, 1 << 20, 32, torch.float32, max_autotune=True, rocm=rocm
        )
        self.assertGreater(len(tuned), 1)
        self.assertEqual(
            tuned[0], self._configs(target, 1 << 20, 32, torch.float32, rocm=rocm)[0]
        )
        for cfg in tuned:
            self._assert_config_invariants(target, cfg, 32)
        # Ragged inner sizes keep the shared-memory-safe tile.
        ragged = self._configs(
            target, 1 << 20, 37, torch.float32, max_autotune=True, rocm=rocm
        )
        for cfg in ragged:
            self._assert_config_invariants(target, cfg, 37)

    def test_rocm_occupancy_configs(self):
        target = ScatterPreaggTarget(64, 64 << 10, 80, fp16_matrix_preagg=True)
        # Narrow fp32 rows take 256 threads and a register cap; wide slices
        # 256 threads.
        (cfg,) = self._configs(target, 1 << 20, 4, torch.float32)
        self.assertGreaterEqual(cfg.num_warps * target.warp_size, 128)
        self.assertGreater(cfg.waves_per_eu, 0)
        (cfg,) = self._configs(target, 1 << 20, 1024, torch.float32)
        self.assertGreaterEqual(cfg.block_d * cfg.d_iters, 256)
        self.assertGreaterEqual(cfg.num_warps * target.warp_size, 128)
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
        self.assertEqual(cfg.num_warps, 1)
        self.assertGreaterEqual(cfg.block_d, target.matrix_min_block_d)

    @parametrize(
        "warp_size,shared_mem_kb,num_sms",
        [(64, 64, 80), (64, 160, 256), (32, 320, 128), (32, 64, 96)],
        name_fn=lambda *a: "_".join(map(str, a)),
    )
    def test_target_configs(self, warp_size, shared_mem_kb, num_sms):
        # Configs keep the lanes per program across warp sizes, scale the ragged
        # tile with shared memory and the small-grid split with the SM count.
        target = ScatterPreaggTarget(warp_size, shared_mem_kb << 10, num_sms)
        (cfg,) = self._configs(target, 50_000, 32, torch.float32)
        self._assert_config_invariants(target, cfg, 32)
        tuned = self._configs(target, 1 << 20, 37, torch.float32, max_autotune=True)
        for candidate in tuned:
            self._assert_config_invariants(target, candidate, 37)
        # Small scatters keep at least two programs per SM.
        (cfg,) = self._configs(target, 256 * num_sms, 16, torch.float32)
        programs = math.ceil(256 * num_sms / cfg.block_t) * math.ceil(16 / cfg.block_d)
        self.assertGreaterEqual(programs, 2 * num_sms)

    def test_copy_tuning_on_target(self):
        # Output copy and sort-threshold choices belong to the target, so a
        # backend overrides them there.
        layout = types.SimpleNamespace(storage_size=lambda: 64 * 32)
        for target in (
            ScatterPreaggTarget(64, 64 << 10, 80, max_copies=2),
            ScatterPreaggTarget(64, 64 << 10, 80),
            ScatterPreaggTarget(64, 64 << 10, 80, max_copies=1),
        ):
            copies = scatter_preagg._num_copies(
                layout, torch.float32, 1 << 20, 32, target
            )
            self.assertEqual(copies & (copies - 1), 0)
            self.assertLessEqual(copies, target.max_copies)
            self.assertLessEqual(
                copies * layout.storage_size() * torch.float32.itemsize,
                target.max_copy_bytes,
            )

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

    def test_target_from_device(self, device):
        device = torch.device(device)
        props = torch.cuda.get_device_properties(device)
        target = V.choices.get_scatter_preagg_target(device)
        self.assertEqual(target.warp_size, props.warp_size)
        self.assertEqual(target.shared_mem_bytes, _shared_memory_per_block(props))
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
        with unittest.mock.patch.object(
            V.choices,
            "get_config_heuristics",
            return_value=ROCmConfigHeuristic(),
        ):
            self.assertTrue(enabled(gpu))
            self.assertFalse(enabled(torch.device("cpu")))
            self.assertFalse(enabled(torch.device("xpu", 0)))
            self.assertFalse(enabled(None))
            with config.patch(scatter_preagg_template=False):
                self.assertFalse(enabled(gpu))
        with unittest.mock.patch.object(
            V.choices,
            "get_config_heuristics",
            return_value=CUDAConfigHeuristic(),
        ):
            self.assertFalse(enabled(gpu))


_DEPRECATED_NO_OP_CONFIGS = [
    "partitioned_scatter_min_partitions",
    "partitioned_scatter_max_partitions",
    "partitioned_scatter_min_index_size",
    "partitioned_scatter_min_contention_ratio",
    "partitioned_scatter_non_model_floor_bytes",
    "partitioned_scatter_fp32_accumulation",
    "partitioned_scatter_force",
]


class TestScatterPreaggDeprecations(TestCase):
    """The partitioned scatter settings the template replaced. Each warns once
    per process, so the warnings are checked in fresh interpreters."""

    hw_classification = HardwareClassification.GENERIC

    def _run(self, script, **env_vars):
        env = {
            k: v
            for k, v in os.environ.items()
            if not k.startswith("TORCHINDUCTOR_") and k != "TORCH_LOGS"
        }
        env.update(env_vars)
        result = subprocess.run(
            [sys.executable, "-c", script],
            env=env,
            text=True,
            capture_output=True,
        )
        self.assertEqual(result.returncode, 0, msg=result.stderr)
        return result

    def _config_warning(self, access):
        """The FutureWarning messages of running ``access`` on the inductor config."""
        script = (
            "import warnings\n"
            "import torch._inductor.config as c\n"
            "with warnings.catch_warnings(record=True) as w:\n"
            "    warnings.simplefilter('always')\n"
            f"    {access}\n"
            "print(*[str(x.message) for x in w if x.category is FutureWarning], sep='\\n')"
        )
        return self._run(script).stdout.strip()

    @parametrize("value", [False, True])
    def test_enabled_aliases_template(self, value):
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", FutureWarning)
            with config.patch(scatter_preagg_template=not value):
                with config.patch(partitioned_scatter_enabled=value):
                    self.assertEqual(config.scatter_preagg_template, value)
            with config.patch(scatter_preagg_template=value):
                self.assertEqual(config.partitioned_scatter_enabled, value)

    def test_enabled_warns(self):
        message = self._config_warning("c.partitioned_scatter_enabled")
        self.assertIn("has aliased scatter_preagg_template since 2.16", message)
        self.assertIn("use that instead", message)
        self.assertIn("2.18", message)
        self.assertIn("https://github.com/pytorch/pytorch/issues/204051", message)
        self.assertEqual(message.count("removed"), 1)

    @parametrize("name", _DEPRECATED_NO_OP_CONFIGS)
    def test_no_op_configs_warn(self, name):
        message = self._config_warning(f"c.{name}")
        self.assertIn(f"{name} is deprecated and has done nothing since 2.16", message)
        self.assertIn("2.18", message)
        self.assertIn("https://github.com/pytorch/pytorch/issues/204051", message)
        # The config module adds when it will be removed.
        self.assertEqual(message.count("removed"), 1)

    def test_no_op_configs_warn_when_set(self):
        name = _DEPRECATED_NO_OP_CONFIGS[0]
        message = self._config_warning(f"c.{name} = 4")
        self.assertIn("has done nothing since 2.16", message)

    @parametrize(
        "legacy,new,expected",
        [
            (None, None, torch.version.hip is not None),
            (None, "1", True),
            ("0", None, False),
            ("1", None, True),
            ("true", None, False),
            ("0", "1", True),
            ("1", "0", False),
        ],
        name_fn=lambda legacy, new, expected: f"legacy_{legacy}_new_{new}",
    )
    def test_enable_env_vars(self, legacy, new, expected):
        # Only "1" enables, and the new name takes precedence.
        env_vars = {
            name: value
            for name, value in [
                ("TORCHINDUCTOR_PARTITIONED_SCATTER_ENABLED", legacy),
                ("TORCHINDUCTOR_SCATTER_PREAGG_TEMPLATE", new),
            ]
            if value is not None
        }
        out = self._run(
            "import torch._inductor.config as c; print(c.scatter_preagg_template)",
            **env_vars,
        )
        self.assertEqual(out.stdout.strip(), str(expected))
        # Importing config must not emit the deferred deprecation warning.
        self.assertNotIn(
            "TORCHINDUCTOR_PARTITIONED_SCATTER_ENABLED is deprecated", out.stderr
        )

    @parametrize(
        "suffix,effect",
        [
            ("ENABLED", "sets scatter_preagg_template unless"),
            ("FP32_ACCUMULATION", "does nothing"),
            ("FORCE", "does nothing"),
        ],
        name_fn=lambda suffix, effect: suffix,
    )
    def test_deprecated_env_var_warns(self, suffix, effect):
        name = f"TORCHINDUCTOR_PARTITIONED_SCATTER_{suffix}"
        out = self._run(
            "import torch\n"
            "torch.compile(lambda x: x + 1, backend='inductor')(torch.ones(1))",
            **{name: "1"},
        )
        self.assertIn(
            f"The environment variable {name} is deprecated since 2.16", out.stderr
        )
        self.assertIn(effect, out.stderr)
        self.assertIn("https://github.com/pytorch/pytorch/issues/204051", out.stderr)
        self.assertIn(os.path.join("torch", "_inductor", "config.py"), out.stderr)

    def test_deprecated_env_var_does_not_warn_during_import(self):
        self._run(
            "import torch",
            PYTHONWARNINGS="error::FutureWarning",
            TORCHINDUCTOR_PARTITIONED_SCATTER_FORCE="1",
            TORCH_LOGS="partitioned_scatter",
        )

    def test_deprecated_logging_artifact(self):
        message = "The 'partitioned_scatter' logging artifact is deprecated"
        compile = (
            "import torch\n"
            "torch.compile(lambda x: x + 1, backend='inductor')(torch.ones(1))"
        )
        out = self._run(compile, TORCH_LOGS="partitioned_scatter")
        self.assertIn(message, out.stderr)
        self.assertIn("Use 'scatter_preagg' instead", out.stderr)
        self.assertIn("https://github.com/pytorch/pytorch/issues/204051", out.stderr)
        out = self._run(compile, TORCH_LOGS="scatter_preagg")
        self.assertNotIn(message, out.stderr)


instantiate_device_type_tests(TestScatterPreagg, globals(), only_for="cuda")
instantiate_device_type_tests(TestScatterPreaggTarget, globals(), only_for="cuda")
instantiate_parametrized_tests(TestScatterPreaggConfigs)
instantiate_parametrized_tests(TestScatterPreaggDeprecations)


if __name__ == "__main__":
    from torch._inductor.test_case import run_tests

    run_tests()
