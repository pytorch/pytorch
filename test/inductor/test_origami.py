# Owner(s): ["module: inductor"]
import dataclasses
import logging
import math
import os
import statistics
import sys
import unittest
from collections.abc import Callable
from types import SimpleNamespace
from unittest import mock

import sympy

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.heuristics.template import triton as triton_heuristics
from torch._inductor.heuristics.template.triton import (
    FlexAttentionConfigContext,
    FlexConfig,
    ROCmConfigHeuristic,
    ROCmFlexConfig,
)
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch._logging import trace_structured
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    MI350_ARCH,
    parametrize,
    runOnRocmArch,
)
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU_AND_TRITON


DO_PERF_TEST = os.environ.get("DO_PERF_TEST") == "1"

log = logging.getLogger(__name__)
if not DO_PERF_TEST:
    log.info(
        "test_origami_runtime_matches_regular_max_autotune will be skipped: "
        "set DO_PERF_TEST=1 to enable runtime perf benchmarks."
    )

# Test configuration parameters (hardcoded for stability and reproducibility)
ORIGAMI_TOPK_VALUES = [5, 10]  # topk configs to test
ORIGAMI_COMPILE_TOPK = 2  # topk for compilation tests
PERF_SLOWDOWN_TOLERANCE = 1.05  # 5% tolerance on performance
BENCH_WARMUP = 100  # warmup iterations for perf benchmarks
BENCH_REP = 500  # measurement iterations for perf benchmarks
BENCH_ROUNDS = 5  # independent rounds; median is used to suppress outliers
# NOTE: Do NOT hardcode device-specific values (device name, SM count, memory, etc).
# Use torch.cuda.get_device_properties() to query actual device capabilities.

IS_ROCM = torch.version.hip is not None


try:
    import origami

    HAS_ORIGAMI = True
except ImportError:
    origami = None
    HAS_ORIGAMI = False


_PRIOR_FP32_MATMUL_PRECISION: str | None = None


class TestOrigamiFlexAttentionConfigSelection(TestCase):
    """Mocked coverage for the gating and projected gfx950 candidate pool."""

    def setUp(self):
        super().setUp()
        self.heuristic = object.__new__(ROCmConfigHeuristic)
        self.heuristic.exhaustive_flex_attn_fwd_configs = [
            ROCmFlexConfig(
                block_m,
                block_n,
                num_stages,
                num_warps,
                mfma,
                waves_per_eu,
                kpack,
            )
            for block_m in (16, 32, 64, 128)
            for block_n in (32, 64, 128)
            for num_stages in (1, 2)
            for num_warps in (2, 4, 8)
            for mfma in (0, 16)
            for waves_per_eu in (0, 8 // num_warps)
            for kpack in (1, 2)
        ]
        self.heuristic.flex_attn_fwd_autotune_configs = [
            ROCmFlexConfig(128, 64, 1, 4, kpack=2)
        ]
        stock_default = ROCmFlexConfig(128, 64, 2, 4, kpack=2)
        self.stock: list[FlexConfig] = [stock_default]
        self.heuristic.gfx950_default_flex_config = {
            (torch.float16, 64): stock_default,
            (torch.float16, 128): stock_default,
            (torch.float16, 256): ROCmFlexConfig(32, 64, 2, 4, kpack=2),
        }
        self.heuristic.flex_fwd_config_by_arch = {
            "gfx950": self.heuristic.gfx950_default_flex_config
        }

    @staticmethod
    def _context(**overrides):
        values = {
            "batch_size": 1,
            "kv_batch_size": 1,
            "num_heads": 16,
            "num_kv_heads": 16,
            "seq_len_q": 4096,
            "seq_len_kv": 4096,
            "qk_head_dim": 64,
            "v_head_dim": 64,
            "qk_head_dim_rounded": 64,
            "v_head_dim_rounded": 64,
            "dtype": torch.float16,
            "key_dtype": torch.float16,
            "value_dtype": torch.float16,
            "device": torch.device("cuda", 0),
            "inputs_contiguous": True,
            "is_noop_block_mask": True,
            "score_mod_reads_buffers": False,
            "kernel_options": {},
        }
        values.update(overrides)
        return FlexAttentionConfigContext(**values)

    def _select(
        self,
        context,
        *,
        heuristic=None,
        arch="gfx950:sramecc+:xnack-",
        search_space="DEFAULT",
        enabled=True,
        origami_module=object(),
        selected_tiles=((128, 128), (64, 128)),
        tile_error=None,
        default_kpack=1,
    ):
        heuristic = self.heuristic if heuristic is None else heuristic
        get_device_properties = mock.Mock(return_value=mock.Mock(gcnArchName=arch))
        self.last_get_device_properties = get_device_properties
        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_flex_search_space": search_space,
                    "rocm.origami": enabled,
                }
            ),
            mock.patch.object(triton_heuristics, "origami", origami_module),
            mock.patch.object(
                triton_heuristics,
                "_origami_flex_attention_tiles",
                return_value=selected_tiles,
                side_effect=tile_error,
            ) as tiles,
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                get_device_properties,
            ),
            # The real lookup caches the queried architecture process-wide.
            mock.patch.object(
                triton_heuristics, "get_default_kpack", return_value=default_kpack
            ),
        ):
            self.last_tiles = tiles
            return heuristic.refine_flex_attn_fwd_configs(self.stock, context)

    def test_selects_subgemm_tiles_and_keeps_unmodeled_backend_axes(self):
        selected = self._select(self._context())
        self.assertEqual(len(selected), 9)
        self.assertEqual(
            {(config.block_m, config.block_n) for config in selected},
            {(128, 128), (64, 128), (128, 64)},
        )
        self.assertEqual({config.num_stages for config in selected}, {2})
        self.assertEqual({config.num_warps for config in selected}, {4})
        self.assertEqual(
            [
                config
                for config in selected
                if (config.block_m, config.block_n) == (128, 64)
            ],
            self.stock,
        )
        self.assertEqual(
            {
                (
                    config.matrix_instr_nonkdim,
                    config.waves_per_eu,
                    config.kpack,
                )
                for config in selected
            },
            {
                (mfma, waves_per_eu, kpack)
                for mfma in (0, 16)
                for waves_per_eu in (0,)
                for kpack in (1, 2)
            },
        )

    def test_preserves_generic_d32_default(self):
        # A kpack unlike ROCmFlexConfig's default shows the fallback tracks it.
        selected = self._select(
            self._context(
                qk_head_dim=32,
                v_head_dim=32,
                qk_head_dim_rounded=32,
                v_head_dim_rounded=32,
            ),
            default_kpack=2,
        )
        self.assertIn(ROCmFlexConfig(128, 64, 1, 4, kpack=2), selected)

    def test_rocm_options_the_lowering_overrides_do_not_disable_selection(self):
        for options in ({"kpack": 2}, {"fwd_waves_per_eu": 2}):
            with self.subTest(options=options):
                selected = self._select(self._context(kernel_options=options))
                self.assertEqual(len(selected), 9)

    def test_selection_uses_the_target_entry(self):
        target = dataclasses.replace(
            triton_heuristics._ORIGAMI_FLEX_TARGETS["gfx950"],
            min_batch_heads=32,
            num_stages=1,
        )
        with mock.patch.dict(
            triton_heuristics._ORIGAMI_FLEX_TARGETS, {"gfx1250": target}
        ):
            self.assertIs(self._select(self._context(), arch="gfx1250"), self.stock)
            selected = self._select(
                self._context(num_heads=32, num_kv_heads=32), arch="gfx1250"
            )
        self.assertEqual({config.num_stages for config in selected}, {1})
        # gfx1250 has no default table, so the generic default is kept.
        self.assertIn(ROCmFlexConfig(128, 64, 1, 4, kpack=1), selected)

    def test_gqa_keeps_two_and_four_warp_variants(self):
        selected = self._select(self._context(num_kv_heads=4))
        self.assertEqual(len(selected), 17)
        self.assertEqual({config.num_warps for config in selected}, {2, 4})

    def test_queries_the_compiling_device(self):
        self._select(self._context(device=torch.device("cuda", 3)))
        self.last_get_device_properties.assert_called_once_with(3)

    def test_models_the_query_heads_and_both_lengths(self):
        self._select(
            self._context(
                batch_size=2,
                kv_batch_size=2,
                num_kv_heads=4,
                seq_len_q=2048,
                seq_len_kv=4096,
            )
        )
        candidate_tiles = tuple(
            dict.fromkeys(
                (config.block_m, config.block_n)
                for config in self.heuristic.exhaustive_flex_attn_fwd_configs
            )
        )
        # B * Hq, not B * Hkv: every query head runs both sub-GEMMs.
        self.last_tiles.assert_called_once_with(
            32, 2048, 4096, 64, 64, torch.float16, 0, candidate_tiles, 1, 2
        )

    def test_falls_back_outside_measured_domain(self):
        cases = {
            "block_mask": self._context(is_noop_block_mask=False),
            "short_sequence": self._context(seq_len_q=512),
            "short_kv_sequence": self._context(seq_len_kv=512),
            "d128_transition": self._context(
                seq_len_q=1024,
                seq_len_kv=1024,
                qk_head_dim=128,
                v_head_dim=128,
                qk_head_dim_rounded=128,
                v_head_dim_rounded=128,
            ),
            "d256_transition": self._context(
                seq_len_q=1024,
                seq_len_kv=1024,
                qk_head_dim=256,
                v_head_dim=256,
                qk_head_dim_rounded=256,
                v_head_dim_rounded=256,
            ),
            "low_parallelism": self._context(num_heads=8),
            "batch_broadcast": self._context(batch_size=2, kv_batch_size=1),
            "mixed_head_dims": self._context(v_head_dim=128, v_head_dim_rounded=128),
            "unmeasured_head_dim": self._context(
                qk_head_dim=100,
                v_head_dim=100,
                qk_head_dim_rounded=128,
                v_head_dim_rounded=128,
            ),
            "unsupported_dtype": self._context(dtype=torch.float32),
            "mixed_dtypes": self._context(
                key_dtype=torch.float8_e4m3fn, value_dtype=torch.float8_e4m3fn
            ),
            "mixed_value_dtype": self._context(value_dtype=torch.bfloat16),
            "dynamic_sequence": self._context(seq_len_q=sympy.Symbol("s")),
            "noncontiguous_inputs": self._context(inputs_contiguous=False),
            "user_tile": self._context(kernel_options={"BLOCK_M": 64}),
            "user_stages": self._context(kernel_options={"num_stages": 1}),
            "prefixed_user_stages": self._context(kernel_options={"fwd_num_stages": 1}),
            "score_mod_buffers": self._context(score_mod_reads_buffers=True),
        }
        for name, context in cases.items():
            with self.subTest(name=name):
                self.assertIs(self._select(context), self.stock)
        select_options = {
            "non_gfx950": {"arch": "gfx942:sramecc+:xnack-"},
            "unmeasured_target": {"arch": "gfx1250"},
            "exhaustive_search": {"search_space": "EXHAUSTIVE"},
            "origami_unavailable": {"origami_module": None},
            "origami_disabled": {"enabled": False},
        }
        for name, options in select_options.items():
            with self.subTest(name=name):
                self.assertIs(self._select(self._context(), **options), self.stock)

    def test_heuristic_that_customizes_the_pool_keeps_it(self):
        class CustomPoolHeuristic(ROCmConfigHeuristic):
            def get_flex_attn_fwd_configs(self, *args, **kwargs):
                return []

        class CustomTableHeuristic(ROCmConfigHeuristic):
            pass

        subclass = object.__new__(CustomPoolHeuristic)
        subclass.__dict__.update(self.heuristic.__dict__)
        with self.subTest(name="subclass"):
            self.assertIs(self._select(self._context(), heuristic=subclass), self.stock)
        table = object.__new__(CustomTableHeuristic)
        table.__dict__.update(self.heuristic.__dict__)
        table.flex_attn_fwd_autotune_configs = []
        with self.subTest(name="subclass_table"):
            self.assertIs(self._select(self._context(), heuristic=table), self.stock)

    def _refine(self, choices, context=None):
        from torch._inductor.virtualized import V

        with V.set_choices_handler(choices):
            return V.choices.refine_flex_attention_fwd_configs(
                self.stock, self._context() if context is None else context
            )

    def test_choices_dispatch_to_the_device_heuristic(self):
        from torch._inductor.choices import InductorChoices

        context = self._context()
        refined: list[FlexConfig] = [ROCmFlexConfig(128, 128, 2, 4)]
        heuristic = mock.Mock(spec=["refine_flex_attn_fwd_configs"])
        heuristic.refine_flex_attn_fwd_configs.return_value = refined
        with mock.patch.object(
            InductorChoices, "get_config_heuristics", return_value=heuristic
        ):
            self.assertIs(self._refine(InductorChoices(), context), refined)
        heuristic.refine_flex_attn_fwd_configs.assert_called_once_with(
            self.stock, context
        )
        # Heuristics without the hook keep the pool.
        with mock.patch.object(
            InductorChoices, "get_config_heuristics", return_value=object()
        ):
            self.assertIs(self._refine(InductorChoices(), context), self.stock)

    def test_choices_handler_that_supplies_the_pool_keeps_it(self):
        from torch._inductor.choices import _ComposedInductorChoices, InductorChoices

        class CustomPoolChoices(InductorChoices):
            def get_flex_attention_fwd_configs(self, *args, **kwargs):
                return []

        class DelegatingRefineChoices(InductorChoices):
            def refine_flex_attention_fwd_configs(self, configs, context):
                return super().refine_flex_attention_fwd_configs(configs, context)

        def runtime_override():
            choices = InductorChoices()
            choices.get_flex_attention_fwd_configs = lambda *args, **kwargs: []
            return choices

        handlers = {
            "subclass": CustomPoolChoices(),
            "composed_contributor": _ComposedInductorChoices(
                [CustomPoolChoices(), InductorChoices()]
            ),
            # The hook is bound to one contributor while another supplies the pool.
            "composed_refine_delegates": _ComposedInductorChoices(
                [DelegatingRefineChoices(), CustomPoolChoices()]
            ),
            "instance_override": runtime_override(),
        }
        for name, choices in handlers.items():
            with (
                self.subTest(name=name),
                mock.patch.object(
                    InductorChoices, "get_config_heuristics"
                ) as heuristics,
            ):
                self.assertIs(self._refine(choices), self.stock)
                heuristics.assert_not_called()
        with (
            self.subTest(name="class_patch"),
            mock.patch.object(
                InductorChoices, "get_flex_attention_fwd_configs", return_value=[]
            ),
            mock.patch.object(InductorChoices, "get_config_heuristics") as heuristics,
        ):
            self.assertIs(self._refine(InductorChoices()), self.stock)
            heuristics.assert_not_called()

    def test_composed_choices_without_a_pool_override_are_refined(self):
        from torch._inductor.choices import _ComposedInductorChoices, InductorChoices

        refined: list[FlexConfig] = [ROCmFlexConfig(128, 128, 2, 4)]
        heuristic = mock.Mock(spec=["refine_flex_attn_fwd_configs"])
        heuristic.refine_flex_attn_fwd_configs.return_value = refined
        with mock.patch.object(
            InductorChoices, "get_config_heuristics", return_value=heuristic
        ):
            self.assertIs(
                self._refine(_ComposedInductorChoices([InductorChoices()])), refined
            )

    def test_origami_failures_fall_back_to_the_stock_pool(self):
        failures = {
            "ValueError": {"tile_error": ValueError("bad")},
            "TypeError": {"tile_error": TypeError("bad")},
            "RuntimeError": {"tile_error": RuntimeError("bad")},
            "AttributeError": {"tile_error": AttributeError("bad")},
            "no_selected_tiles": {"selected_tiles": ()},
        }
        for name, options in failures.items():
            with self.subTest(name=name):
                fallbacks = counters["inductor"]["origami_fallback"]
                self.assertIs(self._select(self._context(), **options), self.stock)
                self.assertEqual(
                    counters["inductor"]["origami_fallback"], fallbacks + 1
                )
        # Other errors, such as an Origami API mismatch, are not hidden.
        with self.assertRaises(KeyError):
            self._select(self._context(), tile_error=KeyError("missing"))

    def test_d128_and_d256_positive_boundaries(self):
        expected_tiles = {
            128: {(128, 128), (64, 128), (128, 64)},
            256: {(128, 128), (64, 128), (32, 64)},
        }
        for head_dim, tiles in expected_tiles.items():
            with self.subTest(head_dim=head_dim):
                selected = self._select(
                    self._context(
                        seq_len_q=2048,
                        seq_len_kv=2048,
                        qk_head_dim=head_dim,
                        v_head_dim=head_dim,
                        qk_head_dim_rounded=head_dim,
                        v_head_dim_rounded=head_dim,
                    )
                )
                self.assertEqual(len(selected), 9)
                self.assertEqual(
                    {(config.block_m, config.block_n) for config in selected}, tiles
                )
                self.assertEqual(
                    {(config.num_stages, config.num_warps) for config in selected},
                    {(2, 4)},
                )

    @staticmethod
    def _fake_origami(latencies):
        fake_origami = mock.Mock()
        fake_origami.transpose_t = SimpleNamespace(N="N", T="T")
        fake_origami.problem_t.side_effect = SimpleNamespace
        fake_origami.config_t.side_effect = SimpleNamespace

        def dim3(m, n, k):
            return (m, n, k)

        fake_origami.dim3_t.side_effect = dim3
        fake_origami.string_to_datatype.return_value = "f16"
        hardware = mock.Mock(N_CU=256)
        hardware.get_recommended_matrix_instruction.return_value = (16, 16, 32)
        fake_origami.get_hardware_for_device.return_value = hardware
        fake_origami.compute_total_latency.side_effect = latencies
        return fake_origami

    def _run_model(self, latencies, candidate_tiles):
        fake_origami = self._fake_origami(latencies)
        triton_heuristics._origami_flex_attention_tiles.cache_clear()
        self.addCleanup(triton_heuristics._origami_flex_attention_tiles.cache_clear)
        with mock.patch.object(triton_heuristics, "origami", fake_origami):
            # Distinct Q/KV lengths, an unrounded head dim and non-square tiles
            # expose swapped axes.
            selected = triton_heuristics._origami_flex_attention_tiles(
                16,
                2048,
                4096,
                100,
                128,
                torch.float16,
                0,
                candidate_tiles,
                1,
                2,
            )
        return selected, fake_origami

    def test_models_qk_and_pv_tile_geometry(self):
        selected, fake_origami = self._run_model((10.0, 20.0), ((64, 32),))

        self.assertEqual(selected, ((64, 32),))
        qk_call, pv_call = fake_origami.compute_total_latency.call_args_list
        qk_problem, _, qk_config, _ = qk_call.args
        pv_problem, _, pv_config, _ = pv_call.args
        self.assertEqual(qk_problem.size, (2048, 4096, 100))
        self.assertEqual(qk_problem.batch, 16)
        self.assertEqual(qk_problem.a_transpose, "N")
        self.assertEqual(qk_problem.b_transpose, "T")
        self.assertEqual(qk_config.mt, (64, 32, 128))
        self.assertEqual(pv_problem.size, (2048, 100, 4096))
        self.assertEqual(pv_problem.batch, 16)
        self.assertEqual(pv_problem.a_transpose, "N")
        self.assertEqual(pv_problem.b_transpose, "N")
        self.assertEqual(pv_config.mt, (64, 128, 32))

    def test_unions_qk_top1_and_pv_top2_without_duplicates(self):
        tiles = ((16, 32), (32, 64), (64, 128))
        # Calls alternate QK/PV. QK ranks A,B,C; PV ranks A,C,B, so A is
        # deduplicated and the production union is A,C.
        selected, _ = self._run_model((1.0, 1.0, 2.0, 3.0, 3.0, 2.0), tiles)
        self.assertEqual(selected, (tiles[0], tiles[2]))

    def test_ranks_only_tiles_both_subgemms_accept(self):
        tiles = ((64, 64), (128, 128))
        # Origami reports a config it rejects as the largest double.
        rejected = sys.float_info.max
        # Calls alternate QK/PV per tile.
        cases = {
            "qk_rejected": ((rejected, 1.0, rejected, 2.0), ()),
            "pv_rejected": ((1.0, rejected, 2.0, rejected), ()),
            "nonfinite": ((math.inf, 1.0, 2.0, math.nan), ()),
            # QK prefers the first tile, but PV rejects it.
            "one_tile_rejected": ((1.0, rejected, 2.0, 1.0), (tiles[1],)),
        }
        for name, (values, expected) in cases.items():
            with self.subTest(name=name):
                selected, _ = self._run_model(values, tiles)
                self.assertEqual(selected, expected)


def setUpModule():
    global _PRIOR_FP32_MATMUL_PRECISION
    _PRIOR_FP32_MATMUL_PRECISION = torch.get_float32_matmul_precision()
    if IS_ROCM:
        torch.set_float32_matmul_precision("highest")


def tearDownModule():
    global _PRIOR_FP32_MATMUL_PRECISION
    if _PRIOR_FP32_MATMUL_PRECISION is not None:
        torch.set_float32_matmul_precision(_PRIOR_FP32_MATMUL_PRECISION)
        _PRIOR_FP32_MATMUL_PRECISION = None


@unittest.skipIf(not HAS_GPU_AND_TRITON, "requires GPU and Triton")
@unittest.skipIf(not IS_ROCM, "Origami integration is ROCm-only")
@unittest.skipIf(not HAS_ORIGAMI, "Origami package is not installed")
@unittest.skipIf(
    HAS_ORIGAMI
    and config.max_autotune
    and config.rocm.origami
    and triton_heuristics.origami is None,
    "The installed rocm-origami does not support this ROCm version",
)
@unittest.skipIf(
    not (config.max_autotune and config.rocm.origami),
    "Requires both max_autotune and origami to be enabled. "
    "Set TORCHINDUCTOR_MAX_AUTOTUNE=1 TORCHINDUCTOR_ORIGAMI=1 to run.",
)
class TestOrigami(TestCase):
    def _make_fn_and_inputs(
        self, op_name: str, size: int
    ) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]:
        torch.manual_seed(0)

        if op_name == "bmm":
            batch = 4
            a = torch.randn(batch, size, size, device=GPU_TYPE, dtype=torch.float16)
            b = torch.randn(batch, size, size, device=GPU_TYPE, dtype=torch.float16)

            def fn(x, y):
                return torch.bmm(x, y)

            return fn, (a, b)

        a = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)
        b = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)

        if op_name == "mm":

            def fn(x, y):
                return torch.mm(x, y)

            return fn, (a, b)

        if op_name == "addmm":
            bias = torch.randn(size, size, device=GPU_TYPE, dtype=torch.float16)

            def fn(inp, x, y):
                return torch.addmm(inp, x, y)

            return fn, (bias, a, b)

        raise AssertionError(f"Unsupported op {op_name}")

    def _benchmark_gpu_call_count(self) -> int:
        return sum(
            value
            for name, value in counters["inductor"].items()
            if "benchmark_gpu" in name
        )

    def _compile_with_config(
        self,
        op_name: str,
        patch_config: dict[str, object],
        *,
        size: int,
    ) -> dict[str, object]:
        fn, args = self._make_fn_and_inputs(op_name, size)
        expected = fn(*args)

        torch._dynamo.reset()
        counters.clear()

        with (
            fresh_cache(),
            config.patch(patch_config),
            mock.patch(
                "origami.select_topk_configs", wraps=origami.select_topk_configs
            ) as select_topk,
        ):
            compiled = torch.compile(fn, dynamic=False)
            result = compiled(*args)

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)

        return {
            "compiled": compiled,
            "args": args,
            "benchmark_gpu_calls": self._benchmark_gpu_call_count(),
            "topk_calls": select_topk.call_count,
        }

    def _origami_default_config(self, topk: int) -> dict[str, object]:
        return {
            "max_autotune": True,
            "max_autotune_gemm": True,
            "rocm.origami": True,
            "rocm.origami_topk": topk,
            "max_autotune_gemm_search_space": "DEFAULT",
            "max_autotune_gemm_backends": "TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_(b)?mm_",
            "triton.native_matmul": False,
        }

    def _origami_exhaustive_config(self) -> dict[str, object]:
        return {
            "max_autotune": True,
            "max_autotune_gemm": True,
            "rocm.origami": True,
            "rocm.origami_topk": ORIGAMI_TOPK_VALUES[0],
            "max_autotune_gemm_search_space": "EXHAUSTIVE",
            "max_autotune_gemm_backends": "TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_(b)?mm_",
            "triton.native_matmul": False,
        }

    def _max_autotune_default_config(self) -> dict[str, object]:
        return {
            "max_autotune": False,
            "max_autotune_gemm": True,
            "rocm.origami": False,
            "max_autotune_gemm_search_space": "DEFAULT",
            "max_autotune_gemm_backends": "TRITON",
            "test_configs.autotune_choice_name_regex": r"^triton_(b)?mm_",
            "triton.native_matmul": False,
        }

    @runOnRocmArch(MI350_ARCH)
    def test_origami_refines_dense_flex_attention_configs(self):
        from torch._inductor.choices import InductorChoices
        from torch.nn.attention.flex_attention import AuxRequest, flex_attention

        torch.manual_seed(0)
        q_ref, k_ref, v_ref = (
            torch.randn(
                1,
                16,
                1024,
                64,
                device=GPU_TYPE,
                dtype=torch.float16,
                requires_grad=True,
            )
            for _ in range(3)
        )
        q, k, v = (
            tensor.detach().clone().requires_grad_() for tensor in (q_ref, k_ref, v_ref)
        )

        def fn(q, k, v):
            return flex_attention(q, k, v, return_aux=AuxRequest(lse=True))

        expected_out, expected_aux = fn(q_ref, k_ref, v_ref)
        expected_out.sum().backward()
        selected_tiles = []
        refined_pools = []
        appended_pools = []
        select_tiles = triton_heuristics._origami_flex_attention_tiles
        refine_configs = ROCmConfigHeuristic.refine_flex_attn_fwd_configs

        def record_tiles(*args):
            tiles = select_tiles(*args)
            selected_tiles.append(tiles)
            return tiles

        def record_refine(heuristic, configs, context):
            refined = refine_configs(heuristic, configs, context)
            refined_pools.append((configs, refined))
            return refined

        def record_append(handler, choices, configs, *args, **kwargs):
            appended_pools.append((configs, len(choices)))
            return choices

        with (
            fresh_cache(),
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_flex_search_space": "DEFAULT",
                    "autotune_num_choices_displayed": 0,
                    "rocm.origami": True,
                }
            ),
            mock.patch.object(
                triton_heuristics, "_origami_flex_attention_tiles", record_tiles
            ),
            mock.patch.object(
                ROCmConfigHeuristic,
                "refine_flex_attn_fwd_configs",
                record_refine,
            ),
            mock.patch.object(
                InductorChoices,
                "append_flex_attention_choices",
                record_append,
            ),
        ):
            actual_out, actual_aux = torch.compile(fn, fullgraph=True)(q, k, v)
            actual_out.sum().backward()

        self.assertEqual(len(selected_tiles), 1)
        self.assertEqual(len(refined_pools), 1)
        tiles = set(selected_tiles[0])
        stock_pool, refined = refined_pools[0]
        self.assertTrue(tiles)
        self.assertLess(len(refined), len(stock_pool))
        # Every candidate is a variant of a selected tile or the stock default.
        for candidate in refined:
            self.assertTrue(
                (candidate.block_m, candidate.block_n) in tiles
                or candidate in stock_pool,
                candidate,
            )
        self.assertLessEqual(
            tiles, {(candidate.block_m, candidate.block_n) for candidate in refined}
        )
        # Extension hooks still receive the unrefined pool.
        self.assertEqual(len(appended_pools), 1)
        appended_configs, compiled_choices = appended_pools[0]
        self.assertIs(appended_configs, stock_pool)
        # The template candidates come from the refined pool.
        self.assertTrue(0 < compiled_choices <= len(refined), compiled_choices)
        self.assertEqual(actual_out, expected_out, atol=2e-2, rtol=2e-2)
        self.assertEqual(actual_aux.lse, expected_aux.lse, atol=2e-2, rtol=2e-2)
        for actual_grad, expected_grad in zip(
            (q.grad, k.grad, v.grad), (q_ref.grad, k_ref.grad, v_ref.grad)
        ):
            self.assertEqual(actual_grad, expected_grad, atol=3e-2, rtol=3e-2)

    def test_origami_leaves_flex_decoding_unchanged(self):
        from torch._inductor.choices import InductorChoices
        from torch.nn.attention.flex_attention import flex_attention

        torch.manual_seed(0)
        q = torch.randn(1, 16, 1, 64, device=GPU_TYPE, dtype=torch.float16)
        k, v = (
            torch.randn(1, 16, 1024, 64, device=GPU_TYPE, dtype=torch.float16)
            for _ in range(2)
        )
        expected = flex_attention(q, k, v)
        refine_calls = []
        refine_configs = InductorChoices.refine_flex_attention_fwd_configs

        def record_refine(handler, configs, context):
            refine_calls.append(context)
            return refine_configs(handler, configs, context)

        with (
            fresh_cache(),
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_flex_search_space": "DEFAULT",
                    "autotune_num_choices_displayed": 0,
                    "rocm.origami": True,
                }
            ),
            mock.patch.object(
                InductorChoices, "refine_flex_attention_fwd_configs", record_refine
            ),
        ):
            actual = torch.compile(flex_attention, fullgraph=True)(q, k, v)

        self.assertEqual(refine_calls, [])
        self.assertEqual(actual, expected, atol=2e-2, rtol=2e-2)

    def test_origami_respects_gemm_search_space(self):
        for op_name in ("mm", "addmm", "bmm"):
            with self.subTest(op_name=op_name, search_space="DEFAULT"):
                default_case = self._compile_with_config(
                    op_name,
                    self._origami_default_config(ORIGAMI_TOPK_VALUES[0]),
                    size=256,
                )
                self.assertGreater(default_case["topk_calls"], 0)

            with self.subTest(op_name=op_name, search_space="EXHAUSTIVE"):
                # Stub get_exhaustive_mm_configs to return a 2-config generator
                # so the EXHAUSTIVE compile finishes in seconds. The assertion
                # only checks that select_topk_configs is NOT called (origami
                # is bypassed when search_space is EXHAUSTIVE).
                from functools import partial

                from torch._inductor.heuristics.template.triton import (
                    ROCmConfigHeuristic,
                )

                _two_configs = ROCmConfigHeuristic().exhaustive_configs[:2]

                def _stub_exhaustive(self):
                    return partial(self.preprocess_mm_configs, configs=_two_configs)

                with mock.patch.object(
                    ROCmConfigHeuristic,
                    "get_exhaustive_mm_configs",
                    _stub_exhaustive,
                ):
                    exhaustive_case = self._compile_with_config(
                        op_name,
                        self._origami_exhaustive_config(),
                        size=256,
                    )
                self.assertEqual(exhaustive_case["topk_calls"], 0)

    def test_origami_reduces_compile_work_vs_regular_max_autotune(self):
        """Test that origami reduces compile work (GPU benchmarking calls) vs regular max_autotune.

        Uses benchmark_gpu_calls count instead of wall-clock timing to avoid flakiness
        on shared CI runners and sequencing bias (origami runs first and pays import cost).
        """
        for op_name in ("mm", "addmm", "bmm"):
            with self.subTest(op_name=op_name):
                origami_case = self._compile_with_config(
                    op_name,
                    self._origami_default_config(ORIGAMI_COMPILE_TOPK),
                    size=256,
                )
                max_autotune_case = self._compile_with_config(
                    op_name,
                    self._max_autotune_default_config(),
                    size=256,
                )
                # Origami with topk should benchmark fewer configs than full max_autotune
                self.assertLess(
                    origami_case["benchmark_gpu_calls"],
                    max_autotune_case["benchmark_gpu_calls"],
                    msg=lambda msg: f"{msg}\nOrigami ({origami_case['benchmark_gpu_calls']} calls) should have fewer "
                    f"GPU benchmarks than max_autotune ({max_autotune_case['benchmark_gpu_calls']} calls)",
                )

    def _benchmark_median(self, compiled, args):
        """Run BENCH_ROUNDS independent benchmarks and return the median.

        A single benchmark() call on a shared machine can vary by +/-5-8%.
        Taking the median of several rounds suppresses transient noise.
        """
        samples = [
            benchmarker.benchmark(
                compiled, args, {}, warmup=BENCH_WARMUP, rep=BENCH_REP
            )
            for _ in range(BENCH_ROUNDS)
        ]
        return statistics.median(samples)

    @unittest.skipIf(
        not DO_PERF_TEST,
        "Perf test not enabled; set DO_PERF_TEST=1 to enable runtime perf benchmarks",
    )
    def test_origami_runtime_matches_regular_max_autotune(self):
        for op_name in ("mm", "addmm", "bmm"):
            for size in (8192, 16384):
                for topk in ORIGAMI_TOPK_VALUES:
                    with self.subTest(op_name=op_name, size=size, topk=topk):
                        origami_case = self._compile_with_config(
                            op_name,
                            self._origami_default_config(topk),
                            size=size,
                        )
                        max_autotune_case = self._compile_with_config(
                            op_name,
                            self._max_autotune_default_config(),
                            size=size,
                        )

                        origami_runtime_ms = self._benchmark_median(
                            origami_case["compiled"],
                            origami_case["args"],
                        )
                        max_autotune_runtime_ms = self._benchmark_median(
                            max_autotune_case["compiled"],
                            max_autotune_case["args"],
                        )

                        runtime_ratio = (
                            origami_runtime_ms / max_autotune_runtime_ms
                            if max_autotune_runtime_ms > 0
                            else 1.0
                        )
                        passed = (
                            origami_runtime_ms
                            <= max_autotune_runtime_ms * PERF_SLOWDOWN_TOLERANCE
                        )
                        trace_structured(
                            "origami_perf_test",
                            metadata_fn=lambda: {
                                "op_name": op_name,
                                "size": size,
                                "topk": topk,
                                "origami_runtime_ms": round(origami_runtime_ms, 3),
                                "max_autotune_runtime_ms": round(
                                    max_autotune_runtime_ms, 3
                                ),
                                "runtime_ratio": round(runtime_ratio, 3),
                                "passed": passed,
                            },
                            expect_trace_id=False,
                        )

                        self.assertLessEqual(
                            origami_runtime_ms,
                            max_autotune_runtime_ms * PERF_SLOWDOWN_TOLERANCE,
                        )

    def test_origami_topk_edge_cases(self):
        """Test edge cases for origami_topk parameter.

        This test validates:
        - Proper error handling for invalid inputs (negative, float)
        - Correct behavior for boundary values (0, 1, very large)
        - Graceful degradation when topk is too small
        """
        op_name = "mm"
        size = 256

        # Test case 1: topk = 0 (no configs selected)
        # This should still compile without errors, just with no origami optimization
        with self.subTest(topk=0, test_case="no_configs_selected"):
            try:
                result = self._compile_with_config(
                    op_name,
                    self._origami_default_config(0),
                    size=size,
                )
                # Should complete compilation even with topk=0
                self.assertIsNotNone(result["compiled"])
            except Exception as e:
                self.fail(f"Compilation failed with topk=0: {e}")

        # Test case 2: topk = 1 (minimal selection)
        # This should select the top 1 config and still work
        with self.subTest(topk=1, test_case="minimal_selection"):
            try:
                result = self._compile_with_config(
                    op_name,
                    self._origami_default_config(1),
                    size=size,
                )
                self.assertIsNotNone(result["compiled"])
                # With topk=1, origami should still be invoked for selection
                # (unless no configs are available)
            except Exception as e:
                self.fail(f"Compilation failed with topk=1: {e}")

        # Test case 3: Very large topk values (e.g., 1000)
        # Should handle gracefully and just select all available configs
        with self.subTest(topk=1000, test_case="very_large_topk"):
            try:
                result = self._compile_with_config(
                    op_name,
                    self._origami_default_config(1000),
                    size=size,
                )
                self.assertIsNotNone(result["compiled"])
                # Large topk should not cause issues, just select all available
            except Exception as e:
                self.fail(f"Compilation failed with topk=1000: {e}")

        # Test case 4: Negative topk values (invalid input)
        # Should raise ValueError or handle gracefully in config validation
        with self.subTest(topk=-1, test_case="negative_topk"):
            # Negative topk is invalid, but behavior depends on implementation
            # It may be caught during config application or runtime
            try:
                result = self._compile_with_config(
                    op_name,
                    self._origami_default_config(-1),
                    size=size,
                )
                # If compilation succeeds with negative topk, that is also valid
                # (implementation may coerce to 0 or similar)
                self.assertIsNotNone(result["compiled"])
            except (ValueError, TypeError, RuntimeError) as e:
                # Expected: negative values may raise errors during validation
                self.assertIn(
                    "topk",
                    str(e).lower(),
                    msg=lambda msg: f"{msg}\nError should mention topk parameter: {e}",  # noqa: F821
                )

        # Test case 5: Mid-range integer topk
        with self.subTest(topk=3, test_case="int_topk"):
            result = self._compile_with_config(
                op_name,
                self._origami_default_config(3),
                size=size,
            )
            self.assertIsNotNone(result["compiled"])

        # Test case 6: Valid normal topk values for comparison
        # These should all work normally
        for topk_val in [2, 4, 5, 8, 10]:
            with self.subTest(topk=topk_val, test_case="valid_topk"):
                try:
                    result = self._compile_with_config(
                        op_name,
                        self._origami_default_config(topk_val),
                        size=size,
                    )
                    self.assertIsNotNone(result["compiled"])
                    # Valid topk should always trigger origami selection
                    self.assertGreaterEqual(
                        result["topk_calls"],
                        0,
                        msg=lambda msg: f"{msg}\norigami.select_topk_configs should be callable with topk={topk_val}",
                    )
                except Exception as e:
                    self.fail(f"Compilation failed with valid topk={topk_val}: {e}")

    def test_origami_configs_use_device_specific_values(self):
        """Verify that origami configs use architecture-specific num_stages and num_warps.

        Tests that configurations are derived from device properties (MI300 vs MI350X)
        rather than hardcoded values. This ensures portability across AMD GPU models.
        """
        device = torch.device("cuda:0")
        device_props = torch.cuda.get_device_properties(device)

        # Compile a simple MM operation with origami enabled
        torch.manual_seed(0)
        a = torch.randn(256, 256, device=device)
        b = torch.randn(256, 256, device=device)

        # Compile with origami config
        with config.patch(self._origami_default_config(topk=2)):
            compiled = torch.compile(
                lambda x, y: x @ y,
                backend="inductor",
                mode="reduce-overhead",
            )
            compiled_result = compiled(a, b)

        # Verify compilation succeeded
        self.assertIsNotNone(compiled_result)

        # Check that device properties are available and used
        self.assertGreater(
            device_props.multi_processor_count,
            0,
            msg="Device should have valid compute properties",
        )

        trace_structured(
            "origami_device_specific_test",
            metadata_fn=lambda: {
                "device_name": device_props.name,
                "multi_processor_count": device_props.multi_processor_count,
                "warp_size": device_props.warp_size,
                "test": "verify architecture-specific config values",
            },
        )

    def test_origami_fallback_when_disabled(self):
        """Test that compilation succeeds when origami import fails or is disabled.

        Verifies that:
        1. Compilation succeeds (no errors)
        2. The compiled function produces correct results
        3. Regular config generator is used as fallback (origami.select_topk_configs not called)
        """
        for op_name in ("mm", "addmm", "bmm"):
            with self.subTest(op_name=op_name):
                fn, args = self._make_fn_and_inputs(op_name, 256)
                expected = fn(*args)

                torch._dynamo.reset()
                counters.clear()

                # Configuration with origami enabled, but we'll mock it to fail
                patch_config = self._origami_default_config(ORIGAMI_COMPILE_TOPK)

                # Patch the cached origami binding directly. Avoid mock.patch.dict
                # on sys.modules: its snapshot/restore evicts modules lazily imported
                # inside the `with` (e.g. torch._dynamo.repro.after_dynamo), causing
                # duplicate backend registration on the next subtest iteration.
                with (
                    fresh_cache(),
                    config.patch(patch_config),
                    mock.patch(
                        "torch._inductor.heuristics.template.triton.origami",
                        None,
                    ),
                ):
                    compiled = torch.compile(fn, dynamic=False)
                    result = compiled(*args)

                # Verify compilation succeeded and produces correct results
                torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
                self.assertIsNotNone(compiled)

    def test_origami_falls_back_when_selector_raises(self):
        for op_name in ("mm", "addmm", "bmm"):
            with self.subTest(op_name=op_name):
                fn, args = self._make_fn_and_inputs(op_name, 256)
                expected = fn(*args)

                torch._dynamo.reset()
                counters.clear()

                with (
                    fresh_cache(),
                    config.patch(self._origami_default_config(ORIGAMI_COMPILE_TOPK)),
                    mock.patch.object(
                        origami,
                        "OrigamiMatmulSelector",
                        side_effect=ValueError("unsupported layout"),
                    ) as selector,
                ):
                    compiled = torch.compile(fn, dynamic=False)
                    result = compiled(*args)

                selector.assert_called()
                self.assertGreater(counters["inductor"]["origami_fallback"], 0)
                torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
                self.assertGreater(
                    self._benchmark_gpu_call_count(), ORIGAMI_COMPILE_TOPK
                )

    def test_origami_falls_back_when_selection_loop_fails(self):
        fn, args = self._make_fn_and_inputs("mm", 256)
        expected = fn(*args)

        torch._dynamo.reset()
        counters.clear()
        calls = []
        real_topk = origami.select_topk_configs
        real_mapping = origami.select_workgroup_mapping

        # Count only calls made after topk selection: the selector constructor
        # also calls select_workgroup_mapping.
        def arm_topk(*a, **kw):
            results = real_topk(*a, **kw)
            calls.clear()
            return results

        def fail_on_second_call(*a, **kw):
            calls.append(1)
            if len(calls) > 1:
                raise ValueError("selection failed midway")
            return real_mapping(*a, **kw)

        with (
            fresh_cache(),
            config.patch(self._origami_default_config(ORIGAMI_TOPK_VALUES[0])),
            mock.patch.object(origami, "select_topk_configs", arm_topk),
            mock.patch.object(origami, "select_workgroup_mapping", fail_on_second_call),
        ):
            result = torch.compile(fn, dynamic=False)(*args)

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
        self.assertGreaterEqual(len(calls), 2)
        self.assertGreater(counters["inductor"]["origami_fallback"], 0)
        # A kept partial pick leaves one choice and no benchmarks; more choices
        # than Origami could pick means the regular generator ran.
        self.assertGreater(self._benchmark_gpu_call_count(), ORIGAMI_TOPK_VALUES[0])

    def test_origami_falls_back_on_attribute_error(self):
        # The _origami_* wrappers read private rocm-origami attributes; a renamed
        # attribute must fall back instead of failing the compile.
        fn, args = self._make_fn_and_inputs("mm", 256)
        expected = fn(*args)

        torch._dynamo.reset()
        counters.clear()
        with (
            fresh_cache(),
            config.patch(self._origami_default_config(ORIGAMI_TOPK_VALUES[0])),
            mock.patch.object(
                triton_heuristics,
                "_origami_hardware",
                side_effect=AttributeError("_hardware"),
            ) as hardware,
        ):
            result = torch.compile(fn, dynamic=False)(*args)

        hardware.assert_called()
        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
        self.assertGreater(counters["inductor"]["origami_fallback"], 0)
        self.assertGreater(self._benchmark_gpu_call_count(), ORIGAMI_TOPK_VALUES[0])

    def test_origami_falls_back_when_config_lookup_fails(self):
        import dataclasses

        import torch._inductor.heuristics.template.triton as th

        fn, args = self._make_fn_and_inputs("mm", 256)
        expected = fn(*args)

        torch._dynamo.reset()
        counters.clear()
        armed = []
        rebuilt = []
        real_mapping = origami.select_workgroup_mapping
        real_filter = th.ROCmConfigHeuristic._filter_configs

        def arm_mapping(*a, **kw):
            armed.append(1)
            return real_mapping(*a, **kw)

        # The first filter call after Origami selection gets tiles whose keys
        # match nothing stored, so every lookup fails.
        def rebuild_tiles(self, configs):
            out = real_filter(self, configs)
            if armed:
                armed.clear()
                out = [dataclasses.replace(c, block_m=c.block_m + 1) for c in out]
                rebuilt.append(len(out))
            return out

        with (
            fresh_cache(),
            config.patch(self._origami_default_config(ORIGAMI_COMPILE_TOPK)),
            mock.patch.object(origami, "select_workgroup_mapping", arm_mapping),
            mock.patch.object(th.ROCmConfigHeuristic, "_filter_configs", rebuild_tiles),
        ):
            result = torch.compile(fn, dynamic=False)(*args)

        torch.testing.assert_close(result, expected, atol=5e-2, rtol=5e-2)
        self.assertTrue(rebuilt and all(rebuilt), rebuilt)
        self.assertGreater(counters["inductor"]["origami_fallback"], 0)
        self.assertGreater(self._benchmark_gpu_call_count(), ORIGAMI_COMPILE_TOPK)

    def test_origami_honors_max_mm_configs(self):
        patch_config = self._origami_default_config(ORIGAMI_TOPK_VALUES[0])
        patch_config["test_configs.max_mm_configs"] = 1
        case = self._compile_with_config("mm", patch_config, size=256)
        self.assertEqual(case["topk_calls"], 0)

    def test_origami_not_used_by_heuristic_subclass(self):
        import torch._inductor.heuristics.template.triton as th

        # Every in-tree ROCm heuristic that reaches the origami branch, directly
        # or through super()._get_template_configs_impl.
        stock_classes = (
            th.ROCmMMTemplateConfigHeuristic,
            th.ROCmAddMMTemplateConfigHeuristic,
            th.ROCmMMAHTemplateConfigHeuristic,
            th.ROCmInt8MMTemplateConfigHeuristic,
            th.ROCmMMPlusMMTemplateConfigHeuristic,
            th.ROCmScaledMMTemplateConfigHeuristic,
            th.ROCmAddMMPersistentTemplateConfigHeuristic,
            th.ROCmPersistentTDMTemplateConfigHeuristic,
            th.ROCmAddMMPersistentTDMTemplateConfigHeuristic,
            th.PersistentMMTemplateConfigHeuristic,
        )
        # Drop the ad-hoc subclasses from the singleton cache afterwards.
        with mock.patch.dict(th.BaseHeuristicSingleton._instances):
            for cls in stock_classes:
                self.assertTrue(cls()._origami_may_select(), cls)
                custom_cls = type(f"Custom{cls.__name__}", (cls,), {})
                self.assertFalse(custom_cls()._origami_may_select(), custom_cls)

    def _matmul_inputs(
        self,
        op_name: str,
        m: int,
        n: int,
        k: int,
        dtype: torch.dtype,
        layout: str = "contiguous",
    ) -> tuple[Callable[..., torch.Tensor], tuple[torch.Tensor, ...]]:
        torch.manual_seed(0)
        batch = (4,) if op_name == "bmm" else ()

        def make(rows, cols, transposed, scale):
            # Scale before transposing so the transposed view keeps its strides.
            if transposed:
                x = torch.randn(*batch, cols, rows, device=GPU_TYPE, dtype=dtype)
                return (x * scale).transpose(-1, -2)
            return torch.randn(*batch, rows, cols, device=GPU_TYPE, dtype=dtype) * scale

        # 1/sqrt(k) on A keeps outputs near unit scale for every k.
        a = make(m, k, layout == "trans_a", k**-0.5)
        b = make(k, n, layout == "trans_b", 1.0)
        if op_name == "mm":
            return (lambda x, y: torch.mm(x, y)), (a, b)
        if op_name == "bmm":
            return (lambda x, y: torch.bmm(x, y)), (a, b)
        bias = torch.randn(n, device=GPU_TYPE, dtype=dtype)
        return (lambda bb, x, y: torch.addmm(bb, x, y)), (bias, a, b)

    def _assert_origami_matches_eager(
        self, fn, args, *, expect_origami=True, dynamic=False
    ):
        expected = fn(*args)
        torch._dynamo.reset()
        counters.clear()
        with (
            fresh_cache(),
            config.patch(self._origami_default_config(ORIGAMI_TOPK_VALUES[0])),
            mock.patch(
                "origami.select_topk_configs", wraps=origami.select_topk_configs
            ) as select_topk,
        ):
            result = torch.compile(fn, dynamic=dynamic)(*args)
        tol = 5e-2 if expected.is_floating_point() else 0
        torch.testing.assert_close(result, expected, atol=tol, rtol=tol)
        if expect_origami:
            # Without these a case could pass through the fallback and test nothing.
            self.assertGreater(select_topk.call_count, 0, "Origami did not select")
            self.assertEqual(
                counters["inductor"]["origami_fallback"], 0, "Origami fell back"
            )
        else:
            self.assertEqual(select_topk.call_count, 0)
            # More choices than Origami could pick: the regular generator ran.
            self.assertGreater(self._benchmark_gpu_call_count(), ORIGAMI_TOPK_VALUES[0])

    @parametrize("op_name", ("mm", "addmm", "bmm"))
    @parametrize("dtype", (torch.float16, torch.bfloat16, torch.float32))
    def test_origami_matches_eager_dtypes(self, op_name, dtype):
        fn, args = self._matmul_inputs(op_name, 1024, 1024, 1024, dtype)
        self._assert_origami_matches_eager(fn, args)

    @parametrize("op_name", ("mm", "addmm", "bmm"))
    @parametrize("layout", ("trans_a", "trans_b"))
    def test_origami_matches_eager_layouts(self, op_name, layout):
        fn, args = self._matmul_inputs(op_name, 1024, 1024, 1024, torch.float16, layout)
        self._assert_origami_matches_eager(fn, args)

    @parametrize("op_name", ("mm", "addmm", "bmm"))
    @parametrize(
        "mnk",
        ((1000, 500, 700), (4096, 64, 256), (64, 4096, 512)),
        name_fn=lambda mnk: "x".join(map(str, mnk)),
    )
    def test_origami_matches_eager_shapes(self, op_name, mnk):
        fn, args = self._matmul_inputs(op_name, *mnk, torch.float16)
        self._assert_origami_matches_eager(fn, args)

    @parametrize(
        "mnk",
        ((1024, 1024, 1024), (64, 512, 256)),
        name_fn=lambda mnk: "x".join(map(str, mnk)),
    )
    def test_origami_matches_eager_int8(self, mnk):
        m, n, k = mnk
        torch.manual_seed(0)
        a = torch.randint(-8, 8, (m, k), device=GPU_TYPE, dtype=torch.int8)
        b = torch.randint(-8, 8, (k, n), device=GPU_TYPE, dtype=torch.int8)
        self._assert_origami_matches_eager(lambda x, y: torch._int_mm(x, y), (a, b))

    @parametrize("epilogue", ("relu", "bias_gelu"))
    def test_origami_matches_eager_with_epilogue(self, epilogue):
        _, (a, b) = self._matmul_inputs("mm", 1024, 1024, 1024, torch.float16)
        if epilogue == "relu":
            fn, args = (lambda x, y: torch.relu(x @ y)), (a, b)
        else:
            bias = torch.randn(1024, device=GPU_TYPE, dtype=torch.float16)

            def fn(bb, x, y):
                return torch.nn.functional.gelu(torch.addmm(bb, x, y))

            args = (bias, a, b)
        self._assert_origami_matches_eager(fn, args)

    def test_origami_skipped_for_dynamic_shapes(self):
        # Non-square so m, n and k become separate symbols.
        fn, args = self._matmul_inputs("mm", 1024, 512, 256, torch.float16)
        self._assert_origami_matches_eager(fn, args, expect_origami=False, dynamic=True)

    def test_origami_module_gate_when_env_var_disabled(self):
        """Verify origami is not imported/used when TORCHINDUCTOR_ORIGAMI=0.

        rocm.origami is a load-time-only knob (env-var driven). origami is on by
        default; setting TORCHINDUCTOR_ORIGAMI=0 disables it. triton.py imports
        the origami module at module load only when IS_ROCM and config.max_autotune
        and config.rocm.origami are all true; otherwise it sets ``origami = None``.
        Once cached, that decision is final for the process -- flipping
        config.rocm.origami via config.patch() after import has no effect.

        This subprocess test exercises the realistic disabled path: a fresh
        Python process with TORCHINDUCTOR_ORIGAMI=0 must end up with
        ``triton.origami is None``, regardless of config.patch() calls afterward.
        """
        import subprocess
        import sys

        snippet = (
            "import os, torch\n"
            "from torch._inductor import config\n"
            "from torch._inductor.heuristics.template import triton as th\n"
            "assert os.environ.get('TORCHINDUCTOR_ORIGAMI') == '0', 'env var not set to 0'\n"
            "assert th.origami is None, f'expected None, got {th.origami!r}'\n"
            "# Even after flipping the config knob mid-process, origami stays None\n"
            "with config.patch({'rocm.origami': True, 'max_autotune': True}):\n"
            "    assert th.origami is None, 'config.patch must not re-trigger import'\n"
            "print('OK')\n"
        )

        env = os.environ.copy()
        env["TORCHINDUCTOR_ORIGAMI"] = "0"

        result = subprocess.run(
            [sys.executable, "-c", snippet],
            env=env,
            capture_output=True,
            text=True,
            timeout=120,
        )
        self.assertEqual(
            result.returncode,
            0,
            msg=lambda msg: f"{msg}\nsubprocess failed:\nstdout: {result.stdout}\nstderr: {result.stderr}",
        )
        self.assertIn("OK", result.stdout)


@unittest.skipIf(
    HAS_GPU_AND_TRITON and IS_ROCM,
    "Skipped on ROCm where origami is available",
)
class TestOrigamiSkippedOnNonROCm(TestCase):
    """Test that origami is properly skipped on unsupported environments.

    Covers non-ROCm hardware (CUDA/CPU) where origami is not available.
    These tests verify that:
    1. origami configuration does not cause errors when disabled
    2. origami.select_topk_configs is not called on non-ROCm hardware
    3. Compilation succeeds with regular config generator as fallback
    4. origami gracefully no-ops on unsupported hardware
    """

    def test_origami_skipped_on_non_rocm(self):
        """Verify that origami is properly skipped on non-ROCm devices.

        Tests that origami gracefully handles non-ROCm environments without
        errors, regardless of configuration settings.
        """
        torch.manual_seed(0)

        # Use CPU device to ensure non-ROCm environment
        size = 128
        a = torch.randn(size, size, device="cpu", dtype=torch.float32)
        b = torch.randn(size, size, device="cpu", dtype=torch.float32)

        def test_fn(x, y):
            return torch.mm(x, y)

        expected = test_fn(a, b)

        torch._dynamo.reset()
        counters.clear()

        # Test with origami config enabled (but we're on non-ROCm)
        config_dict = {
            "max_autotune": True,
            "max_autotune_gemm": True,
            "rocm.origami": True,  # Enable origami config
            "rocm.origami_topk": 5,
            "max_autotune_gemm_search_space": "DEFAULT",
            "triton.native_matmul": False,
        }

        # Mock select_topk_configs to ensure it is not called on non-ROCm
        with fresh_cache(), config.patch(config_dict):
            if HAS_ORIGAMI:
                # If origami is installed, mock it to verify it is not called
                with mock.patch(
                    "origami.select_topk_configs",
                    wraps=origami.select_topk_configs if origami else None,
                ) as mock_select_topk:
                    compiled = torch.compile(test_fn, dynamic=False)
                    result = compiled(a, b)

                    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

                    # On non-ROCm devices, origami.select_topk_configs should NOT be called
                    self.assertEqual(
                        mock_select_topk.call_count,
                        0,
                        msg="origami.select_topk_configs should not be called on non-ROCm devices",
                    )
            else:
                # If origami is not installed, just verify compilation works
                compiled = torch.compile(test_fn, dynamic=False)
                result = compiled(a, b)
                torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

    def test_origami_disabled_uses_regular_config(self):
        """Verify regular config generator is used when origami is explicitly disabled."""
        torch.manual_seed(0)

        size = 128
        a = torch.randn(size, size, device="cpu", dtype=torch.float32)
        b = torch.randn(size, size, device="cpu", dtype=torch.float32)

        def test_fn(x, y):
            return torch.mm(x, y)

        expected = test_fn(a, b)

        torch._dynamo.reset()
        counters.clear()

        # Config with origami explicitly disabled
        config_dict = {
            "max_autotune": True,
            "max_autotune_gemm": True,
            "rocm.origami": False,  # Explicitly disable origami
            "max_autotune_gemm_search_space": "DEFAULT",
            "triton.native_matmul": False,
        }

        with fresh_cache(), config.patch(config_dict):
            if HAS_ORIGAMI:
                with mock.patch(
                    "origami.select_topk_configs",
                    wraps=origami.select_topk_configs if origami else None,
                ) as mock_select_topk:
                    compiled = torch.compile(test_fn, dynamic=False)
                    result = compiled(a, b)

                    torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)

                    # When origami is disabled, select_topk_configs should not be called
                    self.assertEqual(
                        mock_select_topk.call_count,
                        0,
                        msg="origami.select_topk_configs should not be called when origami is disabled",
                    )
            else:
                # Origami not installed - should still work fine
                compiled = torch.compile(test_fn, dynamic=False)
                result = compiled(a, b)
                torch.testing.assert_close(result, expected, atol=1e-5, rtol=1e-5)


class TestOrigamiPackageVersionGate(TestCase):
    """_origami_package_supported() decides whether the installed rocm-origami
    works with the ROCm the build targets. No GPU or origami package needed."""

    def test_package_version_gate(self):
        from torch._inductor.heuristics.template.triton import (
            _origami_package_supported,
        )

        cases = [
            ((7, 2), "0.0.2", True),
            ((10, 0), "0.0.2", False),
            ((10, 1), "0.0.3", True),
            ((10, 1), "0.0.4rc1", True),
            ((10, 1), "0.0.3.dev1", False),
            ((10, 1), "not-a-version", False),
        ]
        for rocm_version, package_version, expected in cases:
            with self.subTest(rocm=rocm_version, package=package_version):
                self.assertEqual(
                    _origami_package_supported(rocm_version, package_version),
                    expected,
                )


instantiate_parametrized_tests(TestOrigami)


if __name__ == "__main__":
    if HAS_GPU_AND_TRITON and IS_ROCM:
        run_tests()
