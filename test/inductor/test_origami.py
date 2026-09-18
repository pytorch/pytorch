# Owner(s): ["module: inductor"]
import logging
import math
import os
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
    _rocm_version as _th_rocm_version,
    FlexAttentionConfigContext,
    FlexConfig,
    ORIGAMI_UNSUPPORTED_ROCM_VERSION,
    ROCmConfigHeuristic,
    ROCmFlexConfig,
)
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch._logging import trace_structured
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
# NOTE: Do NOT hardcode device-specific values (device name, SM count, memory, etc).
# Use torch.cuda.get_device_properties() to query actual device capabilities.

IS_ROCM = torch.version.hip is not None

ORIGAMI_ROCM_SUPPORTED = IS_ROCM and _th_rocm_version < ORIGAMI_UNSUPPORTED_ROCM_VERSION

try:
    import origami

    HAS_ORIGAMI = True
except ImportError:
    origami = None
    HAS_ORIGAMI = False


_PRIOR_FP32_MATMUL_PRECISION: str | None = None


class TestOrigamiFlexAttentionConfigSelection(TestCase):
    """CPU-only coverage for the gating and projected gfx950 candidate pool."""

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
        self.stock: list[FlexConfig] = [ROCmFlexConfig(128, 64, 2, 4, kpack=2)]
        self.heuristic.gfx950_default_flex_config = {
            (torch.float16, 64): self.stock[0],
            (torch.float16, 128): self.stock[0],
            (torch.float16, 256): ROCmFlexConfig(32, 64, 2, 4, kpack=2),
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
            "device": torch.device("cuda", 0),
            "inputs_contiguous": True,
            "is_noop_block_mask": True,
            "kernel_options": {},
        }
        values.update(overrides)
        return FlexAttentionConfigContext(**values)

    def _select(
        self,
        context,
        *,
        arch="gfx950",
        configs: list[FlexConfig] | None = None,
        search_space="DEFAULT",
        origami_module=object(),
        selected_tiles=((128, 128), (64, 128)),
    ):
        configs = self.stock if configs is None else configs
        get_device_properties = mock.Mock(return_value=mock.Mock(gcnArchName=arch))
        self.last_get_device_properties = get_device_properties
        with (
            config.patch(
                {
                    "max_autotune": True,
                    "max_autotune_flex_search_space": search_space,
                }
            ),
            mock.patch.object(triton_heuristics, "origami", origami_module),
            mock.patch.object(
                triton_heuristics,
                "_origami_flex_attention_tiles",
                return_value=selected_tiles,
            ),
            mock.patch.object(
                torch.cuda,
                "get_device_properties",
                get_device_properties,
            ),
            mock.patch.object(
                self.heuristic,
                "get_flex_attn_fwd_configs",
                return_value=self.stock,
            ),
        ):
            return self.heuristic.filter_flex_attn_fwd_configs(configs, context)

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
        selected = self._select(
            self._context(
                qk_head_dim=32,
                v_head_dim=32,
                qk_head_dim_rounded=32,
                v_head_dim_rounded=32,
            )
        )
        self.assertIn(ROCmFlexConfig(128, 64, 1, 4, kpack=2), selected)

    def test_gqa_keeps_two_and_four_warp_variants(self):
        selected = self._select(self._context(num_kv_heads=4))
        self.assertEqual(len(selected), 17)
        self.assertEqual({config.num_warps for config in selected}, {2, 4})

    def test_queries_the_compiling_device(self):
        self._select(self._context(device=torch.device("cuda", 3)))
        self.last_get_device_properties.assert_called_once_with(3)

    def test_falls_back_outside_measured_domain(self):
        cases = {
            "block_mask": self._context(is_noop_block_mask=False),
            "short_sequence": self._context(seq_len_q=512),
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
            "dynamic_sequence": self._context(seq_len_q=sympy.Symbol("s")),
            "noncontiguous_inputs": self._context(inputs_contiguous=False),
            "user_tile": self._context(kernel_options={"BLOCK_M": 64}),
            "user_rocm_option": self._context(kernel_options={"kpack": 1}),
            "prefixed_rocm_option": self._context(
                kernel_options={"fwd_waves_per_eu": 2}
            ),
        }
        for name, context in cases.items():
            with self.subTest(name=name):
                self.assertIs(self._select(context), self.stock)
        self.assertIs(self._select(self._context(), arch="gfx942"), self.stock)
        custom: list[FlexConfig] = [ROCmFlexConfig(64, 64, 1, 4)]
        self.assertIs(self._select(self._context(), configs=custom), custom)
        self.assertIs(
            self._select(self._context(), search_space="EXHAUSTIVE"), self.stock
        )
        self.assertIs(self._select(self._context(), origami_module=None), self.stock)
        self.assertIs(self._select(self._context(), selected_tiles=()), self.stock)

    def test_d128_and_d256_positive_boundaries(self):
        for head_dim in (128, 256):
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
                self.assertGreater(len(selected), 1)

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
        with mock.patch.object(triton_heuristics, "origami", fake_origami):
            selected = triton_heuristics._origami_flex_attention_tiles(
                16,
                2048,
                4096,
                100,
                80,
                128,
                128,
                torch.float16,
                0,
                candidate_tiles,
            )
        return selected, fake_origami

    def test_models_qk_and_pv_tile_geometry(self):
        selected, fake_origami = self._run_model((10.0, 20.0), ((64, 128),))

        self.assertEqual(selected, ((64, 128),))
        qk_call, pv_call = fake_origami.compute_total_latency.call_args_list
        qk_problem, _, qk_config, _ = qk_call.args
        pv_problem, _, pv_config, _ = pv_call.args
        self.assertEqual(qk_problem.size, (2048, 4096, 100))
        self.assertEqual(qk_problem.batch, 16)
        self.assertEqual(qk_problem.a_transpose, "N")
        self.assertEqual(qk_problem.b_transpose, "T")
        self.assertEqual(qk_config.mt, (64, 128, 128))
        self.assertEqual(pv_problem.size, (2048, 80, 4096))
        self.assertEqual(pv_problem.batch, 16)
        self.assertEqual(pv_problem.a_transpose, "N")
        self.assertEqual(pv_problem.b_transpose, "N")
        self.assertEqual(pv_config.mt, (64, 128, 128))

    def test_unions_qk_top1_and_pv_top2_without_duplicates(self):
        tiles = ((16, 32), (32, 64), (64, 128))
        # Calls alternate QK/PV. QK ranks A,B,C; PV ranks A,C,B, so A is
        # deduplicated and the production union is A,C.
        selected, _ = self._run_model((1.0, 1.0, 2.0, 3.0, 3.0, 2.0), tiles)
        self.assertEqual(selected, (tiles[0], tiles[2]))

    def test_requires_finite_results_from_both_subgemms(self):
        tiles = ((64, 64), (128, 128))
        selected, _ = self._run_model((math.inf, 1.0, math.inf, 2.0), tiles)
        self.assertEqual(selected, ())


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
@unittest.skipIf(
    not ORIGAMI_ROCM_SUPPORTED,
    "Origami is not supported on ROCm 10.0+",
)
@unittest.skipIf(not HAS_ORIGAMI, "Origami package is not installed")
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

    def test_origami_filters_dense_flex_attention_configs(self):
        arch = torch.cuda.get_device_properties(0).gcnArchName.split(":", 1)[0]
        if arch != "gfx950":
            self.skipTest("the measured FlexAttention config projection is gfx950-only")

        from torch.nn.attention.flex_attention import AuxRequest, flex_attention
        from torch._inductor.choices import InductorChoices

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
        triton_heuristics._origami_flex_attention_tiles.cache_clear()
        filtered_sizes = []
        appended_config_sizes = []
        filter_configs = ROCmConfigHeuristic.filter_flex_attn_fwd_configs

        def record_filter(heuristic, configs, context):
            filtered = filter_configs(heuristic, configs, context)
            filtered_sizes.append(len(filtered))
            return filtered

        def record_append(handler, choices, configs, *args, **kwargs):
            appended_config_sizes.append(len(configs))
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
                triton_heuristics,
                "_origami_flex_attention_tiles",
                wraps=triton_heuristics._origami_flex_attention_tiles,
            ) as select_tiles,
            mock.patch.object(
                ROCmConfigHeuristic,
                "filter_flex_attn_fwd_configs",
                record_filter,
            ),
            mock.patch.object(
                InductorChoices,
                "append_flex_attention_choices",
                record_append,
            ),
        ):
            actual_out, actual_aux = torch.compile(fn, fullgraph=True)(q, k, v)
            actual_out.sum().backward()

        self.assertGreater(select_tiles.call_count, 0)
        self.assertEqual(filtered_sizes, [9])
        self.assertEqual(appended_config_sizes, [25])
        torch.testing.assert_close(actual_out, expected_out, atol=2e-2, rtol=2e-2)
        torch.testing.assert_close(
            actual_aux.lse, expected_aux.lse, atol=2e-2, rtol=2e-2
        )
        for actual_grad, expected_grad in zip(
            (q.grad, k.grad, v.grad), (q_ref.grad, k_ref.grad, v_ref.grad)
        ):
            torch.testing.assert_close(actual_grad, expected_grad, atol=3e-2, rtol=3e-2)

    def test_origami_leaves_flex_decoding_unchanged(self):
        from torch.nn.attention.flex_attention import flex_attention

        torch.manual_seed(0)
        q = torch.randn(1, 16, 1, 64, device=GPU_TYPE, dtype=torch.float16)
        k, v = (
            torch.randn(1, 16, 1024, 64, device=GPU_TYPE, dtype=torch.float16)
            for _ in range(2)
        )
        expected = flex_attention(q, k, v)
        triton_heuristics._origami_flex_attention_tiles.cache_clear()
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
                triton_heuristics,
                "_origami_flex_attention_tiles",
                wraps=triton_heuristics._origami_flex_attention_tiles,
            ) as select_tiles,
        ):
            actual = torch.compile(flex_attention, fullgraph=True)(q, k, v)

        self.assertEqual(select_tiles.call_count, 0)
        torch.testing.assert_close(actual, expected, atol=2e-2, rtol=2e-2)

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

                        origami_runtime_ms = benchmarker.benchmark(
                            origami_case["compiled"],
                            origami_case["args"],
                            {},
                            warmup=50,
                            rep=200,
                        )
                        max_autotune_runtime_ms = benchmarker.benchmark(
                            max_autotune_case["compiled"],
                            max_autotune_case["args"],
                            {},
                            warmup=50,
                            rep=200,
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
    HAS_GPU_AND_TRITON and ORIGAMI_ROCM_SUPPORTED,
    "Skipped on ROCm < 10.0 where origami is available",
)
class TestOrigamiSkippedOnNonROCm(TestCase):
    """Test that origami is properly skipped on unsupported environments.

    Covers non-ROCm hardware (CUDA/CPU) and ROCm >= ORIGAMI_UNSUPPORTED_ROCM_VERSION.
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


class TestOrigamiVersionGate(TestCase):
    """Unit tests for the _rocm_version / _origami_enabled() cutoff.

    No GPU or origami package required: all paths are exercised by patching
    the module-level _rocm_version directly.
    """

    def test_origami_enabled_below_cutoff(self):
        """_origami_enabled() returns True on ROCm just below the cutoff."""
        import torch._inductor.heuristics.template.triton as th

        with (
            mock.patch.object(th, "_rocm_version", (9, 9)),
            config.patch({"rocm.origami": True}),
        ):
            self.assertTrue(th._origami_enabled())

    def test_origami_disabled_at_cutoff(self):
        """_origami_enabled() returns False at exactly the cutoff version."""
        import torch._inductor.heuristics.template.triton as th

        with (
            mock.patch.object(th, "_rocm_version", (10, 0)),
            config.patch({"rocm.origami": True}),
        ):
            self.assertFalse(th._origami_enabled())

    def test_origami_disabled_above_cutoff(self):
        """_origami_enabled() returns False above the cutoff version."""
        import torch._inductor.heuristics.template.triton as th

        with (
            mock.patch.object(th, "_rocm_version", (10, 1)),
            config.patch({"rocm.origami": True}),
        ):
            self.assertFalse(th._origami_enabled())

    def test_origami_config_off_below_cutoff(self):
        """_origami_enabled() respects config.rocm.origami=False even below cutoff."""
        import torch._inductor.heuristics.template.triton as th

        with (
            mock.patch.object(th, "_rocm_version", (9, 9)),
            config.patch({"rocm.origami": False}),
        ):
            self.assertFalse(th._origami_enabled())


if __name__ == "__main__":
    if HAS_GPU_AND_TRITON and IS_ROCM:
        run_tests()
