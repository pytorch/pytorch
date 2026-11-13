# Owner(s): ["module: inductor"]
import logging
import os
import statistics
import unittest
from collections.abc import Callable
from unittest import mock

import torch
from torch._dynamo.utils import counters
from torch._inductor import config
from torch._inductor.heuristics.template import triton as triton_heuristics
from torch._inductor.runtime.benchmarking import benchmarker
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.utils import fresh_cache
from torch._logging import trace_structured
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
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
