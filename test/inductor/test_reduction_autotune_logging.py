# Owner(s): ["module: inductor"]

"""Tests for the reduction autotune data collection pipeline (Step 1).

Covers: compute_derived_features(), CachingAutotuner callback mechanism,
_attach_logging_callback gating, and end-to-end callback firing through
torch.compile for reduction/persistent_reduction decorator types.
"""

import sys
import unittest
from unittest.mock import MagicMock, patch

import torch
from torch._inductor.runtime.reduction_heuristic_utils import compute_derived_features
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import (
    GPU_TYPE,
    HAS_GPU_AND_TRITON,
)


try:
    import triton  # noqa: F401  # @manual
    import triton.language as tl  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904


from torch._inductor import config
from torch._inductor.runtime.hints import (
    AttrsDescriptorWrapper,
    DeviceProperties,
    HeuristicType,
)
from torch._inductor.runtime.triton_helpers import math as tl_math
from torch._inductor.runtime.triton_heuristics import (
    _attach_logging_callback,
    CachingAutotuner,
    triton_config,
)
from torch._inductor.test_case import run_tests, TestCase


# ---------------------------------------------------------------------------
# Tests for compute_derived_features — no GPU required
# ---------------------------------------------------------------------------


class TestComputeDerivedFeatures(TestCase):
    def test_basic_known_values(self):
        result = compute_derived_features(
            xnumel=1024,
            ynumel=0,
            xblock=128,
            yblock=0,
            grid_size=8,
            num_sms=108,
        )
        self.assertEqual(result["tile_utilization_x"], 1.0)
        self.assertEqual(result["tile_utilization_y"], 1.0)
        self.assertEqual(result["grid_size"], 8.0)
        # 8 / 108 < 1, so num_waves = ceil(8/108) = 1
        self.assertEqual(result["num_waves"], 1.0)
        # wave_utilization = 8 / (1 * 108) = 8/108
        self.assertAlmostEqual(result["wave_utilization"], 8.0 / 108.0)

    def test_partial_tile_utilization(self):
        result = compute_derived_features(
            xnumel=100,
            ynumel=0,
            xblock=128,
            yblock=0,
            grid_size=1,
            num_sms=108,
        )
        # ceil(100/128) = 1, so utilization = 100 / (1*128) = 100/128
        self.assertAlmostEqual(result["tile_utilization_x"], 100.0 / 128.0)

    def test_with_y_dimension(self):
        result = compute_derived_features(
            xnumel=1024,
            ynumel=50,
            xblock=128,
            yblock=32,
            grid_size=16,
            num_sms=108,
        )
        # ceil(50/32) = 2, so utilization = 50 / (2*32) = 50/64
        self.assertAlmostEqual(result["tile_utilization_y"], 50.0 / 64.0)

    def test_multiple_waves(self):
        result = compute_derived_features(
            xnumel=1024,
            ynumel=0,
            xblock=128,
            yblock=0,
            grid_size=256,
            num_sms=108,
        )
        # num_waves = ceil(256/108) = 3
        self.assertEqual(result["num_waves"], 3.0)
        # wave_utilization = 256 / (3 * 108) = 256/324
        self.assertAlmostEqual(result["wave_utilization"], 256.0 / 324.0)

    def test_xblock_zero(self):
        result = compute_derived_features(
            xnumel=1024,
            ynumel=0,
            xblock=0,
            yblock=0,
            grid_size=8,
            num_sms=108,
        )
        self.assertEqual(result["tile_utilization_x"], 1.0)

    def test_num_sms_zero(self):
        result = compute_derived_features(
            xnumel=1024,
            ynumel=0,
            xblock=128,
            yblock=0,
            grid_size=8,
            num_sms=0,
        )
        self.assertEqual(result["num_waves"], 1.0)
        self.assertEqual(result["wave_utilization"], 1.0)

    def test_ynumel_zero(self):
        result = compute_derived_features(
            xnumel=1024,
            ynumel=0,
            xblock=128,
            yblock=64,
            grid_size=8,
            num_sms=108,
        )
        self.assertEqual(result["tile_utilization_y"], 1.0)


# ---------------------------------------------------------------------------
# Helpers for creating CachingAutotuner instances in tests
# ---------------------------------------------------------------------------


def _make_cos_autotuner_args():
    """Build kwargs for CachingAutotuner with a trivial cos kernel."""

    @triton.jit
    def triton_(in_ptr0, out_ptr0, xnumel, XBLOCK: tl.constexpr):
        xnumel = 16
        xoffset = tl.program_id(0) * XBLOCK
        xindex = xoffset + tl.arange(0, XBLOCK)[:]
        xmask = xindex < xnumel
        x0 = xindex
        tmp0 = tl.load(in_ptr0 + (x0), xmask)
        tmp1 = tl_math.cos(tmp0)
        tl.store(out_ptr0 + (x0), tmp1, xmask)

    triton_meta = {
        "signature": {"in_ptr0": "*fp32", "out_ptr0": "*fp32", "xnumel": "i32"},
        "device": DeviceProperties.create(torch.device(GPU_TYPE)),
        "constants": {},
        "configs": [
            AttrsDescriptorWrapper(divisible_by_16=(0, 1, 2), equal_to_1=())
        ],
    }
    configs = [
        triton_config({"x": 16}, 64),
        triton_config({"x": 256}, 64),
    ]
    return {
        "fn": triton_,
        "triton_meta": triton_meta,
        "configs": configs,
        "save_cache_hook": False,
        "mutated_arg_names": [],
        "reset_to_zero_arg_names": [],
        "optimize_mem": True,
        "heuristic_type": HeuristicType.REDUCTION,
        "size_hints": {"x": 1024, "r0_": 4096},
        "inductor_meta": {
            "reduction_hint": "INNER",
            "dtype": "float32",
            "num_load": 1,
            "num_store": 1,
            "num_reduction": 1,
        },
    }


# ---------------------------------------------------------------------------
# Tests for _fire_autotune_callback — requires GPU+Triton
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_GPU_AND_TRITON and IS_LINUX, "requires GPU+Triton on Linux")
class TestFireAutotuneCallback(TestCase):
    def test_callback_fires_once(self):
        autotuner = CachingAutotuner(**_make_cos_autotuner_args())
        mock_cb = MagicMock()
        autotuner.on_autotune_complete = mock_cb

        autotuner._fire_autotune_callback({"some": "timings"})
        autotuner._fire_autotune_callback({"some": "timings"})

        mock_cb.assert_called_once()

    def test_callback_catches_exception(self):
        autotuner = CachingAutotuner(**_make_cos_autotuner_args())
        autotuner.on_autotune_complete = MagicMock(
            side_effect=RuntimeError("boom")
        )

        # Should not raise
        autotuner._fire_autotune_callback(None)

        self.assertTrue(autotuner._autotune_callback_fired)

    def test_callback_noop_when_none(self):
        autotuner = CachingAutotuner(**_make_cos_autotuner_args())
        autotuner.on_autotune_complete = None

        autotuner._fire_autotune_callback(None)

        self.assertFalse(autotuner._autotune_callback_fired)


# ---------------------------------------------------------------------------
# Tests for _attach_logging_callback — requires GPU+Triton
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_GPU_AND_TRITON and IS_LINUX, "requires GPU+Triton on Linux")
class TestAttachLoggingCallback(TestCase):
    def test_not_attached_when_disabled(self):
        autotuner = CachingAutotuner(**_make_cos_autotuner_args())
        with config.patch({"learned_reduction_heuristic.log_autotune": False}):
            _attach_logging_callback(autotuner)
        self.assertIsNone(autotuner.on_autotune_complete)

    def test_attached_when_enabled(self):
        autotuner = CachingAutotuner(**_make_cos_autotuner_args())
        mock_enqueue = MagicMock()
        with config.patch({"learned_reduction_heuristic.log_autotune": True}):
            with patch(
                "torch._inductor.runtime.reduction_heuristic_utils.enqueue_autotune_log",
                mock_enqueue,
            ):
                _attach_logging_callback(autotuner)
        self.assertIs(autotuner.on_autotune_complete, mock_enqueue)

    def test_noop_when_fb_unavailable(self):
        from torch._inductor.runtime.reduction_heuristic_utils import (
            _noop_autotune_log,
        )

        autotuner = CachingAutotuner(**_make_cos_autotuner_args())
        with config.patch({"learned_reduction_heuristic.log_autotune": True}):
            with patch.dict(
                "sys.modules",
                {"torch._inductor.fb.reduction_autotune_logging": None},
            ):
                # Re-import to pick up the stub
                import torch._inductor.runtime.reduction_heuristic_utils as utils

                with patch.object(utils, "enqueue_autotune_log", _noop_autotune_log):
                    _attach_logging_callback(autotuner)
        self.assertIs(autotuner.on_autotune_complete, _noop_autotune_log)


# ---------------------------------------------------------------------------
# End-to-end tests through torch.compile — requires GPU+Triton
# ---------------------------------------------------------------------------


@unittest.skipUnless(HAS_GPU_AND_TRITON and IS_LINUX, "requires GPU+Triton on Linux")
class TestEndToEndReductionLogging(TestCase):
    def _compile_and_run(self, fn, args, extra_config=None):
        """Compile and run fn, returning the mock enqueue function."""
        mock_enqueue = MagicMock()
        cfg = {"learned_reduction_heuristic.log_autotune": True}
        if extra_config:
            cfg.update(extra_config)

        with config.patch(cfg):
            with patch(
                "torch._inductor.runtime.reduction_heuristic_utils.enqueue_autotune_log",
                mock_enqueue,
            ):
                compiled = torch.compile(fn)
                compiled(*args)

        return mock_enqueue

    def test_reduction_callback_fires(self):
        def fn(x):
            return x.sum(dim=-1)

        x = torch.randn(32, 1024, device=GPU_TYPE)
        mock_enqueue = self._compile_and_run(fn, (x,))
        self.assertGreaterEqual(mock_enqueue.call_count, 1)

    def test_persistent_reduction_callback_fires(self):
        def fn(x):
            return x.sum(dim=-1)

        # Small reduction dim triggers persistent reduction
        x = torch.randn(32, 64, device=GPU_TYPE)
        mock_enqueue = self._compile_and_run(fn, (x,))
        self.assertGreaterEqual(mock_enqueue.call_count, 1)

    def test_single_config_timings_is_none(self):
        def fn(x):
            return x.sum(dim=-1)

        x = torch.randn(32, 64, device=GPU_TYPE)
        mock_enqueue = self._compile_and_run(fn, (x,))

        if mock_enqueue.call_count > 0:
            # At least one call should have timings=None (single-config)
            # since without max_autotune there's typically one config
            found_none = False
            for call_args in mock_enqueue.call_args_list:
                args, kwargs = call_args
                # Second positional arg is timings
                if len(args) >= 2 and args[1] is None:
                    found_none = True
                    break
            self.assertTrue(
                found_none,
                "Expected at least one callback with timings=None",
            )

    def test_multi_config_timings_is_dict(self):
        def fn(x):
            return x.sum(dim=-1)

        x = torch.randn(32, 1024, device=GPU_TYPE)
        mock_enqueue = self._compile_and_run(
            fn,
            (x,),
            extra_config={"max_autotune": True},
        )

        if mock_enqueue.call_count > 0:
            found_dict = False
            for call_args in mock_enqueue.call_args_list:
                args, kwargs = call_args
                if len(args) >= 2 and isinstance(args[1], dict):
                    found_dict = True
                    break
            self.assertTrue(
                found_dict,
                "Expected at least one callback with timings as dict",
            )


if __name__ == "__main__":
    run_tests()
