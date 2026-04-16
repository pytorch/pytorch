# Owner(s): ["module: inductor"]

import sys
import unittest
from unittest import mock

import torch
from torch._inductor.runtime import triton_heuristics
from torch._inductor.runtime.hints import TRITON_MAX_BLOCK
from torch._inductor.test_case import run_tests, TestCase
from torch.testing._internal.common_utils import IS_LINUX
from torch.testing._internal.inductor_utils import GPU_TYPE, HAS_GPU


try:
    import triton  # @manual
except ImportError:
    if __name__ == "__main__":
        sys.exit(0)
    raise unittest.SkipTest("requires triton")  # noqa: B904

from torch._inductor import config
from torch._inductor.runtime.coordinate_descent_tuner import CoordescTuner


config.benchmark_kernel = True
config.coordinate_descent_tuning = True

orig_compare_config = CoordescTuner.compare_config


def mock_compare_config_prefer_larger_XBLOCK(
    self, func, candidate_config, best_config, best_timing
):
    """
    self is the CoordescTuner object
    """
    if "XBLOCK" in candidate_config.kwargs:
        if "XBLOCK" not in best_config.kwargs:
            raise AssertionError
        if candidate_config.kwargs["XBLOCK"] < best_config.kwargs["XBLOCK"]:
            func(candidate_config)  # run func so the launcher will be created
            return False, best_timing * 1.1
        elif candidate_config.kwargs["XBLOCK"] > best_config.kwargs["XBLOCK"]:
            func(candidate_config)
            return True, best_timing * 0.9

    return orig_compare_config(self, func, candidate_config, best_config, best_timing)


class TestCoordinateDescentTuner(TestCase):
    def test_abs_function(self):
        """
        The benchmark result is simply abs(XBLOCK - 15)
        """
        tuner = CoordescTuner()
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            return abs(config.kwargs["XBLOCK"] - 15)

        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 16, str(best_config))

    def test_no_neighbors(self):
        """
        Test the case that there is no available neighbor values for a field.
        """

        # size hint for x being 1 limits the max XBLOCK we try to be 1
        tuner = CoordescTuner(size_hints={"x": 1})
        baseline_config = triton.Config({"XBLOCK": 1}, num_warps=8, num_stages=1)

        def func(config):
            return abs(config.kwargs["XBLOCK"] - 15)

        best_config = tuner.autotune(func, baseline_config)
        self.assertTrue(best_config.kwargs.get("XBLOCK") == 1, str(best_config))

    def test_get_neighbour_values(self):
        tuner = CoordescTuner()

        neighbours = tuner.get_neighbour_values("num_stages", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 3, 4})
        neighbours = tuner.get_neighbour_values("num_warps", 2, radius=2)
        self.assertEqual(set(neighbours), {1, 4, 8})

    def test_persistent_reduction(self):
        def f(x):
            return x / x.sum(dim=-1, keepdim=True)

        with mock.patch.object(
            CoordescTuner, "compare_config", mock_compare_config_prefer_larger_XBLOCK
        ):
            x = torch.ones(2, 256).to(GPU_TYPE)
            expected = f(x)
            # the first call get correct result when cache miss. Don't know why yet
            _ = torch.compile(f)(x)
            actual = torch.compile(f)(x)
            self.assertTrue(
                torch.allclose(expected, actual, atol=1e-4, rtol=1e-4),
                f"Expected:\n{expected}\nActual:\n{actual}",
            )

    def test_value_too_large(self):
        # Simulate a reduction
        size_hints = {"x": 2**20, "y": 2**20}

        tuner = CoordescTuner(size_hints=size_hints)

        max_block = TRITON_MAX_BLOCK
        self.assertFalse(tuner.value_too_large("XBLOCK", max_block["X"]))
        self.assertTrue(tuner.value_too_large("XBLOCK", max_block["X"] * 2))
        self.assertFalse(tuner.value_too_large("R0_BLOCK", max_block["R0_"]))
        self.assertTrue(tuner.value_too_large("R0_BLOCK", max_block["R0_"] * 2))

    def test_value_too_large_combo_field_limits(self):
        tuner = CoordescTuner(
            size_hints={"x": 2**20, "r0_": 2**20},
            inductor_meta={
                "combo_coordesc_field_limits": {
                    "XBLOCK_0": 64,
                    "XBLOCK_1": 256,
                    "R0_BLOCK_1": 128,
                }
            },
        )

        self.assertFalse(tuner.value_too_large("XBLOCK_0", 64))
        self.assertTrue(tuner.value_too_large("XBLOCK_0", 128))
        self.assertFalse(tuner.value_too_large("XBLOCK_1", 256))
        self.assertTrue(tuner.value_too_large("XBLOCK_1", 512))
        self.assertFalse(tuner.value_too_large("R0_BLOCK_1", 128))
        self.assertTrue(tuner.value_too_large("R0_BLOCK_1", 256))

    def test_combo_metadata_orders_larger_subkernels_first_for_coordesc(self):
        def make_configs(xblock, yblock):
            return [
                triton.Config(
                    {"XBLOCK": xblock, "YBLOCK": yblock},
                    num_warps=4,
                    num_stages=1,
                )
            ]

        inductor_meta = {
            "combo_grid_meta": {
                "num_kernels": 3,
                "heuristic_0": "pointwise",
                "heuristic_1": "pointwise",
                "heuristic_2": "pointwise",
                "size_hints_0": {"x": 64, "y": 64},
                "size_hints_1": {"x": 256, "y": 256},
                "size_hints_2": {"x": 128, "y": 16},
                "tile_hint_0": "TileHint.SQUARE",
                "tile_hint_1": "TileHint.SQUARE",
                "tile_hint_2": "TileHint.SQUARE",
                "no_x_dim_0": False,
                "no_x_dim_1": False,
                "no_x_dim_2": False,
            }
        }

        configs_by_size = {
            (64, 64): make_configs(64, 32),
            (256, 256): make_configs(256, 64),
            (128, 16): make_configs(128, 16),
        }

        def pointwise_side_effect(size_hints, *args, **kwargs):
            return configs_by_size[(size_hints["x"], size_hints["y"])]

        with mock.patch.object(
            triton_heuristics,
            "pointwise",
            side_effect=pointwise_side_effect,
        ):
            configs = triton_heuristics._handle_combo_kernel_per_subkernel_blocks(
                {"x": 256, "y": 256},
                inductor_meta,
                triton_meta={},
            )

        self.assertIsNotNone(configs)
        self.assertEqual(
            inductor_meta["combo_coordesc_field_order"],
            [
                "XBLOCK_1",
                "YBLOCK_1",
                "XBLOCK_0",
                "YBLOCK_0",
                "XBLOCK_2",
                "YBLOCK_2",
            ],
        )
        self.assertEqual(
            inductor_meta["combo_coordesc_field_limits"],
            {
                "XBLOCK_0": 64,
                "YBLOCK_0": 64,
                "XBLOCK_1": 256,
                "YBLOCK_1": 256,
                "XBLOCK_2": 128,
                "YBLOCK_2": 16,
            },
        )


if __name__ == "__main__":
    if IS_LINUX and HAS_GPU:
        run_tests()
