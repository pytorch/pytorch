# Owner(s): ["oncall: distributed"]

"""
Tests the collection-time GPU requirement resolver and MultiGpuMinFilterPlugin
used by --multigpu-min-gpus.

The requirement comes only from decorator stamps (including through
__wrapped__). Class world_size is not a GPU requirement: on a 4-GPU runner a
literal 4 is indistinguishable from capacity-scaled NUM_DEVICES/DEVICE_COUNT
values that still run on 2 GPUs. distributed_4gpu relies on decorator stamps:
--multigpu-min-gpus 3 keeps tests that declare >= 3 GPUs and drops the rest.

TestSkippedReason covers the `_skipped_reason` stamps that let launchers skip
before spawning ranks.
"""

import os
import sys
import tempfile
import types
import unittest
from unittest.mock import patch

import torch
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    nccl_skip_if_lt_x_gpu,
    require_n_gpus_for_nccl_backend,
    requires_world_size,
    skip_if_lt_x_gpu,
    skip_if_no_gpu,
    skip_if_small_worldsize,
    TEST_SKIPS,
)
from torch.testing._internal.common_utils import IS_SANDCASTLE, run_tests, TestCase


# The marker resolver lives in test/conftest.py. Make it importable whether this
# file is run under pytest (test/ already on sys.path) or directly.
_TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TEST_ROOT not in sys.path:
    sys.path.insert(0, _TEST_ROOT)

from conftest import (
    _decorator_gpu_requirement,
    _resolve_gpu_requirement,
    MultiGpuMinFilterPlugin,
)


def _fake_item(func=None):
    return types.SimpleNamespace(obj=func)


class TestMultiGpuMarker(TestCase):
    def test_decorator_requirement(self):
        @skip_if_lt_x_gpu(4)
        def needs4(self):
            pass

        @requires_world_size(3)
        def needs3(self):
            pass

        @require_n_gpus_for_nccl_backend(4, "nccl")
        def nccl_needs4(self):
            pass

        @nccl_skip_if_lt_x_gpu("nccl", 8)
        def nccl_needs8(self):
            pass

        def plain(self):
            pass

        self.assertEqual(_decorator_gpu_requirement(needs4), 4)
        self.assertEqual(_decorator_gpu_requirement(needs3), 3)
        self.assertEqual(_decorator_gpu_requirement(nccl_needs4), 4)
        self.assertEqual(_decorator_gpu_requirement(nccl_needs8), 8)
        self.assertEqual(_decorator_gpu_requirement(plain), 0)
        self.assertEqual(_decorator_gpu_requirement(None), 0)

    def test_undeclared_requirement_is_zero(self):
        # No stamp means no declared requirement, so no threshold is cleared.
        def plain(self):
            pass

        self.assertEqual(_resolve_gpu_requirement(_fake_item(plain)), 0)
        self.assertEqual(_resolve_gpu_requirement(_fake_item(None)), 0)

    def test_decorator_is_the_only_signal(self):
        # Guards the over-selection fix: capacity-scaled bases (FSDPTest,
        # DTensorTestBase) report the runner's device count as world_size, so a
        # skip_if_lt_x_gpu(2) test must stay at 2 however large the runner is.
        @skip_if_lt_x_gpu(2)
        def needs2(self):
            pass

        self.assertEqual(_resolve_gpu_requirement(_fake_item(needs2)), 2)

    def test_real_gloo_gpu_decorator(self):
        from distributed.test_c10d_gloo import DistributedDataParallelTest

        item = _fake_item(DistributedDataParallelTest.test_gloo_backend_2gpu_module)
        self.assertEqual(_resolve_gpu_requirement(item), 4)

    def test_real_fully_shard_2d_training(self):
        from distributed._composable.test_composability.test_2d_composability import (
            TestFullyShard2DTraining,
        )

        item = _fake_item(TestFullyShard2DTraining.test_train_parity_2d_mlp)
        self.assertEqual(_resolve_gpu_requirement(item), 4)

    def test_class_world_size_is_not_a_requirement(self):
        class Fixed:
            world_size = 4

            def test_fixed(self):
                pass

        fixed = _fake_item(Fixed.test_fixed)
        fixed.cls = Fixed
        self.assertEqual(_resolve_gpu_requirement(fixed), 0)

        calls = []

        class _CountingWorldSize(property):
            def __get__(self, obj, objtype=None):
                calls.append("world_size")
                return 4

        class Scaled:
            world_size = _CountingWorldSize()

            def test_scaled(self):
                pass

        scaled = _fake_item(Scaled.test_scaled)
        scaled.cls = Scaled
        self.assertEqual(_resolve_gpu_requirement(scaled), 0)
        self.assertEqual(calls, [])

    def test_skip_unless_torch_gpu_stamps_floor_not_capacity(self):
        from torch.testing._internal.distributed._tensor import common_dtensor

        with patch.object(common_dtensor, "NUM_DEVICES", 4):

            @common_dtensor.skip_unless_torch_gpu
            def needs_gpu(self):
                pass

        self.assertEqual(_resolve_gpu_requirement(_fake_item(needs_gpu)), 2)
        self.assertEqual(needs_gpu._min_gpus_required, 2)

    def test_requires_multi_gpu_stamps_four(self):
        from distributed.pipelining.test_dtensor_pp_integration import (
            _requires_multi_gpu,
        )

        @_requires_multi_gpu
        def needs4(self):
            pass

        self.assertEqual(_resolve_gpu_requirement(_fake_item(needs4)), 4)
        self.assertEqual(needs4._min_gpus_required, 4)

    def test_writes_final_selection_count(self):
        with tempfile.TemporaryDirectory() as tmp:
            count_file = os.path.join(tmp, "counts")
            with patch.dict(
                os.environ, {"PYTORCH_MULTIGPU_SELECTION_COUNT_FILE": count_file}
            ):
                MultiGpuMinFilterPlugin(3).pytest_collection_finish(
                    types.SimpleNamespace(
                        items=[object(), object()],
                        config=types.SimpleNamespace(getoption=lambda _: -1),
                    )
                )
            with open(count_file) as fp:
                self.assertEqual(fp.read(), "2\n")


def _noop(self):
    pass


class TestSkippedReason(TestCase):
    """Decorators set `_skipped_reason` when every rank would skip."""

    def test_small_worldsize(self):
        with patch.dict(os.environ, {"BACKEND": "gloo", "WORLD_SIZE": "2"}):
            self.assertEqual(
                skip_if_small_worldsize(_noop)._skipped_reason,
                TEST_SKIPS["small_worldsize"].message,
            )
        for env in ({"BACKEND": "gloo", "WORLD_SIZE": "8"}, {"BACKEND": "mpi"}):
            with patch.dict(os.environ, env):
                self.assertFalse(
                    hasattr(skip_if_small_worldsize(_noop), "_skipped_reason")
                )

    def test_no_accelerator(self):
        with patch.object(torch.accelerator, "current_accelerator", return_value=None):
            self.assertEqual(
                skip_if_no_gpu(_noop)._skipped_reason,
                TEST_SKIPS["no_accelerator"].message,
            )
            self.assertEqual(
                skip_if_lt_x_gpu(2)(_noop)._skipped_reason,
                TEST_SKIPS["multi-device-2"].message,
            )
            self.assertFalse(
                hasattr(skip_if_lt_x_gpu(2, allow_cpu=True)(_noop), "_skipped_reason")
            )

        with patch.object(
            torch.accelerator,
            "current_accelerator",
            return_value=torch.device("cuda"),
        ):
            self.assertFalse(hasattr(skip_if_no_gpu(_noop), "_skipped_reason"))
            self.assertFalse(hasattr(skip_if_lt_x_gpu(2)(_noop), "_skipped_reason"))

    @unittest.skipIf(IS_SANDCASTLE, "Sandcastle leaves skips to the ranks")
    def test_dist_backend_skips_before_spawn(self):
        # distributed_test reads BACKEND and WORLD_SIZE at import.
        with patch.dict(os.environ, {"BACKEND": "gloo", "WORLD_SIZE": "2"}):
            from torch.testing._internal.distributed.distributed_test import (
                TestDistBackend,
            )

            class Skipped(TestDistBackend):
                @skip_if_small_worldsize
                def test_fn(self):
                    pass

        with (
            patch.object(MultiProcessTestCase, "setUp") as spawn_setup,
            self.assertRaisesRegex(
                unittest.SkipTest, TEST_SKIPS["small_worldsize"].message
            ),
        ):
            Skipped("test_fn").setUp()
        spawn_setup.assert_not_called()


if __name__ == "__main__":
    run_tests()
