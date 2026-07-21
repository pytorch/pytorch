# Owner(s): ["oncall: distributed"]

import os
import sys
import types

import torch
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
    requires_world_size,
    skip_if_lt_x_gpu,
    STANDARD_DISTRIBUTED_GPUS,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# The marker resolver lives in test/conftest.py. Make it importable whether this
# file is run under pytest (test/ already on sys.path) or directly.
_TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TEST_ROOT not in sys.path:
    sys.path.insert(0, _TEST_ROOT)

from conftest import (  # noqa: E402
    _decorator_gpu_requirement,
    _is_cpu_backed,
    _needs_extra_gpus,
    _probe_world_size,
)


# The resolver reads ``world_size`` off the class via ``cls.__new__(cls)`` (no
# ``__init__``), so these fakes expose it as the same constant-returning property
# the real distributed test classes use.
class _MPws4(MultiProcessTestCase):
    @property
    def world_size(self):
        return 4


class _MPws3(MultiProcessTestCase):
    @property
    def world_size(self):
        return 3


class _MPws2(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2


class _MPws1(MultiProcessTestCase):
    @property
    def world_size(self):
        return 1


class _MPbroken(MultiProcessTestCase):
    @property
    def world_size(self):
        raise RuntimeError("world_size not resolvable at collection time")


# MultiProcContinuousTest declares ``world_size: int = -2`` as an unset sentinel
# populated only at runtime, so an undecorated subclass probes as -2.
class _MPCunset(MultiProcContinuousTest):
    pass


def _fake_item(cls=None, func=None, module_name="test_something"):
    module = types.SimpleNamespace()
    module.__name__ = module_name
    return types.SimpleNamespace(cls=cls, obj=func, module=module)


class TestMultiGpuMarker(TestCase):
    def test_standard_runner_size(self):
        self.assertEqual(STANDARD_DISTRIBUTED_GPUS, 2)

    def test_probe_world_size(self):
        self.assertEqual(_probe_world_size(_MPws4), 4)
        self.assertEqual(_probe_world_size(_MPws3), 3)
        self.assertEqual(_probe_world_size(_MPws2), 2)
        # Unresolvable / non-positive sentinel both collapse to 0.
        self.assertEqual(_probe_world_size(_MPbroken), 0)
        self.assertEqual(_probe_world_size(_MPCunset), 0)

    def test_decorator_requirement(self):
        @skip_if_lt_x_gpu(4)
        def needs4(self):
            pass

        @requires_world_size(3)
        def needs3(self):
            pass

        def plain(self):
            pass

        self.assertEqual(_decorator_gpu_requirement(needs4), 4)
        self.assertEqual(_decorator_gpu_requirement(needs3), 3)
        self.assertEqual(_decorator_gpu_requirement(plain), 0)
        self.assertEqual(_decorator_gpu_requirement(None), 0)

    def test_cpu_backed_detection(self):
        self.assertTrue(_is_cpu_backed(_fake_item(module_name="test_c10d_gloo")))
        self.assertTrue(_is_cpu_backed(_fake_item(module_name="test_foo_cpu")))
        self.assertFalse(_is_cpu_backed(_fake_item(module_name="test_c10d_nccl")))

    def test_world_size_selection(self):
        # >2 GPU tests are selected; <=2 are not.
        self.assertTrue(_needs_extra_gpus(_fake_item(cls=_MPws4), _MPws4))
        self.assertTrue(_needs_extra_gpus(_fake_item(cls=_MPws3), _MPws3))
        self.assertFalse(_needs_extra_gpus(_fake_item(cls=_MPws2), _MPws2))
        self.assertFalse(_needs_extra_gpus(_fake_item(cls=_MPws1), _MPws1))

    def test_ambiguity_favors_coverage(self):
        # Unresolvable world_size routes the test to the larger runner.
        self.assertTrue(_needs_extra_gpus(_fake_item(cls=_MPbroken), _MPbroken))
        self.assertTrue(_needs_extra_gpus(_fake_item(cls=_MPCunset), _MPCunset))

    def test_cpu_backed_high_world_size_excluded(self):
        # A CPU-backed multi-process test spawns ranks, not GPUs, so a high
        # world_size does not qualify it as an extra-GPU test.
        item = _fake_item(cls=_MPws4, module_name="test_c10d_gloo")
        self.assertFalse(_needs_extra_gpus(item, _MPws4))

    def test_explicit_decorator_overrides_cpu_gate(self):
        # An explicit GPU decorator asserts the requirement even for a
        # CPU-named module.
        @skip_if_lt_x_gpu(4)
        def needs4(self):
            pass

        item = _fake_item(cls=_MPws2, func=needs4, module_name="test_c10d_gloo")
        self.assertTrue(_needs_extra_gpus(item, _MPws2))


if __name__ == "__main__":
    run_tests()
