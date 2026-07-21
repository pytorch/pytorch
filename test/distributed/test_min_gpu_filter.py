# Owner(s): ["oncall: distributed"]

import os
import sys
import types
import unittest

import torch
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
    MultiThreadedTestCase,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# The filter under test lives in test/conftest.py. Make it importable whether the
# file is run under pytest (test/ already on sys.path) or directly.
_TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TEST_ROOT not in sys.path:
    sys.path.insert(0, _TEST_ROOT)

from conftest import MinGpuFilterPlugin


# The resolver reads ``world_size`` / ``device`` / ``device_type`` off the class
# via ``cls.__new__(cls)`` (no ``__init__``), so these fakes expose them as the
# same constant-returning properties / attributes the real distributed test
# classes use.
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


class _MPcpuDevWs4(MultiProcessTestCase):
    @property
    def world_size(self):
        return 4

    @property
    def device(self):
        return "cpu"


class _MPcpuDevWs2(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    @property
    def device(self):
        return torch.device("cpu")


class _MPbroken(MultiProcessTestCase):
    @property
    def world_size(self):
        raise RuntimeError("world_size not resolvable at collection time")


# MultiProcContinuousTest declares ``world_size: int = -2`` as an unset sentinel
# populated only at runtime, so an undecorated subclass probes as -2. ``device``
# is not defined on it and ``device_type`` is a classmethod, matching how the
# real class exposes them.
class _MPCunset(MultiProcContinuousTest):
    @classmethod
    def device_type(cls):
        return "cuda"


class _MPCunsetCpu(MultiProcContinuousTest):
    pass


class _MTTws4(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 4


class _MTTws2(MultiThreadedTestCase):
    @property
    def world_size(self):
        return 2


class _FakePlain(unittest.TestCase):
    pass


def _func(min_gpus=None, required_world_size=None):
    def f():
        pass

    if min_gpus is not None:
        f._min_gpus_required = min_gpus
    if required_world_size is not None:
        f._required_world_size = required_world_size
    return f


def _item(*, cls=None, obj=None, name="test_x", module_name="test_foo"):
    return types.SimpleNamespace(
        obj=obj if obj is not None else _func(),
        cls=cls,
        name=name,
        module=types.SimpleNamespace(__name__=module_name),
        nodeid=f"{module_name}.py::{name}",
    )


class TestMinGpuFilter(TestCase):
    # The distributed_4gpu job runs on 4-GPU runners but selects tests needing
    # MORE GPUs than the 2-GPU runners provide, i.e. a threshold of 3 so that
    # tests requiring exactly 3 GPUs (which run nowhere under a strict >= 4
    # filter) are included alongside the 4-GPU tests.
    THRESHOLD = 3

    def setUp(self):
        self.plugin = MinGpuFilterPlugin(self.THRESHOLD)
        # Pin host-dependent state so the resolver is deterministic regardless
        # of the accelerators / device count of the machine running this test.
        self.plugin._accel_tokens = ("cuda",)
        self.plugin._default_world_size = 4

    def _keep(self, item):
        return self.plugin._required_gpus(item) >= self.THRESHOLD

    # --- process-based: world_size ---
    def test_process_world_size_4_kept(self):
        item = _item(cls=_MPws4, module_name="test_c10d_nccl")
        self.assertTrue(self._keep(item))

    def test_process_world_size_3_at_threshold_kept(self):
        # 3-GPU tests need more than the 2-GPU runners provide, so the 4-GPU job
        # (threshold 3) now includes them; a strict >= 4 filter dropped them.
        item = _item(cls=_MPws3, module_name="test_c10d_nccl")
        self.assertTrue(self._keep(item))

    def test_process_world_size_2_below_threshold_dropped(self):
        item = _item(cls=_MPws2, module_name="test_c10d_nccl")
        self.assertFalse(self._keep(item))

    def test_process_world_size_one_dropped(self):
        item = _item(cls=_MPws1, module_name="test_c10d_gloo")
        self.assertFalse(self._keep(item))

    # --- process-based: CPU / gloo gate ---
    def test_gloo_cpu_module_world_size_4_dropped(self):
        # Undecorated gloo test (e.g. test_gloo_backend_cpu_module): the file is
        # gloo-backed so it is treated as CPU and dropped.
        item = _item(cls=_MPws4, module_name="test_c10d_gloo")
        self.assertFalse(self._keep(item))

    def test_gloo_1gpu_decorator_2_dropped(self):
        item = _item(cls=_MPws4, obj=_func(min_gpus=2), module_name="test_c10d_gloo")
        self.assertFalse(self._keep(item))

    def test_gloo_2gpu_decorator_4_kept(self):
        # test_gloo_backend_2gpu_module is @skip_if_lt_x_gpu(4): a real 4-GPU test.
        item = _item(cls=_MPws4, obj=_func(min_gpus=4), module_name="test_c10d_gloo")
        self.assertTrue(self._keep(item))

    def test_cpu_device_property_world_size_4_dropped(self):
        item = _item(cls=_MPcpuDevWs4, module_name="test_c10d_nccl")
        self.assertFalse(self._keep(item))

    def test_cpu_device_property_world_size_2_dropped(self):
        # CommTest-style: device property returns cpu, world_size == 2.
        item = _item(cls=_MPcpuDevWs2, module_name="test_c10d_gloo")
        self.assertFalse(self._keep(item))

    def test_cpu_suffix_module_dropped(self):
        # e.g. distributed/checkpoint/test_file_system_checkpoint_cpu.
        item = _item(cls=_MPws4, module_name="test_file_system_checkpoint_cpu")
        self.assertFalse(self._keep(item))

    def test_cpu_backed_with_decorator_kept(self):
        # An explicit >= threshold decorator overrides the CPU gate.
        item = _item(cls=_MPcpuDevWs4, obj=_func(min_gpus=4))
        self.assertTrue(self._keep(item))

    # --- process-based: robustness when introspection fails ---
    def test_unresolvable_world_size_gloo_dropped(self):
        # world_size property raises -> fall back to default world size, then the
        # gloo module gate drops it.
        item = _item(cls=_MPbroken, module_name="test_c10d_gloo")
        self.assertFalse(self._keep(item))

    def test_unresolvable_world_size_accelerator_kept(self):
        # world_size property raises -> default world size (4) on an
        # accelerator-backed module keeps the test.
        item = _item(cls=_MPbroken, module_name="test_c10d_nccl")
        self.assertTrue(self._keep(item))

    # --- process-based: MultiProcContinuousTest -2 unset sentinel ---
    def test_continuous_unset_world_size_accelerator_kept(self):
        # world_size left at the -2 unset sentinel resolves to the default world
        # size; on an accelerator-reporting class the test is kept.
        item = _item(cls=_MPCunset, module_name="test_c10d_nccl")
        self.assertTrue(self._keep(item))

    def test_continuous_unset_world_size_decorator_kept(self):
        # Even when the class reports as CPU (device_type default), an explicit
        # >= threshold decorator keeps a MultiProcContinuousTest test.
        item = _item(
            cls=_MPCunsetCpu, obj=_func(min_gpus=4), module_name="test_c10d_nccl"
        )
        self.assertTrue(self._keep(item))

    # --- thread-based (MultiThreadedTestCase) ---
    def test_thread_accelerator_variant_world_size_4_kept(self):
        item = _item(cls=_MTTws4, name="test_broadcast_device_cuda")
        self.assertTrue(self._keep(item))

    def test_thread_cpu_variant_dropped(self):
        item = _item(cls=_MTTws4, name="test_broadcast_device_cpu")
        self.assertFalse(self._keep(item))

    def test_thread_no_device_variant_dropped(self):
        item = _item(cls=_MTTws4, name="test_expand_1d_rank_list")
        self.assertFalse(self._keep(item))

    def test_thread_accelerator_variant_world_size_2_dropped(self):
        item = _item(cls=_MTTws2, name="test_broadcast_device_cuda")
        self.assertFalse(self._keep(item))

    def test_thread_with_decorator_below_threshold_dropped(self):
        item = _item(
            cls=_MTTws4, obj=_func(min_gpus=2), name="test_broadcast_device_cuda"
        )
        self.assertFalse(self._keep(item))

    # --- plain TestCase / decorators ---
    def test_decorated_single_process_kept(self):
        item = _item(cls=_FakePlain, obj=_func(min_gpus=4))
        self.assertTrue(self._keep(item))

    def test_requires_world_size_attr_kept(self):
        item = _item(cls=_FakePlain, obj=_func(required_world_size=4))
        self.assertTrue(self._keep(item))

    def test_skip_if_lt_3_gpu_decorator_kept(self):
        # skip_if_lt_x_gpu(3) stamps _min_gpus_required=3: a genuine 3-GPU test.
        item = _item(cls=_FakePlain, obj=_func(min_gpus=3))
        self.assertTrue(self._keep(item))

    def test_skip_if_lt_2_gpu_decorator_dropped(self):
        # 2-GPU tests already run on the 2-GPU runners, so they stay deselected.
        item = _item(cls=_FakePlain, obj=_func(min_gpus=2))
        self.assertFalse(self._keep(item))

    def test_plain_testcase_without_signal_dropped(self):
        item = _item(cls=_FakePlain)
        self.assertFalse(self._keep(item))

    def test_no_class_without_signal_kept(self):
        # Non class-based items are kept unless a decorator says otherwise.
        item = _item(cls=None)
        self.assertTrue(self._keep(item))


if __name__ == "__main__":
    run_tests()
