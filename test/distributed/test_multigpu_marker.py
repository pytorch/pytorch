# Owner(s): ["oncall: distributed"]

import os
import sys
import tempfile
import types
from unittest.mock import patch

from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    MultiProcessTestCase,
    nccl_skip_if_lt_x_gpu,
    require_n_gpus_for_nccl_backend,
    requires_world_size,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TestCase


# The marker resolver lives in test/conftest.py. Make it importable whether this
# file is run under pytest (test/ already on sys.path) or directly.
_TEST_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _TEST_ROOT not in sys.path:
    sys.path.insert(0, _TEST_ROOT)

from conftest import (
    _decorator_gpu_requirement,
    _resolve_gpu_requirement,
    _UNRESOLVED_GPU_REQUIREMENT,
    MultiGpuMinFilterPlugin,
)


# The resolver reads ``world_size`` off the class via ``cls.__new__(cls)`` (no
# ``__init__``), so these fakes expose it as the same constant-returning property
# the real distributed test classes use.
class _MPws4(MultiProcessTestCase):
    @property
    def world_size(self):
        return 4


class _MPws2(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2


class _MPbroken(MultiProcessTestCase):
    @property
    def world_size(self):
        raise RuntimeError("world_size not resolvable at collection time")


# MultiProcContinuousTest declares ``world_size: int = -2`` as an unset sentinel
# populated only at runtime, so an undecorated subclass probes as -2.
class _MPCunset(MultiProcContinuousTest):
    pass


# A LocalTensor simulation class: multi-process base and world_size 4, but runs
# single-process on one GPU. Real *WithLocalTensor classes inherit a
# LocalDTensor* base whose is_local_tensor_enabled property is True; model that
# here so the name-independent probe detects it.
class _MPws4WithLocalTensor(_MPws4):
    @property
    def is_local_tensor_enabled(self):
        return True


def _fake_item(cls=None, func=None, filename="test_something.py"):
    # Model the real collected node: the resolver reads the test-file identity
    # from ``path``/``nodeid`` (as pytest populates them), not module.__name__.
    return types.SimpleNamespace(
        cls=cls, obj=func, path=filename, nodeid=f"{filename}::Cls::method"
    )


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

    def test_ambiguity_favors_coverage(self):
        # Unresolvable world_size clears any threshold (routes to larger runner).
        self.assertEqual(
            _resolve_gpu_requirement(_fake_item(cls=_MPbroken), _MPbroken),
            _UNRESOLVED_GPU_REQUIREMENT,
        )

    def test_continuous_test_scales_to_device_count(self):
        req = _resolve_gpu_requirement(_fake_item(cls=_MPCunset), _MPCunset)
        try:
            import torch

            device_count = max(int(torch.accelerator.device_count()), 0)
        except (AttributeError, RuntimeError, TypeError, ValueError):
            device_count = 0
        if device_count == 0:
            self.assertEqual(req, _UNRESOLVED_GPU_REQUIREMENT)
        else:
            self.assertEqual(req, device_count)

    def test_local_tensor_simulation_requirement_zero(self):
        # A LocalTensor simulation runs single-process on one GPU: its world_size
        # is a simulated mesh size, not a GPU count, so it does not scale.
        item = _fake_item(cls=_MPws4WithLocalTensor)
        self.assertEqual(_resolve_gpu_requirement(item, _MPws4WithLocalTensor), 0)

    def test_cpu_backed_high_world_size_excluded(self):
        # A CPU-backed multi-process test spawns ranks, not GPUs, so a high
        # world_size alone does not scale its GPU requirement.
        item = _fake_item(cls=_MPws4, filename="test_c10d_gloo.py")
        self.assertEqual(_resolve_gpu_requirement(item, _MPws4), 0)

    def test_cpu_gate_honors_explicit_gpu_decorator(self):
        # CPU-backend files still honor skip_if_lt_x_gpu: gloo tests can use
        # multiple GPUs per rank even though world_size is a rank count.
        @skip_if_lt_x_gpu(4)
        def needs4(self):
            pass

        item = _fake_item(cls=_MPws2, func=needs4, filename="test_c10d_gloo.py")
        self.assertEqual(_resolve_gpu_requirement(item, _MPws2), 4)

    def test_decorator_and_world_size_combine(self):
        # The 5 genuine 4-GPU misses: only ``skip_if_lt_x_gpu(2)`` but the class
        # hardcodes world_size=4. The larger of the two signals must win.
        @skip_if_lt_x_gpu(2)
        def needs2(self):
            pass

        item = _fake_item(cls=_MPws4, func=needs2, filename="test_2d_composability.py")
        self.assertEqual(_resolve_gpu_requirement(item, _MPws4), 4)

    def test_real_gloo_gpu_decorator(self):
        from distributed.test_c10d_gloo import DistributedDataParallelTest

        item = _fake_item(
            cls=DistributedDataParallelTest,
            func=DistributedDataParallelTest.test_gloo_backend_2gpu_module,
            filename="test/distributed/test_c10d_gloo.py",
        )
        self.assertEqual(_resolve_gpu_requirement(item, DistributedDataParallelTest), 4)

    def test_real_fully_shard_2d_training(self):
        from distributed._composable.test_composability.test_2d_composability import (
            TestFullyShard2DTraining,
        )

        item = _fake_item(
            cls=TestFullyShard2DTraining,
            func=TestFullyShard2DTraining.test_train_parity_2d_mlp,
            filename="test/distributed/_composable/test_composability/test_2d_composability.py",
        )
        self.assertEqual(_resolve_gpu_requirement(item, TestFullyShard2DTraining), 4)

    def test_real_local_tensor_class(self):
        from torch.testing._internal.distributed._tensor.common_dtensor import (
            create_local_tensor_test_class,
            DTensorTestBase,
        )

        class _Orig(DTensorTestBase):
            world_size = 4

            def test_placeholder(self):
                pass

        local_cls = create_local_tensor_test_class(_Orig)
        item = _fake_item(cls=local_cls)
        self.assertEqual(_resolve_gpu_requirement(item, local_cls), 0)

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


if __name__ == "__main__":
    run_tests()
