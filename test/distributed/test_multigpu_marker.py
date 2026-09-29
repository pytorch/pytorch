# Owner(s): ["oncall: distributed"]

import os
import sys
import tempfile
import types
from unittest.mock import patch

from torch.testing._internal.common_distributed import (
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
