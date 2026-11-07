# Owner(s): ["oncall: distributed"]
import os
import tempfile

import torch
import torch.distributed as dist
from torch.distributed._composable.replicate import replicate
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
    TestCase,
)


class _Boom(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, x):
        raise RuntimeError("simulated OOM")


class _Scale(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.ones(()))

    def forward(self, x):
        return x * self.weight


class ReplicateActiveDDPModuleTest(TestCase):
    hw_classification = HardwareClassification.CPU

    def setUp(self):
        super().setUp()
        fd, self._store_path = tempfile.mkstemp()
        os.close(fd)
        self.addCleanup(os.unlink, self._store_path)
        dist.init_process_group(
            "gloo",
            store=dist.FileStore(self._store_path, 1),
            rank=0,
            world_size=1,
        )
        DistributedDataParallel._active_ddp_module = None

    def tearDown(self):
        DistributedDataParallel._active_ddp_module = None
        dist.destroy_process_group()
        super().tearDown()

    def test_cleared_after_forward_raises(self):
        m = _Boom()
        replicate(m, device_id=torch.device("cpu"))
        with self.assertRaisesRegex(RuntimeError, "simulated OOM"):
            m(torch.ones(()))
        self.assertIsNone(DistributedDataParallel._get_active_ddp_module())

    def test_cleared_after_successful_forward(self):
        m = _Scale()
        replicate(m, device_id=torch.device("cpu"))
        m(torch.ones(()))
        self.assertIsNone(DistributedDataParallel._get_active_ddp_module())

    def test_nested_replicate_restores_outer(self):
        seen = {}

        class Outer(torch.nn.Module):
            def __init__(self):
                super().__init__()
                self.inner = _Scale()

            def forward(self, x):
                seen["before_inner"] = DistributedDataParallel._get_active_ddp_module()
                x = self.inner(x)
                seen["after_inner"] = DistributedDataParallel._get_active_ddp_module()
                return x

        outer = Outer()
        replicate(outer.inner, device_id=torch.device("cpu"))
        replicate(outer, device_id=torch.device("cpu"))
        outer(torch.ones(()))
        outer_ddp = replicate.state(outer)._ddp
        self.assertIs(seen["before_inner"], outer_ddp)
        self.assertIs(seen["after_inner"], outer_ddp)
        self.assertIsNone(DistributedDataParallel._get_active_ddp_module())


if __name__ == "__main__":
    run_tests()
