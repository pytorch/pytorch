# Owner(s): ["oncall: distributed"]

import os
from datetime import timedelta

import torch
import torch.distributed._dist2 as dist2
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TestCase


class ProcessGroupTest(TestCase):
    def test_context_manager(self):
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        pg1 = dist2.new_group(
            backend="gloo", timeout=timedelta(seconds=60), device="cpu", pg_options=None
        )
        pg2 = dist2.new_group(
            backend="gloo", timeout=timedelta(seconds=60), device="cpu", pg_options=None
        )

        self.assertIsNone(dist2.current_process_group())

        with dist2.process_group(pg1):
            self.assertIs(dist2.current_process_group(), pg1)

            with dist2.process_group(pg2):
                self.assertIs(dist2.current_process_group(), pg2)

            self.assertIs(dist2.current_process_group(), pg1)

        self.assertIsNone(dist2.current_process_group())


class ProcessGroupGlooTest(MultiProcessTestCase):
    lazy_init = False

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @requires_gloo()
    def test_new_group(self):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        device = "cpu"

        group = dist2.new_group(
            backend="gloo",
            timeout=timedelta(seconds=60),
            device=device,
            pg_options=None,
        )

        t = torch.rand(10, device=device)
        group.allreduce(t).wait()


class ProcessGroupNCCLTest(MultiProcessTestCase):
    lazy_init = False

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_new_group(self):
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        device = torch.device("cuda", self.rank)

        from torch.distributed import ProcessGroupNCCL

        opts = ProcessGroupNCCL.Options()

        group = dist2.new_group(
            backend="nccl",
            timeout=timedelta(seconds=60),
            device=device,
            pg_options=opts,
        )

        t = torch.rand(10, device=device)
        group.allreduce(t).wait()


if __name__ == "__main__":
    assert not torch.cuda._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    run_tests()
