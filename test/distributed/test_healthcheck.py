# Owner(s): ["oncall: distributed"]

import multiprocessing as mp
import os
import time
import unittest
from datetime import timedelta

# Disable rethrowing errors in the watchdog thread to avoid crashes on teardown.
os.environ["TORCH_NCCL_RETHROW_CUDA_ERRORS"] = "0"

import torch
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
    TestCase,
)
from torch.testing._internal.common_utils import run_tests, TEST_CUDA


class HealthcheckNCCLMultiprocessTest(MultiProcessTestCase):
    @property
    def store_path(self) -> str:
        return f"/tmp/{self.id()}.filestore"

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self) -> None:
        if os.path.exists(self.store_path):
            print("removing file store!", self.store_path)
            os.remove(self.store_path)

        from torch._C._distributed_c10d import FileStore

        store = FileStore(self.store_path, self.world_size)
        store.set("warm", "up")

        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    def test_healthcheck_success(self) -> None:
        from torch._C._distributed_c10d import FileStore, HealthcheckNCCL

        torch.cuda.set_device(self.rank)

        store = FileStore(self.store_path, self.world_size)

        healthcheck = HealthcheckNCCL(
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=1,
            exit_on_error=None,
            interval=timedelta(seconds=1),
            timeout=timedelta(seconds=60),
        )
        while healthcheck.num_failures == -1:
            time.sleep(0.01)
        self.assertEqual(healthcheck.num_failures, 0)

        healthcheck.shutdown()

    @skip_if_lt_x_gpu(2)
    def test_healthcheck_timeout(self) -> None:
        from torch._C._distributed_c10d import FileStore, HealthcheckNCCL

        torch.cuda.set_device(self.rank)

        store = FileStore(self.store_path, self.world_size)

        healthcheck = HealthcheckNCCL(
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            local_world_size=1,
            exit_on_error=None,
            interval=timedelta(seconds=1),
            timeout=timedelta(milliseconds=1),
        )
        while healthcheck.num_failures == -1:
            time.sleep(0.01)
        self.assertEqual(healthcheck.num_failures, 2)

        # NCCL may be in a bad state -- force a clean exit
        os._exit(0)


def healthcheck_exit() -> None:
    from torch._C._distributed_c10d import HashStore, HealthcheckNCCL

    store = HashStore()

    healthcheck = HealthcheckNCCL(
        store=store,
        rank=0,
        world_size=2,
        local_world_size=1,
        exit_on_error=123,
        interval=timedelta(seconds=1),
        timeout=timedelta(milliseconds=1),
    )

    time.sleep(1000)


class HealthcheckNCCLTest(TestCase):
    @unittest.skipIf(not TEST_CUDA, "requires cuda")
    def test_healthcheck_exit(self) -> None:
        mp_context = mp.get_context("spawn")
        p = mp_context.Process(target=healthcheck_exit)
        p.start()
        p.join()

        self.assertEqual(p.exitcode, 123)

    def test_healthcheck_groups(self) -> None:
        from torch._C._distributed_c10d import Healthcheck

        world_size = 8
        local_world_size = 2

        expected = [
            [
                (0, 0, 4),
                (0, 1, 4),
                (0, 2, 4),
                (0, 3, 4),
                (1, 0, 4),
                (1, 1, 4),
                (1, 2, 4),
                (1, 3, 4),
            ],
            [
                (0, 0, 4),
                (0, 1, 4),
                (1, 2, 4),
                (1, 3, 4),
                (1, 0, 4),
                (1, 1, 4),
                (0, 2, 4),
                (0, 3, 4),
            ],
        ]
        for side, infos in enumerate(expected):
            for rank in range(world_size):
                info = Healthcheck.calculate_group_info(
                    side=side,
                    rank=rank,
                    world_size=world_size,
                    local_world_size=local_world_size,
                )
                self.assertEqual(info, infos[rank], f"{side} {rank} {info}")


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
