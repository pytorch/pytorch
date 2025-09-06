# Owner(s): ["oncall: distributed"]

import os
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist
import torch.distributed._dist2 as dist2
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_gloo,
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import run_tests, TestCase


def synchronize_accelerator():
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


class CommunicatorTest(TestCase):
    def test_context_manager(self):
        os.environ["RANK"] = str(0)
        os.environ["WORLD_SIZE"] = str(1)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        comm1 = dist2.new_comm(
            backend="gloo",
            timeout=timedelta(seconds=60),
            device="cpu",
        )
        comm2 = dist2.new_comm(
            backend="gloo",
            timeout=timedelta(seconds=60),
            device="cpu",
        )

        self.assertIsNone(dist2.current_comm())

        with dist2.comm(comm1):
            self.assertIs(dist2.current_comm(), comm1)

            with dist2.comm(comm2):
                self.assertIs(dist2.current_comm(), comm2)

            self.assertIs(dist2.current_comm(), comm1)

        self.assertIsNone(dist2.current_comm())


class Dist2MultiProcessTestCase(MultiProcessTestCase):
    device: torch.device

    @property
    def world_size(self) -> int:
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def new_comm(self) -> torch.distributed.ProcessGroup:
        raise unittest.SkipTest("new_comm() must be implemented by subclasses")

    def test_utility(self) -> None:
        comm = self.new_comm()

    def test_allreduce(self) -> None:
        comm = self.new_comm()

        t = torch.ones(10, device=self.device)
        comm.allreduce(t, timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        self.assertEqual(t, torch.full_like(t, self.world_size))

        comm.shutdown()

    def test_barrier(self) -> None:
        comm = self.new_comm()

        comm.barrier(timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        comm.shutdown()

    def test_broadcast(self) -> None:
        comm = self.new_comm()

        t = torch.full((10,), self.rank, device=self.device)
        comm.broadcast(t, root=0, timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        self.assertEqual(t, torch.full_like(t, 0))

        comm.shutdown()

    def test_allgather(self) -> None:
        comm = self.new_comm()

        t = torch.full((10,), self.rank + 1, device=self.device, dtype=torch.float32)
        out = [torch.zeros(10, device=self.device) for _ in range(self.world_size)]
        comm.allgather(out, t, timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        for i in range(self.world_size):
            self.assertEqual(out[i], torch.full_like(t, i + 1))

        comm.shutdown()

    def test_gather(self) -> None:
        comm = self.new_comm()

        inp = torch.full((10,), self.rank + 1, device=self.device, dtype=torch.float32)
        out = (
            [torch.zeros(10, device=self.device) for _ in range(self.world_size)]
            if self.rank == 0
            else []
        )
        comm.gather(out, inp, root=0, timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        if self.rank == 0:
            for i in range(self.world_size):
                self.assertEqual(out[i], torch.full_like(inp, i + 1))

        comm.shutdown()

    def test_scatter(self) -> None:
        comm = self.new_comm()

        inp = (
            [
                torch.torch.full((10,), i + 1, device=self.device, dtype=torch.float32)
                for i in range(self.world_size)
            ]
            if self.rank == 0
            else []
        )
        out = torch.zeros(10, device=self.device)
        comm.scatter(out, inp, root=0, timeout=timedelta(seconds=30)).wait()

        synchronize_accelerator()

        self.assertEqual(out, torch.full_like(out, self.rank + 1))

        comm.shutdown()

    def test_reduce(self) -> None:
        comm = self.new_comm()

        t = torch.full((10,), 1, device=self.device, dtype=torch.float32)
        comm.reduce(
            t, root=0, op=dist2.ReduceOp.SUM, timeout=timedelta(seconds=30)
        ).wait()

        synchronize_accelerator()

        if self.rank == 0:
            self.assertEqual(t, torch.full_like(t, self.world_size))

        comm.shutdown()

    def test_reduce_scatter(self) -> None:
        comm = self.new_comm()

        inp = [
            torch.full((10,), i + 1, device=self.device, dtype=torch.float32)
            for i in range(self.world_size)
        ]
        out = torch.zeros(10, device=self.device)
        comm.reduce_scatter(
            out, inp, op=dist2.ReduceOp.SUM, timeout=timedelta(seconds=30)
        ).wait()

        synchronize_accelerator()

        self.assertEqual(out, torch.full_like(out, self.world_size * (self.rank + 1)))

        comm.shutdown()

    def test_all_to_all_single(self) -> None:
        comm = self.new_comm()

        out = torch.zeros(self.world_size * 10, device=self.device)
        inp = torch.full(
            (self.world_size * 10,),
            self.rank + 1,
            device=self.device,
            dtype=torch.float32,
        )
        split_comm_sizes = [10 for _ in range(self.world_size)]
        comm.all_to_all_single(
            out, inp, split_comm_sizes, split_comm_sizes, timeout=timedelta(seconds=30)
        ).wait()

        synchronize_accelerator()

        for i in range(self.world_size):
            out_range = out[i * 10 : (i + 1) * 10]
            self.assertEqual(out_range, torch.full_like(out_range, i + 1))

    def test_group_split_comm(self) -> None:
        comm = self.new_comm()
        subcomm = comm.split_comm(
            [0], timeout=timedelta(seconds=30), name="subcomm_1"
        )
        if self.rank == 0:
            assert subcomm is not None
            self.assertEqual(subcomm.size, 1)
            backend = subcomm.unsafe_backend
            self.assertEqual(backend.options._timeout, timedelta(seconds=30))
            self.assertEqual(subcomm.name, "subcomm_1")
        else:
            self.assertEqual(subcomm, None)

    def test_remote_group_merge(self) -> None:
        comm = self.new_comm()
        subcomm_1 = comm.split_comm([0], timeout=timedelta(seconds=30))
        subcomm_2 = comm.split_comm([1], timeout=timedelta(seconds=30))
        if self.rank == 0:
            assert subcomm_1 is not None
            tcp_store = dist.TCPStore(
                host_name=os.environ["MASTER_ADDR"],
                port=29781,
                world_size=2,
                is_master=True,
            )
            merged_comm = subcomm_1.merge_remote_comm(
                tcp_store, 2, timedelta(seconds=40), "merged_comm"
            )
            self.assertEqual(merged_comm.size, 2)
            backend = merged_comm.unsafe_backend
            self.assertEqual(backend.options._timeout, timedelta(seconds=40))
            self.assertEqual(merged_comm.name, "merged_comm")
        else:
            assert subcomm_2 is not None
            tcp_store = dist.TCPStore(
                host_name=os.environ["MASTER_ADDR"],
                port=29781,
                world_size=2,
                is_master=False,
            )
            merged_comm = subcomm_2.merge_remote_comm(
                tcp_store, 2, timedelta(seconds=40), "merged_comm"
            )
            self.assertEqual(merged_comm.size, 2)
            backend = merged_comm.unsafe_backend
            self.assertEqual(backend.options._timeout, timedelta(seconds=40))
            self.assertEqual(merged_comm.name, "merged_comm")


class ProcessGroupGlooTest(Dist2MultiProcessTestCase):
    device = torch.device("cpu")

    @requires_gloo()
    def new_comm(self) -> torch.distributed.ProcessGroup:
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"

        return dist2.new_comm(
            backend="gloo",
            timeout=timedelta(seconds=60),
            device=self.device,
        )


class ProcessGroupNCCLTest(Dist2MultiProcessTestCase):
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def new_comm(self) -> torch.distributed.ProcessGroup:
        os.environ["RANK"] = str(self.rank)
        os.environ["WORLD_SIZE"] = str(self.world_size)
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29501"

        raise unittest.SkipTest("TODO: skip NCCL") 

        self.device = torch.device("cuda", self.rank)

        return dist2.new_comm(
            backend="nccl",
            timeout=timedelta(seconds=60),
            device=self.device,
        )


if __name__ == "__main__":
    assert not torch.cuda._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    run_tests()
