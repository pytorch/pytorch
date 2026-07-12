# Owner(s): ["oncall: distributed"]

import os
import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed.distributed_c10d as c10d
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import run_tests, TestCase


FAULT_TOLERANCE_BACKENDS = [
    ("gloo", "cpu"),
]


class AbstractFaultToleranceTest:
    @property
    def world_size(self):
        return 3

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _create_store(self):
        return dist.FileStore(self.file_name, self.world_size)

    def _init_reconfigurable_pg(self):
        self.store = self._create_store()
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=self.store,
            timeout=timedelta(seconds=30),
            enable_reconfigure=True,
        )
        self.pg = c10d._get_default_group()
        self.backend = self.pg._get_backend(torch.device(self.device))
        self.assertTrue(dist._supports_reconfigure())
        self.assertTrue(self.backend.supports_reconfigure)

    def _collect_handles(self, key_prefix):
        handle = dist._get_reconfigure_handle()
        self.store.set(f"{key_prefix}_{self.rank}", handle)
        return [
            self.store.get(f"{key_prefix}_{rank}").decode("utf-8")
            for rank in range(self.world_size)
        ]

    def _store_barrier(self, key_prefix):
        self.store.set(f"{key_prefix}_{self.rank}", "1")
        for rank in range(self.world_size):
            self.store.get(f"{key_prefix}_{rank}")

    def _reconfigure(self, uuid, handles):
        work = dist._reconfigure(
            uuid,
            handles,
            timeout=timedelta(seconds=30),
        )
        work.wait()

    def _create_reconfigured_pg(self, name, uuid):
        self._init_reconfigurable_pg()
        handles = self._collect_handles(f"{name}_init")
        self._reconfigure(uuid, handles)
        self.assertEqual(dist.get_world_size(), self.world_size)
        self.assertEqual(dist.get_rank(), self.rank)
        return self._collect_handles(f"{name}_post")

    def _assert_all_reduce_sum(self, expected_value):
        tensor = torch.full((4,), dist.get_rank() + 1.0, device=self.device)
        dist.all_reduce(tensor)
        expected = torch.full((4,), expected_value, dtype=tensor.dtype)
        self.assertEqual(tensor.cpu(), expected)

    def test_reconfigure_basic(self):
        self._create_reconfigured_pg("ft_basic", 100)

    def test_reconfigure_then_all_reduce(self):
        self._create_reconfigured_pg("ft_all_reduce", 200)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_then_send_recv(self):
        self._create_reconfigured_pg("ft_send_recv", 300)

        rank = dist.get_rank()
        send_rank = (rank + 1) % self.world_size
        recv_rank = (rank - 1 + self.world_size) % self.world_size
        send_tensor = torch.full((4,), rank + 1.0, device=self.device)
        recv_tensor = torch.zeros(4, device=self.device)

        if rank % 2 == 0:
            send_work = self.backend.send([send_tensor], send_rank, 0)
            recv_work = self.backend.recv([recv_tensor], recv_rank, 0)
        else:
            recv_work = self.backend.recv([recv_tensor], recv_rank, 0)
            send_work = self.backend.send([send_tensor], send_rank, 0)

        send_work.wait()
        recv_work.wait()
        self.assertEqual(recv_tensor.cpu(), torch.full((4,), recv_rank + 1.0))

    def test_shrink_exclude_last_rank(self):
        handles = self._create_reconfigured_pg("ft_shrink_last", 400)
        excluded_rank = self.world_size - 1
        if self.rank == excluded_rank:
            self._store_barrier("ft_shrink_last_done")
            return

        self._reconfigure(401, handles[:excluded_rank])
        self.assertEqual(dist.get_world_size(), self.world_size - 1)
        self.assertEqual(dist.get_rank(), self.rank)
        self._assert_all_reduce_sum(sum(range(1, self.world_size)))

        tensor = torch.zeros(4, device=self.device)
        if dist.get_rank() == 0:
            tensor.fill_(42.0)
        dist.broadcast(tensor, group_src=0)
        self.assertEqual(tensor.cpu(), torch.full((4,), 42.0))
        self._store_barrier("ft_shrink_last_done")

    def test_shrink_exclude_middle_rank(self):
        handles = self._create_reconfigured_pg("ft_shrink_middle", 500)
        excluded_rank = self.world_size // 2
        if self.rank == excluded_rank:
            self._store_barrier("ft_shrink_middle_done")
            return

        surviving_handles = [
            handle for rank, handle in enumerate(handles) if rank != excluded_rank
        ]
        self._reconfigure(501, surviving_handles)

        expected_rank = self.rank if self.rank < excluded_rank else self.rank - 1
        self.assertEqual(dist.get_world_size(), self.world_size - 1)
        self.assertEqual(dist.get_rank(), expected_rank)
        self._assert_all_reduce_sum(sum(range(1, self.world_size)))
        self._store_barrier("ft_shrink_middle_done")

    def test_reconfigure_scale_down_up(self):
        self._init_reconfigurable_pg()
        # Each rank shrinks to its own disjoint group, so each needs a unique uuid.
        self._reconfigure(600 + self.rank, [dist._get_reconfigure_handle()])
        self.assertEqual(dist.get_world_size(), 1)
        self.assertEqual(dist.get_rank(), 0)

        handles = self._collect_handles("ft_scale_down_up")
        self._reconfigure(603, handles)
        self.assertEqual(dist.get_world_size(), self.world_size)
        self.assertEqual(dist.get_rank(), self.rank)

        self._reconfigure(604 + self.rank, [dist._get_reconfigure_handle()])
        self.assertEqual(dist.get_world_size(), 1)
        self.assertEqual(dist.get_rank(), 0)
        self._store_barrier("ft_scale_down_up_done")

    def test_reconfigure_single_to_all(self):
        self._init_reconfigurable_pg()
        # Each rank shrinks to its own disjoint group, so each needs a unique uuid.
        self._reconfigure(700 + self.rank, [dist._get_reconfigure_handle()])

        handles = self._collect_handles("ft_single_to_all")
        self._reconfigure(703, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_identity(self):
        self._create_reconfigured_pg("ft_identity", 800)
        handles = self._collect_handles("ft_identity_again")
        self._reconfigure(801, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_late_join(self):
        self._init_reconfigurable_pg()
        handles = self._collect_handles("ft_late_join_initial")
        initial_world_size = self.world_size // 2
        if self.rank < initial_world_size:
            self._reconfigure(900, handles[:initial_world_size])

        handles = self._collect_handles("ft_late_join_all")
        self._reconfigure(901, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_merge_split(self):
        self._init_reconfigurable_pg()
        handles = self._collect_handles("ft_merge_split_initial")
        split = self.world_size // 2
        if self.rank < split:
            self._reconfigure(1000, handles[:split])
        else:
            self._reconfigure(1001, handles[split:])

        handles = self._collect_handles("ft_merge_split_all")
        self._reconfigure(1002, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_rejects_reused_uuid(self):
        self._init_reconfigurable_pg()
        uuid = 1100 + self.rank
        self._reconfigure(uuid, [dist._get_reconfigure_handle()])
        with self.assertRaisesRegex(RuntimeError, "already used"):
            self._reconfigure(uuid, [dist._get_reconfigure_handle()])


def _make_fault_tolerance_test_class(backend_name, device):
    class FaultToleranceTest(AbstractFaultToleranceTest, MultiProcessTestCase):
        pass

    FaultToleranceTest.backend_name = backend_name
    FaultToleranceTest.device = device
    FaultToleranceTest.__name__ = f"{backend_name.capitalize()}FaultToleranceTest"
    FaultToleranceTest.__qualname__ = FaultToleranceTest.__name__
    return unittest.skipIf(
        not dist.is_backend_available(backend_name),
        f"{backend_name} backend is not available",
    )(FaultToleranceTest)


for backend_name, device in FAULT_TOLERANCE_BACKENDS:
    globals()[f"{backend_name.capitalize()}FaultToleranceTest"] = (
        _make_fault_tolerance_test_class(backend_name, device)
    )


class ReconfigureContractTest(TestCase):
    def test_reconfigure_rejects_multiple_backends(self) -> None:
        pg = dist.ProcessGroup(0, 1)
        pg._register_backend(torch.device("cpu"), dist.ProcessGroup.BackendType.GLOO)
        pg._register_backend(torch.device("cuda"), dist.ProcessGroup.BackendType.NCCL)

        msg = "multiple backends"
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.supports_reconfigure
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.get_reconfigure_handle()
        with self.assertRaisesRegex(RuntimeError, msg):
            pg.reconfigure(c10d.ReconfigureOptions())


if __name__ == "__main__":
    run_tests()
