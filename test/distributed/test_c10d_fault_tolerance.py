# Owner(s): ["oncall: distributed"]

import os
import sys
import time
import unittest
from dataclasses import dataclass
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed.distributed_c10d as c10d
from torch._C._distributed_c10d import WorkResult
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    run_tests,
    TEST_CUDA,
    TEST_WITH_ROCM,
    TestCase,
)


@dataclass(frozen=True)
class FaultToleranceBackend:
    name: str
    device_type: str
    supports_work_result: bool = False
    variant: str = ""
    env: tuple[tuple[str, str], ...] = ()


FAULT_TOLERANCE_BACKENDS = [
    FaultToleranceBackend("gloo", "cpu"),
    FaultToleranceBackend("nccl2", "cuda", supports_work_result=True),
    FaultToleranceBackend(
        "nccl2",
        "cuda",
        supports_work_result=True,
        variant="ShrinkGrow",
        env=(("TORCH_NCCL2_RECONFIGURE_SHRINK_GROW", "1"),),
    ),
]


class AbstractFaultToleranceTest:
    @property
    def world_size(self):
        return 3

    @property
    def device(self):
        if self.device_type == "cuda":
            return f"cuda:{self.rank}"
        return self.device_type

    def setUp(self):
        super().setUp()
        self._saved_env = {key: os.environ.get(key) for key in self.env}
        os.environ.update(self.env)
        self._spawn_processes()

    def tearDown(self):
        if dist.is_initialized():
            dist.destroy_process_group()
        super().tearDown()
        for key, value in self._saved_env.items():
            if value is None:
                os.environ.pop(key, None)
            else:
                os.environ[key] = value
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _create_store(self):
        return dist.FileStore(self.file_name, self.world_size)

    def _init_reconfigurable_pg(self):
        self.store = self._create_store()
        if self.device_type == "cuda":
            torch.cuda.set_device(self.rank)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=self.store,
            timeout=timedelta(seconds=30),
            enable_reconfigure=True,
        )
        self.pg = c10d._get_default_group()
        self.backend = dist.get_backend_impl(self.pg, torch.device(self.device))
        self.assertTrue(dist._supports_reconfigure())
        self.assertTrue(self.backend.supports_reconfigure)

    def _collect_handles(self, key_prefix):
        handle = dist._get_reconfigure_handle()
        self.store.set(f"{key_prefix}_{self.rank}", handle)
        return [
            self.store.get(f"{key_prefix}_{rank}").decode("utf-8")
            for rank in range(self.world_size)
        ]

    def _collect_handles_by_group_rank(self, key_prefix, group_rank=None):
        if group_rank is None:
            group_rank = dist.get_rank()
        self.store.set(f"{key_prefix}_{group_rank}", dist._get_reconfigure_handle())
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

    def test_work_explicit_timeout_includes_prelaunch_stall(self):
        if not self.supports_work_result:
            self.skipTest(f"{self.backend_name} does not report work results")
        self._create_reconfigured_pg("ft_work_timeout", 1300)
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()
        torch.cuda._sleep(int(500 * get_cycles_per_ms()))
        work = dist.all_reduce(torch.ones(4, device=self.device), async_op=True)

        with self.assertRaisesRegex(dist.DistBackendError, "before timing out"):
            work.wait(timeout=timedelta(milliseconds=50))

        self.assertFalse(torch.cuda.current_stream().query())
        self.assertTrue(work.is_completed())
        self.assertEqual(
            WorkResult(work.get_future_result().wait()), WorkResult.TIMEOUT
        )
        torch.cuda.synchronize()

    def test_work_reports_communicator_error(self):
        if not self.supports_work_result:
            self.skipTest(f"{self.backend_name} does not report work results")
        self._create_reconfigured_pg("ft_work_error", 1301)
        dist.all_reduce(torch.ones(1, device=self.device))
        torch.cuda.synchronize()

        if self.rank == 0:
            work = dist.all_reduce(torch.ones(1, device=self.device), async_op=True)
            time.sleep(0.5)
            self.backend.abort()
            self.assertTrue(work.is_completed())
            self.assertFalse(work.is_success())
            self.assertIsInstance(work.exception(), dist.DistBackendError)
            self.assertEqual(
                WorkResult(work.get_future_result().wait()),
                WorkResult.COMM_ERROR,
            )
            with self.assertRaisesRegex(dist.DistBackendError, "NCCL operation failed"):
                work.wait()
        else:
            # Revoke/abort is a local operation: NCCL does not notify peer
            # ranks when rank 0 aborts its side of the communicator. If ranks
            # 1 and 2 don't also abort and instead fall through to
            # tearDown()'s ordinary destroy_process_group(), the communicator
            # ends up in an inconsistent state across ranks -- NCCL's
            # ncclCommDestroy() barrier waits for every rank to enter
            # destroy, but rank 0 already exited via abort and never will, so
            # ranks 1 and 2 hang until the test harness's timeout. All ranks
            # must take the same (abort) path.
            time.sleep(1)
            self.backend.abort()

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

    def test_reconfigure_shrink_then_rejoin_last(self):
        handles = self._create_reconfigured_pg("ft_rejoin_last", 1500)
        last = self.world_size - 1
        if self.rank != last:
            self._reconfigure(1501, handles[:last])
            self._assert_all_reduce_sum(sum(range(1, self.world_size)))

        handles = self._collect_handles("ft_rejoin_last_all")
        self._reconfigure(1502, handles)
        self.assertEqual(dist.get_rank(), self.rank)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_shrink_then_rejoin_middle(self):
        handles = self._create_reconfigured_pg("ft_rejoin_middle", 1600)
        middle = self.world_size // 2
        if self.rank != middle:
            self._reconfigure(1601, handles[:middle] + handles[middle + 1 :])
            self._assert_all_reduce_sum(sum(range(1, self.world_size)))

        handles = self._collect_handles("ft_rejoin_middle_all")
        self._reconfigure(1602, handles)
        self.assertEqual(dist.get_rank(), self.rank)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_repeated_shrink_grow(self):
        self._create_reconfigured_pg("ft_repeated", 1700)
        for i in range(3):
            # Handles in group-rank order; a rejoining rank goes last, matching
            # grow's rank order.
            handles = self._collect_handles_by_group_rank(f"ft_repeated_{i}")
            dropped = i % self.world_size
            rejoining = dist.get_rank() == dropped
            if not rejoining:
                self._reconfigure(
                    1710 + 2 * i, handles[:dropped] + handles[dropped + 1 :]
                )
                self._assert_all_reduce_sum(sum(range(1, self.world_size)))
            last = self.world_size - 1
            handles = self._collect_handles_by_group_rank(
                f"ft_repeated_{i}_rejoin", last if rejoining else None
            )
            self._reconfigure(1711 + 2 * i, handles)
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

    def test_reconfigure_after_abort(self):
        # Port of torchcomms' ReconfigureTest.test_reconfigure_after_abort:
        # abort() (a revoke in reconfigurable mode) must be recoverable by a
        # reconfigure() with a fresh uuid.
        self._create_reconfigured_pg("ft_abort", 1200)
        self.backend.abort()

        from torch._C._distributed_c10d import ErrorType

        is_nccl = self.backend_name == "nccl2"
        expected = ErrorType.COMM_ERROR if is_nccl else ErrorType.SUCCESS
        self.assertEqual(self.backend.get_error(), expected)

        handles = self._collect_handles("ft_abort_recover")
        self._reconfigure(1201, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_after_timeout(self):
        from torch._C._distributed_c10d import ErrorType

        self._create_reconfigured_pg("ft_timeout", 1300)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))
        self.backend.set_timeout(timedelta(milliseconds=50))

        if self.rank == 1:
            tensor = torch.ones(4, device=self.device)
            work = dist.all_reduce(tensor, async_op=True)
            try:
                work.wait()
            except RuntimeError as error:
                self.assertRegex(str(error), "[Tt]imed out|before timing out")
            else:
                deadline = time.monotonic() + 10
                while (
                    self.backend.get_error() == ErrorType.SUCCESS
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.1)
                self.assertEqual(self.backend.get_error(), ErrorType.TIMEOUT)
                with self.assertRaisesRegex(
                    RuntimeError, "[Tt]imed out|before timing out"
                ):
                    dist.all_reduce(tensor, async_op=True)
            del work

        self.backend.set_timeout(timedelta(seconds=30))
        self._store_barrier("ft_timeout_observed")

        handles = self._collect_handles("ft_timeout_recover")
        self._reconfigure(1301, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_rejects_reused_uuid(self):
        self._init_reconfigurable_pg()
        if self.backend_name != "nccl2":
            uuid = 1100 + self.rank
            self._reconfigure(uuid, [dist._get_reconfigure_handle()])
            with self.assertRaisesRegex(RuntimeError, "already used"):
                self._reconfigure(uuid, [dist._get_reconfigure_handle()])
            return

        uuid = 1100
        handles = self._collect_handles("ft_reused_uuid_initial")
        self._reconfigure(uuid, handles)
        handles = self._collect_handles("ft_reused_uuid_current")
        error = "already used" if self.rank == 0 else "Wait timeout"
        with self.assertRaisesRegex(RuntimeError, error):
            dist._reconfigure(
                uuid,
                handles,
                timeout=timedelta(milliseconds=500),
            ).wait()
        self._store_barrier("ft_reused_uuid_rejected")
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_grow_rejects_reused_uuid(self):
        # Rank 0 rejects the reused uuid before growing; the other survivor must
        # time out instead of blocking in commGrow.
        handles = self._create_reconfigured_pg("ft_grow_reused", 1110)
        last = self.world_size - 1
        if self.rank != last:
            self._reconfigure(1111, handles[:last])
        handles = self._collect_handles("ft_grow_reused_all")
        if self.backend_name == "nccl2":
            error = "already used" if self.rank == 0 else "Wait timeout"
            with self.assertRaisesRegex(RuntimeError, error):
                dist._reconfigure(
                    1111, handles, timeout=timedelta(milliseconds=500)
                ).wait()
            handles = self._collect_handles("ft_grow_reused_retry")
        self._reconfigure(1112, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_timeout_is_retryable(self):
        if self.backend_name != "nccl2":
            self.skipTest("nonblocking NCCL initialization behavior")
        self._init_reconfigurable_pg()
        handles = self._collect_handles("ft_timeout_retry_initial")

        if self.rank == 0:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                dist._reconfigure(
                    1400,
                    handles[:2],
                    timeout=timedelta(milliseconds=500),
                ).wait()
        self._store_barrier("ft_timeout_retry_observed")

        handles = self._collect_handles("ft_timeout_retry_current")
        self._reconfigure(1401, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))

    def test_reconfigure_with_dead_peer_raises(self):
        if self.backend_name != "nccl2":
            self.skipTest("nonblocking NCCL initialization behavior")
        # Fail the bootstrap connect to the dead peer quickly.
        os.environ["NCCL_SOCKET_RETRY_CNT"] = "1"
        handles = self._create_reconfigured_pg("ft_dead_peer", 2000)
        self._store_barrier("ft_dead_peer_ready")
        # Exit without teardown: the dead peer leaves the comm unusable.
        if self.rank == 1:
            os._exit(0)
        if self.rank == 2:
            self.store.get("ft_dead_peer_done")
            os._exit(0)

        try:
            with self.assertRaises(RuntimeError):
                dist._reconfigure(
                    2001, handles[:2], timeout=timedelta(seconds=10)
                ).wait()
        finally:
            self.store.set("ft_dead_peer_done", "1")
        os._exit(0)

    def test_reconfigure_after_failed_join_has_unique_handle(self):
        if self.backend_name != "nccl2":
            self.skipTest("nonblocking NCCL initialization behavior")
        self._init_reconfigurable_pg()
        handles = self._collect_handles("ft_failed_join_initial")

        # Rank 1 fails as new rank 0 and has no communicator, like fresh rank 0.
        if self.rank == 1:
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                dist._reconfigure(
                    1900,
                    [handles[1], handles[2]],
                    timeout=timedelta(milliseconds=500),
                ).wait()
        self._store_barrier("ft_failed_join_observed")

        handles = self._collect_handles("ft_failed_join_current")
        self.assertEqual(len(set(handles)), self.world_size)
        self._reconfigure(1901, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)))


def _make_fault_tolerance_test_class(backend):
    class FaultToleranceTest(AbstractFaultToleranceTest, MultiProcessTestCase):
        pass

    FaultToleranceTest.backend_name = backend.name
    FaultToleranceTest.device_type = backend.device_type
    FaultToleranceTest.supports_work_result = backend.supports_work_result
    FaultToleranceTest.env = dict(backend.env)
    FaultToleranceTest.__name__ = (
        f"{backend.name.capitalize()}{backend.variant}FaultToleranceTest"
    )
    FaultToleranceTest.__qualname__ = FaultToleranceTest.__name__
    cls = unittest.skipIf(
        not dist.is_backend_available(backend.name),
        f"{backend.name} backend is not available",
    )(FaultToleranceTest)
    if backend.device_type == "cuda":
        cls = unittest.skipIf(
            not TEST_CUDA or torch.cuda.device_count() < 3,
            "fault tolerance CUDA tests require at least 3 GPUs",
        )(cls)
    if backend.name == "nccl2" and TEST_WITH_ROCM and dist.is_nccl_available():
        cls = unittest.skipIf(
            torch.cuda.nccl.version() < (2, 30, 7),
            "nccl2 reconfigure requires RCCL 2.30.7 or later",
        )(cls)
    return cls


for backend in FAULT_TOLERANCE_BACKENDS:
    cls = _make_fault_tolerance_test_class(backend)
    globals()[cls.__name__] = cls


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


class BackendCapabilityContractTest(TestCase):
    """Ensures base-Backend bindings are safe to call on any backend (e.g. Gloo)
    without throwing, so hasattr-based capability detection stays valid."""

    def test_get_error_returns_success_on_base_backend(self) -> None:
        from torch._C._distributed_c10d import Backend as C10DBackend, ErrorType

        backend = C10DBackend(0, 1)
        self.assertTrue(hasattr(backend, "get_error"))
        self.assertEqual(backend.get_error(), ErrorType.SUCCESS)


if __name__ == "__main__":
    run_tests()
