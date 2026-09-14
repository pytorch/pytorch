# Owner(s): ["oncall: distributed"]

import os
import sys
import time
from datetime import timedelta

import torch
import torch.distributed as dist


if not dist.is_available():
    print("distributed package not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import torch.distributed.distributed_c10d as c10d
from torch._C._distributed_c10d import WorkResult
from torch.testing._internal.common_device_type import (
    Capability,
    instantiate_device_type_tests,
    requires_capabilities,
)
from torch.testing._internal.common_distributed import MultiProcessTestCase
from torch.testing._internal.common_utils import (
    get_cycles_per_ms,
    HardwareClassification,
    run_tests,
    TEST_WITH_ROCM,
    TestCase,
)


class AbstractFaultToleranceTest:
    @property
    def world_size(self):
        return 3

    def _rank_device(self, device):
        dev = torch.device(device)
        if dev.type == "cpu":
            return dev
        return torch.device(dev.type, self.rank)

    @property
    def backend_name(self):
        if self.device_type == "cuda":
            return "nccl2"
        return dist.get_default_backend_for_device(self.device_type)

    def setUp(self):
        super().setUp()
        if self.device_type == "cuda" and TEST_WITH_ROCM:
            self.skipTest("nccl2 reconfigure is not supported with RCCL")
        if self.device_type != "cpu" and torch.accelerator.device_count() < 3:
            self.skipTest("fault tolerance tests require at least 3 accelerators")
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

    def _init_reconfigurable_pg(self, rank_device):
        self.store = self._create_store()
        if self.device_type != "cpu":
            torch.accelerator.set_device_index(self.rank)
        dist.init_process_group(
            self.backend_name,
            world_size=self.world_size,
            rank=self.rank,
            store=self.store,
            timeout=timedelta(seconds=30),
            enable_reconfigure=True,
        )
        self.pg = c10d._get_default_group()
        self.backend = dist.get_backend_impl(self.pg, rank_device)
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

    def _create_reconfigured_pg(self, name, uuid, rank_device):
        self._init_reconfigurable_pg(rank_device)
        handles = self._collect_handles(f"{name}_init")
        self._reconfigure(uuid, handles)
        self.assertEqual(dist.get_world_size(), self.world_size)
        self.assertEqual(dist.get_rank(), self.rank)
        return self._collect_handles(f"{name}_post")

    def _assert_all_reduce_sum(self, expected_value, rank_device):
        tensor = torch.full((4,), dist.get_rank() + 1.0, device=rank_device)
        dist.all_reduce(tensor)
        expected = torch.full((4,), expected_value, dtype=tensor.dtype)
        self.assertEqual(tensor.cpu(), expected)


class FaultToleranceTest(AbstractFaultToleranceTest, MultiProcessTestCase):
    hw_classification = HardwareClassification.ACCELERATOR

    @requires_capabilities(Capability.collective.reconfigure)
    def test_reconfigure_basic(self, device):
        rank_device = self._rank_device(device)
        self._create_reconfigured_pg("ft_basic", 100, rank_device)

    @requires_capabilities(Capability.collective.reconfigure)
    def test_reconfigure_then_all_reduce(self, device):
        rank_device = self._rank_device(device)
        self._create_reconfigured_pg("ft_all_reduce", 200, rank_device)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)), rank_device)

    @requires_capabilities(Capability.collective.reconfigure)
    def test_reconfigure_then_send_recv(self, device):
        rank_device = self._rank_device(device)
        self._create_reconfigured_pg("ft_send_recv", 300, rank_device)

        rank = dist.get_rank()
        send_rank = (rank + 1) % self.world_size
        recv_rank = (rank - 1 + self.world_size) % self.world_size
        send_tensor = torch.full((4,), rank + 1.0, device=rank_device)
        recv_tensor = torch.zeros(4, device=rank_device)

        if rank % 2 == 0:
            send_work = self.backend.send([send_tensor], send_rank, 0)
            recv_work = self.backend.recv([recv_tensor], recv_rank, 0)
        else:
            recv_work = self.backend.recv([recv_tensor], recv_rank, 0)
            send_work = self.backend.send([send_tensor], send_rank, 0)

        send_work.wait()
        recv_work.wait()
        self.assertEqual(recv_tensor.cpu(), torch.full((4,), recv_rank + 1.0))

    @requires_capabilities(
        Capability.collective.reconfigure, Capability.collective.work_result
    )
    def test_work_explicit_timeout_includes_prelaunch_stall(self, device):
        rank_device = self._rank_device(device)
        dev_mod = torch.get_device_module(self.device_type)
        self._create_reconfigured_pg("ft_work_timeout", 1300, rank_device)
        dist.all_reduce(torch.ones(1, device=rank_device))
        torch.accelerator.synchronize()
        dev_mod._sleep(int(500 * get_cycles_per_ms()))
        work = dist.all_reduce(torch.ones(4, device=rank_device), async_op=True)

        with self.assertRaisesRegex(dist.DistBackendError, "timed out"):
            work.wait(timeout=timedelta(milliseconds=50))

        self.assertFalse(dev_mod.current_stream().query())
        self.assertTrue(work.is_completed())
        self.assertEqual(
            WorkResult(work.get_future_result().wait()), WorkResult.TIMEOUT
        )
        torch.accelerator.synchronize()

    @requires_capabilities(
        Capability.collective.reconfigure, Capability.collective.work_result
    )
    def test_work_reports_communicator_error(self, device):
        rank_device = self._rank_device(device)
        self._create_reconfigured_pg("ft_work_error", 1301, rank_device)
        dist.all_reduce(torch.ones(1, device=rank_device))
        torch.accelerator.synchronize()

        if self.rank == 0:
            work = dist.all_reduce(torch.ones(1, device=rank_device), async_op=True)
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
            time.sleep(1)

    @requires_capabilities(Capability.collective.reconfigure)
    def test_shrink_exclude_last_rank(self, device):
        rank_device = self._rank_device(device)
        handles = self._create_reconfigured_pg("ft_shrink_last", 400, rank_device)
        excluded_rank = self.world_size - 1
        if self.rank == excluded_rank:
            self._store_barrier("ft_shrink_last_done")
            return

        self._reconfigure(401, handles[:excluded_rank])
        self.assertEqual(dist.get_world_size(), self.world_size - 1)
        self.assertEqual(dist.get_rank(), self.rank)
        self._assert_all_reduce_sum(sum(range(1, self.world_size)), rank_device)

        tensor = torch.zeros(4, device=rank_device)
        if dist.get_rank() == 0:
            tensor.fill_(42.0)
        dist.broadcast(tensor, group_src=0)
        self.assertEqual(tensor.cpu(), torch.full((4,), 42.0))
        self._store_barrier("ft_shrink_last_done")

    @requires_capabilities(Capability.collective.reconfigure)
    def test_shrink_exclude_middle_rank(self, device):
        rank_device = self._rank_device(device)
        handles = self._create_reconfigured_pg("ft_shrink_middle", 500, rank_device)
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
        self._assert_all_reduce_sum(sum(range(1, self.world_size)), rank_device)
        self._store_barrier("ft_shrink_middle_done")

    @requires_capabilities(Capability.collective.reconfigure)
    def test_reconfigure_scale_down_up(self, device):
        rank_device = self._rank_device(device)
        self._init_reconfigurable_pg(rank_device)
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

    @requires_capabilities(Capability.collective.reconfigure)
    def test_reconfigure_single_to_all(self, device):
        rank_device = self._rank_device(device)
        self._init_reconfigurable_pg(rank_device)
        # Each rank shrinks to its own disjoint group, so each needs a unique uuid.
        self._reconfigure(700 + self.rank, [dist._get_reconfigure_handle()])

        handles = self._collect_handles("ft_single_to_all")
        self._reconfigure(703, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)), rank_device)

    @requires_capabilities(Capability.collective.reconfigure)
    def test_reconfigure_identity(self, device):
        rank_device = self._rank_device(device)
        self._create_reconfigured_pg("ft_identity", 800, rank_device)
        handles = self._collect_handles("ft_identity_again")
        self._reconfigure(801, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)), rank_device)

    @requires_capabilities(Capability.collective.reconfigure)
    def test_reconfigure_late_join(self, device):
        rank_device = self._rank_device(device)
        self._init_reconfigurable_pg(rank_device)
        handles = self._collect_handles("ft_late_join_initial")
        initial_world_size = self.world_size // 2
        if self.rank < initial_world_size:
            self._reconfigure(900, handles[:initial_world_size])

        handles = self._collect_handles("ft_late_join_all")
        self._reconfigure(901, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)), rank_device)

    @requires_capabilities(Capability.collective.reconfigure)
    def test_reconfigure_merge_split(self, device):
        rank_device = self._rank_device(device)
        self._init_reconfigurable_pg(rank_device)
        handles = self._collect_handles("ft_merge_split_initial")
        split = self.world_size // 2
        if self.rank < split:
            self._reconfigure(1000, handles[:split])
        else:
            self._reconfigure(1001, handles[split:])

        handles = self._collect_handles("ft_merge_split_all")
        self._reconfigure(1002, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)), rank_device)

    @requires_capabilities(Capability.collective.reconfigure)
    def test_reconfigure_after_abort(self, device):
        rank_device = self._rank_device(device)
        # Port of torchcomms' ReconfigureTest.test_reconfigure_after_abort:
        # abort() (a revoke in reconfigurable mode) must be recoverable by a
        # reconfigure() with a fresh uuid.
        self._create_reconfigured_pg("ft_abort", 1200, rank_device)
        self.backend.abort()

        from torch._C._distributed_c10d import ErrorType

        is_nccl = self.backend_name == "nccl2"
        expected = ErrorType.COMM_ERROR if is_nccl else ErrorType.SUCCESS
        self.assertEqual(self.backend.get_error(), expected)

        handles = self._collect_handles("ft_abort_recover")
        self._reconfigure(1201, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)), rank_device)

    @requires_capabilities(Capability.collective.reconfigure)
    def test_reconfigure_after_timeout(self, device):
        rank_device = self._rank_device(device)
        from torch._C._distributed_c10d import ErrorType

        self._create_reconfigured_pg("ft_timeout", 1300, rank_device)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)), rank_device)
        self.backend.set_timeout(timedelta(milliseconds=50))

        if self.rank == 1:
            tensor = torch.ones(4, device=rank_device)
            work = dist.all_reduce(tensor, async_op=True)
            try:
                work.wait()
            except RuntimeError as error:
                self.assertRegex(str(error), "[Tt]imed out")
            else:
                deadline = time.monotonic() + 10
                while (
                    self.backend.get_error() == ErrorType.SUCCESS
                    and time.monotonic() < deadline
                ):
                    time.sleep(0.1)
                self.assertEqual(self.backend.get_error(), ErrorType.TIMEOUT)
                with self.assertRaisesRegex(RuntimeError, "timed out"):
                    dist.all_reduce(tensor, async_op=True)
            del work

        self.backend.set_timeout(timedelta(seconds=30))
        self._store_barrier("ft_timeout_observed")

        handles = self._collect_handles("ft_timeout_recover")
        self._reconfigure(1301, handles)
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)), rank_device)

    @requires_capabilities(Capability.collective.reconfigure)
    def test_reconfigure_rejects_reused_uuid(self, device):
        rank_device = self._rank_device(device)
        self._init_reconfigurable_pg(rank_device)
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
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)), rank_device)

    def test_reconfigure_timeout_is_retryable(self, device):
        rank_device = self._rank_device(device)
        if self.backend_name != "nccl2":
            self.skipTest("nonblocking NCCL initialization behavior")
        self._init_reconfigurable_pg(rank_device)
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
        self._assert_all_reduce_sum(sum(range(1, self.world_size + 1)), rank_device)


instantiate_device_type_tests(FaultToleranceTest, globals(), except_for=["hpu", "xpu"])


class ReconfigureContractTest(TestCase):
    hw_classification = HardwareClassification.GENERIC

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

    hw_classification = HardwareClassification.GENERIC

    def test_get_error_returns_success_on_base_backend(self) -> None:
        from torch._C._distributed_c10d import Backend as C10DBackend, ErrorType

        backend = C10DBackend(0, 1)
        self.assertTrue(hasattr(backend, "get_error"))
        self.assertEqual(backend.get_error(), ErrorType.SUCCESS)


if __name__ == "__main__":
    run_tests()
