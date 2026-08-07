# Owner(s): ["oncall: distributed"]
#
# Tests specific to the in-tree torchcomms NCCL backends.

import ctypes
import json
import os
import subprocess
import sys
import tempfile
import time
import unittest
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as dist
from torch._C._distributed_c10d import ErrorType, ReconfigureOptions
from torch.testing._internal.common_distributed import (
    MultiProcContinuousTest,
    requires_nccl,
    requires_nccl_version,
    skip_if_lt_x_gpu,
    skip_if_rocm_ver_atleast_multiprocess,
)
from torch.testing._internal.common_utils import (
    IS_FBCODE,
    IS_SANDCASTLE,
    run_tests,
    TEST_CUDA,
    TestCase,
)


class ProcessGroupNCCL2Test(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_watchdog_does_not_release_python_backed_tensor(self) -> None:
        class TensorSubclass(torch.Tensor):
            pass

        tensor = torch.ones(4, device=self.device).as_subclass(TensorSubclass)
        outputs = [torch.empty(4, device=self.device) for _ in range(self.world_size)]
        work = dist.all_gather(outputs, tensor, async_op=True)
        del tensor
        del work

        torch.cuda.synchronize()
        time.sleep(2)
        dist.barrier()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_shared_options_type(self) -> None:
        self.assertIs(dist.ProcessGroupNCCL2.Options, dist.ProcessGroupNCCL.Options)
        opts = dist.ProcessGroupNCCL2.Options()
        opts.config.cga_cluster_size = 2
        opts.config.max_ctas = 4
        self.assertEqual(opts.config.cga_cluster_size, 2)
        self.assertEqual(opts.config.max_ctas, 4)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduction_semantics(self) -> None:
        tensor = torch.ones(4, dtype=torch.bool, device=self.device)
        dist.all_reduce(tensor, op=dist.ReduceOp.SUM)
        self.assertEqual(
            tensor.view(torch.uint8),
            torch.ones(4, dtype=torch.uint8, device=self.device),
        )

        with self.assertRaisesRegex(TypeError, "ReduceOp.AVG"):
            dist.all_reduce(tensor, op=dist.ReduceOp.AVG)

        for dtype in (torch.float8_e4m3fnuz, torch.float8_e5m2fnuz):
            tensor = torch.ones(4, device=self.device).to(dtype)
            with self.assertRaisesRegex(RuntimeError, "Unsupported Float8"):
                dist.all_reduce(tensor)

        tensor = torch.empty(4, dtype=torch.float4_e2m1fn_x2, device=self.device)
        with self.assertRaisesRegex(RuntimeError, "Unsupported Float4"):
            dist.all_reduce(tensor)

    @requires_nccl()
    @requires_nccl_version((2, 24), "Need NCCL 2.24+ for Float8")
    @skip_if_lt_x_gpu(2)
    def test_float8_reduction(self) -> None:
        if torch.cuda.get_device_capability(self.device) < (9, 0):
            self.skipTest("Float8 reductions require sm90 or newer")
        for dtype in (torch.float8_e4m3fn, torch.float8_e5m2):
            tensor = torch.ones(4, device=self.device).to(dtype)
            dist.all_reduce(tensor)
            self.assertEqual(tensor, torch.full_like(tensor, self.world_size))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_float4_transport(self) -> None:
        tensor = torch.full(
            (4,), self.rank + 1, dtype=torch.uint8, device=self.device
        ).view(torch.float4_e2m1fn_x2)
        dist.broadcast(tensor, src=0)
        self.assertEqual(
            tensor.view(torch.uint8),
            torch.ones(4, dtype=torch.uint8, device=self.device),
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ephemeral_timeout(self) -> None:
        dist.set_timeout(timedelta(seconds=3))

        existing_work = dist.all_reduce(
            torch.ones(4, device=self.device), async_op=True
        )
        dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(timedelta(seconds=10))
        self.assertEqual(existing_work.timeout, timedelta(seconds=3))

        tensor = torch.ones(4, device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        self.assertEqual(work.timeout, timedelta(seconds=13))
        existing_work.wait()
        work.wait()
        torch.cuda.synchronize(self.device)

        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            work = dist.all_reduce(tensor, async_op=True)
            if work.timeout == timedelta(seconds=3):
                work.wait()
                return
            work.wait()
            time.sleep(0.1)
        self.fail("ephemeral timeout was not reset after collective completion")


class _ProcessGroupNCCL2OptionsTest(MultiProcContinuousTest):
    """Base for groups initialized with backend specific options."""

    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    def _check_all_reduce(self) -> None:
        t = torch.full((4,), float(self.rank), device=self.device)
        dist.all_reduce(t)
        expected = float(sum(range(self.world_size)))
        self.assertEqual(t, torch.full((4,), expected, device=self.device))


class ProcessGroupNCCL2EagerNewGroupTest(_ProcessGroupNCCL2OptionsTest):
    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        os.environ["LOCAL_RANK"] = str(rank)
        store = dist.FileStore(rdvz_file, world_size)
        dist.init_process_group(
            backend=cls.backend_str(),
            world_size=world_size,
            rank=rank,
            store=store,
            pg_options=cls.opts(),
            timeout=cls.timeout,
            device_id=torch.device("cuda", rank),
        )
        cls.pg = dist.distributed_c10d._get_default_group()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_new_group_with_nonmembers(self) -> None:
        ranks = [0]
        group = dist.new_group(ranks=ranks)
        if self.rank in ranks:
            tensor = torch.tensor([self.rank + 1], device=self.device)
            dist.all_reduce(tensor, group=group)
            self.assertEqual(tensor, torch.ones_like(tensor))
            dist.destroy_process_group(group)
        else:
            self.assertEqual(group, dist.GroupMember.NON_GROUP_MEMBER)

        store = dist.distributed_c10d._get_default_store()
        store.set(f"eager_new_group/{self.rank}", b"1")
        store.wait([f"eager_new_group/{rank}" for rank in range(self.world_size)])
        self._check_all_reduce()


class ProcessGroupNCCL2ShrinkTest(_ProcessGroupNCCL2OptionsTest):
    @requires_nccl()
    @requires_nccl_version((2, 27), "Need NCCL 2.27+ for communicator shrink")
    @skip_if_lt_x_gpu(2)
    def test_shrink_group(self) -> None:
        tensor = torch.ones(4, device=self.device)
        group = dist.new_group(device_id=self.device)
        dist.barrier(group=group)
        excluded = list(range(1, self.world_size))

        if self.rank in excluded:
            dist.destroy_process_group(group)
            return

        shrunk = dist.shrink_group(excluded, group=group)
        self.assertEqual(shrunk.size(), 1)
        dist.all_reduce(tensor, group=shrunk)
        self.assertEqual(tensor, torch.ones_like(tensor))
        dist.destroy_process_group(shrunk)


class ProcessGroupNCCL2ConfigTest(_ProcessGroupNCCL2OptionsTest):
    @classmethod
    def opts(cls, high_priority_stream=False):
        opts = dist.ProcessGroupNCCL.Options(is_high_priority_stream=True)
        opts.config.cga_cluster_size = 2
        opts.config.max_ctas = 4
        return opts

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_collective_with_config(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        self.assertEqual(backend.options.config.cga_cluster_size, 2)
        self.assertEqual(backend.options.config.max_ctas, 4)
        self.assertTrue(backend.options.is_high_priority_stream)
        self._check_all_reduce()


class ProcessGroupNCCL2NonblockingTest(_ProcessGroupNCCL2OptionsTest):
    @classmethod
    def opts(cls, high_priority_stream=False):
        opts = dist.ProcessGroupNCCL2.Options()
        opts.config.blocking = 0
        return opts

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_collectives_with_nonblocking_communicator(self) -> None:
        self._check_all_reduce()

        tensor = torch.full((4,), float(self.rank), device=self.device)
        gathered = (
            [torch.empty_like(tensor) for _ in range(self.world_size)]
            if self.rank == 0
            else None
        )
        dist.gather(tensor, gathered, dst=0)
        if self.rank == 0:
            for rank, output in enumerate(gathered):
                self.assertEqual(output, torch.full_like(output, rank))

        output = torch.empty_like(tensor)
        scatter_list = (
            [torch.full_like(tensor, float(rank)) for rank in range(self.world_size)]
            if self.rank == 0
            else None
        )
        dist.scatter(output, scatter_list, src=0)
        self.assertEqual(output, tensor)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_split_with_nonblocking_communicator(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        opts = self.opts()
        child = backend.split(dist.distributed_c10d._get_default_store(), [0], opts)
        if self.rank == 0:
            tensor = torch.ones(1, device=self.device)
            child.allreduce([tensor]).wait()
            self.assertEqual(tensor, torch.ones_like(tensor))

    @requires_nccl()
    @requires_nccl_version((2, 27), "Need NCCL 2.27+ for communicator shrink")
    @skip_if_lt_x_gpu(2)
    def test_shrink_with_nonblocking_communicator(self) -> None:
        group = dist.new_group(pg_options=self.opts(), device_id=self.device)
        dist.barrier(group=group)
        excluded = list(range(1, self.world_size))
        if self.rank in excluded:
            dist.destroy_process_group(group)
            return

        shrunk = dist.shrink_group(excluded, group=group)
        tensor = torch.ones(1, device=self.device)
        dist.all_reduce(tensor, group=shrunk)
        self.assertEqual(tensor, torch.ones_like(tensor))
        dist.destroy_process_group(shrunk)


class ProcessGroupNCCL2EnvironmentConfigTest(_ProcessGroupNCCL2OptionsTest):
    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        os.environ["TORCH_NCCL_HIGH_PRIORITY"] = "1"
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_ENABLE_TIMING"] = "1"
        os.environ["TORCH_NCCL_CUDA_EVENT_CACHE"] = "0"
        super()._init_pg(rank, world_size, rdvz_file)

    @classmethod
    def opts(cls, high_priority_stream=False):
        return dist.ProcessGroupNCCL2.Options()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_environment_config(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        self.assertEqual(backend.options.config.blocking, 0)

        tensor = torch.ones(1024, device=self.device)
        work = dist.all_reduce(tensor, async_op=True)
        work.wait()
        torch.cuda.synchronize(self.device)
        self.assertGreater(work._get_duration(), 0)


class ProcessGroupNCCL2NonblockingOptionPrecedenceTest(_ProcessGroupNCCL2OptionsTest):
    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        super()._init_pg(rank, world_size, rdvz_file)

    @classmethod
    def opts(cls, high_priority_stream=False):
        opts = dist.ProcessGroupNCCL2.Options()
        opts.config.blocking = 1
        return opts

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_explicit_option_takes_precedence(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        self.assertEqual(backend.options.config.blocking, 1)
        self._check_all_reduce()


class ProcessGroupNCCLLegacyNonblockingTest(ProcessGroupNCCL2NonblockingTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl-legacy"


class ProcessGroupNCCL2ScalableInitTest(_ProcessGroupNCCL2OptionsTest):
    ranks_per_root = 1

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        os.environ["TORCH_NCCL_RANKS_PER_ROOT"] = str(cls.ranks_per_root)
        super()._init_pg(rank, world_size, rdvz_file)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_collective_with_scalable_init(self) -> None:
        self._check_all_reduce()


class ProcessGroupNCCL2UnevenScalableInitTest(ProcessGroupNCCL2ScalableInitTest):
    world_size = 3
    ranks_per_root = 2

    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_collective_with_scalable_init(self) -> None:
        self._check_all_reduce()


class _ProcessGroupNCCL2SubgroupTest(MultiProcContinuousTest):
    """Base for tests that tear a group down.

    They run on a throwaway subgroup so the class-wide default group survives;
    a collective on that default group afterwards is what proves the process is
    still alive and healthy.
    """

    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    def _new_subgroup(self, timeout=timedelta(seconds=60)):
        return dist.new_group(
            backend="nccl2",
            use_local_synchronization=True,
            timeout=timeout,
        )

    def _check_all_reduce(self, group=None) -> None:
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t, group=group)
        self.assertEqual(t, torch.full_like(t, self.world_size))


class ProcessGroupNCCL2AbortTest(_ProcessGroupNCCL2SubgroupTest):
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_backend_abort_on_healthy_group(self) -> None:
        pg = self._new_subgroup()
        self._check_all_reduce(pg)

        backend = pg._get_backend(self.device)
        # Must return instead of ::abort()ing the process, like stock NCCL.
        backend.abort()
        self.assertEqual(backend.get_error(), ErrorType.COMM_ERROR)

        dist.destroy_process_group(pg)
        self._check_all_reduce()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_abort_process_group_on_healthy_group(self) -> None:
        pg = self._new_subgroup()
        self._check_all_reduce(pg)

        dist.distributed_c10d._abort_process_group(pg)
        self._check_all_reduce()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_destroy_process_group_returns(self) -> None:
        pg = self._new_subgroup()
        self._check_all_reduce(pg)

        dist.destroy_process_group(pg)
        self._check_all_reduce()


class ProcessGroupNCCL2WatchdogNoTearDownTest(_ProcessGroupNCCL2SubgroupTest):
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_timeout_without_process_abort(self) -> None:
        # NoHandling is how a stock NCCL user opts out of the process being
        # taken down by the watchdog; nccl2 reads the same variable, when the
        # backend is constructed.
        env = {"TORCH_NCCL_ASYNC_ERROR_HANDLING": "0"}
        with mock.patch.dict(os.environ, env):
            pg = self._new_subgroup(timeout=timedelta(seconds=5))
        backend = pg._get_backend(self.device)
        self._check_all_reduce(pg)

        if self.rank == 0:
            # Nobody else joins, so this can never complete and the watchdog
            # trips. Without the tear-down the process must survive and the
            # timeout must become readable through get_error().
            dist.all_reduce(torch.ones(1024, device=self.device), group=pg)
            deadline = time.time() + 60
            while time.time() < deadline and backend.get_error() == ErrorType.SUCCESS:
                time.sleep(0.5)
            self.assertEqual(backend.get_error(), ErrorType.TIMEOUT)
            # The next collective on the timed-out group raises rather than
            # silently proceeding on a dead communicator.
            with self.assertRaises(RuntimeError):
                dist.all_reduce(torch.ones(4, device=self.device), group=pg)
        else:
            time.sleep(30)

        dist.destroy_process_group(pg)
        self._check_all_reduce()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_timeout_with_communicator_cleanup(self) -> None:
        env = {"TORCH_NCCL_ASYNC_ERROR_HANDLING": "2"}
        with mock.patch.dict(os.environ, env):
            pg = self._new_subgroup(timeout=timedelta(seconds=5))
        backend = pg._get_backend(self.device)
        self._check_all_reduce(pg)

        if self.rank == 0:
            dist.all_reduce(torch.ones(1024, device=self.device), group=pg)
            deadline = time.time() + 60
            while time.time() < deadline and backend.get_error() == ErrorType.SUCCESS:
                time.sleep(0.5)
            self.assertEqual(backend.get_error(), ErrorType.TIMEOUT)
            with self.assertRaises(RuntimeError):
                dist.all_reduce(torch.ones(4, device=self.device), group=pg)
        else:
            time.sleep(30)

        dist.destroy_process_group(pg)
        self._check_all_reduce()


class ProcessGroupNCCL2BlockingWaitTest(_ProcessGroupNCCL2SubgroupTest):
    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        super()._init_pg(rank, world_size, rdvz_file)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_wait_times_out_and_process_survives(self) -> None:
        pg = self._new_subgroup(timeout=timedelta(seconds=5))
        self._check_all_reduce(pg)

        if self.rank == 0:
            work = dist.all_reduce(
                torch.ones(1024, device=self.device), group=pg, async_op=True
            )
            with self.assertRaisesRegex(RuntimeError, "timed out"):
                work.wait()
        else:
            time.sleep(30)

        dist.destroy_process_group(pg)
        self._check_all_reduce()


class ProcessGroupNCCL2ExpandableSegmentsTest(MultiProcContinuousTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        torch._C._accelerator_setAllocatorSettings("expandable_segments:True")
        super()._init_pg(rank, world_size, rdvz_file)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm_ver_atleast_multiprocess([7, 14])
    def test_large_in_place_all_gather(self) -> None:
        numel = 16 * 1024 * 1024
        output = torch.empty(
            self.world_size * numel, dtype=torch.bfloat16, device=self.device
        )
        input = output.narrow(0, self.rank * numel, numel)
        input.fill_(self.rank)
        self.assertTrue(
            any(segment["is_expandable"] for segment in torch.cuda.memory_snapshot())
        )

        dist.all_gather_single(output, input)

        for rank, chunk in enumerate(output.chunk(self.world_size)):
            self.assertEqual(chunk, torch.full_like(chunk, rank))


class ProcessGroupNCCL2MemPoolTest(MultiProcContinuousTest):
    """register_mem_pool / deregister_mem_pool: the NCCL user-buffer path that
    Megatron's nccl_allocator drives (megatron/core/nccl_allocator.py)."""

    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        # The device is set here rather than in setUp because _init_pg is the
        # only hook that runs in the worker processes; setUp runs in the
        # dispatcher, on rank -1. MemPool and use_mem_pool bind to the current
        # device, so without this every rank would pool on device 0 while its
        # tensors live on cuda:rank -- only rank 0 would register anything and
        # the collective below would hang.
        torch.cuda.set_device(rank)
        super()._init_pg(rank, world_size, rdvz_file)

    def _backend(self):
        # nccl2 bootstraps its communicator on the first collective, and
        # register_mem_pool needs it up (as it does on the stock backend, whose
        # test eagerly inits via init_process_group(device_id=...) instead).
        dist.all_reduce(torch.zeros(1, device=self.device))
        torch.cuda.synchronize()
        return dist.get_backend_impl(device=self.device)

    @property
    def _all_reduced(self) -> float:
        return float(sum(range(self.world_size)))

    def _pool_tensor(self, pool, numel: int = 1024 * 1024) -> torch.Tensor:
        with torch.cuda.use_mem_pool(pool):
            return torch.full((numel,), float(self.rank), device=self.device)

    def _check_all_reduce_over_pool(self, symm: bool) -> None:
        backend = self._backend()
        pool = torch.cuda.MemPool(backend.mem_allocator)
        tensor = self._pool_tensor(pool)
        backend.register_mem_pool(pool, symm=symm)
        try:
            dist.all_reduce(tensor)
            torch.cuda.synchronize()
            self.assertEqual(tensor, torch.full_like(tensor, self._all_reduced))
        finally:
            backend.deregister_mem_pool(pool)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_register_mem_pool(self) -> None:
        self._check_all_reduce_over_pool(symm=False)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_register_mem_pool_symmetric(self) -> None:
        self._check_all_reduce_over_pool(symm=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_register_mem_pool_round_trip(self) -> None:
        # Megatron re-enters its allocator context once per bucket: deregister,
        # grow the pool, register again. Neither leg may trip over a segment
        # the caching-allocator hook already registered, and the second
        # register has to pick up segments the hook never saw (reused ones).
        backend = self._backend()
        pool = torch.cuda.MemPool(backend.mem_allocator)
        first = self._pool_tensor(pool)
        backend.register_mem_pool(pool, symm=True)
        backend.deregister_mem_pool(pool)
        second = self._pool_tensor(pool)
        backend.register_mem_pool(pool, symm=True)
        try:
            for tensor in (first, second):
                dist.all_reduce(tensor)
            torch.cuda.synchronize()
            for tensor in (first, second):
                self.assertEqual(tensor, torch.full_like(tensor, self._all_reduced))
        finally:
            backend.deregister_mem_pool(pool)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_deregister_unregistered_mem_pool_raises(self) -> None:
        backend = self._backend()
        pool = torch.cuda.MemPool(backend.mem_allocator)
        with self.assertRaisesRegex(RuntimeError, "not previously registered"):
            backend.deregister_mem_pool(pool)


class ProcessGroupNCCLLazyTest(ProcessGroupNCCL2Test):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl-lazy"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reconfigure_not_supported(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        self.assertFalse(backend.supports_reconfigure)
        with self.assertRaisesRegex(
            RuntimeError, "does not support get_reconfigure_handle"
        ):
            backend.get_reconfigure_handle()
        with self.assertRaisesRegex(RuntimeError, "does not support reconfigure"):
            backend.reconfigure(ReconfigureOptions())

        opts = dist.ProcessGroupNCCL.Options()
        opts.enable_reconfigure = True
        with self.assertRaisesRegex(
            RuntimeError, "nccl-lazy does not support enable_reconfigure"
        ):
            dist.ProcessGroupNCCLLazy(
                dist.HashStore(), self.rank, self.world_size, opts
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_lazy_pair_channels(self) -> None:
        backend = dist.get_backend_impl(device=self.device)
        before_collective = backend._num_active_channels()
        t = torch.full((4,), 1.0, device=self.device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        self.assertEqual(backend._num_active_channels(), before_collective)

        send_t = torch.full((4,), float(self.rank), device=self.device)
        recv_t = torch.empty((4,), device=self.device)
        nxt = (self.rank + 1) % self.world_size
        prev = (self.rank - 1) % self.world_size
        if self.rank % 2 == 0:
            dist.send(send_t, nxt)
            dist.recv(recv_t, prev)
        else:
            dist.recv(recv_t, prev)
            dist.send(send_t, nxt)
        torch.cuda.synchronize()
        self.assertEqual(recv_t, torch.full((4,), float(prev), device=self.device))

        expected = 1 if nxt == prev else 2
        self.assertGreaterEqual(backend._num_active_channels(), expected)


class ProcessGroupNCCLLazyNonblockingTest(ProcessGroupNCCL2NonblockingTest):
    @classmethod
    def backend_str(cls) -> str:
        return "nccl-lazy"

    def test_shrink_with_nonblocking_communicator(self) -> None:
        self.skipTest("nccl-lazy does not support communicator shrink")


def _live_env(name: str) -> str | None:
    # os.environ is a snapshot taken at interpreter startup and does NOT observe
    # values set later by C++ via setenv (which is how the nccl2 backend forces
    # NCCL_ALGO under deterministic mode), so read the live environ via libc.
    libc = ctypes.CDLL(None)
    libc.getenv.restype = ctypes.c_char_p
    val = libc.getenv(name.encode())
    return val.decode() if val is not None else None


class _DeterministicNVLSTest(MultiProcContinuousTest):
    """Base for the nccl2 deterministic-mode NVLS-disable hook.

    Under torch deterministic mode, if the user has not set NCCL_ALGO the nccl2
    backend forces NCCL_ALGO=^NVLS (NVLS can give non-deterministic reductions),
    mirroring the stock ProcessGroupNCCL. The hook runs during PG init and the
    env var persists once set, so each scenario is its own subclass: every class
    gets a freshly spawned process pool (clean deterministic flag / NCCL_ALGO)
    and configures the env in _init_pg before the PG (and thus the hook) runs.
    """

    deterministic: bool = False
    preset_nccl_algo: str | None = None

    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file) -> None:
        if cls.preset_nccl_algo is not None:
            os.environ["NCCL_ALGO"] = cls.preset_nccl_algo
        else:
            os.environ.pop("NCCL_ALGO", None)
        if cls.deterministic:
            torch.use_deterministic_algorithms(True)
        super()._init_pg(rank, world_size, rdvz_file)

    def _all_reduce_and_read_algo(self) -> str | None:
        # Force the (possibly lazy) nccl2 comm to initialize so the hook runs,
        # and confirm the reduction is numerically correct under the chosen algo.
        t = torch.ones(4, device=self.device)
        dist.all_reduce(t)
        torch.cuda.synchronize()
        self.assertEqual(t, torch.full_like(t, self.world_size))
        return _live_env("NCCL_ALGO")


class ProcessGroupNCCL2DeterministicNVLSDisabledTest(_DeterministicNVLSTest):
    deterministic = True

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_deterministic_disables_nvls(self) -> None:
        self.assertEqual(self._all_reduce_and_read_algo(), "^NVLS")


class ProcessGroupNCCL2DeterministicNVLSPresetTest(_DeterministicNVLSTest):
    deterministic = True
    preset_nccl_algo = "Ring"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_deterministic_respects_user_nccl_algo(self) -> None:
        self.assertEqual(self._all_reduce_and_read_algo(), "Ring")


class ProcessGroupNCCL2NonDeterministicNVLSTest(_DeterministicNVLSTest):
    deterministic = False

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_non_deterministic_does_not_force_nccl_algo(self) -> None:
        self.assertIsNone(self._all_reduce_and_read_algo())


class ProcessGroupNCCL2ObservabilityTest(MultiProcContinuousTest):
    """What nccl2 reports about a collective to profilers and flight recorders."""

    @classmethod
    def backend_str(cls) -> str:
        return "nccl2"

    @classmethod
    def device_type(cls) -> str:
        return "cuda"

    @property
    def device(self) -> torch.device:
        return torch.device("cuda", self.rank)

    def setUp(self) -> None:
        super().setUp()
        torch.cuda.set_device(self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_profiler_sequence_number_is_per_group(self) -> None:
        # TracingGuard used to record a process-global counter of its own
        # instead of the issuing group's sequence_number_, so a rank's
        # numbering depended on how it interleaved its other groups'
        # collectives -- and profilers pair collectives across ranks by
        # (process group, sequence number).
        groups = [
            dist.new_group(
                backend=self.backend_str(),
                use_local_synchronization=True,
                timeout=timedelta(seconds=60),
            )
            for _ in range(2)
        ]
        t = torch.ones(4, device=self.device)
        try:
            # Bootstrap both communicators outside the profiled region.
            for group in groups:
                dist.all_reduce(t, group=group)
            torch.cuda.synchronize()
            # Three collectives on the first group, one on the second: a
            # process-global counter would number the second group's collective
            # after all of the first group's.
            counts = [3, 1]
            base = [group._get_sequence_number_for_group() for group in groups]

            with torch.profiler.profile(
                activities=[
                    torch.profiler.ProfilerActivity.CPU,
                    torch.profiler.ProfilerActivity.CUDA,
                ],
                record_shapes=True,
            ) as prof:
                for group, count in zip(groups, counts):
                    for _ in range(count):
                        dist.all_reduce(t, group=group)
                torch.cuda.synchronize()

            with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
                trace_path = f.name
            try:
                prof.export_chrome_trace(trace_path)
                with open(trace_path) as f:
                    trace = json.load(f)
            finally:
                os.unlink(trace_path)

            recorded: dict[str, set[int]] = {}
            for ev in trace.get("traceEvents", []):
                args = ev.get("args", {})
                if "Process Group Name" in args and "Seq" in args:
                    recorded.setdefault(args["Process Group Name"], set()).add(
                        args["Seq"]
                    )
            for group, count, first in zip(groups, counts, base):
                self.assertEqual(
                    sorted(recorded.get(group.group_name, set())),
                    [first + i + 1 for i in range(count)],
                )
        finally:
            for group in groups:
                dist.destroy_process_group(group)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_split_group_child_global_ranks(self) -> None:
        # nccl2's split() copied everything into the child's options except the
        # parent-global rank map, which FlightRecorderHook then falls back to
        # identity ranks for -- so a child's recorded membership claimed world
        # ranks instead of its own.
        pg = dist.distributed_c10d._get_default_group()
        # Bring the parent communicator up before splitting it.
        dist.all_reduce(torch.ones(4, device=self.device))
        half = self.world_size // 2
        halves = (
            list(range(half))
            if self.rank < half
            else list(range(half, self.world_size))
        )
        child = pg.split_group(halves, group_name=f"halves{halves}")
        child_options = child._get_backend(self.device).options
        self.assertEqual(child_options.global_ranks_in_group, halves)
        if hasattr(child_options.config, "comm_name"):
            # split() points config.commName at the group_name string for the
            # ncclCommSplit call; storing that pointer left the child holding
            # one that outlives its string.
            self.assertIsNone(child_options.config.comm_name)

        # Splitting the child again must map its ranks through the parent's map
        # rather than treating them as world ranks.
        mine = [halves.index(self.rank)]
        grandchild = child.split_group(mine, group_name=f"single{self.rank}")
        self.assertEqual(
            grandchild._get_backend(self.device).options.global_ranks_in_group,
            [self.rank],
        )

        t = torch.full((4,), float(self.rank), device=self.device)
        dist.all_reduce(t, group=child)
        expected = float(sum(halves))
        self.assertEqual(t, torch.full_like(t, expected))


_UNINITIALIZED_CUDA_SCRIPT = """\
import torch
import torch.distributed as dist

# Nothing above this point may touch torch.cuda: the caching allocator is
# brought up lazily by the Python torch.cuda layer, and neither CUDAGuard nor
# stream/event creation does it.
dist.init_process_group(
    "cpu:gloo,cuda:nccl2",
    rank=0,
    world_size=1,
    store=dist.HashStore(),
    device_id=torch.device("cuda:0"),
)
assert not torch.cuda.is_initialized(), "creating a process group initialized CUDA"
{extra}
dist.destroy_process_group()
"""


class ProcessGroupNCCL2UninitializedCudaTest(TestCase):
    """Constructing a process group must not need the CUDA caching allocator up.

    Eager init (`init_process_group(device_id=...)`) allocated the barrier
    buffer from the caching allocator, so a process that had made no torch.cuda
    call died inside init_process_group with "Allocator not initialized for
    device 0". Any external launcher or library that builds the PG before
    touching CUDA hits this. Runs in a subprocess because the harness (and
    every other test here) calls torch.cuda.set_device in setUp, which hides it.
    """

    def _run_child(self, extra: str = "") -> None:
        try:
            subprocess.check_output(
                [sys.executable, "-c", _UNINITIALIZED_CUDA_SCRIPT.format(extra=extra)],
                stderr=subprocess.STDOUT,
                cwd=os.path.dirname(os.path.realpath(__file__)),
                timeout=300,
            )
        except subprocess.TimeoutExpired:
            self.fail("child process timed out")
        except subprocess.CalledProcessError as e:
            self.fail(f"child process failed with:\n{e.output.decode()}")

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "subprocess test fails in fbcode")
    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_eager_init(self) -> None:
        self._run_child()

    @unittest.skipIf(IS_FBCODE or IS_SANDCASTLE, "subprocess test fails in fbcode")
    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_barrier_after_init(self) -> None:
        # The allocation is merely deferred, so barrier() is the second entrance
        # into the same allocator: without lazyInitDevice() it hits the identical
        # assert one call later.
        self._run_child("dist.barrier()")


if __name__ == "__main__":
    if TEST_CUDA:
        run_tests()
