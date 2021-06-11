import os
import sys
import unittest
from datetime import timedelta

import torch
import torch.distributed as c10d

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_nccl,
    requires_gloo,
    skip_if_lt_x_gpu,
    with_dist_debug_levels,
    create_device,
)
from torch.testing._internal.common_utils import (
    run_tests,
    TEST_WITH_TSAN,
)
from test_c10d_common import LOOPBACK


class AbstractProcessGroupWrapperTest(MultiProcessTestCase):
    def setUp(self):
        super(AbstractProcessGroupWrapperTest, self).setUp()
        # For Windows platform, Python does not support fork, change it to spawn here.
        if sys.platform == "win32":
            self._spawn_processes()
        else:
            self._fork_processes()

    def _test_collective_hang(self, wrapper_pg, use_cuda=False):
        # All ranks besides 1 call allreduce and wrapper_pg should detect a hang
        # and report an issue with rank 1.
        faulty_rank = 1
        if self.rank != faulty_rank:
            tensor = torch.randn(20, 10)
            if use_cuda:
                tensor = tensor.to(self.rank)

            if self.rank == 0:
                # Rank 0 reports faulty ranks
                err = f"Ranks {faulty_rank} failed to pass monitoredBarrier"
            else:
                err = "Please check rank 0 logs for faulty rank"
            with self.assertRaisesRegex(RuntimeError, err):
                wrapper_pg.allreduce([tensor])

    def _test_collectives_op_mismatch(self, wrapper_pg, use_cuda=False):
        tensor = torch.randn(20, 10)
        if use_cuda:
            tensor = tensor.to(self.rank)
        works = []
        # Run a few successful collectives
        for _ in range(10):
            work = wrapper_pg.allreduce([tensor])
            works.append(work)

        for w in works:
            w.wait()

        # Simulate mismatch: allreduce vs reduce.
        with self.assertRaisesRegex(
            RuntimeError, "Mismatch between collective operation types"
        ):
            if self.rank == 0:
                wrapper_pg.allreduce([tensor])
            else:
                wrapper_pg.reduce([tensor])

        # Check additional mismatches

        with self.assertRaisesRegex(
            RuntimeError, "Mismatch between collective operation types"
        ):
            if self.rank == 0:
                wrapper_pg.reduce([tensor])
            else:
                wrapper_pg.barrier()

        with self.assertRaisesRegex(
            RuntimeError, "Mismatch between collective operation types"
        ):
            scatter_result = [torch.ones(4) * i for i in range(self.world_size)]
            scattered_tensor = torch.empty(4)
            if self.rank == 0:
                wrapper_pg.scatter(scattered_tensor, scatter_result, 0)
            else:
                wrapper_pg.reduce_scatter(scattered_tensor, scatter_result)

        with self.assertRaisesRegex(
            RuntimeError, "Mismatch between collective operation types"
        ):
            if self.rank == 0:
                wrapper_pg.broadcast(tensor, 0)
            else:
                output_tensors = [
                    torch.zeros_like(tensor) for _ in range(self.world_size)
                ]
                wrapper_pg.allgather([output_tensors], [tensor])

    def _test_collective_shape_mismatch(self, wrapper_pg, use_cuda=False):
        wrapper_pg.barrier()
        dim = 2 if self.rank == 0 else 10
        tensor = torch.randn(20, dim)
        if use_cuda:
            tensor = tensor.to(self.rank)
        with self.assertRaisesRegex(RuntimeError, "Error when verifying shape tensors"):
            wrapper_pg.allreduce([tensor])
        # Check errors are raised when dimensionality of shapes is different
        tensor = torch.randn(20, 10, 2) if self.rank == 0 else torch.randn(20, 10)
        if use_cuda:
            tensor = tensor.to(self.rank)
        with self.assertRaisesRegex(RuntimeError, "Error when verifying shape tensors"):
            wrapper_pg.allreduce([tensor])

        # Check shape errors with scatter
        input = [
            torch.tensor(
                [self.rank] if self.rank == 0 else [self.rank, self.rank],
                device=self.rank if use_cuda else "cpu",
            )
            for _ in range(self.world_size)
        ]
        outputs = [
            torch.tensor(
                [-1] if self.rank == 0 else [-1, -1],
                device=self.rank if use_cuda else "cpu",
            )
            for _ in range(self.world_size)
        ]
        root_rank = 0
        opts = c10d.ScatterOptions()
        opts.rootRank = root_rank
        with self.assertRaisesRegex(RuntimeError, "Error when verifying shape tensors"):
            if self.rank == root_rank:
                wrapper_pg.scatter([outputs[self.rank]], [input], opts).wait()
            else:
                wrapper_pg.scatter([outputs[self.rank]], [], opts).wait()


@requires_gloo()
@requires_nccl()
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class ProcessGroupNCCLWrapperTest(AbstractProcessGroupWrapperTest):
    def setUp(self):
        self.num_gpus = torch.cuda.device_count()
        if self.num_gpus < 2:
            raise unittest.SkipTest("NCCL test requires 2+ GPUs")
        super(AbstractProcessGroupWrapperTest, self).setUp()
        self._spawn_processes()
        # NCCL_BLOCKING_WAIT overrides NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["NCCL_ASYNC_ERROR_HANDLING"] = "1"

    @property
    def world_size(self) -> int:
        return 2

    def _create_wrapper_pg(self, with_new_group=False, timeout=10.0):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
            timeout=timedelta(seconds=timeout),
        )
        if with_new_group:
            pg = c10d.new_group(backend="nccl", timeout=timedelta(seconds=timeout))
        else:
            _pg = c10d.ProcessGroupNCCL(
                store, self.rank, self.world_size, timeout=timedelta(seconds=timeout)
            )
            pg = c10d._create_process_group_wrapper(
                _pg,
                "unused",
                store,
                self.rank,
                self.world_size,
                timeout=timeout,
            )
        return pg

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_collective_hang(self):
        pg = self._create_wrapper_pg(timeout=2.0)
        self._test_collective_hang(pg)

    # NOTE: these tests are separated by debug level instead of combined into
    # one due to https://github.com/pytorch/pytorch/issues/55967, they can be
    # combined after that is resolved.
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collectives_op_mismatch_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_collectives_op_mismatch(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collective_shape_mismatch_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg, use_cuda=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_collective_shape_mismatch(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg, use_cuda=True)


@requires_gloo()
@unittest.skipIf(
    TEST_WITH_TSAN,
    "TSAN is not fork-safe since we're forking in a multi-threaded environment",
)
class ProcessGroupGlooWrapperTest(AbstractProcessGroupWrapperTest):
    def setUp(self):
        super(ProcessGroupGlooWrapperTest, self).setUp()

    def opts(self, threads=2, timeout=10.0):
        opts = c10d.ProcessGroupGloo._Options()
        opts._timeout = timeout
        opts._devices = [create_device(interface=LOOPBACK)]
        opts._threads = threads
        return opts

    def _create_wrapper_pg(self, with_new_group=False, timeout=10.0):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="gloo", rank=self.rank, world_size=self.world_size, store=store
        )
        if with_new_group:
            pg = c10d.new_group(backend="gloo")
        else:
            _pg = c10d.ProcessGroupGloo(
                store, self.rank, self.world_size, self.opts(timeout=timeout)
            )
            pg = c10d._create_process_group_wrapper(
                _pg,
                "unused",
                store,
                self.rank,
                self.world_size,
                timeout=timeout,
            )
        return pg

    def test_collective_hang(self):
        pg = self._create_wrapper_pg(timeout=2.0)
        self._test_collective_hang(pg)

    # NOTE: these tests are separated by debug level instead of combined into
    # one due to https://github.com/pytorch/pytorch/issues/55967, they can be
    # combined after that is resolved.
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collectives_op_mismatch_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg)

    @with_dist_debug_levels(levels=["OFF"])
    def test_collectives_op_mismatch(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg)

    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collective_shape_mismatch_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg)

    @with_dist_debug_levels(levels=["OFF"])
    def test_collective_shape_mismatch(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collectives_op_mismatch_cuda_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["OFF"])
    def test_collectives_op_mismatch_cuda(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collectives_op_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_collective_shape_mismatch_cuda_debug_mode(self):
        pg = self._create_wrapper_pg(with_new_group=True)
        self._test_collective_shape_mismatch(pg, use_cuda=True)

    @skip_if_lt_x_gpu(4)
    @with_dist_debug_levels(levels=["OFF"])
    def test_collective_shape_mismatch_cuda(self):
        pg = self._create_wrapper_pg(with_new_group=False)
        self._test_collective_shape_mismatch(pg, use_cuda=True)

if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_pg_wrapper must not have initialized CUDA context on main process"

    run_tests()
