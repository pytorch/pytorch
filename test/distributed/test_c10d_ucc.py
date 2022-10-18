# Owner(s): ["oncall: distributed"]

import copy
import logging
import math
import operator
import os
import random
import sys
import tempfile
from functools import reduce
from itertools import groupby

import torch
import torch.distributed as c10d

if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import test_c10d_common
import torch.distributed as dist
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from test_c10d_common import (
    LOOPBACK,
    gpus_for_rank,
    Task,
    ModuleForDdpCommHook,
    SparseGradientModule,
)
from torch import nn
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    requires_ucc,
    skip_if_lt_x_gpu,
    simple_sparse_reduce_tests,
    skip_if_win32,
    create_device,
    verify_ddp_error_logged,
)
from torch.testing._internal.common_utils import (
    TestCase,
    run_tests,
    retry_on_connect_failures,
    sandcastle_skip,
    sandcastle_skip_if,
)

def simple_reduce_tests(rank, world_size):
    tests = [
        (
            c10d.ReduceOp.SUM,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(world_size * (world_size + 1) / 2)]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            torch.tensor([rank + 1.0]),
            torch.tensor([float(math.factorial(world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            torch.tensor([rank + 1.0]),
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            torch.tensor([rank + 1.0]),
            torch.tensor([world_size]),
        ),
    ]

    # Generate tests for BAND.
    # The bit that is set changes in every iteration to check
    # that the output changes accordingly.
    for i in range(4):
        vin = rank | (1 << i)
        vout = 1 << i
        tests.append(
            (
                c10d.ReduceOp.BAND,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for BOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-OR'ed.
    for i in range(1, 5):
        vin = reduce(operator.or_, [rank * i + j for j in range(i)])
        vout = reduce(operator.or_, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    # Generate tests for XOR.
    # These emulate a larger world size per iteration by having every
    # rank contribute multiple values that are pre-XOR'ed.
    for i in range(1, 5):
        vin = reduce(operator.xor, [rank * i + j for j in range(i)])
        vout = reduce(operator.xor, range(world_size * i))
        tests.append(
            (
                c10d.ReduceOp.BXOR,
                torch.tensor([vin], dtype=torch.int32),
                torch.tensor([vout], dtype=torch.int32),
            ),
        )

    return tests


def simple_coalesced_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.tensor([rank + 1]), torch.tensor([(rank + 1) ** 2])],
            [
                torch.tensor([float(world_size * (world_size + 1) / 2)]),
                torch.tensor(
                    [float(world_size * (world_size + 1) * (2 * world_size + 1) / 6)]
                ),
            ],
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.tensor([rank + 1.0]), torch.tensor([rank + 2.0])],
            [
                torch.tensor([float(math.factorial(world_size))]),
                torch.tensor([float(math.factorial(world_size + 1))]),
            ],
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.tensor([rank + x]) for x in [0.0, 1.0]],
            [torch.tensor([0.0]), torch.tensor([1.0])],
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.tensor([rank + x]) for x in [1.0, 2.0]],
            [torch.tensor([world_size]), torch.tensor([world_size + 1.0])],
        ),
    ]


def simple_multi_input_reduce_tests(rank, world_size):
    return [
        (
            c10d.ReduceOp.SUM,
            [torch.tensor([2 * rank + 0.0]), torch.tensor([2 * rank + 1.0])],
            torch.tensor([float(world_size * (2 * world_size - 1))]),
        ),
        (
            c10d.ReduceOp.PRODUCT,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([float(math.factorial(2 * world_size))]),
        ),
        (
            c10d.ReduceOp.MIN,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([1.0]),
        ),
        (
            c10d.ReduceOp.MAX,
            [torch.tensor([2 * rank + 1.0]), torch.tensor([2 * rank + 2.0])],
            torch.tensor([2 * world_size]),
        ),
    ]

class RendezvousEnvTest(TestCase):
    @requires_ucc()
    @retry_on_connect_failures
    def test_logging_init(self):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = str(common.find_free_port())
        os.environ["RANK"] = "0"

        previous_handlers = logging.root.handlers

        c10d.init_process_group(backend="ucc", init_method="env://")

        current_handlers = logging.root.handlers
        self.assertEqual(len(previous_handlers), len(current_handlers))
        for current, previous in zip(current_handlers, previous_handlers):
            self.assertEqual(current, previous)

        c10d.destroy_process_group()

class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    @requires_ucc()
    @retry_on_connect_failures
    def test_default_store_timeout_ucc(self):
        self._test_default_store_timeout("ucc")

class ProcessGroupUCCTest(MultiProcessTestCase):
    def _create_process_group_ucc(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = c10d.ProcessGroupUCC(store, self.rank, self.world_size)
        dist.barrier(group=pg)
        return pg

    def setUp(self):
        super(ProcessGroupUCCTest, self).setUp()
        self._spawn_processes()

    def tearDown(self):
        super(ProcessGroupUCCTest, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @requires_ucc()
    def test_empty_tensors(self):
        pg = self._create_process_group_ucc()

        xs = [torch.FloatTensor([])]
        fut = pg.broadcast(xs).get_future()
        fut.wait()
        output = fut.value()
        self.assertEqual(0, output[0].numel())
        self.assertEqualIgnoreType(xs[0], output[0])

    # TODO: add error check testing

    def _test_broadcast_basics(self, fn):
        pg = self._create_process_group_ucc()

        def broadcast(xs, rootRank, rootTensor):
            opts = c10d.BroadcastOptions()
            opts.rootRank = rootRank
            opts.rootTensor = rootTensor
            fut = pg.broadcast(xs, opts).get_future()
            fut.wait()
            return fut.value()

        # Every rank is root once
        for i in range(self.world_size):
            # Run with 1 input tensor
            x = fn(torch.tensor([self.rank]))
            output = broadcast([x], i, 0)
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(torch.tensor([i]), output[0])
            
            # TODO: UCC currently does not support multi tensor input

        # Test overloaded convenience function
        x = torch.tensor([self.rank + 1.0])
        fut = pg.broadcast(x, root=0).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(torch.tensor([1.0]), result[0])

    @requires_ucc()
    def test_broadcast_basics(self):
        self._test_broadcast_basics(lambda t: t.clone())

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_broadcast_basics_cuda(self):
        self._test_broadcast_basics(lambda t: t.clone().cuda())

    def _test_allreduce_basics(self, fn):
        pg = self._create_process_group_ucc()

        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for (op, input, expected) in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input)
            fut = pg.allreduce([tensor], opts).get_future()
            fut.wait()
            result = fut.value()
            # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
            self.assertEqualIgnoreType(expected, result[0])

        # TODO: UCC currently does not support multi tensor input

        # Test overloaded convenience function (defaults to using sum)
        x = fn(torch.tensor([self.rank + 1.0]))
        fut = pg.allreduce(x).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(
            torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]),
            result[0],
        )

    @requires_ucc()
    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())

    # TODO: test_allreduce_basics_cuda times out locally

    # TODO: allgather tests from gloo use multi tensor input, which ucc does not support currently

    def _test_reduce_basics(self, fn):
        pg = self._create_process_group_ucc()
        for (op, input, output) in simple_reduce_tests(self.rank, self.world_size):
            for root in range(self.world_size):
                opts = c10d.ReduceOptions()
                opts.reduceOp = op
                opts.rootRank = root
                tmp = fn(input)
                fut = pg.reduce([tmp], opts).get_future()
                fut.wait()
                result = fut.value()
                if root == self.rank:
                    # TODO(#38095): Replace assertEqualIgnoreType. See issue #38095
                    self.assertEqualIgnoreType(output, result[0])

    @requires_ucc()
    def test_reduce_basics(self):
        self._test_reduce_basics(lambda t: t.clone())

    # TODO: test_reduce_basics_cuda times out locally

    @requires_ucc()
    def test_send_recv_all_to_all(self):
        pg = self._create_process_group_ucc()

        # Preallocate tensors for input/output
        inputs = [torch.tensor([self.rank]) for _ in range(self.world_size)]
        outputs = [torch.tensor([-1]) for _ in range(self.world_size)]

        # Issue sends
        send_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            send_work.append(pg.send([inputs[i]], i, 0))

        # Issue recvs
        recv_work = []
        for i in range(self.world_size):
            if i == self.rank:
                continue
            recv_work.append(pg.recv([outputs[i]], i, 0))

        # Wait for sends to complete
        for work in send_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Wait for recvs to complete
        for work in recv_work:
            work.wait()
            self.assertTrue(work.is_completed())

        # Test that every output other than our own contains the respective rank
        for i in range(self.world_size):
            if i == self.rank:
                continue
            self.assertEqual(torch.tensor([i]), outputs[i])
    
    # TODO: test_barrier_implies_wait seems to fail even after Sergey's barrier blocking fix


class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):
    @property
    def device(self):
        return "cpu"

    def setUp(self):
        super(CommTest, self).setUp()
        self._spawn_processes()

    def tearDown(self):
        super(CommTest, self).tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    # TODO: merge in sequence number support PR first
    """
    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_default_pg_ucc(self):
        self._test_sequence_num_set_default_pg(backend="ucc")

    @requires_ucc()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_ucc_new_group(self):
        self._test_sequence_num_set_new_group(backend="ucc")

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_sequence_num_incremented_ucc_default(self):
        self._test_sequence_num_incremented_default_group("ucc")

    @skip_if_lt_x_gpu(4)
    @requires_ucc()
    def test_sequence_num_incremented_ucc_subgroup(self):
        if self.world_size < 4:
            return sandcastle_skip("Test requires world_size of at least 4")
        self._test_sequence_num_incremented_subgroup("ucc")
    """

    @requires_ucc()
    def test_ucc_barrier_device_ids(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="ucc", rank=self.rank, world_size=self.world_size, store=store
        )

        with self.assertRaisesRegex(RuntimeError, "device_ids not supported"):
            c10d.barrier(device_ids=[self.rank])

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_ucc_warn_not_in_group(self):
        self._test_warn_not_in_group(backend="ucc")

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_ucc_rank_membership(self):
        self._test_rank_membership(backend="ucc")

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_tensor_dtype_mismatch(self):
        self._test_tensor_dtype_mismatch(backend="ucc")

    @skip_if_lt_x_gpu(2)
    @requires_ucc()
    def test_tensor_dtype_complex(self):
        self._test_tensor_dtype_complex(backend="ucc")

class CompilerTest(test_c10d_common.CompilerTest):

    @property
    def world_size(self):
        return 2

    def _get_default_group(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="ucc",
            rank=self.rank,
            world_size=self.world_size,
            store=store,
        )
        return dist.distributed_c10d._get_default_group()

    
    @skip_if_lt_x_gpu(2)
    def test_allreduce_work_wait_gpu(self):
        self._test_allgather_work_wait(
            torch.ones(2, 2, device=self.rank) * self.rank,
        )

    @skip_if_lt_x_gpu(2)
    def test_allgather_work_wait_gpu(self):
        self._test_allgather_work_wait(
            torch.ones(2, 2, device=self.rank) * self.rank
        )

    @skip_if_lt_x_gpu(2)
    def test_broadcast_work_wait_gpu(self):
        self._test_broadcast_work_wait(
            torch.ones(2, 2, device=self.rank) * self.rank
        )

    @skip_if_lt_x_gpu(2)
    def test_nested_comm_tensor_wrapping(self):
        self._test_nested_comm_tensor_wrapping(
            torch.ones(2, 2, device=self.rank) * self.rank
        )

    @skip_if_lt_x_gpu(2)
    def test_consecutive_comm_work_wait_gpu(self):
        self._test_consecutive_comm_work_wait(
            torch.ones(2, 2, device=self.rank) * self.rank
        )

class UccProcessGroupWithDispatchedCollectivesTests(test_c10d_common.ProcessGroupWithDispatchedCollectivesTests):
    @requires_ucc()
    @skip_if_lt_x_gpu(1)
    def test_collectives(self):
        self._test_collectives(backend="ucc")


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
