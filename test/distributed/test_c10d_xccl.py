# Owner(s): ["oncall: distributed"]

import math
import os
import sys
import time
from datetime import timedelta
from unittest import mock

import torch
import torch.distributed as c10d


if not c10d.is_available() or not c10d.is_xccl_available():
    print("c10d XCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

import test_c10d_common

import torch.distributed as dist
import torch.testing._internal.common_utils as common
from torch.testing._internal.common_distributed import (
    init_multigpu_helper,
    MultiProcessTestCase,
    requires_xccl,
)
from torch.testing._internal.common_utils import (
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle_if,
    TEST_XPU,
    TestCase,
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

    return tests


TEST_MULTIXPU = torch.xpu.device_count() > 1


class RendezvousEnvTest(TestCase):
    @retry_on_connect_failures
    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_XPU, "No GPUs available, skipping test")
    def test_common_errors(self):
        vars = {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "MASTER_ADDR": "127.0.0.1",
            "MASTER_PORT": str(common.find_free_port()),
        }

        class Env:
            def __init__(self, vars):
                self.env_patcher = mock.patch.dict(os.environ, vars, clear=True)

            def __enter__(self):
                self.env_patcher.start()

            def __exit__(self, type, value, traceback):
                self.env_patcher.stop()

        def without(d, key):
            d = d.copy()
            d.pop(key)
            return d

        def withouts(d, keys):
            d = d.copy()
            for key in keys:
                d.pop(key)
            return d

        with Env(without(vars, "WORLD_SIZE")):
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            with self.assertRaisesRegex(ValueError, "WORLD_SIZE expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend="xccl", world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "RANK")):
            self.assertEqual(None, os.environ.get("RANK"))
            with self.assertRaisesRegex(ValueError, "RANK expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend="xccl", rank=0)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            self.assertEqual(None, os.environ.get("RANK"))
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            c10d.init_process_group(backend="xccl", rank=0, world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(vars):
            c10d.init_process_group(backend="xccl")
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "MASTER_ADDR")):
            self.assertEqual(None, os.environ.get("MASTER_ADDR"))
            with self.assertRaisesRegex(ValueError, "MASTER_ADDR expected"):
                gen = c10d.rendezvous("env://")
                next(gen)

        with Env(without(vars, "MASTER_PORT")):
            self.assertEqual(None, os.environ.get("MASTER_PORT"))
            with self.assertRaisesRegex(ValueError, "MASTER_PORT expected"):
                gen = c10d.rendezvous("env://")
                next(gen)

        with Env(without(vars, "WORLD_SIZE")):
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            gen = c10d.rendezvous(f"env://?world_size={1}")
            _, _, size = next(gen)
            self.assertEqual(size, 1)

        with Env(without(vars, "RANK")):
            self.assertEqual(None, os.environ.get("RANK"))
            gen = c10d.rendezvous(f"env://?rank={0}")
            _, rank, _ = next(gen)
            self.assertEqual(rank, 0)

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            self.assertEqual(None, os.environ.get("RANK"))
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            gen = c10d.rendezvous(f"env://?rank={0}&world_size={1}")
            _, rank, size = next(gen)
            self.assertEqual(rank, 0)
            self.assertEqual(size, 1)


class TimeoutTest(test_c10d_common.AbstractTimeoutTest, TestCase):
    @requires_xccl()
    @retry_on_connect_failures
    @skip_but_pass_in_sandcastle_if(not TEST_XPU, "No GPUs available, skipping test")
    def test_default_store_timeout_nccl(self):
        self._test_default_store_timeout("xccl")


class ProcessGroupXCCLTest(MultiProcessTestCase):
    def _create_process_group_xccl(
        self, timeout=timedelta(seconds=600), device_id=None
    ):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timeout,
            device_id=device_id,
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 2

    @property
    def rank_to_GPU(self):
        # return rank to GPU map
        return init_multigpu_helper(self.world_size, "xccl")

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(
        torch.xpu.device_count() < 2, "XCCL test requires 2+ GPUs"
    )
    def test_close_multi_pg_unordered(self):
        pg = self._create_process_group_xccl()
        device = self.rank_to_GPU[self.rank][0]
        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize default PG's communicator.
        pg.allreduce(t).wait()
        new_pg1 = c10d.new_group([0, 1])
        new_pg2 = c10d.new_group([0, 1])
        if self.rank == 0 or self.rank == 1:
            t1 = torch.rand(10, 10, device=device)
            t2 = torch.rand(10, 10, device=device)
            new_pg1.allreduce(t1).wait()
            new_pg2.allreduce(t2).wait()
        if self.rank == 0:
            dist.destroy_process_group(new_pg2)
            # force destruction of pg2 first
            del new_pg2
            dist.destroy_process_group(new_pg1)
            del new_pg1
        if self.rank == 1:
            c10d.destroy_process_group(new_pg1)
            # force destruction of pg1 first
            del new_pg1
            dist.destroy_process_group(new_pg2)
            del new_pg2
        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(
        torch.xpu.device_count() < 2, "XCCL test requires 2+ GPUs"
    )
    def test_file_store_check(self):
        # self.file_name is created using "delete=False"
        # e.g., self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="xccl", rank=self.rank, world_size=self.world_size, store=store
        )
        pg = dist.distributed_c10d._get_default_group()
        self.assertEqual(pg.rank(), self.rank)
        self.assertEqual(pg.size(), self.world_size)
        # give enough time for check() to be executed multiple times
        time.sleep(2)
        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIXPU, "XCCL test requires 2+ GPUs")
    def test_set_process_group_desc(self):
        device = torch.device(f"xpu:{self.rank}")
        pg_default = self._create_process_group_xccl(device_id=device)
        self.assertEqual(pg_default.group_desc, "default_pg")
        pg_1 = c10d.new_group([0, 1], group_desc="test_purpose")
        self.assertEqual(pg_1.group_desc, "test_purpose")
        pg_2 = c10d.new_group([0, 1])
        self.assertEqual(pg_2.group_desc, "undefined")

    def _test_allreduce_basics(self, fn):
        pg = self._create_process_group_xccl()
        device = torch.device("xpu:" + str(self.rank))
        # Single input tests
        tests = simple_reduce_tests(self.rank, self.world_size)
        for op, input, expected in tests:
            opts = c10d.AllreduceOptions()
            opts.reduceOp = op
            tensor = fn(input.to(device))
            fut = pg.allreduce([tensor], opts).get_future()
            fut.wait()
            result = fut.value()
            self.assertEqual(expected, result[0], exact_dtype=False)

        x = fn(torch.tensor([self.rank + 1.0], device=device))
        fut = pg.allreduce(x).get_future()
        fut.wait()
        result = fut.value()
        self.assertEqual(
            torch.tensor([float(self.world_size * (self.world_size + 1) / 2)]),
            result[0],
        )

    @requires_xccl()
    def test_allreduce_basics(self):
        self._test_allreduce_basics(lambda t: t.clone())


if __name__ == "__main__":
    assert (
        not torch.xpu._initialized
    ), "test_distributed must not have initialized XPU context on main process"

    run_tests()
