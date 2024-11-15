# Owner(s): ["oncall: distributed"]

import copy
import json
import os
import pickle
import random
import re
import signal
import sys
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import auto, Enum
from itertools import chain, product
from unittest import mock, SkipTest

import torch
import torch.distributed as c10d
import torch.distributed._functional_collectives as _functional_collectives


if not c10d.is_available() or not c10d.is_nccl_available():
    print("c10d NCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from typing import Dict, List

import test_c10d_common
from test_c10d_common import ConvNet, DoubleGpuNet, gpus_for_rank, ModuleForDdpCommHook

import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch._C._distributed_c10d import ErrorType, OpType, WorkResult
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    get_timeout,
    init_multigpu_helper,
    MultiProcessTestCase,
    requires_gloo,
    requires_nccl,
    requires_nccl_version,
    skip_if_lt_x_gpu,
    skip_if_rocm_multiprocess,
    sm_is_or_higher_than,
    TEST_SKIPS,
    with_dist_debug_levels,
    with_nccl_blocking_wait,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_CUDA,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_WITH_ROCM,
    TestCase,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)

# bfloat16 is only supported by CUDA 11+
BFLOAT16_AVAILABLE = torch.cuda.is_available() and (
    (torch.version.cuda is not None and int(torch.version.cuda.split(".")[0]) >= 11)
    or torch.version.hip is not None
)


class RendezvousEnvTest(TestCase):
    @retry_on_connect_failures
    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
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
            c10d.init_process_group(backend="nccl", world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(without(vars, "RANK")):
            self.assertEqual(None, os.environ.get("RANK"))
            with self.assertRaisesRegex(ValueError, "RANK expected"):
                gen = c10d.rendezvous("env://")
                next(gen)
            c10d.init_process_group(backend="nccl", rank=0)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(withouts(vars, ["RANK", "WORLD_SIZE"])):
            self.assertEqual(None, os.environ.get("RANK"))
            self.assertEqual(None, os.environ.get("WORLD_SIZE"))
            c10d.init_process_group(backend="nccl", rank=0, world_size=1)
            self.assertEqual(c10d.get_rank(), 0)
            self.assertEqual(c10d.get_world_size(), 1)
            c10d.destroy_process_group()

        with Env(vars):
            c10d.init_process_group(backend="nccl")
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
    @requires_nccl()
    @retry_on_connect_failures
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    def test_default_store_timeout_nccl(self):
        self._test_default_store_timeout("nccl")


class ProcessGroupNCCLNoGPUTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        self.file = tempfile.NamedTemporaryFile(delete=False)

    def tearDown(self):
        pass

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(TEST_CUDA, "GPUs are available, skipping test")
    def test_init_no_gpus(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        with self.assertRaisesRegex(
            ValueError, "ProcessGroupNCCL is only supported with GPUs, no GPUs found!"
        ):
            c10d.ProcessGroupNCCL(store, self.rank, self.world_size)


class ProcessGroupNCCLGroupTest(MultiProcessTestCase):
    def _create_process_group_nccl(self, store, opts, device_id=None):
        # create nccl processgroup with opts
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=opts,
            device_id=device_id,
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def opts(self, high_priority_stream=False):
        opts = c10d.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = high_priority_stream
        return opts

    def setUp(self):
        super().setUp()
        # Need to skip return code checking for these tests since the child
        # processes don't exit cleanly in some cuda versions
        self.skip_return_code_checks = [
            self.test_nan_assert_float16.__wrapped__,
            self.test_nan_assert_float32.__wrapped__,
            self.test_nan_assert_float64.__wrapped__,
            self.test_nan_assert_bfloat16.__wrapped__,
            self.test_nan_assert_float8_e4m3fn.__wrapped__,
            self.test_nan_assert_float8_e5m2.__wrapped__,
        ]

        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # self.num_gpus = torch.cuda.device_count()
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
        return init_multigpu_helper(self.world_size, "nccl")

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 1 GPU")
    @skip_if_lt_x_gpu(1)
    def test_nccl_dist_backend_error(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_nccl(store, self.opts())

        # Both rank 0 and 1 will use the same CUDA device resulting in ncclInvalidUsage
        with self.assertRaises(dist.DistBackendError) as cm:
            dist.broadcast(torch.tensor([1, 2, 3]).cuda(), 0)
        self.assertTrue(isinstance(cm.exception, dist.DistError))

        self.assertIsInstance(cm.exception, RuntimeError)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_abort_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        dist.all_reduce(t)

        def abortpg():
            c10d.distributed_c10d._get_default_group()._get_backend(
                torch.device(device)
            )._shutdown()

        # Initialize DDP to ensure "destroy_process_group" will not call
        # ProcessGroupNCCL destructor since DDP holds a reference to process group.
        # Run a single iteration of DDP to initialize state.
        model = DistributedDataParallel(
            torch.nn.Linear(10, 10).to(device), device_ids=[device]
        )
        model(t).sum().backward()

        # Now simulate collective getting stuck and abort gets us unstuck
        if self.rank == 0:
            dist.all_reduce(t)

            # Schedule thread before we get stuck to abort pg.
            thread = threading.Thread(target=abortpg)
            thread.start()

            # We would get stuck here due to d2h if we didn't abort.
            t_cpu = t.cpu()

            thread.join()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("eager_init", [True, False])
    def test_close_pg(self, eager_init: bool):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device if eager_init else None,
        )

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        dist.all_reduce(t)

        # Destroy pg and validate pg is no longer valid
        dist.destroy_process_group()
        with self.assertRaises(ValueError):
            dist.all_reduce(t)

    @requires_nccl()
    @skip_if_rocm_multiprocess
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_restart_pg(self):
        # Note: restart test passes steadily only for blocking mode for now.
        # TODO: expand this test to non-blocking mode
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")

        # initialize pg for the first time
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        t0 = torch.rand(10, 10, device=device)
        # First allreduce to lazy initialize default pg
        dist.all_reduce(t0)
        torch.cuda.synchronize()
        # Destroy pg
        dist.destroy_process_group()

        # re-initialize pg
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        t1 = torch.rand(5, 5, device=device)
        dist.all_reduce(t1)
        torch.cuda.synchronize()
        dist.destroy_process_group()
        # validate default pg is no longer valid
        with self.assertRaises(ValueError):
            dist.all_reduce(t1)

    CUDA_12_AND_ABOVE = torch.cuda.is_available() and (
        torch.version.cuda is not None and int(torch.version.cuda.split(".")[0]) >= 12
    )

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        not (TEST_MULTIGPU and CUDA_12_AND_ABOVE),
        "NCCL test requires 2+ GPUs and Device side assert could cause unexpected errors in lower versions of CUDA",
    )
    @parametrize(
        "type",
        [
            torch.float16,
            torch.float32,
            torch.float64,
            torch.bfloat16,
            torch.float8_e4m3fn,
            torch.float8_e5m2,
        ],
    )
    @skip_if_rocm_multiprocess
    def test_nan_assert(self, type):
        # Expecting a device-side error when NaN is detected
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]
        # Cover different buffer sizes
        if type == torch.float64:
            size = (1024,)  # 1K elements
        elif type == torch.float32:
            size = (1024, 1024)  # 1M elements
        elif type == torch.float16:
            size = (1024, 1024, 1024)  # 1G elements
        else:
            size = (1,)  # 1 element

        # Note: currently we cannot fill values into a FP8 tensor, thus we
        # create the NaN tensor in float32 type and cast it to FP8
        if type == torch.float8_e4m3fn or type == torch.float8_e5m2:
            init_type = torch.float32
        else:
            init_type = type

        nan_tensor = torch.zeros(*size, dtype=init_type, device=device)
        # randomly pick an nan element
        index = tuple([random.randrange(size[i]) for i in range(len(size))])
        nan_tensor[index] = float("nan")
        if init_type != type:
            # Now cast to the targeted dtype
            nan_tensor = nan_tensor.to(type)

        output = torch.empty(self.world_size, *size, dtype=type, device=device)
        with self.assertRaises(RuntimeError):
            # Note: using all-gather here bc FP8 types do not support reduce ops
            # at the moment
            pg._allgather_base(output, nan_tensor)
        dist.destroy_process_group()
        # reset env
        os.environ["TORCH_NCCL_NAN_CHECK"] = "0"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nan_rank_filter(self):
        # Putting NaN at recv buffer, program should not fail as NaN checker
        # should not check on receive buffer
        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device("cuda:%d" % self.rank)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        t = torch.ones(3, 4, dtype=torch.bfloat16, device=device)
        if self.rank != 0:
            # Putting NaN at recv buffer
            t[1, 1] = float("nan")
        # Against broadcast
        c10d.broadcast(t, 0)
        # Against P2P
        if self.rank == 0:
            c10d.send(t, 1)
        elif self.rank == 1:
            c10d.recv(t, 0)
        c10d.destroy_process_group()
        # reset env
        os.environ["TORCH_NCCL_NAN_CHECK"] = "0"

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nan_check(self):
        # Not expecting an error, NaN check should not make legit code fail
        device = torch.device("cuda:%d" % self.rank)
        if not sm_is_or_higher_than(device, 8, 0):
            self.skipTest("bf16 requires sm >= 8.0")

        os.environ["TORCH_NCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        x = torch.ones((10,), dtype=torch.bfloat16, device=device) * self.rank
        t = torch.ones(3, 4, dtype=torch.bfloat16, device=device)
        c10d.broadcast(x, src=0)
        c10d.all_reduce(t)
        c10d.barrier()
        c10d.destroy_process_group()
        # reset env
        os.environ["TORCH_NCCL_NAN_CHECK"] = "0"

    def _helper_test_extra_cuda_context_by_nvml(self):
        """
        A helper for `test_extra_cuda_context`, if pynvml is avaiable.
        pynvml provides python bindings for NVIDIA NVML functionalities.
        Here we are interested in: nvmlDeviceGetComputeRunningProcesses
        """
        import pynvml

        pynvml.nvmlInit()

        device = torch.device("cuda:%d" % self.rank)
        x = torch.empty((1,), device=device)
        work = c10d.all_reduce(x, async_op=True)

        # Wait for non-0 ranks to garbage collect Work -- this is the latest
        # point where extra CUDA context can be created
        if self.rank == 0:
            time.sleep(5)
        del work
        handle = pynvml.nvmlDeviceGetHandleByIndex(self.rank)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        nprocs = len(processes)

        # A barrier for non-0 ranks
        c10d.all_reduce(x)
        torch.cuda.synchronize(device)
        c10d.destroy_process_group()
        self.assertEqual(
            nprocs,
            1,
            f"Found {nprocs} processes creating contexts on {device}, expecting 1 only",
        )

    def _helper_test_extra_cuda_context_by_memory(self):
        """
        A helper for `test_extra_cuda_context`, if pynvml is NOT avaiable.
        If extra context is created, it would manifest into device 0's memory usage.
        """
        device = torch.device("cuda:%d" % self.rank)
        x = torch.empty((1,), device=device)
        # Rank 0 takes a snapshot before collective -- this snapshot should have
        # included rank 0's own context.
        if self.rank == 0:
            free, total = torch.cuda.mem_get_info(device)
            used_before = float(total - free)

        work = c10d.all_reduce(x, async_op=True)

        # Wait for non-0 ranks to garbage collect Work -- this is the latest
        # point where extra CUDA context can be created
        if self.rank == 0:
            time.sleep(5)
            free, total = torch.cuda.mem_get_info(device)
            used_after = float(total - free)
        del work

        # A barrier for non-0 ranks
        c10d.all_reduce(x)
        torch.cuda.synchronize(device)
        c10d.destroy_process_group()
        if self.rank == 0:
            # If non-0 rank creates a context on device 0, this assert would
            # fail because one context takes about 1 GB -- much more than the
            # tensor size created in this test.
            self.assertTrue(
                used_after < used_before * 1.5,
                f"{device} used {used_after} bytes after collective, "
                f"50% more than the status before ({used_before} bytes). "
                f"Extra CUDA context may have been created.",
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_extra_cuda_context(self):
        # Check if non-0 ranks would create extra CUDA context on device 0
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device("cuda:%d" % self.rank)
        c10d.init_process_group(
            backend="nccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )
        try:
            self._helper_test_extra_cuda_context_by_nvml()
        except ModuleNotFoundError:
            self._helper_test_extra_cuda_context_by_memory()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_destruct_before_terminate_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        pg.allreduce(t)
        # force destruction before terminating comms, destructor would terminate comms
        del pg

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_abort_in_destroy_pg(self):
        # Disable ASYNC_ERROR_HANDLING for this test to ensure we can programmatically
        # abort the process group.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"

        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        pg.allreduce(t)

        # Destroy pg and validate pg is NOT in working condition since
        # we have shutdown comms
        dist.destroy_process_group()
        with self.assertRaises(dist.DistBackendError):
            pg.allreduce([t])

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_close_multi_pg_unordered(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
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

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_abort_in_destroy_multi_pgs(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]
        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize default PG's communicator.
        pg.allreduce(t).wait()
        new_pg1 = c10d.new_group([0, 1])
        new_pg2 = c10d.new_group([0, 1])
        t1 = torch.rand(10, 10, device=device)
        t2 = torch.rand(10, 10, device=device)
        new_pg1.allreduce(t1).wait()
        new_pg2.allreduce(t2).wait()
        backend = pg._get_backend(torch.device(device))
        # default PG's backend should have a split count of 0 because
        # it's not eager initialized
        self.assertEqual(backend.comm_split_count(), 0)
        # shutdown all NCCL PGs in one shot
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_abort_in_destroy_mixed_empty_pgs(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_nccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]
        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize default PG's communicator.
        pg.allreduce(t).wait()
        # PG1 is an PG without comms initialized, since we don't call collective on it
        new_pg1 = c10d.new_group([0, 1])
        new_pg2 = c10d.new_group([0, 1])
        t2 = torch.rand(10, 10, device=device)

        new_pg2.allreduce(t2).wait()
        backend = pg._get_backend(torch.device(device))
        # default PG's backend should have a split count of 0
        self.assertEqual(backend.comm_split_count(), 0)
        # shutdown all NCCL PGs in one shot
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.device_count() < 2, "NCCL test requires 2+ GPUs"
    )
    def test_file_store_check(self):
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["TORCH_NCCL_ENABLE_MONITORING"] = "0"
        # FileStore check() would be executed
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "0"

        # self.file_name is created using "delete=False"
        # e.g., self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )
        pg = dist.distributed_c10d._get_default_group()
        self.assertEqual(pg.rank(), self.rank)
        self.assertEqual(pg.size(), self.world_size)
        # give enough time for check() to be executed multiple times
        time.sleep(2)
        dist.destroy_process_group()

    def _check_nccl_timeout(self, expected_timeout):
        pg = dist.distributed_c10d._get_default_group()
        options = pg._get_backend(torch.device(f"cuda:{self.rank}")).options
        self.assertEqual(options._timeout, expected_timeout)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_CUDA, "No GPUs available, skipping test")
    def test_init_process_group_nccl_timeout(self):
        # nccl is handled 'specially' inside init_process_group and its options class is different from the options
        # used by the other PG's.  There are specific edge cases for nccl that need to be tested.

        store = c10d.FileStore(self.file_name, self.world_size)
        base_opts = dict(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )

        # test the default value coming from the `init_process_group` kwarg default
        dist.init_process_group(**base_opts)
        self._check_nccl_timeout(torch.distributed.constants.default_pg_nccl_timeout)
        dist.destroy_process_group()

        # test that `kwarg` timeout takes effect
        new_timeout = timedelta(seconds=123)
        dist.init_process_group(**base_opts, timeout=new_timeout)
        self._check_nccl_timeout(new_timeout)
        dist.destroy_process_group()

        # test that timeout value provided via `pg_options` kwarg is ignored and issues warning,
        # 'timeout' kwarg (or its kwdefault) taking precedence
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=123)
        with warnings.catch_warnings(record=True) as w:
            dist.init_process_group(**base_opts, pg_options=opts)
            # TODO(whc) i verified that we are indeed emitting this warning, and i can't figure out why i can't catch it.
            # self.assertEqual(len(w), 1)
            # self.assertTrue("pg_options._timeout was specified" in str(w[-1].message))
        self._check_nccl_timeout(torch.distributed.constants.default_pg_nccl_timeout)
        dist.destroy_process_group()

        # test that timeout value provided via `pg_options` kwarg is ignored and issues warning,
        # 'timeout' kwarg taking precedence
        opts = dist.ProcessGroupNCCL.Options()
        opts._timeout = timedelta(seconds=123)
        dist.init_process_group(
            **base_opts, pg_options=opts, timeout=timedelta(seconds=1240)
        )
        self._check_nccl_timeout(timedelta(seconds=1240))
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("backend", [None, "nccl"])
    def test_set_nccl_pg_timeout(self, backend):
        store = c10d.FileStore(self.file_name, self.world_size)
        opts = dict(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=123),
        )
        dist.init_process_group(**opts)
        pg = dist.distributed_c10d._get_default_group()
        pg.allreduce(torch.rand(10).cuda(self.rank))
        self._check_nccl_timeout(timedelta(seconds=123))
        pg._get_backend(torch.device(f"cuda:{self.rank}"))._set_default_timeout(
            timedelta(seconds=23)
        )
        self._check_nccl_timeout(timedelta(seconds=23))
        pg.allreduce(torch.rand(10).cuda(self.rank))
        c10d.distributed_c10d._set_pg_timeout(timedelta(seconds=252), pg)
        self._check_nccl_timeout(timedelta(seconds=252))

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("backend", [None, "nccl"])
    def test_extend_nccl_pg_timeout(self, backend):
        torch.cuda.set_device(self.rank)
        store = c10d.FileStore(self.file_name, self.world_size)
        opts = dict(
            backend=backend,
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            timeout=timedelta(seconds=123),
        )
        dist.init_process_group(**opts)
        pg = dist.distributed_c10d._get_default_group()
        bankend = pg._get_backend(torch.device(f"cuda:{self.rank}"))
        w = pg.allreduce(torch.rand(10).cuda(self.rank))
        self.assertTrue(bankend._verify_work_timeout(w, timedelta(seconds=123)))
        w.wait()
        bankend._set_default_timeout(timedelta(seconds=3))
        if self.rank == 0:
            # Ideally we want to sleep for a very long time, but this is not
            # feasible in unit test. So this is only a very tiny case.
            time.sleep(5)
            pg.allreduce(torch.rand(10).cuda(self.rank))
            time.sleep(5)
            pg.allreduce(torch.rand(5).cuda(self.rank))
            w = pg.allreduce(torch.rand(10).cuda(self.rank))
            self.assertTrue(bankend._verify_work_timeout(w, timedelta(seconds=3)))
            w.wait()
        else:
            dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(
                timedelta(seconds=10)
            )
            w1 = pg.allreduce(torch.rand(10).cuda(self.rank))
            w2 = pg.allreduce(torch.rand(5).cuda(self.rank))
            self.assertTrue(bankend._verify_work_timeout(w1, timedelta(seconds=13)))
            self.assertTrue(bankend._verify_work_timeout(w2, timedelta(seconds=13)))
            w1.wait()
            dist.distributed_c10d._add_ephemeral_timeout_for_all_pgs(
                timedelta(seconds=5)
            )
            # Since we are not block wait so use a sync here to leave enough time
            # for watchdog to reset first timeout extension.
            torch.cuda.synchronize(torch.device(f"cuda:{self.rank}"))
            w = pg.allreduce(torch.rand(10).cuda(self.rank))
            self.assertTrue(bankend._verify_work_timeout(w, timedelta(seconds=8)))
            w.wait()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("eager_init", [True, False])
    def test_new_group(self, eager_init: bool):
        # Test the optimization of new groups that contain all world
        # ranks use the "transparent" `ncclCommSplit` optimization.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device if eager_init else None,
        )
        ng = c10d.new_group()
        tensor = torch.tensor([self.rank], device=device)
        dist.broadcast(tensor, 0)
        dist.broadcast(tensor, 0, group=ng)
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @skip_but_pass_in_sandcastle_if(
        torch.cuda.nccl.version()[-1] == "x", "NCCL test not for NCCLX"
    )
    def test_comm_split_subgroup(self):
        # Test `ncclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        tensor = torch.full((1,), self.rank).cuda(device)
        original_tensor = tensor.clone()
        ng = c10d.new_group([0])

        # comm split happens eagerly since device_id is passed to init_process_group.
        self.assertEqual(backend.comm_split_count(), 1)
        if self.rank == 0:
            dist.broadcast(tensor, 0, group=ng)

        # no additional comm split happens after a collective.
        self.assertEqual(backend.comm_split_count(), 1)
        self.assertEqual(tensor, original_tensor)
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_comm_eager_init_subgroup(self):
        # Test `ncclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        # default PG comm is not initialized yet
        pg = self._create_process_group_nccl(store, self.opts())
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend._is_initialized(), False)
        # create a subgroup eagerly
        new_group = c10d.new_group([0, 1], device_id=device)
        tensor = torch.full((1,), self.rank).cuda(device)
        dist.broadcast(tensor, 0, group=new_group)
        # the default group should stay lazy
        self.assertEqual(backend._is_initialized(), False)
        torch.cuda.synchronize()
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_comm_split_group(self):
        # Test `ncclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        tensor = torch.full((1,), self.rank).cuda(device)
        # Create subgroup between ranks 0, 1
        subg_ranks = [0, 1]
        ng1 = c10d.split_group(pg, [subg_ranks])
        backend1 = ng1._get_backend(torch.device(device))

        # check basic options are the same between parent and child
        self.assertEqual(backend.options._timeout, backend1.options._timeout)
        self.assertEqual(
            backend.options.is_high_priority_stream,
            backend1.options.is_high_priority_stream,
        )
        self.assertEqual(ng1.group_desc, "default_pg:split:0")

        # comm split happens eagerly since device_id is passed to init_process_group.
        self.assertEqual(backend.comm_split_count(), 1)
        # dist.get_process_group_ranks returns the global ranks in the subgroup.
        self.assertEqual(
            dist.get_process_group_ranks(ng1),
            subg_ranks if self.rank in subg_ranks else [],
        )

        # is part of ng1; otherwise, -1
        if dist.get_rank(ng1) >= 0:
            dist.broadcast(tensor, dist.get_global_rank(ng1, 0), group=ng1)
            self.assertEqual(tensor, torch.full((1,), 0))

        ng2 = c10d.split_group(pg, [subg_ranks])
        self.assertEqual(ng2.group_desc, "default_pg:split:1")
        self.assertEqual(backend.comm_split_count(), 2)

        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_non_blocking_init(self):
        # Test creating a pg using nonblocking mode but not eagerly
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "100"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = self.rank_to_GPU[self.rank][0]
        pg = self._create_process_group_nccl(store, self.opts())
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend.comm_split_count(), 0)
        reduce_tensor = torch.rand(10, 10, device=device)
        # Run an allreduce, which should trigger a comm init for pg
        pg.allreduce(reduce_tensor).wait()
        new_pg = c10d.new_group()
        # even after pg's collective call, new pg's comm is not initialized until its own collectcive calls
        self.assertEqual(backend.comm_split_count(), 0)
        broadcast_tensor = torch.tensor([self.rank]).cuda(device)
        new_pg.broadcast(broadcast_tensor, 0).wait()
        self.assertEqual(backend.comm_split_count(), 0)
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_non_blocking_with_eager_init(self):
        # Test creating a pg eagerly with nonblocking mode when
        # we've passed a specific device_id to init_process_group.
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "100"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        # bound device to triger eager init mode
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend.comm_split_count(), 0)
        reduce_tensor = torch.rand(10, 10, device=device)
        # Run an allreduce, comm should have already started initilizaing,
        # but allreduce is issued to CUDA STREAM only after the initialization is a success
        pg.allreduce(reduce_tensor).wait()
        new_pg = c10d.new_group()
        # new pg's comm is initialized eagerly
        self.assertEqual(backend.comm_split_count(), 1)
        broadcast_tensor = torch.tensor([self.rank]).cuda(device)
        new_pg.broadcast(broadcast_tensor, 0).wait()
        self.assertEqual(backend.comm_split_count(), 1)
        dist.destroy_process_group()

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_non_blocking_p2p(self):
        # Test creating a pg using nonblocking mode but not eagerly
        os.environ["TORCH_NCCL_USE_COMM_NONBLOCKING"] = "1"
        os.environ["TORCH_NCCL_NONBLOCKING_TIMEOUT"] = "100"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = self.rank_to_GPU[self.rank][0]
        self._create_process_group_nccl(store, self.opts())
        # Generate the same tensor
        send_tensor = torch.ones(10, 10, device=device)
        if self.rank == 0:
            dist.send(send_tensor, 1)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, device=device)
            dist.recv(recv_tensor, 0)
            self.assertEqual(send_tensor, recv_tensor)
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_get_uid(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        from torch.distributed.distributed_c10d import _get_process_group_uid

        self.assertEqual(_get_process_group_uid(pg), 0)
        pg_2 = c10d.new_group([0, 1])
        self.assertEqual(_get_process_group_uid(pg_2), 1)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_set_process_group_desc(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg_default = self._create_process_group_nccl(
            store, self.opts(), device_id=device
        )
        self.assertEqual(pg_default.group_desc, "default_pg")
        pg_1 = c10d.new_group([0, 1], group_desc="test_purpose")
        self.assertEqual(pg_1.group_desc, "test_purpose")
        pg_2 = c10d.new_group([0, 1])
        self.assertEqual(pg_2.group_desc, "undefined")


class DistributedDataParallelTest(
    test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase
):
    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def _get_process_group(self):
        store = self._get_store()
        c10d.init_process_group(
            "nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        return c10d.distributed_c10d._get_default_group()

    def _test_nccl_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        process_group = self._get_process_group()
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_propagate_error_reason(self):
        # Need to use TORCH_NCCL_BLOCKING_WAIT and not ASYNC_ERROR_HANDLING,
        # otherwise process will be taken down and we can't check for errors.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"
        # Need to disable TORCH_NCCL_DUMP_ON_TIMEOUT otherwise this test times out
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        # provide sufficient timeout to initialize NCCL comm.
        pg = c10d.ProcessGroupNCCL(
            store, self.rank, self.world_size, timeout=timedelta(seconds=15)
        )
        pg_gloo = c10d.ProcessGroupGloo(store, self.rank, self.world_size)
        pg.barrier().wait(timedelta(seconds=5))
        # Simulate stuckness in rank 0.
        if self.rank == 0:
            pg_gloo.barrier().wait()
        inp = torch.ones(1).cuda(self.rank)

        if self.rank != 0:
            # Time out due to rank 0 not calling into allreduce.
            with self.assertRaises(dist.DistBackendError):
                pg.allreduce([inp]).wait(timedelta(seconds=5))

            # Now when nonzero rank attempts to use communicator, original failure reason should be logged.
            try:
                pg.allreduce([torch.ones(2).cuda(self.rank)]).wait()
            except dist.DistBackendError as e:
                self.assertTrue("aborted" in str(e))
            else:
                self.fail("Expected error to be raised!")

            # Unblock rank 0
            pg_gloo.barrier().wait()

        # TODO: We can also test that if rank 0 attempts to use the communicator,
        # then we should error out with the info that it was aborted due to
        # timeout on another rank. Although this would only be the case after
        # the watchdog has run on the rank, and there is no reliable way
        # to confirm it has run.

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_multi_device_ids_not_allowed(self):
        int_devices = list(range(torch.cuda.device_count()))
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            self._test_nccl_backend(devices, int_devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_single_device_module_device_ids_None(self):
        self._test_nccl_backend(None, None)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_single_device_module_empty_device_ids(self):
        # This tests the backward compatibility of accepting an empty list as `device_ids`,
        # although we no longer document this in favor of the default value of `None`,
        # which is consistent with multi-device modules and CPU modules.
        self._test_nccl_backend(None, [])

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_backend_multi_device_module_device_ids_None(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_1gpu_module_device_ids_integer_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, int_devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_backend_1gpu_module_device_ids_torch_device_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, devices)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_backend_2gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(8)
    def test_nccl_backend_4gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        self._test_nccl_backend(devices, None, multi_device=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_ddp_multi_device_module_config(self):
        gpus = gpus_for_rank(self.world_size)[self.rank]

        self.assertTrue(len(gpus) >= 2, "expecting at least 2 gpus per process")

        process_group = self._get_process_group()

        gpus = gpus[:2]
        model = DoubleGpuNet(gpus)

        with self.assertRaisesRegex(
            ValueError,
            "DistributedDataParallel device_ids and output_device arguments only work with "
            "single-device/multiple-device GPU modules or CPU modules",
        ):
            ddp_model = DistributedDataParallel(
                model, output_device=gpus[1], process_group=process_group
            )

        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            ddp_model = DistributedDataParallel(
                model, device_ids=gpus, process_group=process_group
            )

        with self.assertRaisesRegex(
            ValueError, "input module must be on the same type of devices"
        ):
            model.fc1 = model.fc1.cpu()
            ddp_model = DistributedDataParallel(model, process_group=process_group)

        model = model.cpu()
        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            ddp_model = DistributedDataParallel(
                model, device_ids=gpus, process_group=process_group
            )

    def _test_fp16(self, gradient_as_bucket_view=False):
        process_group = self._get_process_group()

        gpus = gpus_for_rank(self.world_size)[self.rank]
        model = nn.Linear(1, 1, bias=False).cuda(gpus[0]).half()
        nn.init.constant_(model.weight, 1)
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[gpus[0]],
            process_group=process_group,
            bucket_cap_mb=0.001,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        # Input 2**15, so that the gradients will overflow with a
        # world_size of 2, unless we normalize the gradient by the
        # world_size before the reduction
        input = torch.tensor([[2**15]]).cuda(gpus[0]).half()

        # Step model
        ddp_model.train()
        output = ddp_model(input)
        loss = output.sum()
        loss.backward()

        self.assertFalse(any(torch.isinf(p.grad).any() for p in ddp_model.parameters()))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16(self):
        self._test_fp16()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_grad_is_view(self):
        self._test_fp16(gradient_as_bucket_view=True)

    def _test_arbitrary_forward_return_value(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        process_group = self._get_process_group()

        class ForwardReturnValueModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x, fn):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # The first softmax does NOT include fc3 in its autograd graph
                # whereas the second softmax DOES. If we pass only the first
                # tensor we see in the output to the reducer, it marks the
                # gradient for fc3 as ready (because it doesn't show up). If
                # downstream uses of this return value choose to differentiate
                # against the second output tensor, it would still receive a
                # gradient and a callback for this tensor, resulting in a crash.
                return fn(
                    F.softmax(x, dim=1),
                    F.softmax(self.fc3(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            ForwardReturnValueModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # Always run "backward" to ensure the reducer is called by autograd.
        # If we don't correctly capture the output tensors from the return value,
        # the reducer won't see a hook for the unused parameter, and throw an error.
        # The correct capture is what we're testing in this function.
        def test(box, unbox):
            output = model(input, fn=box)
            loss = criterion(unbox(output), target)
            loss.backward()

        # Test with identity return value
        test(
            box=lambda x, y: (x, y),
            unbox=lambda obj: obj[1],
        )

        # Test with list return value
        test(
            box=lambda x, y: ["foo", x, "bar", y],
            unbox=lambda obj: obj[3],
        )

        # Test with tuple return value
        test(
            box=lambda x, y: ("foo", x, "bar", y),
            unbox=lambda obj: obj[3],
        )

        # Test with dict return value
        test(
            box=lambda x, y: {"foo": "bar", "a": x, "b": y},
            unbox=lambda obj: obj["b"],
        )

        # Test with list with dict return value
        test(
            box=lambda x, y: ["foo", "bar", {"a": x, "b": y}],
            unbox=lambda obj: obj[2]["b"],
        )

        # Test with dict with list return value
        test(
            box=lambda x, y: {"foo": "bar", "list": [0, x, 1, y]},
            unbox=lambda obj: obj["list"][3],
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value(self):
        self._test_arbitrary_forward_return_value()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value_grad_is_view(self):
        self._test_arbitrary_forward_return_value(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_with_lazy_parameters(self):
        process_group = self._get_process_group()
        with self.assertRaisesRegex(
            RuntimeError, "Modules with uninitialized parameters"
        ):
            DistributedDataParallel(
                torch.nn.LazyLinear(10), process_group=process_group
            )

    def _test_find_unused_parameters_kwarg(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        torch.cuda.set_device(self.rank)
        dist.init_process_group(
            backend="nccl",
            world_size=self.world_size,
            rank=self.rank,
            init_method=f"file://{self.file_name}",
        )
        process_group = c10d.distributed_c10d._get_default_group()

        class FindUnusedParametersModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.fc3 = nn.Linear(4, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                # Return the fc3 module so that the caller can invoke it
                # outside of the forward function. While this is bad practice,
                # we can use it to trigger a reducer error.
                return (F.softmax(x, dim=1), self.fc3)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        ddp_model = None

        def test_find_unused_parameters(
            find_unused_parameters, test_default=False, gradient_as_bucket_view=False
        ):
            if test_default:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                    gradient_as_bucket_view=gradient_as_bucket_view,
                )
            else:
                model = DistributedDataParallel(
                    FindUnusedParametersModule().float().to(device_id),
                    device_ids=[device_id],
                    process_group=process_group,
                    find_unused_parameters=find_unused_parameters,
                    gradient_as_bucket_view=gradient_as_bucket_view,
                )
            nonlocal ddp_model
            ddp_model = model

            output, fc3 = model(input)
            output = fc3(output)
            loss = criterion(output, target)
            loss.backward()

        # First test that finding unused params under these conditions is to
        # trigger an error when `backward` is called (because fc3 is an unused
        # parameter and will therefore be marked ready twice).
        try:
            test_find_unused_parameters(
                True, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.assertTrue(
                str(ex).startswith(
                    "Expected to mark a variable ready only once.",
                )
            )
            unused_index = 2
            unused_index_str = f"Parameter at index {unused_index}"
            model = ddp_model.module
            for module_name, module in model.named_modules():
                if module == model.fc3:
                    for parameter_name, _ in module.named_parameters(recurse=False):
                        unused_fqn = f"{module_name}.{parameter_name}"
                        # Only one such parameter in model.fc3, since bias=False
                        break

            if dist.get_debug_level() != dist.DebugLevel.OFF:
                unused_index_str += f" with name {unused_fqn}"

            self.assertTrue(unused_index_str in str(ex))
        else:
            self.fail("Expected exception")

        dist.barrier(process_group)

        # Then test that the default behavior can be overridden by setting
        # `find_unused_parameters=False`.
        try:
            test_find_unused_parameters(
                False, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.fail(f"Unexpected exception: {ex}")

        # Test find_unused_parameters defaults to False
        try:
            test_find_unused_parameters(
                True, test_default=True, gradient_as_bucket_view=gradient_as_bucket_view
            )
        except Exception as ex:
            self.fail(f"Unexpected exception: {ex}")

    # TODO: Combine the following tests once https://github.com/pytorch/pytorch/issues/55967
    # is resolved.
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_find_unused_parameters_kwarg_debug_detail(self):
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_find_unused_parameters_kwarg_debug_info(self):
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_find_unused_parameters_kwarg_debug_off(self):
        self._test_find_unused_parameters_kwarg()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_detail(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_info(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_off(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    def _test_multiple_outputs_multiple_backward(self, gradient_as_bucket_view=False):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        process_group = self._get_process_group()

        class MultipleOutputModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()

                def define_module():
                    return nn.Sequential(
                        nn.Linear(2, 10, bias=False),
                        nn.ReLU(),
                        nn.Linear(10, 4, bias=False),
                        nn.ReLU(),
                    )

                self.module0 = define_module()
                self.module1 = define_module()

            def forward(self, x):
                return (
                    F.softmax(self.module0(x), dim=1),
                    F.softmax(self.module1(x), dim=1),
                )

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            MultipleOutputModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        # Compute loss and gradients for both outputs
        output1, output2 = model(input)
        loss1 = criterion(output1, target)
        loss1.backward()
        loss2 = criterion(output2, target)
        loss2.backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward(self):
        self._test_multiple_outputs_multiple_backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward_grad_is_view(self):
        self._test_multiple_outputs_multiple_backward(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_no_grad(self):
        """
        Note: this test can be sped up by only running it on a CPU module
        once DistributedDataParallel supports them.
        """
        process_group = self._get_process_group()

        class NoGradModule(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = DistributedDataParallel(
            NoGradModule().float().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        input = torch.rand([batch_size, 2], dtype=torch.float)

        def check_no_grads():
            for p in model.parameters():
                self.assertTrue(p.requires_grad)
                self.assertIsNone(p.grad)

        # After initialization, no parameter has their gradient set.
        check_no_grads()

        # Run `forward` function with torch.no_grad()
        with torch.no_grad():
            output = model(input)
            self.assertTrue(isinstance(output, torch.Tensor))

        # No parameter should have their gradient set.
        check_no_grads()

    def _test_accumulate_gradients_module(self, gradient_as_bucket_view=False):
        # This is NOT the recommended way to implement accumulating grads, but
        # we would like to make sure DDP does not mess up with the underlying
        # module.
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("cuda:" + str(i)) for i in int_devices]
        process_group = self._get_process_group()
        global_batch_size = self.world_size

        model, ddp_model, input, target = self._prepare_single_device_module(
            process_group, devices, devices, global_batch_size, gradient_as_bucket_view
        )

        def step_model(model, input, target):
            model.train()
            output = model(input)
            loss = F.mse_loss(output, target.to(output.device))
            loss.backward()

        # ensure accumulate grads works with no_grad
        with torch.no_grad():
            ddp_model.train()
            ddp_model.module(input)

        # Check two model parameters over 4 iterations.
        # Use 4 iterations because we alternate between reducing and
        # not reducing and want to make sure we switch both ways.
        for iteration in range(4):
            step_model(model, input, target)

            if iteration % 2 == 0:
                # Skip gradients sync without calling prepare_for_backward
                step_model(
                    ddp_model.module,
                    input[self.rank : (self.rank + 1)],
                    target[self.rank : (self.rank + 1)],
                )
                for i, j in zip(model.parameters(), ddp_model.parameters()):
                    self.assertNotEqual(i.grad, j.grad)
            else:
                step_model(
                    ddp_model,
                    input[self.rank : (self.rank + 1)],
                    target[self.rank : (self.rank + 1)],
                )
                for i, j in zip(model.parameters(), ddp_model.parameters()):
                    self.assertEqual(i.grad, j.grad, rtol=1.3e-06, atol=5e-5)

            # Shuffle the input so that DDP input is different
            torch.manual_seed(1337 + iteration)
            input = input[torch.randperm(global_batch_size)]

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_accumulate_gradients_module(self):
        self._test_accumulate_gradients_module()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_accumulate_gradients_module_with_grad_is_view(self):
        self._test_accumulate_gradients_module(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_failure_recovery(self):
        process_group = self._get_process_group()

        # need to create a separate file for the recovered FileStore, because
        # the original one will be deleted when destructing the first FileStore.
        recovery_filename = self.file_name + "_recovery"

        if self.rank == 0:
            # the file will be deleted by the recovered FileStore
            open(recovery_filename, "w").close()

        # not necessary to run barrier here, as DDP will synchronize

        class TestModel(nn.Module):
            def __init__(self) -> None:
                super().__init__()
                self.fc1 = nn.Linear(2, 10, bias=False)
                self.fc2 = nn.Linear(10, 4, bias=False)
                self.relu = nn.ReLU()

            def forward(self, x):
                x = self.relu(self.fc1(x))
                x = self.relu(self.fc2(x))
                return F.softmax(x, dim=1)

        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        model = TestModel().float().to(device_id)
        ddp = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
        )

        batch_size = 4
        criterion = nn.CrossEntropyLoss()
        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )

        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()

        del ddp
        c10d.destroy_process_group(process_group)

        store = c10d.FileStore(recovery_filename, self.world_size)
        c10d.init_process_group(
            "nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        ddp = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
        )

        input = torch.rand([batch_size, 2], dtype=torch.float)
        target = torch.LongTensor([random.randrange(4) for _ in range(batch_size)]).to(
            device_id
        )
        for _ in range(6):
            output = ddp(input)
            loss = criterion(output, target)
            loss.backward()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_pass_default_pg(self):
        dist.init_process_group(
            "nccl",
            init_method=f"file://{self.file_name}",
            world_size=self.world_size,
            rank=self.rank,
        )

        default_pg = c10d.distributed_c10d._get_default_group()
        dist.destroy_process_group(default_pg)
        self.assertFalse(dist.is_initialized())

    def _test_grad_layout(self, replica_devices, layer_devs, local_batch_size):
        process_group = self._get_process_group()

        global_batch_size = local_batch_size * self.world_size

        # Carry out some trials with small buckets and some with big buckets.
        bucketsizes = (0.000001, 25)
        # Tuples of lists.  Each list describes per-layer characteristics for one trial.
        layer_formats = (
            [torch.contiguous_format] * 4,
            [torch.channels_last] * 2 + [torch.contiguous_format] * 2,
            [torch.channels_last] * 4,
        )
        layer_dtypes = (
            [torch.float] * 4,
            [torch.float] * 2 + [torch.half] * 2,
            [torch.half] * 4,
        )

        input_dev = layer_devs[0] if isinstance(layer_devs, list) else layer_devs
        target_dev = layer_devs[-1] if isinstance(layer_devs, list) else layer_devs
        input = torch.randn(
            (global_batch_size, 8, 8, 8), device=input_dev, dtype=torch.float
        )
        target = torch.randn(
            (global_batch_size, 8, 4, 4), device=target_dev, dtype=torch.float
        )
        local_batch_start = self.rank * local_batch_size
        local_batch_end = (self.rank + 1) * local_batch_size

        # Reducer.cpp sneakily creates one "initial bucket" that ignores the "bucket_cap_mb"
        # argument.  The following makes sure the initial bucket also complies.
        @contextmanager
        def first_bucket_size(ddp_bucket_mb):
            old_DEFAULT_FIRST_BUCKET_BYTES = dist._DEFAULT_FIRST_BUCKET_BYTES
            dist._DEFAULT_FIRST_BUCKET_BYTES = int(ddp_bucket_mb * 1.0e6)
            try:
                yield
            finally:
                dist._DEFAULT_FIRST_BUCKET_BYTES = old_DEFAULT_FIRST_BUCKET_BYTES

        with torch.backends.cudnn.flags(
            enabled=True, deterministic=True, benchmark=False
        ):
            for formats, dtypes, bucketsize in product(
                layer_formats, layer_dtypes, bucketsizes
            ):
                with first_bucket_size(bucketsize):
                    model_msg = f"rank = {self.rank} formats = {formats} dtypes = {dtypes} bucketsize = {bucketsize} "
                    try:
                        m = ConvNet(layer_devs, formats, dtypes)
                        m_ddp = DistributedDataParallel(
                            copy.deepcopy(m),
                            device_ids=replica_devices,
                            process_group=process_group,
                            bucket_cap_mb=bucketsize,
                        )
                        opt = torch.optim.SGD(m.parameters(), lr=0.1)
                        opt_ddp = torch.optim.SGD(m_ddp.parameters(), lr=0.1)
                        has_half = any(p.dtype is torch.half for p in m.parameters())
                        tol = 1.0e-3 if has_half else 1.0e-5
                    except BaseException:
                        # Prints case-specific debugging info to narrow down failing case.
                        print(
                            "Caught exception during model creation for " + model_msg,
                            flush=True,
                        )
                        raise
                    # 3 iters:  First iter creates grads, second iter retests after rebucketing,
                    # third iter tries zeroed grads.
                    for it in range(3):
                        iter_msg = f"iter = {it} " + model_msg
                        named_msg = iter_msg
                        try:
                            F.mse_loss(m(input).float(), target).backward()
                            F.mse_loss(
                                m_ddp(input[local_batch_start:local_batch_end]).float(),
                                target[local_batch_start:local_batch_end],
                            ).backward()
                            for i, ((layer_name, m_child), m_ddp_child) in enumerate(
                                zip(m.named_children(), m_ddp.module.children())
                            ):
                                named_msg = layer_name + ".weight" + " " + iter_msg
                                self.assertTrue(
                                    m_child.weight.grad.is_contiguous(
                                        memory_format=formats[i]
                                    ),
                                    named_msg,
                                )
                                self.assertTrue(
                                    m_ddp_child.weight.grad.is_contiguous(
                                        memory_format=formats[i]
                                    ),
                                    named_msg,
                                )
                                for j, ((param_name, p), p_ddp) in enumerate(
                                    zip(
                                        m_child.named_parameters(),
                                        m_ddp_child.parameters(),
                                    )
                                ):
                                    named_msg = (
                                        layer_name + "." + param_name + " " + iter_msg
                                    )
                                    self.assertEqual(
                                        p.grad, p_ddp.grad, rtol=tol, atol=tol
                                    )
                            opt.step()
                            opt_ddp.step()
                            if it == 0:
                                for p, p_ddp in zip(m.parameters(), m_ddp.parameters()):
                                    p.grad = None
                                    p_ddp.grad = None
                            else:
                                m.zero_grad()
                                m_ddp.zero_grad()
                        except BaseException:
                            # Makes sure we still get info if an error occurred somewhere other than the asserts.
                            print(
                                "Caught exception during iterations at " + named_msg,
                                flush=True,
                            )
                            raise

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_grad_layout_1devicemodule_1replicaperprocess(self):
        dev0 = torch.device("cuda:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        # Tells DDP to use just one device.
        replica_devices = [dev0]
        # Tells _test_grad_layout to construct ConvNet with all layers on this process's first assigned device.
        layer_devs = dev0
        local_batch_size = 8
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @skip_if_rocm_multiprocess
    def test_grad_layout_2devicemodule(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        dev0 = torch.device("cuda:" + str(int_devices[0]))
        dev1 = torch.device("cuda:" + str(int_devices[1]))
        # DDP's default behavior for a multi-device module is "don't replicate."
        replica_devices = None
        # Tells _test_grad_layout to constructs this process's ConvNet on 2 devices, with 2 layers on each device.
        layer_devs = [dev0] * 2 + [dev1] * 2
        local_batch_size = 8
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_param_layout_mismatch_error(self):
        process_group = self._get_process_group()

        dev0 = torch.device("cuda:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        layer_devs = dev0
        layer_formats = (
            [torch.contiguous_format] * 4
            if self.rank == 0
            else [torch.channels_last] * 4
        )
        layer_dtypes = [torch.float] * 4

        m = ConvNet(layer_devs, layer_formats, layer_dtypes)
        if self.rank == 0:
            m_ddp = DistributedDataParallel(
                m, device_ids=[dev0], process_group=process_group
            )
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                ".* appears not to match strides of the same param in process 0",
            ):
                m_ddp = DistributedDataParallel(
                    m, device_ids=[dev0], process_group=process_group
                )

    def _gpu_model_with_ddp_comm_hook(
        self,
        process_group,
        hook=None,
        gradient_as_bucket_view=False,
        state=None,
        static_graph=False,
    ):
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        gpu_model = DistributedDataParallel(
            ModuleForDdpCommHook().to(device_id),
            device_ids=[device_id],
            process_group=process_group,
            gradient_as_bucket_view=gradient_as_bucket_view,
            static_graph=static_graph,
        )

        # Register a DDP communication hook if any.
        if hook is not None:
            gpu_model.register_comm_hook(state, hook)

        return gpu_model

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_nccl(self):
        """
        This unit test verifies whether the Future object is passed properly using nccl backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        process_group = self._get_process_group()

        # Get GPU model with simple_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # check whether the grads are equal to what simple_hook's then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    def _test_ddp_comm_hook_allreduce_hook_nccl(
        self, gradient_as_bucket_view=False, static_graph=False
    ):
        """
        This unit test verifies whether a DDP communication hook that just calls
        allreduce gives the same result with the case of no hook registered.
        Without the then callback, the future_value in reducer is no longer
        a PyObject, and this unit test verifies future_value is properly checked.
        """
        process_group = self._get_process_group()

        def allreduce_hook(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            tensors = [bucket.buffer() / self.world_size]
            return (
                process_group.allreduce(tensors)
                .get_future()
                .then(lambda fut: fut.value()[0])
            )

        # Get GPU model with allreduce_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_hook, gradient_as_bucket_view, static_graph
        )

        # check whether the grads are equal to what DDP without hook would return.
        self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_default_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether default Python DDP communication hooks ALLREDUCE, FP16_COMPRESS
        and BF16_COMPRESS, can give the same result with the case of no hook registered.
        """
        process_group = self._get_process_group()

        # For these default DDP comm hooks, the only state is process group.
        state = process_group
        hook_options = [default.allreduce_hook, default.fp16_compress_hook]
        if (
            not TEST_WITH_ROCM
            and BFLOAT16_AVAILABLE
            and c10d.is_nccl_available()
            and torch.cuda.nccl.version() >= (2, 10)
        ):
            hook_options.append(default.bf16_compress_hook)
        for hook in hook_options:
            # Get GPU model with the hook registered.
            # The first arg 'process_group' is used for initializing the test environment,
            # so it cannot be replaced by 'state', although they have the same value.
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group, hook, gradient_as_bucket_view, state
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_fp16_compress_wrapper(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether wrapping the ALLREDUCE and POWER_SGD hooks with
        the FP16_WRAPPER can give the same result as when there is no hook registered.
        """
        process_group = self._get_process_group()
        powerSGD_state = powerSGD.PowerSGDState(process_group=process_group)

        hook_args = [
            (powerSGD.powerSGD_hook, powerSGD_state),
            (default.allreduce_hook, process_group),
        ]

        for hook, state in hook_args:
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group,
                default.fp16_compress_wrapper(hook),
                gradient_as_bucket_view,
                state,
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_bf16_compress_wrapper(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether wrapping the ALLREDUCE and POWER_SGD hooks with
        the BF16_WRAPPER can give the same result as when there is no hook registered.
        """
        process_group = self._get_process_group()
        powerSGD_state = powerSGD.PowerSGDState(process_group=process_group)

        hook_args = [
            (powerSGD.powerSGD_hook, powerSGD_state),
            (default.allreduce_hook, process_group),
        ]

        for hook, state in hook_args:
            gpu_model = self._gpu_model_with_ddp_comm_hook(
                process_group,
                default.bf16_compress_wrapper(hook),
                gradient_as_bucket_view,
                state,
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_powerSGD_ddp_comm_hook_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether Python DDP communication hook POWER_SGD
        can give the same result with the case of no hook registered.
        """
        process_group = self._get_process_group()

        # Get GPU model with the hook registered.
        # Test the hook with different algorithmic configs.
        for use_error_feedback, warm_start, batch_tensors_with_same_shape in product(
            [True, False],
            [True, False],
            [True, False],
        ):
            state = powerSGD.PowerSGDState(
                process_group=process_group,
                matrix_approximation_rank=1,
                use_error_feedback=use_error_feedback,
                warm_start=warm_start,
                batch_tensors_with_same_shape=batch_tensors_with_same_shape,
            )
            for hook in [powerSGD.powerSGD_hook, powerSGD.batched_powerSGD_hook]:
                gpu_model = self._gpu_model_with_ddp_comm_hook(
                    process_group, hook, gradient_as_bucket_view, state
                )

                # check whether the grads are equal to what DDP without hook would return.
                self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    def _test_builtin_ddp_comm_hooks_nccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether built-in C++ DDP communication hooks ALLREDUCE and FP16_COMPRESS
        can give the same result with the case of no hook registered.
        """
        process_group = self._get_process_group()

        for comm_hook_type in [
            dist.BuiltinCommHookType.ALLREDUCE,
            dist.BuiltinCommHookType.FP16_COMPRESS,
        ]:
            # Get GPU model with the built-in communication hook.
            gpu_model = self._gpu_model_with_builtin_ddp_comm_hook(
                process_group, comm_hook_type, gradient_as_bucket_view
            )

            # check whether the grads are equal to what DDP without hook would return.
            self._run_and_verify_hook(gpu_model, 8, 0.25 * torch.ones(2, 2))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl(self):
        self._test_ddp_comm_hook_allreduce_hook_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_nccl(self):
        self._test_default_ddp_comm_hooks_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_wrapper_nccl(self):
        self._test_fp16_compress_wrapper()

    @requires_nccl()
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for BF16_COMPRESS")
    @skip_but_pass_in_sandcastle_if(
        not BFLOAT16_AVAILABLE,
        "BFloat16 is only supported by CUDA 11+",
    )
    @skip_if_lt_x_gpu(2)
    def test_bf16_compress_wrapper_nccl(self):
        self._test_bf16_compress_wrapper()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_nccl(self):
        self._test_builtin_ddp_comm_hooks_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_nccl(self):
        self._test_powerSGD_ddp_comm_hook_nccl()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl_grad_is_view(self):
        self._test_ddp_comm_hook_allreduce_hook_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_nccl_static_graph(self):
        self._test_ddp_comm_hook_allreduce_hook_nccl(static_graph=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_nccl_is_view(self):
        self._test_default_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_wrapper_is_view(self):
        self._test_fp16_compress_wrapper(gradient_as_bucket_view=True)

    @requires_nccl()
    @requires_nccl_version((2, 10), "Need NCCL 2.10+ for BF16_COMPRESS")
    @skip_but_pass_in_sandcastle_if(
        not BFLOAT16_AVAILABLE,
        "BFloat16 is only supported by CUDA 11+",
    )
    @skip_if_lt_x_gpu(2)
    def test_bf16_compress_wrapper_is_view(self):
        self._test_bf16_compress_wrapper(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_nccl_grad_is_view(self):
        self._test_builtin_ddp_comm_hooks_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_nccl_grad_is_view(self):
        self._test_powerSGD_ddp_comm_hook_nccl(gradient_as_bucket_view=True)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_with_then_hook_nccl(self):
        """
        This unit test verifies whether a DDP communication hook that calls allreduce and then
        multiplies the result by ten and divides by two gives the expected result.
        """
        process_group = self._get_process_group()

        def allreduce_with_then_hook(
            state: object, bucket: dist.GradBucket
        ) -> torch.futures.Future[torch.Tensor]:
            tensors = [bucket.buffer() / self.world_size]
            fut = process_group.allreduce(tensors).get_future()

            def mult(fut):
                # Multiply the result by 10.
                return 10 * fut.value()[0]

            def div(fut):
                # Divide the result by 2.
                return 0.5 * fut.value()

            return fut.then(mult).then(div)

        # Get GPU model with allreduce_with_then_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(
            process_group, allreduce_with_then_hook
        )

        # check whether the grads are equal to what allreduce returns multiplied by 5.
        # without the comm_hook, result would be still 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 1.25 * torch.ones(2, 2))

    class AcceptsParam(torch.nn.Module):
        def __init__(self, p, factor):
            super().__init__()
            self.a = p
            self.f = factor

        def forward(self, input):
            return input + self.a * self.f

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_weight_sharing(self):
        process_group = self._get_process_group()

        size = 2048 * 2048
        dev = self.rank
        world = self.world_size

        p = torch.nn.Parameter(torch.randn(size, requires_grad=True))

        for try_set_to_none, use_bucket_view in product((False, True), (False, True)):
            m = torch.nn.Sequential(
                self.AcceptsParam(p, dev + 1), self.AcceptsParam(p, dev + 1)
            ).cuda(dev)

            m = torch.nn.parallel.DistributedDataParallel(
                m,
                bucket_cap_mb=1,
                gradient_as_bucket_view=use_bucket_view,
                device_ids=[dev],
                process_group=process_group,
            )

            for i in range(3):
                m.zero_grad(set_to_none=try_set_to_none)
                m(1).sum().backward()

                # Each param value is multiplied by "rank + 1" twice in forward, so the grad
                # values produced by a particular rank should be 2. * (rank + 1).
                # Summing these over ranks and dividing by world size gives the expected result:
                analytic = torch.full_like(
                    p, 2.0 * (world * (world + 1.0) / 2.0) / world, device=dev
                )
                for name, p in m.named_parameters():
                    self.assertEqual(
                        p.grad,
                        analytic,
                        "mismatch at "
                        + name
                        + ".grad for "
                        + f"set_to_none = {try_set_to_none}, use_bucket_view = {use_bucket_view}",
                    )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_packed_sequence(self):
        """
        Tests that DDP with ``device_ids`` specified can run a forward and
        backward pass with ``PackedSequence`` s with parity compared to a local
        version of the model.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        seqs = ["sequence_sequence", "seq", "sequence"]
        vocab = ["<pad>"] + sorted({ch for seq in seqs for ch in seq})
        vectorized_seqs = [[vocab.index(tok) for tok in seq] for seq in seqs]
        # Set the seed to make the embedding and LSTM deterministic (even
        # across ranks since DDP broadcasts parameters from rank 0)
        torch.manual_seed(0)
        embed = nn.Embedding(len(vocab), 4)  # keep on CPU
        lstm = nn.LSTM(input_size=4, hidden_size=2, batch_first=True).to(self.rank)
        lstm_ddp = DistributedDataParallel(
            copy.deepcopy(lstm),
            device_ids=[self.rank],
            process_group=process_group,
        )
        for p1, p2 in zip(lstm.parameters(), lstm_ddp.module.parameters()):
            self.assertEqual(p1, p2)
        seq_lengths = torch.LongTensor(list(map(len, vectorized_seqs)))
        seq_tensor = torch.Tensor(
            torch.zeros((len(vectorized_seqs), seq_lengths.max()))
        ).long()
        for i, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
            seq_tensor[i, :seq_len] = torch.LongTensor(seq)
        seq_lengths, permutation_idx = seq_lengths.sort(0, descending=True)
        seq_tensor = seq_tensor[permutation_idx]
        embedded_seq_tensor = embed(seq_tensor)
        packed_input = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_seq_tensor,
            seq_lengths,
            batch_first=True,
        )
        packed_input_ddp = torch.nn.utils.rnn.pack_padded_sequence(
            embedded_seq_tensor.detach().clone(),
            seq_lengths,
            batch_first=True,
        )
        # Move the input to GPU explicitly for the local model
        packed_output, (ht, ct) = lstm(packed_input.to(self.rank))
        # Let DDP move the input to GPU internally
        packed_output_ddp, (ht_ddp, ct_ddp) = lstm_ddp(packed_input_ddp)
        self.assertEqual(packed_output.data, packed_output_ddp.data)
        self.assertEqual(ht, ht_ddp)
        self.assertEqual(ct, ct_ddp)
        packed_output.data.sum().backward()
        packed_output_ddp.data.sum().backward()
        for p1, p2 in zip(lstm.parameters(), lstm_ddp.parameters()):
            self.assertEqual(p1.grad, p2.grad)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_channels_last_contig(self):
        process_group = self._get_process_group()
        device = torch.device(f"cuda:{self.rank}")
        tensor = torch.ones((2, 16, 768, 1152), dtype=torch.float32, device=device).to(
            memory_format=torch.channels_last
        )
        process_group.broadcast([tensor]).wait()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_complex_params(self):
        class FFTModel(nn.Module):
            def __init__(self, hin, win, n_features):
                super().__init__()
                self.hin = hin
                self.win = win
                self.weight = nn.Parameter(
                    torch.ones(
                        (n_features, n_features, hin, win // 2 + 1), dtype=torch.cfloat
                    )
                )

            def forward(self, x):
                xc = torch.fft.rfft2(
                    x, s=(self.hin, self.win), dim=(-2, -1), norm="ortho"
                )
                xcw = torch.einsum("nchw,cohw->nohw", xc, self.weight)
                x = torch.fft.irfft2(xcw, dim=(-2, -1), norm="ortho")
                return x

        process_group = self._get_process_group()
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        N, C, H, W = 1, 16, 64, 64
        ddp_model = DistributedDataParallel(
            FFTModel(hin=H, win=W, n_features=C).to(device_id),
            device_ids=[device_id],
            process_group=process_group,
        )
        optimizer = torch.optim.Adam(ddp_model.parameters(), lr=0.001)

        inp = torch.ones((N, C, H, W), dtype=torch.float32)

        # train step
        out = ddp_model(inp)
        loss = torch.sum(out)
        loss.backward()
        optimizer.step()

        torch.cuda.synchronize(device=device_id)


class WorkHookTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        # set TORCH_NCCL_ENABLE_TIMING to enable timing for CUDAEvents
        # in ProcessGroup Work
        os.environ["TORCH_NCCL_ENABLE_TIMING"] = "1"
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        del os.environ["TORCH_NCCL_ENABLE_TIMING"]
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _get_store(self):
        return dist.FileStore(self.file_name, self.world_size)

    def _get_process_group(self):
        store = self._get_store()
        c10d.init_process_group(
            "nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        return c10d.distributed_c10d._get_default_group()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_broadcast(self):
        pg = self._get_process_group()
        num_hook_fired = 0
        durations: List[float] = []

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, durations
            num_hook_fired += 1
            durations.append(work_info.active_duration.total_seconds())

        pg._register_on_completion_hook(hook)
        tensor = torch.ones([2, 3]).cuda(self.rank) * self.rank
        pg.broadcast([tensor]).wait()
        pg.broadcast([tensor]).wait()

        # N.B.: destroy_process_group is necessary to wait for
        # all pending works to finish.
        c10d.destroy_process_group(pg)

        self.assertEqual(num_hook_fired, 2)
        self.assertEqual(len(durations), 2)
        for duration in durations:
            self.assertTrue(duration > 0)

        self.assertEqual(tensor, torch.zeros([2, 3]).cuda(self.rank))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_mixed_ops(self):
        pg = self._get_process_group()
        num_hook_fired = 0
        durations: List[float] = []

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, durations
            num_hook_fired += 1
            durations.append(work_info.active_duration.total_seconds())

        pg._register_on_completion_hook(hook)
        tensor = torch.ones([2, 3]).cuda(self.rank)
        tensor_list = [torch.empty_like(tensor) for _ in range(self.world_size)]
        # intentionally using async ops.
        pg.allreduce(tensor)
        pg.allgather(tensor_list, tensor)
        pg.allreduce(tensor)

        # N.B.: destroy_process_group is necessary to wait for
        # all pending works to finish.
        c10d.destroy_process_group(pg)

        self.assertEqual(num_hook_fired, 3)
        self.assertEqual(len(durations), 3)
        for duration in durations:
            self.assertTrue(duration > 0)

        self.assertEqual(
            tensor,
            torch.ones([2, 3]).cuda(self.rank) * self.world_size * self.world_size,
        )

        self.assertEqual(
            tensor_list,
            [
                torch.ones([2, 3]).cuda(self.rank) * self.world_size
                for _ in range(self.world_size)
            ],
        )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_with_ddp(self):
        pg = self._get_process_group()
        num_hook_fired: Dict[int, int] = {}
        durations: Dict[OpType, List[float]] = {}

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, durations
            op_type = work_info.op_type
            if op_type not in num_hook_fired:
                num_hook_fired[op_type] = 0
                durations[op_type] = []
            num_hook_fired[op_type] += 1
            durations[op_type].append(work_info.active_duration.total_seconds())

        pg._register_on_completion_hook(hook)

        nlayers = 10
        net = nn.Sequential(
            *[nn.Linear(1000, 1000, bias=False) for _ in range(nlayers)]
        ).to(self.rank)

        ddp = DistributedDataParallel(
            net,
            device_ids=[self.rank],
            process_group=pg,
            bucket_cap_mb=1,
        )

        pg._wait_for_pending_works()

        # DDP is expected to synchronize model parameter by broadcasting
        # from rank0 to other ranks. However, this is DDP's internal implementation,
        # which is subject to change in future versions.
        self.assertTrue(num_hook_fired[OpType.BROADCAST] > 0)
        ctor_allreduce = (
            num_hook_fired[OpType.ALLREDUCE]
            if OpType.ALLREDUCE in num_hook_fired
            else 0
        )

        x = torch.zeros(2, 1000).cuda(self.rank)
        ddp(x).sum().backward()

        c10d.destroy_process_group(pg)

        self.assertTrue(OpType.ALLREDUCE in num_hook_fired)
        # The number of allreduce ops depend on DDP internal implementation, but
        # there should be at least one allreduce.
        self.assertTrue(num_hook_fired[OpType.ALLREDUCE] - ctor_allreduce > 0)
        self.assertTrue(all(duration > 0 for duration in chain(*(durations.values()))))

    # Not testing FSDP due to https://github.com/pytorch/pytorch/issues/90848.
    # We cannot disable workCleanupLoop() as hooks are fired in that thread.

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_all_gather_object(self):
        torch.cuda.set_device(self.rank)

        pg = self._get_process_group()
        num_hook_fired: Dict[int, int] = {}
        durations: Dict[OpType, List[float]] = {}

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, durations
            op_type = work_info.op_type
            if op_type not in num_hook_fired:
                num_hook_fired[op_type] = 0
                durations[op_type] = []
            num_hook_fired[op_type] += 1
            durations[op_type].append(work_info.active_duration.total_seconds())

        pg._register_on_completion_hook(hook)

        obj = {"rank": self.rank, "world_size": self.world_size}
        obj_list = [None for _ in range(self.world_size)]

        c10d.all_gather_object(obj_list, obj, group=pg)

        for r, o in enumerate(obj_list):
            self.assertTrue(isinstance(o, dict))
            self.assertTrue(set(o.keys()), {"rank", "world_size"})
            self.assertEqual(o["rank"], r)
            self.assertEqual(o["world_size"], self.world_size)

        c10d.destroy_process_group(pg)

        self.assertTrue(OpType.ALLGATHER in num_hook_fired)
        self.assertEqual(len(num_hook_fired), 1)
        # two allgathers, one for size and another for values
        self.assertEqual(num_hook_fired[OpType.ALLGATHER], 2)
        self.assertTrue(all(duration > 0 for duration in durations[OpType.ALLGATHER]))

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_on_completion_hook_seq(self):
        pg = self._get_process_group()
        num_hook_fired = 0
        seq: int = -1
        work: int = 0

        def hook(work_info: torch._C._distributed_c10d.WorkInfo):
            nonlocal num_hook_fired, seq
            num_hook_fired += 1
            seq = work_info.seq

        pg._register_on_completion_hook(hook)
        tensor = torch.ones([2, 3]).cuda(self.rank) * self.rank
        work_count = 3
        for i in range(work_count):
            work += 1
            pg.broadcast([tensor]).wait()

        # N.B.: destroy_process_group is necessary to wait for
        # all pending works to finish.
        c10d.destroy_process_group(pg)

        self.assertEqual(num_hook_fired, work_count)
        self.assertEqual(work, seq)


class NcclErrorHandlingTest(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # Need to skip return code checking for these tests since the child
        # processes don't exit cleanly.
        self.skip_return_code_checks = [
            self.test_nccl_errors_blocking_abort.__wrapped__,
            self.test_nccl_errors_blocking_sigkill.__wrapped__,
            self.test_nccl_errors_blocking_sigterm.__wrapped__,
            self.test_nccl_errors_blocking_nonzero_exit.__wrapped__,
        ]
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def op_timeout_sec(self):
        return 3

    @property
    def world_size(self):
        return 3

    @property
    def blocking_wait_error_msg(self):
        return "timeout"

    def _run_all_reduce(self, pg):
        pg.allreduce(torch.rand(10).cuda(self.rank))

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm_multiprocess
    @skip_but_pass_in_sandcastle("Test does not pass when run locally")
    def test_nccl_errors_nonblocking(self):
        # Note: we unset and restore TORCH_NCCL_ASYNC_ERROR_HANDLING for this test
        # since test_c10d_common runs with async error handling by default, but this
        # tests behavior when it is not enabled.
        prev_nccl_async_error_handling = os.environ.get(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING", None
        )
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)
        process_group.allreduce(torch.rand(10).cuda(self.rank))
        if self.rank == 0:
            # This allreduce does not block Python thread as allreduce enqueues
            # the cuda operation, and then wait only blocks the current cuda
            # stream.
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            work.wait()

            # Now the work scheduled next should hang forever since the previous
            # allreduce will never complete.
            t = threading.Thread(target=self._run_all_reduce, args=(process_group,))
            t.daemon = True
            t.start()
            t.join(int(get_timeout(self.id()) / 5))
            self.assertTrue(t.is_alive())

        if prev_nccl_async_error_handling is not None:
            os.environ[
                "TORCH_NCCL_ASYNC_ERROR_HANDLING"
            ] = prev_nccl_async_error_handling

    def _test_nccl_errors_blocking(self, func):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=10),
        )
        process_group.allreduce(torch.rand(10).cuda(self.rank))
        if self.rank == 0:
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                # It seems the error message would be different depending on
                # whether the test is run on CI machine and devGPU.  Skipping
                # the error message check to make both sides happy.
                work.wait(timeout=timedelta(seconds=self.op_timeout_sec))
            # Run some GPU operations to make sure cuda has not gotten stuck.
            # It was observed cuda could get stuck if NCCL communicators were
            # not properly aborted before throwing RuntimeError.
            a = torch.rand(10).cuda(self.rank)
        elif self.rank == 1:
            # Clean up structures (ex: files for FileStore before going down)
            del process_group
            func()

    def _test_barrier_error(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=10),
        )
        process_group.barrier().wait()
        if self.rank == 0:
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                # It seems the error message would be different depending on
                # whether the test is run on CI machine and devGPU.  Skipping
                # the error message check to make both sides happy.
                process_group.barrier().wait(
                    timeout=timedelta(seconds=self.op_timeout_sec)
                )

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm_multiprocess
    def test_nccl_errors_blocking_clean_exit(self):
        self._test_nccl_errors_blocking(lambda: sys.exit(0))

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm_multiprocess
    def test_nccl_errors_blocking_nonzero_exit(self):
        self._test_nccl_errors_blocking(lambda: sys.exit(1))

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm_multiprocess
    @skip_but_pass_in_sandcastle(
        "Frequently times out see https://github.com/pytorch/pytorch/issues/58920"
    )
    def test_nccl_errors_blocking_abort(self):
        self._test_nccl_errors_blocking(lambda: os.abort())

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm_multiprocess
    def test_nccl_errors_blocking_sigkill(self):
        self._test_nccl_errors_blocking(lambda: os.kill(os.getpid(), signal.SIGKILL))

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    @skip_if_rocm_multiprocess
    def test_nccl_errors_blocking_sigterm(self):
        self._test_nccl_errors_blocking(lambda: os.kill(os.getpid(), signal.SIGTERM))

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_nccl_blocking_wait_with_barrier(self):
        self._test_barrier_error()

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_nccl_non_blocking_wait_with_barrier(self):
        # test the barrier behavior in the non blocking wait setting
        prev_nccl_async_error_handling = os.environ.get(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING", None
        )
        # avoid watchdog thread interference
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        self._test_barrier_error()
        if prev_nccl_async_error_handling is not None:
            os.environ[
                "TORCH_NCCL_ASYNC_ERROR_HANDLING"
            ] = prev_nccl_async_error_handling

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(3)
    def test_error_detection_and_propagation(self):
        def assert_fut_success(fut):
            self.assertEqual(WorkResult(fut.value()), WorkResult.SUCCESS)

        # test the barrier behavior in the non blocking wait setting
        prev_nccl_async_error_handling = os.environ.get(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING", None
        )
        # avoid watchdog thread interference
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=2),
        )
        self.assertEqual(process_group.get_error(), ErrorType.SUCCESS)
        barrier_work = process_group.barrier()
        barrier_work.wait()
        barrier_result = barrier_work.get_future_result().wait()
        self.assertEqual(WorkResult(barrier_result), WorkResult.SUCCESS)
        ar_work = process_group.allreduce(torch.rand(10).cuda(self.rank))
        ar_work.wait()
        fut = ar_work.get_future_result()
        # test adding a callback function
        fut.then(assert_fut_success)
        if self.rank == 0:
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            work.wait()
            result = work.get_future_result().wait()
            self.assertEqual(WorkResult(result), WorkResult.TIMEOUT)
            self.assertEqual(process_group.get_error(), ErrorType.TIMEOUT)
        else:
            # other ranks not exiting before rank 0 timeout, this is to avoid
            # nccl error happening before rank 0 timeouts
            time.sleep(4)
            self.assertEqual(process_group.get_error(), ErrorType.REMOTE_ERROR)

        if prev_nccl_async_error_handling is not None:
            os.environ[
                "TORCH_NCCL_ASYNC_ERROR_HANDLING"
            ] = prev_nccl_async_error_handling

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_rocm_multiprocess
    @skip_if_lt_x_gpu(3)
    def test_restart_pg_after_error(self):
        # test the barrier behavior in the non blocking wait setting
        prev_nccl_async_error_handling = os.environ.get(
            "TORCH_NCCL_ASYNC_ERROR_HANDLING", None
        )
        # avoid watchdog thread interference
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "0"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank % torch.cuda.device_count()}")
        # initialize pg for the first time
        c10d.init_process_group(
            "nccl",
            timeout=timedelta(seconds=2),
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        pg = c10d.distributed_c10d._get_default_group()
        nccl_backend = pg._get_backend(torch.device(device))
        self.assertEqual(nccl_backend.get_error(), ErrorType.SUCCESS)
        barrier_work = nccl_backend.barrier()
        barrier_work.wait()
        barrier_result = barrier_work.get_future_result().wait()
        self.assertEqual(nccl_backend.get_error(), ErrorType.SUCCESS)
        if self.rank == 0:
            work = nccl_backend.allreduce(torch.rand(10).cuda(self.rank))
            work.wait()
            result = work.get_future_result().wait()
            self.assertEqual(nccl_backend.get_error(), ErrorType.TIMEOUT)
            # we need a brand new fileStore for the new PG
            # the new file name is shared through the old fileStore
            new_file_name = tempfile.NamedTemporaryFile(delete=False).name
            store.set("file", new_file_name)
        else:
            # other ranks not exiting before rank 0 timeout, this is to avoid
            # nccl error happening before rank 0 timeouts
            time.sleep(4)
            self.assertEqual(nccl_backend.get_error(), ErrorType.REMOTE_ERROR)
            new_file_name = store.get("file").decode()

        # all ranks restart using a new store after detecting the timeout error
        dist.destroy_process_group()

        new_store = c10d.FileStore(new_file_name, self.world_size)
        # re-initialize pg
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=new_store,
        )

        new_pg = c10d.distributed_c10d._get_default_group()
        new_nccl_backend = new_pg._get_backend(torch.device(device))
        t = torch.rand(5, 5, device=device)
        dist.all_reduce(t)
        self.assertEqual(new_nccl_backend.get_error(), ErrorType.SUCCESS)
        torch.cuda.synchronize()
        dist.destroy_process_group()

        # give some time for other ranks to exit first before destroying FileStore
        if self.rank == 0:
            time.sleep(4)
            os.remove(new_file_name)

        if prev_nccl_async_error_handling is not None:
            os.environ[
                "TORCH_NCCL_ASYNC_ERROR_HANDLING"
            ] = prev_nccl_async_error_handling

    def _run_invalid_nccl_blocking_wait_env(self, val):
        os.environ["TORCH_NCCL_BLOCKING_WAIT"] = val
        store = c10d.FileStore(self.file_name, self.world_size)
        with self.assertRaises(RuntimeError):
            process_group = c10d.ProcessGroupNCCL(store, self.rank, self.world_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(3)
    def test_invalid_nccl_blocking_wait_env(self):
        self._run_invalid_nccl_blocking_wait_env("abc")
        self._run_invalid_nccl_blocking_wait_env("-1")
        self._run_invalid_nccl_blocking_wait_env("2147483647")
        self._run_invalid_nccl_blocking_wait_env("4294967295")

    @with_nccl_blocking_wait
    @requires_nccl()
    @requires_gloo()
    @skip_if_lt_x_gpu(3)
    def test_nccl_timeout(self):
        store = c10d.FileStore(self.file_name, self.world_size)

        # Initialize process_group.
        process_group = c10d.ProcessGroupNCCL(
            store, self.rank, self.world_size, timeout=timedelta(seconds=10)
        )
        # Control gloo pg used as go-ahead signal/barrier
        # to coordinate btwn ranks.
        pg_gloo = c10d.ProcessGroupGloo(store, self.rank, self.world_size)
        failed_collective_timeout = timedelta(milliseconds=100)
        process_group.allreduce(torch.rand(10).cuda(self.rank)).wait(
            timeout=timedelta(seconds=5)
        )

        if self.rank == 0:
            # This should timeout in about 1 second.
            # Watchdog may abort timed out work resulting in NCCL error instead of operation timed out.
            with self.assertRaisesRegex(
                dist.DistBackendError, self.blocking_wait_error_msg
            ):
                process_group.allreduce(torch.rand(10).cuda(self.rank)).wait(
                    timeout=failed_collective_timeout
                )
            # Now do a barrier to tell other rank to go ahead.
            pg_gloo.barrier().wait()
        else:
            # Wait on rank 0 to fail.
            try:
                pg_gloo.barrier().wait()
            except Exception as e:
                raise ValueError(
                    f"Rank {self.rank} barrier timed out waiting for rank 0 with error: {str(e)}"
                ) from e


class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):
    @property
    def device(self):
        return f"cuda:{self.rank}"

    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def _test_broadcast_coalesced(self, process_group, device, root_rank):
        half = torch.float16

        # No support for float16 for CPU tensors
        if device == torch.device("cpu"):
            half = torch.float32

        target = torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float64, device=device).chunk(5)
        target += torch.arange(60, dtype=half, device=device).chunk(5)
        target += torch.arange(60, dtype=torch.float32, device=device).chunk(5)

        # The tensors to pass to broadcast are identical to the target
        # only on the process that is the root of the broadcast.
        if self.rank == root_rank:
            tensors = [tensor.clone() for tensor in target]
        else:
            tensors = [torch.zeros_like(tensor) for tensor in target]

        if self.rank != root_rank:
            self.assertNotEqual(tensors, target)

        c10d._broadcast_coalesced(
            process_group, tensors, buffer_size=256, src=root_rank
        )

        if self.rank != root_rank:
            self.assertEqual(tensors, target)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_coalesced_nccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device("cuda:%d" % self.rank)
        ranks = [0, 1]
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_nccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device("cuda:%d" % self.rank)
        tensors = [
            torch.full((60 + i,), self.rank + 1 + i, device=device, dtype=torch.float)
            for i in range(5)
        ]
        torch.distributed.all_reduce_coalesced(tensors, group=process_group)
        for i, t in enumerate(tensors):
            self.assertEqual(
                t,
                torch.full_like(
                    t, self.world_size * (i + (self.world_size + 1.0) / 2.0)
                ),
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_nccl_float8_errors(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device("cuda:%d" % self.rank)
        tensors = [
            torch.full(
                (60 + i,), self.rank + 1 + i, device=device, dtype=torch.float
            ).to(torch.float8_e4m3fn)
            for i in range(5)
        ]
        with self.assertRaisesRegex(
            RuntimeError,
            "Float8 dtypes are not currenlty supported for NCCL reductions",
        ):
            torch.distributed.all_reduce_coalesced(tensors, group=process_group)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_manager_nccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device("cuda:%d" % self.rank)
        tensors = [
            torch.full((60 + i,), self.rank + 1 + i, device=device, dtype=torch.float)
            for i in range(5)
        ]
        with torch.distributed._coalescing_manager(
            group=process_group, device=device, async_ops=True
        ) as cm:
            for tensor in tensors:
                torch.distributed.all_reduce(tensor)
        self.assertEqual(len(cm.works), 1)
        cm.wait()
        for i, t in enumerate(tensors):
            self.assertEqual(
                t,
                torch.full_like(
                    t, self.world_size * (i + (self.world_size + 1.0) / 2.0)
                ),
            )

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm_multiprocess
    def test_intra_node_comm_all_reduce(self):
        from torch._C._distributed_c10d import _get_intra_node_comm_usage_counter
        from torch.testing._internal.common_cuda import SM80OrLater

        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            if not torch._C._cuda_canDeviceAccessPeer(self.rank, peer):
                raise SkipTest("Test requires p2p access")

        if not SM80OrLater:
            raise SkipTest("Test requires sm>=80")

        store = c10d.FileStore(self.file_name, self.world_size)
        os.environ["ENABLE_INTRA_NODE_COMM"] = "1"
        os.environ["TEST_INTRA_NODE_COMM"] = "1"
        torch.cuda.set_device(self.rank)
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )
        expect = self.world_size * (self.world_size - 1) // 2

        # IntraNodeComm currently only supports sum and bf16.
        # Verify that it is not used in the next two configurations.
        t = torch.full((4 * 1024 // 2,), self.rank).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 0)

        t = torch.full((4 * 1024 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.AVG)
        self.assertEqual(_get_intra_node_comm_usage_counter(), 0)

        # Verify that IntraNodeComm is used up to 10MB
        t = torch.full((4 * 1024 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 1)

        t = torch.full((512 * 1024 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 2)

        t = torch.full((10 * 1024**2 // 2,), self.rank, dtype=torch.bfloat16).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 3)

        # Verify that IntraNodeComm is not used beyond 10MB
        t = torch.full(
            (10 * 1024**2 // 2 + 1,), self.rank, dtype=torch.bfloat16
        ).cuda()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())
        self.assertEqual(_get_intra_node_comm_usage_counter(), 3)

        c10d.destroy_process_group()

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_default_pg_nccl(self):
        torch.cuda.set_device(self.rank)
        self._test_sequence_num_set_default_pg(backend="nccl")

    @skip_if_lt_x_gpu(2)
    @requires_nccl()
    def test_sequence_num_incremented_nccl_default(self):
        self._test_sequence_num_incremented_default_group("nccl")

    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sequence_num_incremented_nccl_subgroup(self):
        if self.world_size < 4:
            return skip_but_pass_in_sandcastle("Test requires world_size of at least 4")
        self._test_sequence_num_incremented_subgroup("nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_nccl_new_group(self):
        torch.cuda.set_device(self.rank)
        self._test_sequence_num_set_new_group(backend="nccl")

    def _test_pass_nccl_options(self, pg_opts):
        store = c10d.FileStore(self.file_name, self.world_size)
        # Test init_process_group accepts options
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=pg_opts,
        )

        # Test with new_group
        pg = c10d.new_group([0, 1], pg_options=pg_opts)
        # test the process group works as expected
        t = torch.tensor([self.rank + 1] * 10).cuda(self.rank)
        pg.allreduce(t).wait()
        expected_tensor = torch.tensor([3] * 10).cuda(self.rank)
        self.assertEqual(expected_tensor, t)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_pass_nccl_options_high_priority_stream(self):
        pg_opts = c10d.ProcessGroupNCCL.Options()
        pg_opts.is_high_priority_stream = True
        self._test_pass_nccl_options(pg_opts)

    @requires_nccl()
    @requires_nccl_version(
        (2, 18), "Need NCCL 2.17+ for configuring NCCL communicators"
    )
    @skip_if_lt_x_gpu(2)
    def test_pass_nccl_options_config(self):
        pg_opts = c10d.ProcessGroupNCCL.Options()
        pg_opts.config.max_ctas = 4
        pg_opts.config.min_ctas = 2
        pg_opts.config.cga_cluster_size = 2
        pg_opts.config.net_name = "Socket"
        pg_opts.config.split_share = 1
        nccl_debug_file = tempfile.NamedTemporaryFile()
        os.environ["NCCL_DEBUG"] = "INFO"
        os.environ["NCCL_DEBUG_FILE"] = nccl_debug_file.name

        # Tests functionality when passing nccl config
        self._test_pass_nccl_options(pg_opts)

        # Tests if comms were configured
        nccl_debug_file_content = nccl_debug_file.read()
        max_ctas = re.search(rb"Max CTAs.*(\d+)|$", nccl_debug_file_content).group(1)
        min_ctas = re.search(rb"Min CTAs.*(\d+)|$", nccl_debug_file_content).group(1)
        split_share = re.search(
            rb"Split share.*(\d+)|$", nccl_debug_file_content
        ).group(1)
        cga_cluster_size = re.search(
            rb"CGA cluster.*(\d+)|$", nccl_debug_file_content
        ).group(1)
        net_name = re.search(
            rb"Using network.([a-zA-z]+)|$", nccl_debug_file_content
        ).group(1)
        self.assertEqual(pg_opts.config.max_ctas, int(max_ctas))
        self.assertEqual(pg_opts.config.min_ctas, int(min_ctas))
        self.assertEqual(pg_opts.config.cga_cluster_size, int(cga_cluster_size))
        self.assertEqual(pg_opts.config.net_name, net_name.decode())
        self.assertEqual(pg_opts.config.split_share, int(split_share))

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_nccl_barrier(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
        c10d.all_reduce(t)
        expected_tensor = torch.tensor([3] * 10).cuda(2 * self.rank)
        self.assertEqual(expected_tensor, t)

        # Test with new_group
        pg = c10d.new_group([0, 1])
        t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
        pg.allreduce(t).wait()
        self.assertEqual(expected_tensor, t)

        pg = c10d.new_group([0])
        if self.rank == 0:
            t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)

        pg = c10d.new_group([1])
        if self.rank == 1:
            t = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).cuda(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_barrier_device_ids(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        c10d.barrier(device_ids=[self.rank])

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nccl_barrier_device_ids_function_argument(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        with self.assertRaisesRegex(TypeError, "Invalid function argument"):
            c10d.barrier(device_ids=self.rank)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_unwaited(self) -> None:
        # Verify that the process can terminate gracefully
        # even with unwaited tensors
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        # Case 1: Run collectives under context manager, and don't call wait on them.
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            input = torch.full(
                (10240, 10240), float(self.rank), device=f"cuda:{self.rank}"
            )
            dist.all_reduce(input, op=dist.ReduceOp.SUM, async_op=True)
            # Non-functional collectives run under the context manager is registered in the work registry.
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            # Running another collective on the same tensor should still work
            dist.all_reduce(input, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 2)

        # Case 2: Run collectives not under context manager, and don't call wait on them.
        # NOTE: Here we intentionally test memory-stressed case.
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 2)
        for _ in range(50000):
            input = torch.full(
                (1024, 1024), float(self.rank), device=f"cuda:{self.rank}"
            )
            dist.all_reduce(input, op=dist.ReduceOp.SUM, async_op=True)
        # Work registry size is unchanged, since non-functional collectives not run under
        # the context manager is not registered in the work registry.
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 2)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_wait_tensor(self) -> None:
        # Verify that c10d_functional.wait_tensor() can be invoked on
        # output tensor of non-functional collective
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="nccl", rank=self.rank, world_size=self.world_size, store=store
        )

        # Case 1: under context manager (i.e. work is registered in registry)
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            input1 = torch.full((10, 10), float(self.rank), device=f"cuda:{self.rank}")
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            dist.all_reduce(input1, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            torch.ops.c10d_functional.wait_tensor(input1)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

            input2 = torch.full((10, 10), float(self.rank), device=f"cuda:{self.rank}")
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            work = dist.all_reduce(input2, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            work.wait()
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            self.assertEqual(input1, input2)

        # Case 2: not under context manager (i.e. work is not registered in registry)
        input1 = torch.full((10, 10), float(self.rank), device=f"cuda:{self.rank}")
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        dist.all_reduce(input1, op=dist.ReduceOp.SUM, async_op=True)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        # this does not take effect, since the underlying wait_tensor() logic would not
        # be able to find the corresponding work object (because it's not registered in registry)
        torch.ops.c10d_functional.wait_tensor(input1)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

        input2 = torch.full((10, 10), float(self.rank), device=f"cuda:{self.rank}")
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        work = dist.all_reduce(input2, op=dist.ReduceOp.SUM, async_op=True)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        work.wait()
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        self.assertEqual(input1, input2)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_nccl_warn_not_in_group_debug_detail(self):
        self._test_warn_not_in_group(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_nccl_warn_not_in_group_debug_info(self):
        self._test_warn_not_in_group(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_nccl_warn_not_in_group_debug_off(self):
        self._test_warn_not_in_group(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_nncl_rank_membership(self):
        self._test_rank_membership(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_tensor_dtype_mismatch(self):
        self._test_tensor_dtype_mismatch(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_tensor_dtype_complex(self):
        self._test_tensor_dtype_complex(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_base_k(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        output_tensor = torch.zeros(2, dtype=torch.int64).to(self.rank)
        input_tensors = torch.arange(self.world_size * 2, dtype=torch.int64).to(
            self.rank
        )
        input_tensors = torch.reshape(input_tensors, (self.world_size, 2))
        dist.reduce_scatter_tensor(output_tensor, input_tensors)
        self.assertEqual(output_tensor, input_tensors[self.rank] * self.world_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_coalesced(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        output_tensors = torch.zeros(2, 2).to(self.rank)
        input_tensors = [torch.ones(2, 2).to(self.rank) for _ in range(self.world_size)]
        with dist._coalescing_manager():
            for i in range(self.world_size):
                dist.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
        self.assertEqual(output_tensors, input_tensors[self.rank] * self.world_size)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_base_k_float8_errors(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        output_tensor = (
            torch.zeros(2, dtype=torch.float32).to(torch.float8_e4m3fn).to(self.rank)
        )
        input_tensors = (
            torch.arange(self.world_size * 2, dtype=torch.float32)
            .to(torch.float8_e4m3fn)
            .to(self.rank)
        )
        input_tensors = torch.reshape(input_tensors, (self.world_size, 2))
        with self.assertRaisesRegex(
            RuntimeError,
            "Float8 dtypes are not currenlty supported for NCCL reductions",
        ):
            dist.reduce_scatter_tensor(output_tensor, input_tensors)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_coalesced_float8_errors(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        output_tensors = torch.zeros(2, 2).to(torch.float8_e5m2).to(self.rank)
        input_tensors = [
            torch.ones(2, 2).to(torch.float8_e5m2).to(self.rank)
            for _ in range(self.world_size)
        ]

        with self.assertRaisesRegex(
            RuntimeError,
            "Float8 dtypes are not currenlty supported for NCCL reductions",
        ):
            with dist._coalescing_manager():
                for i in range(self.world_size):
                    dist.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
            self.assertEqual(output_tensors, input_tensors[self.rank])


class SetDeviceMethod(Enum):
    TORCH_CUDA_SET = auto()  # torch.cuda.set_device
    COLLECTIVE_ARGUMENT = auto()  # broadcast_object_list(device=)


class NcclProcessGroupWithDispatchedCollectivesTests(
    test_c10d_common.ProcessGroupWithDispatchedCollectivesTests
):
    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_collectives(self):
        self._test_collectives(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_allreduce_coalesced(self):
        self._test_allreduce_coalesced(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_all_to_all_single(self):
        self._test_all_to_all_single(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_allgather_base(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "cuda"
        tensor = torch.ones(10, 10, device=torch.device(device))
        output_tensor = torch.zeros(10, 10, device=torch.device(device))
        dist.all_gather_into_tensor(output_tensor, tensor)
        self.assertEqual(output_tensor, tensor)

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    @parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_allgather_float8(self, float8_dtype):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "cuda"
        tensor = torch.ones(10, 16, device=torch.device(device)).to(float8_dtype)
        output_tensor = torch.zeros(10, 16, device=torch.device(device)).to(
            float8_dtype
        )
        dist.all_gather_into_tensor(output_tensor, tensor)
        self.assertEqual(output_tensor.view(torch.float32), tensor.view(torch.float32))


instantiate_parametrized_tests(NcclProcessGroupWithDispatchedCollectivesTests)


class LargeCommTest(test_c10d_common.AbstractLargeCommTest, MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def device(self):
        return self.rank

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync(self):
        self._test_new_group_local_sync(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync_sanity_check(self):
        self._test_new_group_local_sync_sanity_check(backend="nccl")

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync_duplicated_pg(self):
        self._test_new_group_local_sync_duplicate_pg(backend="nccl")

    def _init_two_pg2_subgroups(self, world_size: int = 4):
        if world_size != 4:
            raise NotImplementedError(
                f"need world size of 4 to get 2 subgroup PGs, but got world size of {world_size}"
            )
        store = c10d.FileStore(self.file_name, world_size)
        c10d.init_process_group(
            backend="nccl", store=store, rank=self.rank, world_size=world_size
        )
        # every rank creates the same sub groups
        # including unused sub groups in the current rank
        a_group = c10d.new_group([0, 1])
        b_group = c10d.new_group([2, 3])
        return a_group if self.rank < 2 else b_group

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_gather_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            # just easier to write the test for exactly 4 gpus, even if this test class increased to 8gpu later
            return

        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device("cuda:%d" % self.rank)
        input = torch.ones((10,), device=device) * self.rank
        if self.rank == 0 or self.rank == 2:
            gather_list = [torch.empty_like(input) for _ in range(subgroup.size())]
            if group_rank:
                # global_dst=0 group_dst=0 my_global_rank=2 gather_list is not None=True
                torch.distributed.gather(
                    input,
                    gather_list=gather_list,
                    group_dst=0,
                    group=subgroup,
                    async_op=False,
                )
            else:
                torch.distributed.gather(
                    input,
                    gather_list=gather_list,
                    dst=self.rank,
                    group=subgroup,
                    async_op=False,
                )
            for src in range(len(gather_list)):
                expected = (torch.ones_like(input) * self.rank) + src
                self.assertEqual(gather_list[src], expected)
        else:
            if group_rank:
                torch.distributed.gather(
                    input,
                    gather_list=None,
                    group_dst=0,
                    group=subgroup,
                    async_op=False,
                )
            else:
                torch.distributed.gather(
                    input,
                    gather_list=None,
                    dst=self.rank - 1,
                    group=subgroup,
                    async_op=False,
                )

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_gather_object_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            # just easier to write the test for exactly 4 gpus, even if this test class increased to 8gpu later
            return

        subgroup = self._init_two_pg2_subgroups(world_size)

        # discrepancy #1
        # have to set device or else gather_object gets wrong device from 'current_device = _get_pg_default_device(group)
        torch.cuda.set_device(self.rank)

        input = {"rank": self.rank}
        if self.rank == 0 or self.rank == 2:
            # discrepancy #2
            # another weird thing- what's the point of making me specify some empty objects in my list?
            # empty list should be valid imo.  (but it throws an error)
            gather_list = [{}, {}]
            if group_rank:
                torch.distributed.gather_object(
                    input, object_gather_list=gather_list, group_dst=0, group=subgroup
                )
            else:
                torch.distributed.gather_object(
                    input, object_gather_list=gather_list, dst=self.rank, group=subgroup
                )
            for src in range(len(gather_list)):
                self.assertEqual(gather_list[src]["rank"], self.rank + src)
        else:
            if group_rank:
                torch.distributed.gather_object(
                    input, object_gather_list=None, group_dst=0, group=subgroup
                )
            else:
                torch.distributed.gather_object(
                    input, object_gather_list=None, dst=self.rank - 1, group=subgroup
                )

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_reduce_subgroup(self):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device("cuda:%d" % self.rank)
        x = torch.ones((10,), device=device) * self.rank
        if self.rank == 0 or self.rank == 2:
            expected = x + torch.ones((10,), device=device) * (self.rank + 1)
            c10d.reduce(x, dst=self.rank, group=subgroup, async_op=False)
            self.assertEqual(x, expected)
        else:
            c10d.reduce(x, dst=self.rank - 1, group=subgroup, async_op=False)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    @parametrize("async_op", [True, False])
    def test_send_recv_subgroup(self, async_op, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device("cuda:%d" % self.rank)
        if self.rank == 0 or self.rank == 2:
            x = torch.empty((10,), device=device)
            if async_op:
                if group_rank:
                    c10d.irecv(x, group_src=1, group=subgroup).wait()
                else:
                    c10d.irecv(x, src=self.rank + 1, group=subgroup).wait()
            else:
                if group_rank:
                    c10d.recv(x, group_src=1, group=subgroup)
                else:
                    c10d.recv(x, src=self.rank + 1, group=subgroup)
            expected = torch.ones((10,), device=device) * (self.rank + 1)
            self.assertEqual(x, expected)
        else:
            x = torch.ones((10,), device=device) * self.rank
            if async_op:
                if group_rank:
                    c10d.isend(x, group_dst=0, group=subgroup).wait()
                else:
                    c10d.isend(x, dst=self.rank - 1, group=subgroup).wait()
            else:
                if group_rank:
                    c10d.send(x, group_dst=0, group=subgroup)
                else:
                    c10d.send(x, dst=self.rank - 1, group=subgroup)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    def test_broadcast_subgroup(self):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device("cuda:%d" % self.rank)
        if self.rank == 0 or self.rank == 2:
            x = torch.empty((10,), device=device)
            c10d.broadcast(x, src=self.rank + 1, group=subgroup)
            expected = torch.ones((10,), device=device) * (self.rank + 1)
            self.assertEqual(x, expected)
        else:
            x = torch.ones((10,), device=device) * self.rank
            c10d.broadcast(x, src=self.rank, group=subgroup)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "set_device",
        [SetDeviceMethod.TORCH_CUDA_SET, SetDeviceMethod.COLLECTIVE_ARGUMENT],
    )
    def test_send_recv_object_list_subgroup(self, set_device: SetDeviceMethod):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        if set_device == SetDeviceMethod.TORCH_CUDA_SET:
            torch.cuda.set_device(self.rank)
            device = None
        else:
            device = torch.device("cuda:%d" % self.rank)
        if self.rank == 0 or self.rank == 2:
            x = [{}]
            c10d.recv_object_list(x, src=self.rank + 1, group=subgroup, device=device)
            expected = [{"rank": self.rank + 1}]
            self.assertEqual(x, expected)
        else:
            x = [{"rank": self.rank}]
            c10d.send_object_list(x, dst=self.rank - 1, group=subgroup, device=device)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "set_device",
        [SetDeviceMethod.TORCH_CUDA_SET, SetDeviceMethod.COLLECTIVE_ARGUMENT],
    )
    def test_broadcast_object_list_subgroup(self, set_device: SetDeviceMethod):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        if set_device == SetDeviceMethod.TORCH_CUDA_SET:
            torch.cuda.set_device(self.rank)
            device = None
        else:
            device = torch.device("cuda:%d" % self.rank)
        if self.rank == 0 or self.rank == 2:
            x = [{}]
            c10d.broadcast_object_list(
                x, src=self.rank + 1, group=subgroup, device=device
            )
            expected = [{"rank": self.rank + 1}]
            self.assertEqual(x, expected)
        else:
            x = [{"rank": self.rank}]
            c10d.broadcast_object_list(x, src=self.rank, group=subgroup, device=device)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_scatter_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device("cuda:%d" % self.rank)
        x = torch.empty((10,), device=device)
        expected = torch.ones((10,), device=device) * self.rank
        if self.rank == 0 or self.rank == 2:
            if group_rank:
                c10d.scatter(x, scatter_list=None, group_src=1, group=subgroup)
            else:
                c10d.scatter(x, scatter_list=None, src=self.rank + 1, group=subgroup)
        else:
            scatter_list = [
                torch.ones((10,), device=device) * (self.rank - 1),
                torch.ones((10,), device=device) * self.rank,
            ]
            if group_rank:
                c10d.scatter(x, scatter_list=scatter_list, group_src=1, group=subgroup)
            else:
                c10d.scatter(
                    x, scatter_list=scatter_list, src=self.rank, group=subgroup
                )
        self.assertEqual(x, expected)

    @requires_nccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_scatter_object_list_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        torch.cuda.set_device(self.rank)
        scatter_object_output_list = [None]
        expected = [{"rank": self.rank}]
        if self.rank == 0 or self.rank == 2:
            if group_rank:
                c10d.scatter_object_list(
                    scatter_object_output_list=scatter_object_output_list,
                    scatter_object_input_list=None,
                    group_src=1,
                    group=subgroup,
                )
            else:
                c10d.scatter_object_list(
                    scatter_object_output_list=scatter_object_output_list,
                    scatter_object_input_list=None,
                    src=self.rank + 1,
                    group=subgroup,
                )

        else:
            scatter_object_input_list = [
                {"rank": self.rank - 1},
                {"rank": self.rank},
            ]
            if group_rank:
                c10d.scatter_object_list(
                    scatter_object_output_list=scatter_object_output_list,
                    scatter_object_input_list=scatter_object_input_list,
                    group_src=1,
                    group=subgroup,
                )
            else:
                c10d.scatter_object_list(
                    scatter_object_output_list=scatter_object_output_list,
                    scatter_object_input_list=scatter_object_input_list,
                    src=self.rank,
                    group=subgroup,
                )
        self.assertEqual(scatter_object_output_list, expected)


instantiate_parametrized_tests(LargeCommTest)


class SparseCollective(MultiProcessTestCase):
    @property
    def world_size(self):
        return 1

    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # self.num_gpus = torch.cuda.device_count()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    class ToyModel(nn.Module):
        def __init__(self, rank, vocab_size, embedding_dim):
            super().__init__()
            self.embedding = nn.Embedding(vocab_size, embedding_dim, sparse=True).to(
                rank
            )
            self.linear = nn.Linear(embedding_dim, 1).to(rank)

        def forward(self, inputs):
            embedded = self.embedding(inputs)
            # embedded shape: (batch_size, sequence_length, embedding_dim)
            flattened = torch.mean(embedded, dim=1)
            # flattened shape: (batch_size, embedding_dim)
            output = self.linear(flattened)
            # output shape: (batch_size, 1)
            return output

    @requires_nccl()
    @skip_if_lt_x_gpu(1)
    def test_ddp_set_sparse_metadata(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )

        vocab_size = 5

        model = SparseCollective.ToyModel(
            self.rank, vocab_size=vocab_size, embedding_dim=10
        )
        ddp_model = DistributedDataParallel(model)
        inputs = torch.tensor([[1, 0, 0], [0, 0, 0], [0, 0, 0]]).to(self.rank)
        # set sparse metadata on the DDP model
        indices = torch.Tensor(list(range(vocab_size)))
        ddp_model._set_sparse_metadata({"embedding.weight": indices})
        # forward pass
        try:
            output = ddp_model(inputs)
            loss = output.sum()

            # backward pass
            loss.backward()
            self.assertTrue(ddp_model.module.embedding.weight.grad.indices, indices)
        except RuntimeError as e:
            if "NCCL does not support all_reduce with sparse tensors" in str(e):
                pass
            else:
                # Rethrow the exception if it's a different error
                raise


class NCCLTraceTestBase(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        os.environ[
            "TORCH_NCCL_ENABLE_TIMING"
        ] = "0"  # see 'timing_enabled' parametrized tests
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000"
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["TORCH_NCCL_DEBUG_INFO_TEMP_FILE"] = self._trace_basename()
        os.environ["TORCH_NCCL_DEBUG_INFO_PIPE_FILE"] = self._trace_basename()
        self._spawn_processes()

    @classmethod
    def _run(
        cls,
        parent_conn,
        rank: int,
        test_name: str,
        file_name: str,
        parent_pipe,
        **kwargs,
    ) -> None:
        cls.parent = parent_conn
        super()._run(rank, test_name, file_name, parent_pipe)

    @property
    def local_device(self):
        return torch.device("cuda", self.rank_to_GPU[self.rank][0])

    def _join_processes(self, fn):
        # We need to patch sys.exit() as skip_if will use sys.exit() and
        # the exit code from the this process will not be catched.
        with mock.patch("sys.exit") as exit_mock:
            fn()
        super()._join_processes(fn)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process
        self.children_pipes = []
        parent_pipes = []
        for i in range(self.world_size):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            self.children_pipes.append(child_conn)
            parent_pipes.append(parent_conn)
        piter = iter(parent_pipes)

        def wrap(*positional, args, **kwargs):
            args = (next(piter), *args)
            return proc(*positional, args=args, **kwargs)

        self._start_processes(wrap)

    def _create_process_group_nccl(self):
        store = dist.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "nccl", world_size=self.world_size, rank=self.rank, store=store
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

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
        return init_multigpu_helper(self.world_size, "nccl")

    def _trace_basename(self):
        # we pass the base to the env, and the dump util will append rank
        return os.path.join(self.tempdir.name, "trace_")

    def _trace_name(self, rank):
        return self._trace_basename() + str(rank)

    def started_or_scheduled(self, timing_enabled):
        return "started" if timing_enabled else "scheduled"


class NCCLTraceTest(NCCLTraceTestBase):
    def _verify_trace(self, t, include_collectives, timing_enabled, is_json):
        ver = t["version"]
        self.assertEqual(ver, "2.4")
        pg_config = t["pg_config"]
        self.assertEqual(len(pg_config), 1)
        default_pg_info = pg_config["0"]
        self.assertIn("name", default_pg_info)
        self.assertIn("desc", default_pg_info)
        self.assertIn("ranks", default_pg_info)
        pg_status = t["pg_status"]
        self.assertEqual(len(pg_status), 1)
        self.assertEqual(str(pg_status["0"]["last_enqueued_collective"]), "2")
        self.assertEqual(str(pg_status["0"]["last_completed_collective"]), "2")
        self.assertEqual(
            str(pg_status["0"]["last_started_collective"]),
            "2" if timing_enabled else "-1",
        )
        global_ranks = pg_config["0"]["ranks"]
        self.assertEqual(len(json.loads(global_ranks)), self.world_size)
        if include_collectives:
            self.assertEqual(len(t["entries"]), 2)
            t = t["entries"]
            last = t[-1]
            self.assertEqual(last["process_group"], ("0", "default_pg"))
            self.assertEqual(last["state"], "completed")
            s = last["time_discovered_started_ns"]
            f = last["time_discovered_completed_ns"]
            self.assertEqual(last["record_id"], 1)
            self.assertIsNotNone(f)
            if timing_enabled:
                self.assertIsNotNone(s)
                self.assertTrue(s <= f)
            # we don't collect stack traces in JSON at the moment
            if not is_json:
                self.assertIn("test_c10d_nccl.py", str(last["frames"]))
            self.assertEqual(last["input_sizes"], ((3, 4),))
            self.assertEqual(last["input_dtypes"], ["Float"])
            self.assertEqual(last["output_sizes"], ((3, 4),))
            self.assertEqual(last["output_dtypes"], ["Float"])
            self.assertEqual(last["collective_seq_id"], 2)
            self.assertEqual(last["timeout_ms"], 600000)
            now = datetime.now()
            event_created_time = datetime.fromtimestamp(
                last["time_created_ns"] / 1000000000
            )
            before_test = now - timedelta(minutes=1)
            self.assertTrue(before_test < event_created_time < now)
            if timing_enabled:
                # very loose bounds, measured 0.036 ms on devgpu
                self.assertTrue(0 < last["duration_ms"] < 100)
            else:
                self.assertTrue("duration_ms" not in last)
        else:
            self.assertTrue("entries" not in t)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("timing_enabled", [True, False])
    @parametrize("include_collectives", [True, False])
    def test_short_json(self, timing_enabled, include_collectives):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for i in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.cuda.synchronize(device=device)
        # gah ok so now the duration_ms is populated best-effort since it can only happen outside "dump()" api
        time.sleep(1)
        t = json.loads(
            torch._C._distributed_c10d._dump_nccl_trace_json(
                includeCollectives=include_collectives
            )
        )
        self._verify_trace(t, include_collectives, timing_enabled, True)
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("timing_enabled", [True, False])
    @parametrize("include_collectives", [True, False])
    def test_short_pickle(self, timing_enabled, include_collectives):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for i in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.cuda.synchronize(device=device)
        # gah ok so now the duration_ms is populated best-effort since it can only happen outside "dump()" api
        time.sleep(1)
        t = pickle.loads(
            torch._C._distributed_c10d._dump_nccl_trace(
                includeCollectives=include_collectives
            )
        )
        self._verify_trace(
            t,
            include_collectives=include_collectives,
            timing_enabled=timing_enabled,
            is_json=True,
        )
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_dump_pipe(self):
        def open_file_with_timeout(file_path, mode, timeout=1.0):
            start_time = time.time()
            while time.time() - start_time < timeout:
                if os.path.exists(file_path):
                    return open(file_path, mode)
                time.sleep(0.1)
            raise FileNotFoundError

        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")

            dump_file = self._trace_name(rank=0)
            pipe_file = dump_file + ".pipe"
            with open_file_with_timeout(pipe_file, "w") as f:
                f.write("1\n")
            with open_file_with_timeout(dump_file, "rb", timeout=10.0) as f:
                self.assertTrue("all_reduce" in str(pickle.load(f)))

            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_nccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for i in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.cuda.synchronize(device=device)
        self.parent.send("next")
        self.parent.recv()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_long(self):
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for i in range(2):
            # test some other primitives to make sure
            # their strings are valid
            xs = [torch.ones(3, 4, device=device)]
            pg.broadcast(xs).wait()
            pg.allreduce(xs).wait()
            pg.reduce(xs).wait()
            ys = [[torch.empty(3, 4, device=device) for _ in range(self.world_size)]]
            pg.allgather(ys, xs).wait()
            pg.reduce_scatter(xs, ys).wait()
            f = pg.allreduce(a)
        f.wait()
        torch.cuda.synchronize(device=device)
        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 10)
        first = t[0]
        last = t[-1]
        self.assertEqual(last["profiling_name"], "nccl:all_reduce")
        self.assertEqual(last["state"], "completed")
        self.assertIn("test_c10d_nccl.py", str(last["frames"]))
        self.assertEqual(last["input_sizes"], ((3, 4),))
        self.assertEqual(last["input_dtypes"], ["Float"])
        self.assertEqual(last["output_sizes"], ((3, 4),))
        self.assertEqual(last["output_dtypes"], ["Float"])
        self.assertEqual(last["timeout_ms"], 600000)
        self.assertEqual(last["collective_seq_id"] - first["collective_seq_id"], 9)
        dist.destroy_process_group()

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_trace_while_all_works_retired(self):
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        device = self.local_device
        # send more works than the buffer size to overwrite the previous entry
        for i in range(12):
            a = [torch.ones(3, 4, device=device)]
            pg.broadcast(a).wait()
        torch.cuda.synchronize(device=device)

        # wait for all works to be retired
        pg._wait_for_pending_works()
        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 10)
        last = t[-1]
        self.assertEqual(last["retired"], True)
        self.assertEqual(last["state"], "completed")

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("timing_enabled", [True, False])
    @parametrize("only_active", [True, False])
    def test_trace_while_active(self, timing_enabled, only_active):
        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")
            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        with torch.cuda.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            e = torch.cuda.Event()
            e.record()
            if self.rank != 0:
                pg.allreduce(a).wait()
            e.synchronize()
            t = pickle.loads(
                torch._C._distributed_c10d._dump_nccl_trace(onlyActive=only_active)
            )
            t = t["entries"]
            if only_active:
                if self.rank == 0:
                    self.assertEqual(len(t), 0)
                else:
                    self.assertEqual(len(t), 1)
            if not only_active:
                if self.rank == 0:
                    self.assertEqual(t[-1]["profiling_name"], "nccl:all_reduce")
                    self.assertEqual(t[-1]["collective_seq_id"], 1)
                    self.assertEqual(t[-1]["state"], "completed")
                else:
                    self.assertEqual(t[-1]["profiling_name"], "nccl:all_reduce")
                    self.assertEqual(t[-1]["collective_seq_id"], 2)
                    self.assertEqual(
                        t[-1]["state"], self.started_or_scheduled(timing_enabled)
                    )

            self.parent.send("next")
            self.assertEqual("next", self.parent.recv())
            if self.rank == 0:
                pg.allreduce(a).wait()
            torch.cuda.synchronize(device=device)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize("timing_enabled", [True, False])
    def test_trace_while_stuck(self, timing_enabled):
        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")
            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        device = self.local_device
        with torch.cuda.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            e = torch.cuda.Event()
            e.record()

            def gather_trace():
                e.synchronize()
                # give the other thread some time to fill the cuda buffer
                time.sleep(5)
                t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
                t = t["entries"]
                self.assertEqual(t[-1]["profiling_name"], "nccl:all_reduce")
                if self.rank == 0:
                    self.assertEqual(t[-1]["collective_seq_id"], 1)
                    self.assertEqual(t[-1]["state"], "completed")
                else:
                    self.assertEqual(t[-1]["collective_seq_id"], 2)
                    self.assertEqual(
                        t[-1]["state"], self.started_or_scheduled(timing_enabled)
                    )
                    self.assertIsNone(t[-1]["time_discovered_completed_ns"])
                # this will eventually cause the missing rank 0
                # to continue which will unblock the non-zero ranks
                self.parent.send("next")

            if self.rank != 0:
                pg.allreduce(a).wait()
                th = threading.Thread(target=gather_trace)
                th.start()
                # fill the cuda buffer, at around 1024 events
                # this will stall
                for i in range(2000):
                    a = a + a
                th.join()
            else:
                gather_trace()

            self.assertEqual("next", self.parent.recv())
            if self.rank == 0:
                pg.allreduce(a).wait()
            torch.cuda.synchronize(device=device)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize(
        "op_sizes_per_coalesce",
        [
            [(2, 3)],
            [(2, 3), (5, 5), (1,)],
        ],
    )
    @parametrize("timing_enabled", [True, False])
    def test_batched_send_recv(self, op_sizes_per_coalesce, timing_enabled):
        """
        'WorkEnqueue' was skipped for isendirecv, leading to segfault on dump_entries when update_state tried to use
        a destructed Work obj's cuda events
        """

        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        num_coalesced_ops = 20
        ops_per_coalesce = len(op_sizes_per_coalesce)
        for i in range(num_coalesced_ops):
            ops = []
            for input_sizes in op_sizes_per_coalesce:
                tensor = torch.zeros(input_sizes).to(self.local_device)
                if self.rank == 0:
                    ops.append(dist.P2POp(dist.irecv, tensor, 1))
                elif self.rank == 1:
                    tensor *= 2
                    ops.append(dist.P2POp(dist.isend, tensor, 0))

            dist.batch_isend_irecv(ops).pop().wait()

        torch.cuda.synchronize(device=self.local_device)

        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        self.assertEqual(len(t["entries"]), num_coalesced_ops * (ops_per_coalesce + 1))

        expected_record_id = 0
        expected_seq = 1
        expected_op_id = 1
        for seq in range(num_coalesced_ops):
            first_op = seq * (ops_per_coalesce + 1)
            coalesced_op = first_op + ops_per_coalesce
            for p2p_op_idx, input_sizes in zip(
                range(first_op, coalesced_op, 1), op_sizes_per_coalesce
            ):
                # the indivudal ops inside the coalescing group the individual op metadata,
                # but not the timing info coming from the actual coalesced kernel
                profiling_name = (
                    "nccl:recv 0<-1" if self.rank == 0 else "nccl:send 1->0"
                )
                self.assertEqual(
                    t["entries"][p2p_op_idx]["record_id"], expected_record_id
                )
                expected_record_id += 1
                self.assertEqual(
                    t["entries"][p2p_op_idx]["profiling_name"], profiling_name
                )
                # we don't increment collective_seq_id for p2p ops.
                self.assertEqual(t["entries"][p2p_op_idx]["collective_seq_id"], 0)
                self.assertEqual(t["entries"][p2p_op_idx]["p2p_seq_id"], expected_seq)
                self.assertEqual(t["entries"][p2p_op_idx]["op_id"], expected_op_id)
                expected_op_id += 1
                self.assertEqual(t["entries"][p2p_op_idx]["input_sizes"], [input_sizes])
                self.assertEqual(
                    t["entries"][p2p_op_idx]["output_sizes"], [input_sizes]
                )
                # duration doesn't get tagged onto individual ops yet, nor is their state updated
                self.assertEqual(t["entries"][p2p_op_idx]["state"], "scheduled")
                self.assertTrue("duration_ms" not in t["entries"][p2p_op_idx])

            # the coalesced op has no metadata but indicates that coalescing was used,
            # and accurately reflects the timing and state info for the whole group
            self.assertEqual(
                t["entries"][coalesced_op]["record_id"], expected_record_id
            )
            expected_record_id += 1
            self.assertEqual(
                t["entries"][coalesced_op]["profiling_name"], "nccl:coalesced"
            )
            self.assertEqual(t["entries"][coalesced_op]["p2p_seq_id"], expected_seq)
            expected_seq += 1
            self.assertEqual(t["entries"][coalesced_op]["state"], "completed")
            self.assertEqual(t["entries"][coalesced_op]["input_sizes"], [])
            self.assertEqual(t["entries"][coalesced_op]["output_sizes"], [])
            if timing_enabled:
                duration = t["entries"][coalesced_op]["duration_ms"]
                self.assertTrue(0.001 < duration < 10000, duration)
            else:
                self.assertTrue("duration_ms" not in t["entries"][coalesced_op])
            self.assertEqual(t["entries"][coalesced_op]["timeout_ms"], 600000)

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    @parametrize(
        "op_sizes",
        [
            [(2, 3)],
            [(2, 3), (5, 5), (1,)],
        ],
    )
    @parametrize("timing_enabled", [True, False])
    def test_individual_send_recv(self, op_sizes, timing_enabled):
        """
        'WorkEnqueue' was skipped for isendirecv, leading to segfault on dump_entries when update_state tried to use
        a destructed Work obj's cuda events
        """

        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        num_repeats = 10
        ops_per_repeat = len(op_sizes)
        for i in range(num_repeats):
            for input_sizes in op_sizes:
                tensor = torch.zeros(input_sizes).to(self.local_device)
                if self.rank == 0:
                    dist.recv(tensor, 1)
                elif self.rank == 1:
                    tensor *= 2
                    dist.send(tensor, 0)

        torch.cuda.synchronize(device=self.local_device)
        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())
        self.assertEqual(len(t["entries"]), num_repeats * (ops_per_repeat))
        expected_seq = 1
        expected_op_id = 1
        for seq in range(num_repeats * ops_per_repeat):
            input_sizes = op_sizes[seq % ops_per_repeat]
            profiling_name = "nccl:recv 0<-1" if self.rank == 0 else "nccl:send 1->0"
            self.assertEqual(t["entries"][seq]["profiling_name"], profiling_name)
            # we don't increment collective_seq_id for p2p ops.
            self.assertEqual(t["entries"][seq]["collective_seq_id"], 0)
            self.assertEqual(t["entries"][seq]["p2p_seq_id"], expected_seq)
            expected_seq += 1
            self.assertEqual(t["entries"][seq]["op_id"], expected_op_id)
            expected_op_id += 1
            self.assertEqual(t["entries"][seq]["input_sizes"], [input_sizes])
            self.assertEqual(t["entries"][seq]["output_sizes"], [input_sizes])
            self.assertEqual(t["entries"][seq]["state"], "completed")

            if timing_enabled:
                duration = t["entries"][seq]["duration_ms"]
                self.assertTrue(0.001 < duration < 10000, duration)
            else:
                self.assertTrue("duration_ms" not in t["entries"][seq])

    # TODO(whc) support and test coalesced collectives that use the c++ start/end group thingy instead of python
    # coalescing manager

    # TODO(whc) test out other ops (And combinations of ops, if that's valid?)
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("timing_enabled", [True, False])
    def test_coalescing_manager_collective(self, timing_enabled):
        """
        The coalescing manager api works by accumulating operations in python via a contextmanager, and then making
        one call into c++ to an <op>_coalesced API.  It has limited support for ops and has been added recently to
        avoid overheads of making individual py-cpp calls.  This complicates flight recording..

        For now, flight recording of coalescing_manager collectives is less detailed than cpp coalesced collectives.
        """
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_nccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        output_tensors = torch.zeros(2, 2).to(self.rank)
        input_tensors = [torch.ones(2, 2).to(self.rank) for _ in range(self.world_size)]

        # TODO(whc) make this work with bigger world or something
        self.assertEqual(self.world_size, 2, self.world_size)

        with dist._coalescing_manager():
            for i in range(self.world_size):
                dist.reduce_scatter_tensor(output_tensors[i], input_tensors[i])
        self.assertEqual(output_tensors, input_tensors[self.rank] * self.world_size)

        torch.cuda.synchronize(device=self.rank)

        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch._C._distributed_c10d._dump_nccl_trace())

        self.assertEqual(
            len(t["entries"]), 1
        )  # one for the reduce_scatter_tensor_coalesced
        self.assertEqual(
            t["entries"][0]["profiling_name"], "nccl:reduce_scatter_tensor_coalesced"
        )
        # collective_seq_id should be incremented once.
        self.assertEqual(t["entries"][0]["collective_seq_id"], 1)
        self.assertEqual(t["entries"][0]["input_sizes"], [[2, 2], [2, 2]])
        self.assertEqual(
            t["entries"][0]["output_sizes"],
            [
                [
                    2,
                ],
                [
                    2,
                ],
            ],
        )
        self.assertEqual(t["entries"][0]["state"], "completed")
        if timing_enabled:
            duration = t["entries"][0]["duration_ms"]
            self.assertTrue(0.001 < duration < 10000, duration)
        else:
            self.assertTrue("duration_ms" not in t["entries"][0])


def check_if_test_is_skipped(fn):
    def wrapper(self, *args, **kwargs):
        for skip in TEST_SKIPS.values():
            if self.processes[0].exitcode == skip.exit_code:
                return MultiProcessTestCase._check_return_codes(self, *args, **kwargs)
        return fn(self, *args, **kwargs)

    return wrapper


class NCCLTraceTestDumpOnTimeoutBase(NCCLTraceTestBase):
    timeout_sec = 1

    def _create_process_group_nccl(self):
        store = dist.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=NCCLTraceTestDumpOnTimeoutBase.timeout_sec),
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to abort and rank1 to exit cleanly in this test
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, 0)

    def _wait_process(self, rank, timeout):
        try:
            self.processes[rank].join(timeout)
            return self.processes[rank].exitcode
        except TimeoutError:
            return None


@skip_but_pass_in_sandcastle
class NCCLTraceTestDumpOnTimeout(NCCLTraceTestDumpOnTimeoutBase):
    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("timing_enabled", [True, False])
    def test_timeout_dumps(self, timing_enabled):
        # dump on heartbeatmonitor thread
        os.environ["TORCH_NCCL_COORD_CHECK_MILSEC"] = "1000"
        # need rank0 to crash before looking for its output file
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for rank0 to crash before looking for its output file
            # we rely on rank0 holding off its abort long enough to dump the debug info
            self.assertEqual(self._wait_process(0, timeout=90), -6)
            with open(self._trace_name(rank=0), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 2)
                self.assertEqual(t[0]["collective_seq_id"], 1)
                self.assertEqual(t[0]["state"], "completed")
                self.assertEqual(t[1]["collective_seq_id"], 2)
                self.assertEqual(
                    t[1]["state"], self.started_or_scheduled(timing_enabled)
                )

            self.assertFalse(os.path.exists(self._trace_name(rank=1)))

            return

        pg = self._create_process_group_nccl()
        if timing_enabled:
            # we force disabled timing in setup, since there is no 'disable' function
            pg._enable_collectives_timing()

        device = self.local_device
        with torch.cuda.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            if self.rank == 0:
                pg.allreduce(a).wait()

            # rank 0 will crash before it passes the sync, but rank1 will exit quickly and cleanly
            torch.cuda.synchronize(device=device)


instantiate_parametrized_tests(ProcessGroupNCCLGroupTest)
instantiate_parametrized_tests(NCCLTraceTestDumpOnTimeout)
instantiate_parametrized_tests(NCCLTraceTest)


@skip_but_pass_in_sandcastle
class NCCLTraceTestTimeoutDumpOnStuckRanks(NCCLTraceTestDumpOnTimeoutBase):
    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to abort and rank1 to exit cleanly in this test
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, -6)

    @requires_nccl()
    @skip_if_lt_x_gpu(2)
    def test_timeout_dumps_on_stuck_ranks(self):
        # need rank0 to crash quicker after detecting timeout
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "1"
        # restore this env var to its prior default in case another test changed it
        os.environ["TORCH_NCCL_COORD_CHECK_MILSEC"] = "1000"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for both rank0 and 1 to crash before looking for both ranks' output
            # file, and we rely on rank1 to sleep long enough to dump the debug info.
            self.assertEqual(self._wait_process(0, timeout=90), -6)
            self.assertEqual(self._wait_process(1, timeout=90), -6)
            self.assertTrue(os.path.exists(self._trace_name(rank=1)))
            self.assertTrue(os.path.exists(self._trace_name(rank=0)))
            with open(self._trace_name(rank=0), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 2)
            with open(self._trace_name(rank=1), "rb") as f:
                t = pickle.load(f)
                t = t["entries"]
                self.assertEqual(len(t), 1)
                self.assertEqual(t[0]["collective_seq_id"], 1)
                self.assertEqual(t[0]["state"], "completed")
            return

        pg = self._create_process_group_nccl()
        device = self.local_device
        with torch.cuda.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            if self.rank == 0:
                pg.allreduce(a).wait()

            # rank 0 will get stuck, timeout and then signal a timeout to all ranks.
            torch.cuda.synchronize(device=device)

            if self.rank == 1:
                # Force rank 1 to idle so that it will eventually timeout as well after
                # getting the global signal to dump the debugging info.
                time.sleep(600)


@skip_but_pass_in_sandcastle
class NcclErrorDumpTest(NCCLTraceTestBase):
    def _wait_process(self, rank, timeout):
        try:
            self.processes[rank].join(timeout)
            return self.processes[rank].exitcode
        except TimeoutError:
            return None

    @check_if_test_is_skipped
    def _check_return_codes(self, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to abort with exception and rank1 to exit with exit 1
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, 1)

    @requires_nccl()
    @requires_nccl_version((2, 4, 0), "Need NCCL 2.4+ for error checking")
    @skip_if_lt_x_gpu(2)
    @skip_if_rocm_multiprocess
    def test_nccl_errors_dump(self):
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        os.environ["TORCH_NCCL_TRACE_BUFFER_SIZE"] = "1000"
        os.environ["TORCH_NCCL_DUMP_ON_TIMEOUT"] = "1"
        # need rank0 to dump before abort
        os.environ["TORCH_NCCL_HEARTBEAT_TIMEOUT_SEC"] = "5"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for both rank0 and 1 to crash before looking for dump
            self.assertEqual(self._wait_process(0, timeout=90), -6)
            self.assertEqual(self._wait_process(1, timeout=90), 1)
            # verify that the trace file exists for rank0
            self.assertTrue(os.path.exists(self._trace_name(rank=0)))
            return

        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupNCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=10),
        )
        process_group.allreduce(torch.rand(10).cuda(self.rank))
        if self.rank == 0:
            work = process_group.allreduce(torch.rand(10).cuda(self.rank))
            # expect an error to be raised
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                # Block the current stream on the NCCL stream
                work.wait()
                # Run some GPU operations
                a = torch.rand(10).cuda(self.rank)
        elif self.rank == 1:
            # Clean up structures (ex: files for FileStore before going down)
            del process_group
            sys.exit(1)


# tests that needs to be run with a larger world size
class ProcessGroupNCCLLargerScaleTest(MultiProcessTestCase):
    def _create_process_group_nccl(self, store, opts, device_id=None):
        # create nccl processgroup with opts
        c10d.init_process_group(
            "nccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=opts,
            device_id=device_id,
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def opts(self, high_priority_stream=False):
        opts = c10d.ProcessGroupNCCL.Options()
        opts.is_high_priority_stream = high_priority_stream
        return opts

    def setUp(self):
        super().setUp()
        # TORCH_NCCL_BLOCKING_WAIT overrides TORCH_NCCL_ASYNC_ERROR_HANDLING hence tests
        # that use TORCH_NCCL_BLOCKING_WAIT will test it as expected.
        os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = "1"
        # self.num_gpus = torch.cuda.device_count()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return 8

    @property
    def rank_to_GPU(self):
        # return rank to GPU map
        return init_multigpu_helper(self.world_size, "nccl")

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_if_lt_x_gpu(8)
    def test_comm_split_group_larger_scale(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        tensor = torch.full((1,), self.rank).cuda(device)
        ng1 = c10d.split_group(pg, [[0, 1], [2, 3, 4, 5, 6, 7]])
        backend1 = ng1._get_backend(torch.device(device))

        # comm split happens eagerly since device_id is passed to init_process_group.
        self.assertEqual(backend.comm_split_count(), 1)
        # dist.broadcast take Source rank on global process group
        if self.rank < 2:
            dist.broadcast(tensor, 0, group=ng1)
            self.assertEqual(tensor, torch.full((1,), 0))
        else:
            dist.broadcast(tensor, 2, group=ng1)
            self.assertEqual(tensor, torch.full((1,), 2))

        # test split with only one colored group, other ranks should be no color split.
        ng2 = c10d.split_group(pg, [[5, 6, 7]])
        self.assertEqual(backend.comm_split_count(), 2)

        if self.rank >= 5:
            tensor2 = torch.full((1,), self.rank).cuda(device)
            dist.broadcast(tensor2, 7, group=ng2)
            self.assertEqual(tensor2, torch.full((1,), 7))
        else:
            self.assertEqual(ng2, None)
        # a barrier and a cuda sync before destroying all pgs.
        dist.barrier(pg)
        torch.cuda.synchronize()
        dist.destroy_process_group()

    @requires_nccl_version((2, 18), "Need NCCL 2.18+ for ncclCommSplit")
    @skip_if_lt_x_gpu(8)
    def test_comm_recursive_split_group(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"cuda:{self.rank}")
        pg = self._create_process_group_nccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        # split the default PG into 2 subgroups, each subgroup (ng1) has 4 ranks.
        tensor1 = torch.full((1,), self.rank).cuda(device)
        ng1 = c10d.split_group(pg, [[0, 1, 2, 3], [4, 5, 6, 7]])
        backend1 = ng1._get_backend(torch.device(device))
        if self.rank < 4:
            dist.broadcast(tensor1, 0, group=ng1)
            self.assertEqual(tensor1, torch.full((1,), 0))
        else:
            dist.broadcast(tensor1, 4, group=ng1)
            self.assertEqual(tensor1, torch.full((1,), 4))

        # comm split happens eagerly since device_id is passed to init_process_group.
        self.assertEqual(backend.comm_split_count(), 1)
        self.assertEqual(backend1.comm_split_count(), 0)

        # further split ng1 into 2 subgroups, each subgroup (ng2) has 2 ranks.
        tensor2 = torch.full((1,), self.rank).cuda(device)
        ng2 = c10d.split_group(ng1, [[0, 1], [2, 3]])
        backend2 = ng2._get_backend(torch.device(device))
        self.assertEqual(backend.comm_split_count(), 1)
        self.assertEqual(backend1.comm_split_count(), 1)
        self.assertEqual(backend2.comm_split_count(), 0)

        # execute collective calls within each 2-rank pg
        if self.rank == 0 or self.rank == 1:
            dist.broadcast(tensor2, 1, group=ng2)
            self.assertEqual(tensor2, torch.full((1,), 1))

        if self.rank == 2 or self.rank == 3:
            dist.broadcast(tensor2, 2, group=ng2)
            self.assertEqual(tensor2, torch.full((1,), 2))

        if self.rank == 4 or self.rank == 5:
            dist.broadcast(tensor2, 5, group=ng2)
            self.assertEqual(tensor2, torch.full((1,), 5))

        if self.rank == 6 or self.rank == 7:
            dist.broadcast(tensor2, 6, group=ng2)
            self.assertEqual(tensor2, torch.full((1,), 6))
        # a barrier and a cuda sync before destroying all pgs.
        dist.barrier(pg)
        torch.cuda.synchronize()
        dist.destroy_process_group()


if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
