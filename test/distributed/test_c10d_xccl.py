# Owner(s): ["oncall: distributed"]

import copy
import json
import logging
import os
import pickle
import random
import signal
import sys
import tempfile
import threading
import time
import warnings
from contextlib import contextmanager
from datetime import datetime, timedelta
from enum import auto, Enum
from itertools import product
from unittest import mock, SkipTest

import torch
import torch.distributed as c10d
import torch.distributed._functional_collectives as _functional_collectives


if not c10d.is_available() or not c10d.is_xccl_available():
    print("c10d XCCL/XCCL not available, skipping tests", file=sys.stderr)
    sys.exit(0)


import test_c10d_common
from test_c10d_common import (
    ConvNet,
    DoubleGpuNet,
    FFTModel,
    gpus_for_rank,
    ModuleForDdpCommHook,
)

import torch.distributed as dist
import torch.distributed.algorithms.ddp_comm_hooks.default_hooks as default
import torch.distributed.algorithms.ddp_comm_hooks.powerSGD_hook as powerSGD
import torch.nn.functional as F
import torch.testing._internal.common_utils as common
from torch import nn
from torch._C._distributed_c10d import ErrorType, WorkResult
from torch.nn.parallel import DistributedDataParallel
from torch.testing._internal.common_distributed import (
    get_required_world_size,
    get_timeout,
    init_multigpu_helper,
    MultiProcessTestCase,
    requires_xccl,
    skip_if_lt_x_gpu,
    TEST_SKIPS,
    with_dist_debug_levels,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    IS_SANDCASTLE,
    parametrize,
    retry_on_connect_failures,
    run_tests,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    skipIfXpu,
    TEST_MULTIACCELERATOR,
    TEST_WITH_DEV_DBG_ASAN,
    TEST_XPU,
    TestCase,
)


if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip ASAN as torch + multiprocessing spawn have known issues", file=sys.stderr
    )
    sys.exit(0)


_start_time = time.time()
_logger = logging.getLogger(__name__)
DEFAULT_PG_TIMEOUT = int(torch._C._distributed_c10d._DEFAULT_PG_TIMEOUT.seconds * 1000)


def _ts():
    return time.time() - _start_time


def configure(level=logging.INFO, force=False):
    try:
        logging.basicConfig(
            level=level,
            format="%(asctime)s %(name)s %(levelname)s: %(message)s",
            force=force,
        )
    except TypeError:
        logging.basicConfig(
            level=level, format="%(asctime)s %(name)s %(levelname)s: %(message)s"
        )


def log_test_info(rank, message):
    _logger.info("[%7.3fs][Rank %s] %s", _ts(), rank, message)


def log_test_success(rank, message):
    _logger.info("[%7.3fs][Rank %s] ✅ %s", _ts(), rank, message)


def log_test_validation(rank, message):
    _logger.info("[%7.3fs][Rank %s] ✓ %s", _ts(), rank, message)


def log_test_warning(rank, message):
    _logger.warning("[%7.3fs][Rank %s] ⚠️ %s", _ts(), rank, message)


def log_test_error(rank, message):
    _logger.error("[%7.3fs][Rank %s] ✗ %s", _ts(), rank, message)


_log_configure = configure


_log_configure(level=logging.INFO, force=True)


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
    def test_default_store_timeout_xccl(self):
        self._test_default_store_timeout("xccl")


class ProcessGroupXCCLNoGPUTest(TestCase):
    MAIN_PROCESS_RANK = 0

    def setUp(self):
        super().setUp()
        self.rank = self.MAIN_PROCESS_RANK
        self.world_size = 1
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.file = f

    def tearDown(self):
        pass

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(TEST_XPU, "GPUs are available, skipping test")
    def test_init_no_gpus(self):
        store = c10d.FileStore(self.file.name, self.world_size)
        with self.assertRaisesRegex(
            ValueError, "ProcessGroupXCCL is only supported with GPUs, no GPUs found!"
        ):
            c10d.ProcessGroupXCCL(store, self.rank, self.world_size)


class ProcessGroupXCCLInitTest(MultiProcessTestCase):
    device_type = "xpu"

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
        dm = torch.get_device_module(self.device_type)
        return dm.device_count()

    @property
    def device(self):
        return torch.device(self.device_type, self.rank % self.world_size)

    # A helper with the must-needed init args for test infra.
    # kwargs can be filled in by individual init tests.
    def _init_process_group(self, **kwargs):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            rank=self.rank,
            world_size=self.world_size,
            store=store,
            **kwargs,
        )

    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_init_wo_backend_str(self):
        self._init_process_group(device_id=self.device)
        x = torch.empty(1, device=self.device)
        c10d.all_reduce(x)

    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_scalable_init(self):
        self._init_process_group(device_id=self.device)
        x = torch.empty(1, device=self.device)
        c10d.all_reduce(x)


class ProcessGroupXCCLGroupTest(MultiProcessTestCase):
    def _create_process_group_xccl(self, store, opts, device_id=None):
        # create xccl processgroup with opts
        c10d.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=opts,
            device_id=device_id,
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def opts(self, high_priority_stream=False):
        opts = c10d.ProcessGroupXCCL.Options()
        opts.is_high_priority_stream = high_priority_stream
        return opts

    def setUp(self):
        super().setUp()

        # These tests are expected to throw SIGABRT(6);
        # But if we are in Sandcastle, `skip_but_pass_in_sandcastle` would return 0.
        #
        # CUDA: Uses native __trap() instruction → CUDA runtime catches it →
        #       clean exit(6) → exit code 6
        # ROCm: No native trap instruction, uses assert(0) (NanCheck.cu:24-27) →
        #       calls abort() → OS sends SIGABRT signal → process killed by signal →
        #       exit code -6
        # XPU: Uses assert(0) (NanCheck_XPU.cpp) → calls abort() →
        #       OS sends SIGABRT signal → process killed by signal → exit code -6
        TEST_NAN_ASSERT_RETURN = (
            0
            if (IS_SANDCASTLE and not TEST_MULTIACCELERATOR)
            else (
                -signal.SIGABRT if (torch.version.hip or TEST_XPU) else signal.SIGABRT
            )
        )
        self.special_return_code_checks = {
            self.test_nan_assert_float16.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float32.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float64.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_bfloat16.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float8_e4m3fn.__wrapped__: TEST_NAN_ASSERT_RETURN,
            self.test_nan_assert_float8_e5m2.__wrapped__: TEST_NAN_ASSERT_RETURN,
        }

        # self.num_gpus = torch.xpu.device_count()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @property
    def world_size(self):
        return get_required_world_size(self, 2)

    @property
    def rank_to_GPU(self):
        # return rank to GPU map
        return init_multigpu_helper(self.world_size, "xccl")

    @property
    def destroy_pg_upon_exit(self) -> bool:
        # This TestCase focuses on creation, destroy and abort of PG's. So it
        # does not need auto-destroy upon exit.
        return False

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 1 GPU")
    @skip_if_lt_x_gpu(1)
    def test_xccl_dist_backend_error(self):
        self.skipTest("Skipping due to no oneCCL error reporting")
        # TODO: expose proper error reporting in xccl backend
        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_xccl(store, self.opts())

        # Both rank 0 and 1 will use the same XPU device resulting in xcclInvalidUsage
        with self.assertRaises(dist.DistBackendError) as cm:
            dist.broadcast(torch.tensor([1, 2, 3]).xpu(), 0)
        self.assertTrue(isinstance(cm.exception, dist.DistError))

        self.assertIsInstance(cm.exception, RuntimeError)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("eager_init", [True, False])
    def test_close_pg(self, eager_init: bool):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank % torch.xpu.device_count()}")
        c10d.init_process_group(
            "xccl",
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

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    def test_restart_pg(self):
        # Note: restart test passes steadily only for blocking mode for now.
        # TODO: expand this test to non-blocking mode
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank % torch.xpu.device_count()}")

        # initialize pg for the first time
        c10d.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        t0 = torch.rand(10, 10, device=device)
        # First allreduce to lazy initialize default pg
        dist.all_reduce(t0)
        torch.xpu.synchronize()
        # Destroy pg
        dist.destroy_process_group()

        # we need a new Store for the new PG, achieving it by adding prefix
        new_store = c10d.PrefixStore("2nd", store)

        # re-initialize pg
        c10d.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=new_store,
        )
        t1 = torch.rand(5, 5, device=device)
        dist.all_reduce(t1)
        torch.xpu.synchronize()
        dist.destroy_process_group()
        # validate default pg is no longer valid
        with self.assertRaises(ValueError):
            dist.all_reduce(t1)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    def test_xpu_event_cache_mthd_race(self):
        # This unit test is to test the case when the collective is launched in
        # a side thread and the thread dies before the cache has been fully recycled.
        # More details can be found in this issue: https://github.com/pytorch/pytorch/issues/143470.

        # initiate collectives here
        def init_collective_task(t):
            dist.all_reduce(t)
            dist.all_reduce(t)
            dist.all_reduce(t)

        os.environ["TORCH_XCCL_XPU_EVENT_CACHE"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        self._create_process_group_xccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        dist.all_reduce(t)
        dist.all_reduce(t)
        dist.all_reduce(t)
        side_thread = threading.Thread(target=init_collective_task, args=(t,))
        side_thread.start()
        side_thread.join()
        torch.xpu.synchronize()

        # reset ENV
        os.environ["TORCH_XCCL_XPU_EVENT_CACHE"] = "0"

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(
        not TEST_MULTIACCELERATOR,
        "Test requires 2+ GPUs",
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
    def test_nan_assert(self, type):
        # Expecting a device-side error when NaN is detected
        os.environ["TORCH_XCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_xccl(store, self.opts())
        backend = pg._get_backend(torch.device("xpu"))

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

        # confirm enable/disable flag works
        backend._set_enable_nan_check(False)
        # Note: using all-gather here bc some XCCL/SM version does not support
        # FP8 reduction
        # temporarily skip due to https://github.com/pytorch/pytorch/issues/153479
        # pg._allgather_base(output, nan_tensor)

        backend._set_enable_nan_check(True)
        try:
            pg._allgather_base(output, nan_tensor)
        except Exception:
            sys.exit(signal.SIGABRT)

        dist.destroy_process_group()

        # reset env
        os.environ["TORCH_XCCL_NAN_CHECK"] = "0"

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_nan_rank_filter(self):
        # Putting NaN at recv buffer, program should not fail as NaN checker
        # should not check on receive buffer
        os.environ["TORCH_XCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank:d}")
        c10d.init_process_group(
            backend="xccl", store=store, rank=self.rank, world_size=self.world_size
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
        os.environ["TORCH_XCCL_NAN_CHECK"] = "0"

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_nan_check(self):
        # Not expecting an error, NaN check should not make legit code fail
        device = torch.device(f"xpu:{self.rank:d}")
        os.environ["TORCH_XCCL_NAN_CHECK"] = "1"
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", store=store, rank=self.rank, world_size=self.world_size
        )
        x = torch.ones((10,), device=device) * self.rank
        t = torch.ones(3, 4, device=device)
        c10d.broadcast(x, src=0)
        c10d.all_reduce(t)
        c10d.barrier()
        c10d.destroy_process_group()
        # reset env
        os.environ["TORCH_XCCL_NAN_CHECK"] = "0"

    def _helper_test_extra_xpu_context_by_memory(self):
        """
        A helper for `test_extra_xpu_context`, if pynvml is NOT available.
        If extra context is created, it would manifest into device 0's memory usage.
        """
        device = torch.device(f"xpu:{self.rank:d}")
        x = torch.empty((1,), device=device)
        # Rank 0 takes a snapshot before collective -- this snapshot should have
        # included rank 0's own context.
        if self.rank == 0:
            free, total = torch.xpu.mem_get_info(device)
            used_before = float(total - free)

        work = c10d.all_reduce(x, async_op=True)

        # Wait for non-0 ranks to garbage collect Work -- this is the latest
        # point where extra CUDA context can be created
        if self.rank == 0:
            time.sleep(5)
            free, total = torch.xpu.mem_get_info(device)
            used_after = float(total - free)
        del work

        # A barrier for non-0 ranks
        c10d.all_reduce(x)
        torch.xpu.synchronize(device)
        c10d.destroy_process_group()
        if self.rank == 0:
            # If non-0 rank creates a context on device 0, this assert would
            # fail because one context takes about 1 GB -- much more than the
            # tensor size created in this test.
            self.assertTrue(
                # Bump the heuristic from 1.5 to 1.7 due to
                # https://github.com/pytorch/pytorch/issues/153122
                used_after < used_before * 1.7,
                f"{device} used {used_after} bytes after collective, "
                f"70% more than the status before ({used_before} bytes). "
                f"Extra CUDA context may have been created.",
            )

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_extra_xpu_context(self):
        self.skipTest("XPU context test not supported")
        # TODO: Use xpu-smi to detect XPU contexts
        # Check if non-0 ranks would create extra XPU context on device 0
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank:d}")
        c10d.init_process_group(
            backend="xccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )
        try:
            self._helper_test_extra_xpu_context_by_nvml()
        except ModuleNotFoundError:
            self._helper_test_extra_xpu_context_by_memory()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_extra_xpu_context_sync_ops(self):
        self.skipTest("XPU context test not supported")
        # TODO: Use xpu-smi to detect XPU contexts
        # Loop a bunch of sync ops and see if any of them creates extra context.
        # Requires nvml to check number of processes resident on a device.
        try:
            import pynvml

            pynvml.nvmlInit()
        except Exception:
            self.skipTest("pynvml not available")

        # Check if non-0 ranks would create extra XPU context on device 0
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank:d}")
        c10d.init_process_group(
            backend="xccl",
            store=store,
            rank=self.rank,
            world_size=self.world_size,
            device_id=device,
        )

        x = torch.empty((1,), device=device)
        y = torch.empty((self.world_size,), device=device)

        c10d.all_reduce(x)
        c10d.reduce(x, dst=0)
        c10d.broadcast(x, src=0)
        c10d.all_gather_into_tensor(y, x)
        c10d.reduce_scatter_tensor(x, y)
        c10d.barrier()

        # Wait a bit for remote processes to touch my device
        if self.rank == 0:
            time.sleep(5)

        handle = pynvml.nvmlDeviceGetHandleByIndex(self.rank)
        processes = pynvml.nvmlDeviceGetComputeRunningProcesses(handle)
        nprocs = len(processes)

        # Don't exit till rank 0 is done with the nvml detection
        c10d.barrier()
        c10d.destroy_process_group()
        self.assertLessEqual(
            nprocs,
            1,
            f"Found {nprocs} processes creating contexts on {device}, expecting 1 at most",
        )

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    def test_destruct_before_terminate_pg(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        pg = self._create_process_group_xccl(store, self.opts())
        device = self.rank_to_GPU[self.rank][0]

        t = torch.rand(10, 10, device=device)
        # First allreduce to initialize state.
        pg.allreduce(t)
        # force destruction before terminating comms, destructor would terminate comms
        del pg

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(
        torch.xpu.device_count() < 2, "XCCL test requires 2+ XPUs"
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

    def _check_xccl_timeout(self, expected_timeout):
        pg = dist.distributed_c10d._get_default_group()
        options = pg._get_backend(torch.device(f"xpu:{self.rank}")).options
        self.assertEqual(options._timeout, expected_timeout)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_XPU, "No XPUs available, skipping test")
    def test_init_process_group_xccl_timeout(self):
        # xccl is handled 'specially' inside init_process_group and its options class is different from the options
        # used by the other PG's.  There are specific edge cases for xccl that need to be tested.

        store = c10d.FileStore(self.file_name, self.world_size)
        base_opts = dict(
            backend="xccl", store=store, rank=self.rank, world_size=self.world_size
        )

        # test the default value coming from the `init_process_group` kwarg default
        dist.init_process_group(**base_opts)
        self._check_xccl_timeout(torch.distributed.constants.default_pg_timeout)
        dist.destroy_process_group()

        # test that `kwarg` timeout takes effect
        new_timeout = timedelta(seconds=123)
        dist.init_process_group(**base_opts, timeout=new_timeout)
        self._check_xccl_timeout(new_timeout)
        dist.destroy_process_group()

        # test that timeout value provided via `pg_options` kwarg is ignored and issues warning,
        # 'timeout' kwarg (or its kwdefault) taking precedence
        opts = dist.ProcessGroupXCCL.Options()
        opts._timeout = timedelta(seconds=123)
        with warnings.catch_warnings(record=True):
            dist.init_process_group(**base_opts, pg_options=opts)
            # TODO(whc) i verified that we are indeed emitting this warning, and i can't figure out why i can't catch it.
            # self.assertEqual(len(w), 1)
            # self.assertTrue("pg_options._timeout was specified" in str(w[-1].message))
        self._check_xccl_timeout(torch.distributed.constants.default_pg_timeout)
        dist.destroy_process_group()

        # test that timeout value provided via `pg_options` kwarg is ignored and issues warning,
        # 'timeout' kwarg taking precedence
        opts = dist.ProcessGroupXCCL.Options()
        opts._timeout = timedelta(seconds=123)
        dist.init_process_group(
            **base_opts, pg_options=opts, timeout=timedelta(seconds=1240)
        )
        self._check_xccl_timeout(timedelta(seconds=1240))
        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("backend", [None, "xccl"])
    def test_set_xccl_pg_timeout(self, backend):
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
        pg.allreduce(torch.rand(10).xpu(self.rank))
        self._check_xccl_timeout(timedelta(seconds=123))
        pg._get_backend(torch.device(f"xpu:{self.rank}"))._set_default_timeout(
            timedelta(seconds=23)
        )
        self._check_xccl_timeout(timedelta(seconds=23))
        pg.allreduce(torch.rand(10).xpu(self.rank))
        c10d.distributed_c10d._set_pg_timeout(timedelta(seconds=252), pg)
        self._check_xccl_timeout(timedelta(seconds=252))

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("eager_init", [True, False])
    def test_new_group(self, eager_init: bool):
        # Test the optimization of new groups that contain all world
        # ranks use the "transparent" `xcclCommSplit` optimization.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank % torch.xpu.device_count()}")
        c10d.init_process_group(
            "xccl",
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

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @skipIfXpu(msg="XCCL doesn't currently support comm split, skipping test")
    def test_comm_split_subgroup(self):
        # Test `xcclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank}")
        pg = self._create_process_group_xccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        tensor = torch.full((1,), self.rank).xpu(device)
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

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    def test_comm_eager_init_subgroup(self):
        # Test `xcclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank}")
        # default PG comm is not initialized yet
        pg = self._create_process_group_xccl(store, self.opts())
        backend = pg._get_backend(torch.device(device))
        self.assertEqual(backend._is_initialized(), False)
        # create a subgroup eagerly
        new_group = c10d.new_group([0, 1], device_id=device)
        tensor = torch.full((1,), self.rank).xpu(device)
        dist.broadcast(tensor, 0, group=new_group)
        # the default group should stay lazy
        self.assertEqual(backend._is_initialized(), False)
        torch.xpu.synchronize()
        dist.destroy_process_group()

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @skipIfXpu(msg="XCCL doesn't currently support comm split, skipping test")
    def test_comm_split_group(self):
        # Test `xcclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank}")
        pg = self._create_process_group_xccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        tensor = torch.full((1,), self.rank).xpu(device)
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

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @skipIfXpu(msg="XCCL doesn't currently support comm split, skipping test")
    def test_comm_split_group_mixed_backend(self):
        # Test `xcclCommSplit` for smaller subgroups of the world when
        # we've passed a specific device_id to init_process_group.
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank}")
        # pg = self._create_process_group_xccl(store, self.opts(), device_id=device)
        # create xccl processgroup with opts
        c10d.init_process_group(
            "cpu:gloo,xpu:xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=self.opts(),
            device_id=device,
        )
        pg = c10d.distributed_c10d._get_default_group()
        backend = pg._get_backend(torch.device(device))

        xpu_tensor = torch.full((1,), self.rank).xpu(device)
        cpu_tensor = torch.full((1,), self.rank)
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
            dist.broadcast(xpu_tensor, dist.get_global_rank(ng1, 0), group=ng1)
            self.assertEqual(xpu_tensor, torch.full((1,), 0))
            dist.broadcast(cpu_tensor, dist.get_global_rank(ng1, 0), group=ng1)
            self.assertEqual(cpu_tensor, torch.full((1,), 0))

        ng2 = c10d.split_group(pg, [subg_ranks])
        self.assertEqual(ng2.group_desc, "default_pg:split:1")
        self.assertEqual(backend.comm_split_count(), 2)

        dist.destroy_process_group()

    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("eager_init", [True, False])
    def test_subgroup_p2p(self, eager_init: bool):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank % torch.xpu.device_count()}")
        c10d.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device if eager_init else None,
        )
        send_tensor = torch.ones(10, 10, device=device)
        group = dist.new_group()
        if self.rank == 0:
            dist.send(send_tensor, 1, group=group)
        if self.rank == 1:
            recv_tensor = torch.rand(10, 10, device=device)
            dist.recv(recv_tensor, 0, group=group)
            self.assertEqual(send_tensor, recv_tensor)
        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    def test_get_uid(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank}")
        pg = self._create_process_group_xccl(store, self.opts(), device_id=device)
        from torch.distributed.distributed_c10d import _get_process_group_uid

        self.assertEqual(_get_process_group_uid(pg), 0)
        pg_2 = c10d.new_group([0, 1])
        self.assertEqual(_get_process_group_uid(pg_2), 1)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    def test_set_process_group_desc(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank}")
        pg_default = self._create_process_group_xccl(
            store, self.opts(), device_id=device
        )
        self.assertEqual(pg_default.group_desc, "default_pg")
        pg_1 = c10d.new_group([0, 1], group_desc="test_purpose")
        self.assertEqual(pg_1.group_desc, "test_purpose")
        pg_2 = c10d.new_group([0, 1])
        self.assertEqual(pg_2.group_desc, "undefined")

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    def test_deterministic_mode_no_break(self):
        torch.use_deterministic_algorithms(True)
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank}")
        self._create_process_group_xccl(store, self.opts(), device_id=device)
        tensor = torch.empty(10, 10, device=device)
        dist.all_reduce(tensor)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    def test_init_with_idx(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device_idx = self.rank
        dist.init_process_group(
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device_idx,
        )
        dist.all_reduce(torch.empty(1, device=torch.device("xpu", device_idx)))

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    def test_block_current_stream(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank}")
        pg = self._create_process_group_xccl(store, self.opts(), device_id=device)

        t = torch.rand(10, device=device)
        work = pg.allreduce(t)
        work.block_current_stream()

        torch.xpu.current_stream().synchronize()
        work.wait()
        torch.xpu.synchronize()


class DistributedDataParallelTest(
    test_c10d_common.CommonDistributedDataParallelTest, MultiProcessTestCase
):
    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def _get_process_group(self):
        store = self._get_store()
        c10d.init_process_group(
            "xccl", store=store, rank=self.rank, world_size=self.world_size
        )
        return c10d.distributed_c10d._get_default_group()

    def _test_xccl_backend(
        self, devices, device_ids, multi_device=False, gradient_as_bucket_view=False
    ):
        process_group = self._get_process_group()
        self._test_ddp_with_process_group(
            process_group, devices, device_ids, multi_device, gradient_as_bucket_view
        )

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_complex_params_and_grads(self):
        # test ddp with complex parameters and gradients
        process_group = self._get_process_group()
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        device = torch.device(f"xpu:{device_id}")

        torch.manual_seed(42 + self.rank)
        model = nn.Sequential(
            nn.Linear(4, 8, dtype=torch.cfloat),
            nn.Linear(8, 2, dtype=torch.cfloat),
        ).to(device)

        torch.manual_seed(42 + self.rank)
        ref_model = nn.Sequential(
            nn.Linear(4, 8, dtype=torch.cfloat),
            nn.Linear(8, 2, dtype=torch.cfloat),
        ).to(device)

        # 0.001 forces tiny buckets, creating multiple buckets, stress-testing bucketing
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
            bucket_cap_mb=0.001,
        )

        torch.manual_seed(100)
        batch_size = 16
        input_dim = 4
        output_dim = 2

        x = torch.randn(batch_size, input_dim, dtype=torch.cfloat, device=device)
        y = torch.randn(batch_size, output_dim, dtype=torch.cfloat, device=device)

        optimizer_ddp = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
        optimizer_ref = torch.optim.SGD(ref_model.parameters(), lr=0.01)

        for iteration in range(5):
            optimizer_ddp.zero_grad()
            output_ddp = ddp_model(x)
            loss_ddp = torch.mean(torch.abs(output_ddp - y) ** 2)
            loss_ddp.backward()

            optimizer_ref.zero_grad()
            with torch.no_grad():
                for p_ddp, p_ref in zip(ddp_model.parameters(), ref_model.parameters()):
                    p_ref.copy_(p_ddp)

            output_ref = ref_model(x)
            loss_ref = torch.mean(torch.abs(output_ref - y) ** 2)
            loss_ref.backward()

            for param in ref_model.parameters():
                if param.grad is not None:
                    dist.all_reduce(
                        param.grad.data, op=dist.ReduceOp.SUM, group=process_group
                    )
                    param.grad.data /= self.world_size

            for name, (p_ddp, p_ref) in enumerate(
                zip(ddp_model.parameters(), ref_model.parameters())
            ):
                self.assertIsNotNone(
                    p_ddp.grad,
                    f"DDP gradient is None at iteration {iteration}, param {name}",
                )

                self.assertIsNotNone(
                    p_ref.grad,
                    f"Reference gradient is None at iteration {iteration}, param {name}",
                )

                self.assertTrue(
                    p_ddp.grad.is_complex(),
                    f"DDP gradient lost complex dtype at iteration {iteration}, param {name}",
                )

                self.assertTrue(
                    p_ref.grad.is_complex(),
                    f"Reference gradient lost complex dtype at iteration {iteration}, param {name}",
                )

                self.assertFalse(
                    torch.allclose(p_ddp.grad.imag, torch.zeros_like(p_ddp.grad.imag)),
                    f"DDP imaginary gradient is all zeros at iteration {iteration}, param {name}! "
                    f"This indicates the complex gradient bug.",
                )

                self.assertTrue(
                    torch.allclose(
                        p_ddp.grad.real, p_ref.grad.real, rtol=1e-5, atol=1e-5
                    ),
                    f"Real gradient mismatch at iteration {iteration}, param {name}\n"
                    f"DDP real: {p_ddp.grad.real.mean():.6f}, "
                    f"Ref real: {p_ref.grad.real.mean():.6f}",
                )

                self.assertTrue(
                    torch.allclose(
                        p_ddp.grad.imag, p_ref.grad.imag, rtol=1e-5, atol=1e-5
                    ),
                    f"Imaginary gradient mismatch at iteration {iteration}, param {name}\n"
                    f"DDP imag: {p_ddp.grad.imag.mean():.6f}, "
                    f"Ref imag: {p_ref.grad.imag.mean():.6f}",
                )

            optimizer_ddp.step()
            optimizer_ref.step()

        for p_ddp, p_ref in zip(ddp_model.parameters(), ref_model.parameters()):
            self.assertTrue(
                torch.allclose(p_ddp, p_ref, rtol=1e-4, atol=1e-4),
                "Final model parameters don't match after training",
            )

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_mixed_real_and_complex_params(self):
        # test ddp with mixed real and complex parameters and gradients
        process_group = self._get_process_group()
        device_id = gpus_for_rank(self.world_size)[self.rank][0]
        device = torch.device(f"xpu:{device_id}")

        class MixedModule(nn.Module):
            def __init__(self):
                super().__init__()
                self.complex_fc = nn.Linear(4, 4, dtype=torch.cfloat)
                self.real_fc = nn.Linear(4, 4, dtype=torch.float32)
                self.final_fc = nn.Linear(4, 2, dtype=torch.cfloat)

            def forward(self, x_complex, x_real):
                complex_branch = self.complex_fc(x_complex)
                real_branch = self.real_fc(x_real)
                real_as_complex = torch.complex(
                    real_branch, torch.zeros_like(real_branch)
                )
                return self.final_fc(complex_branch + real_as_complex)

        torch.manual_seed(42 + self.rank)
        model = MixedModule().to(device)
        ref_model = MixedModule().to(device)

        # 100 forces large bucket, forcing the BucketKey mechanism to segregate buckets, testing bucket segregation by dtype
        ddp_model = DistributedDataParallel(
            model,
            device_ids=[device_id],
            process_group=process_group,
            bucket_cap_mb=100,
        )

        optimizer_ddp = torch.optim.SGD(ddp_model.parameters(), lr=0.01)
        optimizer_ref = torch.optim.SGD(ref_model.parameters(), lr=0.01)

        torch.manual_seed(100)
        x_complex = torch.randn(8, 4, dtype=torch.cfloat, device=device)
        x_real = torch.randn(8, 4, dtype=torch.float32, device=device)
        target = torch.randn(8, 2, dtype=torch.cfloat, device=device)

        for iteration in range(5):
            optimizer_ddp.zero_grad()
            loss_ddp = torch.mean(torch.abs(ddp_model(x_complex, x_real) - target) ** 2)
            loss_ddp.backward()

            optimizer_ref.zero_grad()
            with torch.no_grad():
                for p_ddp, p_ref in zip(ddp_model.parameters(), ref_model.parameters()):
                    p_ref.copy_(p_ddp)
            loss_ref = torch.mean(torch.abs(ref_model(x_complex, x_real) - target) ** 2)
            loss_ref.backward()
            for param in ref_model.parameters(5):
                if param.grad is not None and param.grad.is_floating_point():
                    dist.all_reduce(
                        param.grad.data,
                        op=dist.ReduceOp.SUM,
                        group=process_group,
                    )
                    param.grad.data /= self.world_size

            for name, (p_ddp, p_ref) in enumerate(
                zip(ddp_model.parameters(), ref_model.parameters())
            ):
                self.assertIsNotNone(
                    p_ddp.grad,
                    f"DDP gradient is None at iteration {iteration}, param {name}",
                )
                self.assertIsNotNone(
                    p_ref.grad,
                    f"Reference gradient is None at iteration {iteration}, param {name}",
                )

                self.assertTrue(
                    p_ddp.grad.is_complex() == p_ref.grad.is_complex(),
                    f"Gradient dtype mismatch at iteration {iteration}, param {name}",
                )

                if p_ddp.grad.is_complex():
                    self.assertFalse(
                        torch.allclose(
                            p_ddp.grad.imag, torch.zeros_like(p_ddp.grad.imag)
                        ),
                        f"DDP imaginary gradient is all zeros at iteration {iteration}, param {name}",
                    )
                    self.assertTrue(
                        torch.allclose(
                            p_ddp.grad.real, p_ref.grad.real, rtol=1e-5, atol=1e-5
                        ),
                        f"Real gradient mismatch at iteration {iteration}, param {name}",
                    )
                    self.assertTrue(
                        torch.allclose(
                            p_ddp.grad.imag, p_ref.grad.imag, rtol=1e-5, atol=1e-5
                        ),
                        f"Imaginary gradient mismatch at iteration {iteration}, param {name}",
                    )
                else:
                    self.assertTrue(
                        torch.allclose(p_ddp.grad, p_ref.grad, rtol=1e-5, atol=1e-5),
                        f"Real gradient mismatch at iteration {iteration}, param {name}",
                    )

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_xccl_backend_multi_device_ids_not_allowed(self):
        int_devices = list(range(torch.xpu.device_count()))
        devices = [torch.device("xpu:" + str(i)) for i in int_devices]
        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            self._test_xccl_backend(devices, int_devices)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_xccl_backend_single_device_module_device_ids_None(self):
        self._test_xccl_backend(None, None)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_xccl_backend_single_device_module_empty_device_ids(self):
        # This tests the backward compatibility of accepting an empty list as `device_ids`,
        # although we no longer document this in favor of the default value of `None`,
        # which is consistent with multi-device modules and CPU modules.
        self._test_xccl_backend(None, [])

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    def test_xccl_backend_multi_device_module_device_ids_None(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("xpu:" + str(i)) for i in int_devices]
        self._test_xccl_backend(devices, None, multi_device=True)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_xccl_backend_1gpu_module_device_ids_integer_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("xpu:" + str(i)) for i in int_devices]
        self._test_xccl_backend(devices, int_devices)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_xccl_backend_1gpu_module_device_ids_torch_device_list(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:1]
        devices = [torch.device("xpu:" + str(i)) for i in int_devices]
        self._test_xccl_backend(devices, devices)

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    def test_xccl_backend_2gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        devices = [torch.device("xpu:" + str(i)) for i in int_devices]
        self._test_xccl_backend(devices, None, multi_device=True)

    @requires_xccl()
    @skip_if_lt_x_gpu(8)
    def test_xccl_backend_4gpu_module(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:4]
        devices = [torch.device("xpu:" + str(i)) for i in int_devices]
        self._test_xccl_backend(devices, None, multi_device=True)

    @requires_xccl()
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
            DistributedDataParallel(
                model, output_device=gpus[1], process_group=process_group
            )

        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            DistributedDataParallel(model, device_ids=gpus, process_group=process_group)

        with self.assertRaisesRegex(
            ValueError, "input module must be on the same type of devices"
        ):
            model.fc1 = model.fc1.cpu()
            DistributedDataParallel(model, process_group=process_group)

        model = model.cpu()
        with self.assertRaisesRegex(
            ValueError, "device_ids can only be None or contain a single element."
        ):
            DistributedDataParallel(model, device_ids=gpus, process_group=process_group)

    def _test_fp16(self, gradient_as_bucket_view=False):
        process_group = self._get_process_group()

        gpus = gpus_for_rank(self.world_size)[self.rank]
        model = nn.Linear(1, 1, bias=False).xpu(gpus[0]).half()
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
        input = torch.tensor([[2**15]]).xpu(gpus[0]).half()

        # Step model
        ddp_model.train()
        output = ddp_model(input)
        loss = output.sum()
        loss.backward()

        self.assertFalse(any(torch.isinf(p.grad).any() for p in ddp_model.parameters()))

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16(self):
        self._test_fp16()

    @requires_xccl()
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value(self):
        self._test_arbitrary_forward_return_value()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_arbitrary_forward_return_value_grad_is_view(self):
        self._test_arbitrary_forward_return_value(gradient_as_bucket_view=True)

    @requires_xccl()
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
        torch.xpu.set_device(self.rank)
        dist.init_process_group(
            backend="xccl",
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
    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_find_unused_parameters_kwarg_debug_detail(self):
        self._test_find_unused_parameters_kwarg()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_find_unused_parameters_kwarg_debug_info(self):
        self._test_find_unused_parameters_kwarg()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_find_unused_parameters_kwarg_debug_off(self):
        self._test_find_unused_parameters_kwarg()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_detail(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_find_unused_parameters_kwarg_grad_is_view_debug_info(self):
        self._test_find_unused_parameters_kwarg(gradient_as_bucket_view=True)

    @requires_xccl()
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward(self):
        self._test_multiple_outputs_multiple_backward()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_multiple_outputs_multiple_backward_grad_is_view(self):
        self._test_multiple_outputs_multiple_backward(gradient_as_bucket_view=True)

    @requires_xccl()
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
        devices = [torch.device("xpu:" + str(i)) for i in int_devices]
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_accumulate_gradients_module(self):
        self._test_accumulate_gradients_module()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_accumulate_gradients_module_with_grad_is_view(self):
        self._test_accumulate_gradients_module(gradient_as_bucket_view=True)

    @requires_xccl()
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
            "xccl", store=store, rank=self.rank, world_size=self.world_size
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_pass_default_pg(self):
        dist.init_process_group(
            "xccl",
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
                        tol = 3.0e-3 if has_half else 1.0e-5
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
                                for (param_name, p), p_ddp in zip(
                                    m_child.named_parameters(),
                                    m_ddp_child.parameters(),
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_grad_layout_1devicemodule_1replicaperprocess(self):
        dev0 = torch.device("xpu:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        # Tells DDP to use just one device.
        replica_devices = [dev0]
        # Tells _test_grad_layout to construct ConvNet with all layers on this process's first assigned device.
        layer_devs = dev0
        local_batch_size = 16
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    def test_grad_layout_2devicemodule(self):
        int_devices = gpus_for_rank(self.world_size)[self.rank][:2]
        dev0 = torch.device("xpu:" + str(int_devices[0]))
        dev1 = torch.device("xpu:" + str(int_devices[1]))
        # DDP's default behavior for a multi-device module is "don't replicate."
        replica_devices = None
        # Tells _test_grad_layout to constructs this process's ConvNet on 2 devices, with 2 layers on each device.
        layer_devs = [dev0] * 2 + [dev1] * 2
        local_batch_size = 16
        self._test_grad_layout(replica_devices, layer_devs, local_batch_size)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_param_layout_mismatch_error(self):
        self.skipTest("Skipping test due to no oneCCL error reporting")
        # TODO: expose proper error reporting in xccl backend
        process_group = self._get_process_group()

        dev0 = torch.device("xpu:" + str(gpus_for_rank(self.world_size)[self.rank][0]))
        layer_devs = dev0
        layer_formats = (
            [torch.contiguous_format] * 4
            if self.rank == 0
            else [torch.channels_last] * 4
        )
        layer_dtypes = [torch.float] * 4

        m = ConvNet(layer_devs, layer_formats, layer_dtypes)
        if self.rank == 0:
            DistributedDataParallel(m, device_ids=[dev0], process_group=process_group)
        else:
            with self.assertRaisesRegex(
                RuntimeError,
                ".* appears not to match strides of the same param in process 0",
            ):
                DistributedDataParallel(
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_future_passing_gpu_xccl(self):
        """
        This unit test verifies whether the Future object is passed properly using xccl backend.
        The hook callback function creates a Future object and sets a value to it.
        """
        process_group = self._get_process_group()

        # Get GPU model with simple_hook registered.
        gpu_model = self._gpu_model_with_ddp_comm_hook(process_group, self._simple_hook)

        # check whether the grads are equal to what simple_hook's then callback returns.
        # without the comm_hook, result would be 0.25 * torch.ones(2, 2).
        self._run_and_verify_hook(gpu_model, 8, 2 * torch.ones(2, 2))

    def _test_ddp_comm_hook_allreduce_hook_xccl(
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

    def _test_default_ddp_comm_hooks_xccl(self, gradient_as_bucket_view=False):
        """
        This unit test verifies whether default Python DDP communication hooks ALLREDUCE, FP16_COMPRESS
        and BF16_COMPRESS, can give the same result with the case of no hook registered.
        """
        process_group = self._get_process_group()

        # For these default DDP comm hooks, the only state is process group.
        state = process_group
        hook_options = [default.allreduce_hook, default.fp16_compress_hook]
        if c10d.is_xccl_available():
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

    def _test_powerSGD_ddp_comm_hook_xccl(self, gradient_as_bucket_view=False):
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

    def _test_builtin_ddp_comm_hooks_xccl(self, gradient_as_bucket_view=False):
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_xccl(self):
        self._test_ddp_comm_hook_allreduce_hook_xccl()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_xccl(self):
        self._test_default_ddp_comm_hooks_xccl()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_wrapper_xccl(self):
        self._test_fp16_compress_wrapper()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_bf16_compress_wrapper_xccl(self):
        self._test_bf16_compress_wrapper()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_xccl(self):
        self._test_builtin_ddp_comm_hooks_xccl()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_xccl(self):
        self._test_powerSGD_ddp_comm_hook_xccl()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_xccl_grad_is_view(self):
        self._test_ddp_comm_hook_allreduce_hook_xccl(gradient_as_bucket_view=True)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_hook_xccl_static_graph(self):
        self._test_ddp_comm_hook_allreduce_hook_xccl(static_graph=True)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_default_ddp_comm_hooks_xccl_is_view(self):
        self._test_default_ddp_comm_hooks_xccl(gradient_as_bucket_view=True)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_fp16_compress_wrapper_is_view(self):
        self._test_fp16_compress_wrapper(gradient_as_bucket_view=True)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_bf16_compress_wrapper_is_view(self):
        self._test_bf16_compress_wrapper(gradient_as_bucket_view=True)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_builtin_ddp_comm_hooks_xccl_grad_is_view(self):
        self._test_builtin_ddp_comm_hooks_xccl(gradient_as_bucket_view=True)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_powerSGD_ddp_comm_hook_xccl_grad_is_view(self):
        self._test_powerSGD_ddp_comm_hook_xccl(gradient_as_bucket_view=True)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_comm_hook_allreduce_with_then_hook_xccl(self):
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

    @requires_xccl()
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
            ).xpu(dev)

            m = torch.nn.parallel.DistributedDataParallel(
                m,
                bucket_cap_mb=1,
                gradient_as_bucket_view=use_bucket_view,
                device_ids=[dev],
                process_group=process_group,
            )

            for _ in range(3):
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_packed_sequence(self):
        """
        Tests that DDP with ``device_ids`` specified can run a forward and
        backward pass with ``PackedSequence`` s with parity compared to a local
        version of the model.
        """
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = dist.init_process_group(
            "xccl",
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_channels_last_contig(self):
        process_group = self._get_process_group()
        device = torch.device(f"xpu:{self.rank}")
        tensor = torch.ones((2, 16, 768, 1152), dtype=torch.float32, device=device).to(
            memory_format=torch.channels_last
        )
        process_group.broadcast([tensor]).wait()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_ddp_complex_params(self):
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

        torch.xpu.synchronize(device=device_id)


class XcclErrorHandlingTest(MultiProcessTestCase):
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
    def op_timeout_sec(self):
        return 3

    @property
    def world_size(self):
        return 3

    @property
    def blocking_wait_error_msg(self):
        return "timeout"

    def _run_all_reduce(self, pg):
        pg.allreduce(torch.rand(10).xpu(self.rank))

    @requires_xccl()
    @skip_if_lt_x_gpu(3)
    def test_send_recv_non_dense_tensor(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device("xpu", self.rank % torch.xpu.device_count())
        dist.init_process_group(
            rank=self.rank, world_size=self.world_size, store=store, device_id=device
        )
        full = torch.empty((64, 64), device=device).fill_(self.rank)
        # Take a slice in col dimension, making it non-dense
        block = full[:, 16:32]
        if self.rank == 0:
            with self.assertRaises(ValueError):
                dist.send(block, dst=1)
        elif self.rank == 1:
            with self.assertRaises(ValueError):
                dist.recv(block, src=0)

    @requires_xccl()
    @skip_if_lt_x_gpu(3)
    @skip_but_pass_in_sandcastle("Test does not pass when run locally")
    def test_xccl_errors_nonblocking(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupXCCL(store, self.rank, self.world_size)
        process_group.allreduce(torch.rand(10).xpu(self.rank))
        if self.rank == 0:
            # This allreduce does not block Python thread as allreduce enqueues
            # the xpu operation, and then wait only blocks the current xpu
            # stream.
            work = process_group.allreduce(torch.rand(10).xpu(self.rank))
            work.wait()

            # Now the work scheduled next should hang forever since the previous
            # allreduce will never complete.
            t = threading.Thread(target=self._run_all_reduce, args=(process_group,))
            t.daemon = True
            t.start()
            t.join(int(get_timeout(self.id()) / 5))
            self.assertTrue(t.is_alive())

    @requires_xccl()
    @skip_if_lt_x_gpu(3)
    def test_xccl_errors_blocking(self):
        # TODO: expose proper error reporting in xccl backend
        self.skipTest("Skipping test due to no oneCCL error reporting")
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupXCCL(
            store,
            self.rank,
            self.world_size,
        )
        x = torch.rand(1024 * 1024).xpu(self.rank)
        process_group.allreduce(x)
        if self.rank == 0:
            work = process_group.allreduce(x)
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                work.wait(timeout=timedelta(seconds=self.op_timeout_sec))

    def _test_barrier_error(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupXCCL(
            store,
            self.rank,
            self.world_size,
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

    @requires_xccl()
    @skip_if_lt_x_gpu(3)
    def test_xccl_blocking_wait_with_barrier(self):
        # TODO: expose proper error reporting in xccl backend
        self.skipTest("Skipping test due to no oneCCL error reporting")
        self._test_barrier_error()

    @requires_xccl()
    @skip_if_lt_x_gpu(3)
    def test_xccl_non_blocking_wait_with_barrier(self):
        # TODO: expose proper error reporting in xccl backend
        self.skipTest("Skipping test due to no oneCCL error reporting")
        # test the barrier behavior in the non blocking wait setting
        prev_xccl_async_error_handling = os.environ.get(
            "TORCH_XCCL_ASYNC_ERROR_HANDLING", None
        )
        # avoid watchdog thread interference
        os.environ["TORCH_XCCL_ASYNC_ERROR_HANDLING"] = "0"
        self._test_barrier_error()
        if prev_xccl_async_error_handling is not None:
            os.environ["TORCH_XCCL_ASYNC_ERROR_HANDLING"] = (
                prev_xccl_async_error_handling
            )

    @requires_xccl()
    @skip_if_lt_x_gpu(3)
    def test_error_detection_and_propagation(self):
        # TODO: expose proper error reporting in xccl backend
        self.skipTest("Skipping test due to no oneCCL error reporting")

        def assert_fut_success(fut):
            self.assertEqual(WorkResult(fut.value()), WorkResult.SUCCESS)

        # test the barrier behavior in the non blocking wait setting
        prev_xccl_async_error_handling = os.environ.get(
            "TORCH_XCCL_ASYNC_ERROR_HANDLING", None
        )
        # avoid watchdog thread interference
        os.environ["TORCH_XCCL_ASYNC_ERROR_HANDLING"] = "0"
        os.environ["TORCH_XCCL_PROPAGATE_ERROR"] = "1"
        # set heartbeat timeout to a small value so that we don't wait too long for things to shutdown
        os.environ["TORCH_XCCL_HEARTBEAT_TIMEOUT_SEC"] = "5"
        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupXCCL(
            store,
            self.rank,
            self.world_size,
        )
        self.assertEqual(process_group.get_error(), ErrorType.SUCCESS)
        barrier_work = process_group.barrier()
        barrier_work.wait()
        barrier_result = barrier_work.get_future_result().wait()
        self.assertEqual(WorkResult(barrier_result), WorkResult.SUCCESS)
        ar_work = process_group.allreduce(torch.rand(10).xpu(self.rank))
        ar_work.wait()
        fut = ar_work.get_future_result()
        # test adding a callback function
        fut.then(assert_fut_success)
        if self.rank == 0:
            work = process_group.allreduce(torch.rand(10).xpu(self.rank))
            work.wait()
            result = work.get_future_result().wait()
            self.assertEqual(WorkResult(result), WorkResult.TIMEOUT)
            self.assertEqual(process_group.get_error(), ErrorType.TIMEOUT)
        else:
            # other ranks not exiting before rank 0 timeout, this is to avoid
            # xccl error happening before rank 0 timeouts
            time.sleep(4)
            self.assertEqual(process_group.get_error(), ErrorType.REMOTE_ERROR)

        # Mimicking all ranks sensing the timeout, abort
        process_group.abort()

        if prev_xccl_async_error_handling is not None:
            os.environ["TORCH_XCCL_ASYNC_ERROR_HANDLING"] = (
                prev_xccl_async_error_handling
            )

    @requires_xccl()
    @skip_if_lt_x_gpu(3)
    def test_restart_pg_after_error(self):
        # TODO: expose proper error reporting in xccl backend
        self.skipTest("Skipping test due to no oneCCL error reporting")
        # test the barrier behavior in the non blocking wait setting
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank % torch.xpu.device_count()}")
        # initialize pg for the first time
        c10d.init_process_group(
            "xccl",
            timeout=timedelta(seconds=2),
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        pg = c10d.distributed_c10d._get_default_group()
        xccl_backend = pg._get_backend(torch.device(device))
        self.assertEqual(xccl_backend.get_error(), ErrorType.SUCCESS)
        barrier_work = xccl_backend.barrier()
        barrier_work.wait()
        barrier_result = barrier_work.get_future_result().wait()
        self.assertEqual(WorkResult(barrier_result), WorkResult.SUCCESS)
        self.assertEqual(xccl_backend.get_error(), ErrorType.SUCCESS)
        if self.rank == 0:
            work = xccl_backend.allreduce(torch.rand(10).xpu(self.rank))
            work.wait()
            result = work.get_future_result().wait()
            self.assertEqual(WorkResult(result), WorkResult.TIMEOUT)
            self.assertEqual(xccl_backend.get_error(), ErrorType.TIMEOUT)
            # we need a brand new fileStore for the new PG
            # the new file name is shared through the old fileStore
            with tempfile.NamedTemporaryFile(delete=False) as f:
                new_file_name = f.name
                store.set("file", new_file_name)
        else:
            # other ranks not exiting before rank 0 timeout, this is to avoid
            # xccl error happening before rank 0 timeouts
            time.sleep(4)
            self.assertEqual(xccl_backend.get_error(), ErrorType.REMOTE_ERROR)
            new_file_name = store.get("file").decode()

        # all ranks restart using a new store after detecting the timeout error
        xccl_backend.abort()
        dist.destroy_process_group()

        new_store = c10d.FileStore(new_file_name, self.world_size)
        # re-initialize pg
        c10d.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=new_store,
        )

        new_pg = c10d.distributed_c10d._get_default_group()
        new_xccl_backend = new_pg._get_backend(torch.device(device))
        t = torch.rand(5, 5, device=device)
        dist.all_reduce(t)
        self.assertEqual(new_xccl_backend.get_error(), ErrorType.SUCCESS)
        torch.xpu.synchronize()
        dist.destroy_process_group()

        # give some time for other ranks to exit first before destroying FileStore
        if self.rank == 0:
            time.sleep(4)
            os.remove(new_file_name)

    def _run_invalid_xccl_blocking_wait_env(self, val):
        os.environ["TORCH_XCCL_BLOCKING_WAIT"] = val
        store = c10d.FileStore(self.file_name, self.world_size)
        with self.assertRaises(RuntimeError):
            c10d.ProcessGroupXCCL(store, self.rank, self.world_size)

    @requires_xccl()
    @skip_if_lt_x_gpu(3)
    def test_invalid_xccl_blocking_wait_env(self):
        self._run_invalid_xccl_blocking_wait_env("abc")
        self._run_invalid_xccl_blocking_wait_env("-1")
        self._run_invalid_xccl_blocking_wait_env("2147483647")
        self._run_invalid_xccl_blocking_wait_env("4294967295")


class CommTest(test_c10d_common.AbstractCommTest, MultiProcessTestCase):
    @property
    def device(self):
        return f"xpu:{self.rank}"

    def setUp(self):
        super().setUp()
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_broadcast_coalesced_xccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device(f"xpu:{self.rank:d}")
        ranks = [0, 1]
        for root_rank in ranks:
            self._test_broadcast_coalesced(process_group, device, root_rank)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_xccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device(f"xpu:{self.rank:d}")
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_all_reduce_coalesced_manager_xccl(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", store=store, rank=self.rank, world_size=self.world_size
        )
        process_group = c10d.distributed_c10d._get_default_group()
        device = torch.device(f"xpu:{self.rank:d}")
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_intra_node_comm_all_reduce(self):
        from torch.testing._internal.common_cuda import SM80OrLater

        for peer in range(self.world_size):
            if peer == self.rank:
                continue
            if not torch._C._xpu_canDeviceAccessPeer(self.rank, peer):
                raise SkipTest("Test requires p2p access")

        if not SM80OrLater:
            raise SkipTest("Test requires sm>=80")

        store = c10d.FileStore(self.file_name, self.world_size)
        os.environ["ENABLE_INTRA_NODE_COMM"] = "1"
        os.environ["TEST_INTRA_NODE_COMM"] = "1"
        torch.xpu.set_device(self.rank)
        c10d.init_process_group(
            backend="xccl", rank=self.rank, world_size=self.world_size, store=store
        )
        expect = self.world_size * (self.world_size - 1) // 2

        # IntraNodeComm currently only supports sum and bf16.
        # Verify that it is not used in the next two configurations.
        t = torch.full((4 * 1024 // 2,), self.rank).xpu()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())

        t = torch.full((4 * 1024 // 2,), self.rank, dtype=torch.bfloat16).xpu()
        c10d.all_reduce(t, c10d.ReduceOp.AVG)

        # Verify that IntraNodeComm is used up to 10MB
        t = torch.full((4 * 1024 // 2,), self.rank, dtype=torch.bfloat16).xpu()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())

        t = torch.full((512 * 1024 // 2,), self.rank, dtype=torch.bfloat16).xpu()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())

        t = torch.full((10 * 1024**2 // 2,), self.rank, dtype=torch.bfloat16).xpu()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())

        # Verify that IntraNodeComm is not used beyond 10MB
        t = torch.full((10 * 1024**2 // 2 + 1,), self.rank, dtype=torch.bfloat16).xpu()
        c10d.all_reduce(t, c10d.ReduceOp.SUM)
        self.assertTrue(t.eq(expect).all())

        c10d.destroy_process_group()

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_default_pg_xccl(self):
        torch.xpu.set_device(self.rank)
        self._test_sequence_num_set_default_pg(backend="xccl")

    @skip_if_lt_x_gpu(2)
    @requires_xccl()
    def test_sequence_num_incremented_xccl_default(self):
        self._test_sequence_num_incremented_default_group("xccl")

    @skip_if_lt_x_gpu(4)
    @requires_xccl()
    def test_sequence_num_incremented_xccl_subgroup(self):
        if self.world_size < 4:
            return skip_but_pass_in_sandcastle("Test requires world_size of at least 4")
        self._test_sequence_num_incremented_subgroup("xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_sequence_num_set_xccl_new_group(self):
        torch.xpu.set_device(self.rank)
        self._test_sequence_num_set_new_group(backend="xccl")

    def _test_pass_xccl_options(self, pg_opts):
        store = c10d.FileStore(self.file_name, self.world_size)
        # Test init_process_group accepts options
        dist.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=pg_opts,
        )

        # Test with new_group
        pg = c10d.new_group([0, 1], pg_options=pg_opts)
        # test the process group works as expected
        t = torch.tensor([self.rank + 1] * 10).xpu(self.rank)
        pg.allreduce(t).wait()
        expected_tensor = torch.tensor([3] * 10).xpu(self.rank)
        self.assertEqual(expected_tensor, t)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_pass_xccl_options_high_priority_stream(self):
        pg_opts = c10d.ProcessGroupXCCL.Options()
        pg_opts.is_high_priority_stream = True
        self._test_pass_xccl_options(pg_opts)

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    def test_xccl_barrier(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", rank=self.rank, world_size=self.world_size, store=store
        )

        t = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
        c10d.all_reduce(t)
        expected_tensor = torch.tensor([3] * 10).xpu(2 * self.rank)
        self.assertEqual(expected_tensor, t)

        # Test with new_group
        pg = c10d.new_group([0, 1])
        t = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
        pg.allreduce(t).wait()
        self.assertEqual(expected_tensor, t)

        pg = c10d.new_group([0])
        if self.rank == 0:
            t = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)

        pg = c10d.new_group([1])
        if self.rank == 1:
            t = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
            expected_tensor = torch.tensor([self.rank + 1] * 10).xpu(2 * self.rank)
            pg.allreduce(t).wait()
            self.assertEqual(expected_tensor, t)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_xccl_barrier_device_ids(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", rank=self.rank, world_size=self.world_size, store=store
        )

        c10d.barrier(device_ids=[self.rank])

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_unwaited(self) -> None:
        # Verify that the process can terminate gracefully
        # even with unwaited tensors
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", rank=self.rank, world_size=self.world_size, store=store
        )

        # Case 1: Run collectives under context manager, and don't call wait on them.
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            input = torch.full(
                (10240, 10240), float(self.rank), device=f"xpu:{self.rank}"
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
        for _ in range(500):
            input = torch.full(
                (1024, 1024), float(self.rank), device=f"xpu:{self.rank}"
            )
            dist.all_reduce(input, op=dist.ReduceOp.SUM, async_op=True)
        # Work registry size is unchanged, since non-functional collectives not run under
        # the context manager is not registered in the work registry.
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 2)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_wait_tensor(self) -> None:
        # Verify that c10d_functional.wait_tensor() can be invoked on
        # output tensor of non-functional collective
        store = c10d.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            backend="xccl", rank=self.rank, world_size=self.world_size, store=store
        )

        # Case 1: under context manager (i.e. work is registered in registry)
        with _functional_collectives.allow_inflight_collective_as_graph_input_ctx():
            input1 = torch.full((10, 10), float(self.rank), device=f"xpu:{self.rank}")
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            dist.all_reduce(input1, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            torch.ops.c10d_functional.wait_tensor(input1)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

            input2 = torch.full((10, 10), float(self.rank), device=f"xpu:{self.rank}")
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            work = dist.all_reduce(input2, op=dist.ReduceOp.SUM, async_op=True)
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 1)
            work.wait()
            self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
            self.assertEqual(input1, input2)

        # Case 2: not under context manager (i.e. work is not registered in registry)
        input1 = torch.full((10, 10), float(self.rank), device=f"xpu:{self.rank}")
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        dist.all_reduce(input1, op=dist.ReduceOp.SUM, async_op=True)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        # this does not take effect, since the underlying wait_tensor() logic would not
        # be able to find the corresponding work object (because it's not registered in registry)
        torch.ops.c10d_functional.wait_tensor(input1)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)

        input2 = torch.full((10, 10), float(self.rank), device=f"xpu:{self.rank}")
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        work = dist.all_reduce(input2, op=dist.ReduceOp.SUM, async_op=True)
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        work.wait()
        self.assertEqual(torch._C._distributed_c10d._get_work_registry_size(), 0)
        self.assertEqual(input1, input2)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["DETAIL"])
    def test_xccl_warn_not_in_group_debug_detail(self):
        self._test_warn_not_in_group(backend="xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["INFO"])
    def test_xccl_warn_not_in_group_debug_info(self):
        self._test_warn_not_in_group(backend="xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    @with_dist_debug_levels(levels=["OFF"])
    def test_xccl_warn_not_in_group_debug_off(self):
        self._test_warn_not_in_group(backend="xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_nncl_rank_membership(self):
        self._test_rank_membership(backend="xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_tensor_dtype_mismatch(self):
        self._test_tensor_dtype_mismatch(backend="xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_tensor_dtype_complex(self):
        self._test_tensor_dtype_complex(backend="xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_base_k(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
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

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_reduce_scatter_tensor_coalesced(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
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


class SetDeviceMethod(Enum):
    TORCH_XPU_SET = auto()  # torch.xpu.set_device
    COLLECTIVE_ARGUMENT = auto()  # broadcast_object_list(device=)


class XcclProcessGroupWithDispatchedCollectivesTests(
    test_c10d_common.ProcessGroupWithDispatchedCollectivesTests
):
    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_collectives(self):
        self._test_collectives(backend="xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_allreduce_coalesced(self):
        self._test_allreduce_coalesced(backend="xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_all_to_all_single(self):
        self._test_all_to_all_single(backend="xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_allgather_base(self):
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "xpu"
        tensor = torch.ones(10, 10, device=torch.device(device))
        output_tensor = torch.zeros(10, 10, device=torch.device(device))
        dist.all_gather_into_tensor(output_tensor, tensor)
        self.assertEqual(output_tensor, tensor)

    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    @parametrize("float8_dtype", [torch.float8_e4m3fn, torch.float8_e5m2])
    def test_allgather_float8(self, float8_dtype):
        device = torch.device(f"xpu:{self.rank:d}")
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
        )
        device = "xpu"
        tensor = torch.ones(10, 16, device=torch.device(device)).to(float8_dtype)
        output_tensor = torch.zeros(10, 16, device=torch.device(device)).to(
            float8_dtype
        )
        dist.all_gather_into_tensor(output_tensor, tensor)
        self.assertEqual(output_tensor.view(torch.float32), tensor.view(torch.float32))


instantiate_parametrized_tests(XcclProcessGroupWithDispatchedCollectivesTests)


class LargeCommTest(test_c10d_common.AbstractLargeCommTest, MultiProcessTestCase):
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
    def device(self):
        return self.rank

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync(self):
        self._test_new_group_local_sync(backend="xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync_sanity_check(self):
        self._test_new_group_local_sync_sanity_check(backend="xccl")

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    def test_new_group_local_sync_duplicated_pg(self):
        self._test_new_group_local_sync_duplicate_pg(backend="xccl")

    def _init_two_pg2_subgroups(self, world_size: int = 4):
        if world_size != 4:
            raise NotImplementedError(
                f"need world size of 4 to get 2 subgroup PGs, but got world size of {world_size}"
            )
        store = c10d.FileStore(self.file_name, world_size)
        c10d.init_process_group(
            backend="xccl", store=store, rank=self.rank, world_size=world_size
        )
        # every rank creates the same sub groups
        # including unused sub groups in the current rank
        a_group = c10d.new_group([0, 1])
        b_group = c10d.new_group([2, 3])
        return a_group if self.rank < 2 else b_group

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_gather_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            # just easier to write the test for exactly 4 gpus, even if this test class increased to 8gpu later
            return

        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"xpu:{self.rank:d}")
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

    @requires_xccl()
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
        torch.xpu.set_device(self.rank)

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

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_reduce_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"xpu:{self.rank:d}")
        x = torch.ones((10,), device=device) * self.rank
        if self.rank == 0 or self.rank == 2:
            expected = x + torch.ones((10,), device=device) * (self.rank + 1)
            if group_rank:
                c10d.reduce(x, group_dst=0, group=subgroup, async_op=False)
            else:
                c10d.reduce(x, dst=self.rank, group=subgroup, async_op=False)
            self.assertEqual(x, expected)
        else:
            if group_rank:
                c10d.reduce(x, group_dst=0, group=subgroup, async_op=False)
            else:
                c10d.reduce(x, dst=self.rank - 1, group=subgroup, async_op=False)

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    @parametrize("async_op", [True, False])
    def test_send_recv_subgroup(self, async_op, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"xpu:{self.rank:d}")
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

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_batch_send_recv_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"xpu:{self.rank:d}")
        ops = []
        if self.rank == 0 or self.rank == 2:
            x = torch.empty((10,), device=device)
            if group_rank:
                ops.append(c10d.P2POp(dist.irecv, x, group=subgroup, group_peer=1))
            else:
                ops.append(
                    c10d.P2POp(dist.irecv, x, peer=self.rank + 1, group=subgroup)
                )

            for work in dist.batch_isend_irecv(ops):
                work.wait()
            expected = torch.ones((10,), device=device) * (self.rank + 1)
            self.assertEqual(x, expected)
        else:
            x = torch.ones((10,), device=device) * self.rank
            if group_rank:
                ops.append(c10d.P2POp(dist.isend, x, group=subgroup, group_peer=0))
            else:
                ops.append(
                    c10d.P2POp(dist.isend, x, peer=self.rank - 1, group=subgroup)
                )
            for work in dist.batch_isend_irecv(ops):
                work.wait()

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_broadcast_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"xpu:{self.rank:d}")
        if self.rank == 0 or self.rank == 2:
            x = torch.empty((10,), device=device)
            if group_rank:
                c10d.broadcast(x, group_src=1, group=subgroup)
            else:
                c10d.broadcast(x, src=self.rank + 1, group=subgroup)
            expected = torch.ones((10,), device=device) * (self.rank + 1)
            self.assertEqual(x, expected)
        else:
            x = torch.ones((10,), device=device) * self.rank
            if group_rank:
                c10d.broadcast(x, group_src=1, group=subgroup)
            else:
                c10d.broadcast(x, src=self.rank, group=subgroup)

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "set_device",
        [SetDeviceMethod.TORCH_XPU_SET, SetDeviceMethod.COLLECTIVE_ARGUMENT],
    )
    @parametrize("group_rank", [True, False])
    def test_send_recv_object_list_subgroup(
        self, set_device: SetDeviceMethod, group_rank
    ):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        if set_device == SetDeviceMethod.TORCH_XPU_SET:
            torch.xpu.set_device(self.rank)
            device = None
        else:
            device = torch.device(f"xpu:{self.rank:d}")
        if self.rank == 0 or self.rank == 2:
            x = [{}]
            if group_rank:
                c10d.recv_object_list(x, group_src=1, group=subgroup, device=device)
            else:
                c10d.recv_object_list(
                    x, src=self.rank + 1, group=subgroup, device=device
                )
            expected = [{"rank": self.rank + 1}]
            self.assertEqual(x, expected)
        else:
            x = [{"rank": self.rank}]
            if group_rank:
                c10d.send_object_list(x, group_dst=0, group=subgroup, device=device)
            else:
                c10d.send_object_list(
                    x, dst=self.rank - 1, group=subgroup, device=device
                )

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    @parametrize(
        "set_device",
        [SetDeviceMethod.TORCH_XPU_SET, SetDeviceMethod.COLLECTIVE_ARGUMENT],
    )
    @parametrize("group_rank", [True, False])
    def test_broadcast_object_list_subgroup(
        self, set_device: SetDeviceMethod, group_rank
    ):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        if set_device == SetDeviceMethod.TORCH_XPU_SET:
            torch.xpu.set_device(self.rank)
            device = None
        else:
            device = torch.device(f"xpu:{self.rank:d}")
        if self.rank == 0 or self.rank == 2:
            x = [{}]
            if group_rank:
                c10d.broadcast_object_list(
                    x, group_src=1, group=subgroup, device=device
                )
            else:
                c10d.broadcast_object_list(
                    x, src=self.rank + 1, group=subgroup, device=device
                )
            expected = [{"rank": self.rank + 1}]
            self.assertEqual(x, expected)
        else:
            x = [{"rank": self.rank}]
            if group_rank:
                c10d.broadcast_object_list(
                    x, group_src=1, group=subgroup, device=device
                )
            else:
                c10d.broadcast_object_list(
                    x, src=self.rank, group=subgroup, device=device
                )

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_scatter_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        device = torch.device(f"xpu:{self.rank:d}")
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

    @requires_xccl()
    @skip_if_lt_x_gpu(4)
    @parametrize("group_rank", [True, False])
    def test_scatter_object_list_subgroup(self, group_rank):
        world_size = 4
        if self.rank >= world_size:
            return
        subgroup = self._init_two_pg2_subgroups(world_size)
        torch.xpu.set_device(self.rank)
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

    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_ddp_set_sparse_metadata(self):
        self.skipTest("XCCL does not support sparse allreduce")
        # TODO: Support sparse allreduce in XCCL
        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
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
            if "XCCL does not support all_reduce with sparse tensors" in str(e):
                pass
            else:
                # Rethrow the exception if it's a different error
                raise


class ProcessGroupXCCLOneRankTest(MultiProcessTestCase):
    @property
    def world_size(self):
        return 1

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        super().tearDown()
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    @requires_xccl()
    @skip_if_lt_x_gpu(1)
    def test_reduce_scatter(self):
        device = torch.device(f"xpu:{self.rank:d}")

        store = dist.FileStore(self.file_name, self.world_size)
        dist.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            device_id=device,
        )

        size = 8192 + 1  # known bad size
        input_tensor = torch.randn(size, dtype=torch.bfloat16, device=device)

        with self.subTest("reduce_scatter"):
            output_tensor = torch.zeros(size, dtype=torch.bfloat16, device=device)
            dist.reduce_scatter(
                output=output_tensor,
                input_list=[input_tensor],
                op=dist.ReduceOp.AVG,
            )
            torch.testing.assert_close(output_tensor, input_tensor)

        with self.subTest("reduce_scatter_tensor"):
            output_tensor = torch.zeros(size, dtype=torch.bfloat16, device=device)
            dist.reduce_scatter_tensor(
                output=output_tensor,
                input=input_tensor,
                op=dist.ReduceOp.AVG,
            )
            torch.testing.assert_close(output_tensor, input_tensor)

        with self.subTest("reduce_scatter_tensor_coalesced"):
            output_tensor = torch.zeros(size, dtype=torch.bfloat16, device=device)
            with dist._coalescing_manager():
                dist.reduce_scatter_tensor(
                    output=output_tensor,
                    input=input_tensor,
                    op=dist.ReduceOp.AVG,
                )
            torch.testing.assert_close(output_tensor, input_tensor)


class XCCLTraceTestBase(MultiProcessTestCase):
    def setUp(self):
        super().setUp()
        os.environ["TORCH_XCCL_ENABLE_TIMING"] = (
            "0"  # see 'timing_enabled' parametrized tests
        )
        os.environ["TORCH_FR_BUFFER_SIZE"] = "1000"
        self.tempdir = tempfile.TemporaryDirectory()
        os.environ["TORCH_FR_DUMP_TEMP_FILE"] = self._trace_basename()
        os.environ["TORCH_FR_DEBUG_INFO_PIPE_FILE"] = self._trace_basename()
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
        return torch.device("xpu", self.rank_to_GPU[self.rank][0])

    def _join_processes(self, fn):
        # We need to patch sys.exit() as skip_if will use sys.exit() and
        # the exit code from the this process will not be caught.
        with mock.patch("sys.exit"):
            fn()
        super()._join_processes(fn)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process
        self.children_pipes = []
        parent_pipes = []
        for _ in range(self.world_size):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            self.children_pipes.append(child_conn)
            parent_pipes.append(parent_conn)
        piter = iter(parent_pipes)

        def wrap(*positional, args, **kwargs):
            args = (next(piter), *args)
            return proc(*positional, args=args, **kwargs)

        self._start_processes(wrap)

    def _create_process_group_xccl(self):
        store = dist.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "xccl", world_size=self.world_size, rank=self.rank, store=store
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
        return init_multigpu_helper(self.world_size, "xccl")

    def _trace_basename(self):
        # we pass the base to the env, and the dump util will append rank
        return os.path.join(self.tempdir.name, "trace_")

    def _trace_name(self, rank):
        return self._trace_basename() + str(rank)

    def started_or_scheduled(self, timing_enabled):
        return "started" if timing_enabled else "scheduled"


class XCCLTraceTest(XCCLTraceTestBase):
    def _verify_trace(self, t, include_collectives, timing_enabled, is_json):
        ver = t["version"]
        self.assertEqual(ver, "2.10")
        comm_lib_version = t["comm_lib_version"]
        torch_comm_lib_version = torch._C._distributed_c10d.get_xccl_version()
        self.assertEqual(comm_lib_version, torch_comm_lib_version)
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
        global_ranks = pg_config["0"]["ranks"]
        self.assertEqual(len(json.loads(global_ranks)), self.world_size)
        if include_collectives:
            self.assertEqual(len(t["entries"]), 2)
            t = t["entries"]
            last = t[-1]
            self.assertEqual(last["thread_id"], str(threading.current_thread().ident))
            self.assertEqual(last["thread_name"], "fr_test_thread")
            self.assertEqual(last["process_group"], ("0", "default_pg"))
            # TODO: Mark completed in PGXCCL so that the "state" field can be asserted here
            # self.assertEqual(last["state"], "completed")
            self.assertEqual(last["record_id"], 1)
            # TODO: Discovery not supported in PGXCCL work queue
            # s = last["time_discovered_started_ns"]
            # f = last["time_discovered_completed_ns"]
            # self.assertIsNotNone(f)
            # if timing_enabled:
            #   self.assertIsNotNone(s)
            #   self.assertTrue(s <= f)
            # we don't collect stack traces in JSON at the moment
            if not is_json:
                self.assertIn("test_c10d_xccl.py", str(last["frames"]))
            self.assertEqual(last["input_sizes"], ((3, 4),))
            self.assertEqual(last["input_dtypes"], ["Float"])
            self.assertEqual(last["output_sizes"], ((3, 4),))
            self.assertEqual(last["output_dtypes"], ["Float"])
            self.assertEqual(last["collective_seq_id"], 2)
            self.assertEqual(last["timeout_ms"], DEFAULT_PG_TIMEOUT)
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

    def load_libpthread_or_libc(self):
        import ctypes.util

        for base in ("pthread", "c"):
            path = ctypes.util.find_library(base)
            if path:
                try:
                    return ctypes.CDLL(path)
                except OSError:
                    continue
        raise RuntimeError("Could not load pthread or libc")

    # Directly set thread name using threading.current_thread().name does not work
    # because we use pthread_getname_np to get the thread’s OS-level name in C++
    def set_thread_name(self, name):
        import ctypes

        lib = self.load_libpthread_or_libc()
        pthread_self = lib.pthread_self
        pthread_self.restype = ctypes.c_void_p
        pthread_setname_np = lib.pthread_setname_np
        pthread_setname_np.argtypes = [ctypes.c_void_p, ctypes.c_char_p]

        # Get current pthread handle
        tid = pthread_self()

        # Set name
        pthread_setname_np(tid, name.encode())

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("timing_enabled", [True, False])
    @parametrize("include_collectives", [True, False])
    def test_short_json(self, timing_enabled, include_collectives):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        self.set_thread_name("fr_test_thread")
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        # gah ok so now the duration_ms is populated best-effort since it can only happen outside "dump()" api
        time.sleep(1)
        t = json.loads(
            torch._C._distributed_c10d._dump_xccl_trace_json(
                includeCollectives=include_collectives
            )
        )
        self._verify_trace(t, include_collectives, timing_enabled, True)
        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("timing_enabled", [True, False])
    @parametrize("include_collectives", [True, False])
    def test_short_pickle(self, timing_enabled, include_collectives):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        self.set_thread_name("fr_test_thread")
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        # gah ok so now the duration_ms is populated best-effort since it can only happen outside "dump()" api
        time.sleep(1)
        t = pickle.loads(
            torch._C._distributed_c10d._dump_xccl_trace(
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

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("timing_enabled", [True, False])
    def test_fr_record_reset(self, timing_enabled):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        self.set_thread_name("fr_test_thread")
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(5):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        # gah ok so now the duration_ms is populated best-effort since it can only happen outside "dump()" api
        time.sleep(1)
        torch._C._distributed_c10d._reset_fr_recording_xccl()
        for _ in range(4):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        time.sleep(1)
        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        self.assertEqual(len(t["entries"]), 4)
        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
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

        pg = self._create_process_group_xccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        self.parent.send("next")
        self.parent.recv()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    def test_long(self):
        os.environ["TORCH_FR_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_xccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        for _ in range(2):
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
        torch.xpu.synchronize(device=device)
        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 10)
        first = t[0]
        last = t[-1]
        self.assertEqual(last["profiling_name"], "xccl:all_reduce")
        # TODO: Mark completed in PGXCCL so that the "state" field can be asserted here
        # self.assertEqual(last["state"], "completed")
        self.assertIn("test_c10d_xccl.py", str(last["frames"]))
        self.assertEqual(last["input_sizes"], ((3, 4),))
        self.assertEqual(last["input_dtypes"], ["Float"])
        self.assertEqual(last["output_sizes"], ((3, 4),))
        self.assertEqual(last["output_dtypes"], ["Float"])
        self.assertEqual(last["timeout_ms"], DEFAULT_PG_TIMEOUT)
        self.assertEqual(last["collective_seq_id"] - first["collective_seq_id"], 9)
        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    def test_barrier_profiling(self):
        os.environ["TORCH_FR_BUFFER_SIZE"] = "10"
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_xccl()
        device = self.local_device
        a = torch.full((3, 4), float(self.rank), device=device)
        f = pg.barrier()
        f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        t = t["entries"]
        self.assertEqual(len(t), 2)
        first = t[0]
        last = t[-1]
        self.assertEqual(first["profiling_name"], "xccl:all_reduce_barrier")
        self.assertEqual(last["profiling_name"], "xccl:all_reduce")
        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @skipIfXpu(msg="XCCL doesn't currently support onlyActive filtering")
    @parametrize("timing_enabled", [True, False])
    @parametrize("only_active", [True, False])
    def test_trace_while_active(self, timing_enabled, only_active):
        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")
            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        with torch.xpu.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            e = torch.xpu.Event()
            e.record()
            if self.rank != 0:
                pg.allreduce(a).wait()
            e.synchronize()
            t = pickle.loads(
                torch._C._distributed_c10d._dump_xccl_trace(onlyActive=only_active)
            )
            t = t["entries"]
            if only_active:
                if self.rank == 0:
                    self.assertEqual(len(t), 0)
                else:
                    self.assertEqual(len(t), 1)
            if not only_active:
                if self.rank == 0:
                    self.assertEqual(t[-1]["profiling_name"], "xccl:all_reduce")
                    self.assertEqual(t[-1]["collective_seq_id"], 1)
                    self.assertEqual(t[-1]["state"], "completed")
                else:
                    self.assertEqual(t[-1]["profiling_name"], "xccl:all_reduce")
                    self.assertEqual(t[-1]["collective_seq_id"], 2)

            if self.rank == 0:
                pg.allreduce(a).wait()
            self.parent.send("next")
            self.assertEqual("next", self.parent.recv())
            torch.xpu.synchronize(device=device)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("timing_enabled", [True, False])
    def test_trace_while_stuck(self, timing_enabled):
        if self.rank == self.MAIN_PROCESS_RANK:
            for c in self.children_pipes:
                self.assertEqual(c.recv(), "next")
            for c in self.children_pipes:
                c.send("next")
            return

        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        device = self.local_device
        with torch.xpu.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            e = torch.xpu.Event()
            e.record()

            def gather_trace():
                e.synchronize()
                # give the other thread some time to fill the xpu buffer
                time.sleep(5)
                t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
                t = t["entries"]
                self.assertEqual(t[-1]["profiling_name"], "xccl:all_reduce")
                if self.rank == 0:
                    self.assertEqual(t[-1]["collective_seq_id"], 1)
                    # TODO: Mark completed in PGXCCL so that the "state" field can be asserted here
                    # self.assertEqual(t[-1]["state"], "completed")
                else:
                    self.assertEqual(t[-1]["collective_seq_id"], 2)
                    # self.assertEqual(
                    #     t[-1]["state"], self.started_or_scheduled(timing_enabled)
                    # )
                    # self.assertIsNone(t[-1]["time_discovered_completed_ns"])
                # this will eventually cause the missing rank 0
                # to continue which will unblock the non-zero ranks
                self.parent.send("next")

            if self.rank != 0:
                pg.allreduce(a).wait()
                th = threading.Thread(target=gather_trace)
                th.start()
                # fill the xpu buffer, at around 1024 events
                # this will stall
                for _ in range(2000):
                    a = a + a
                th.join()
            else:
                gather_trace()

            if self.rank == 0:
                pg.allreduce(a).wait()
            self.assertEqual("next", self.parent.recv())
            torch.xpu.synchronize(device=device)

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
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
        a destructed Work obj's xpu events
        """
        if timing_enabled:
            self.skipTest("XCCL timing is not consistent, skipping")

        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        num_coalesced_ops = 20
        ops_per_coalesce = len(op_sizes_per_coalesce)
        for _ in range(num_coalesced_ops):
            ops = []
            for input_sizes in op_sizes_per_coalesce:
                tensor = torch.zeros(input_sizes).to(self.local_device)
                if self.rank == 0:
                    ops.append(dist.P2POp(dist.irecv, tensor, 1))
                elif self.rank == 1:
                    tensor *= 2
                    ops.append(dist.P2POp(dist.isend, tensor, 0))

            dist.batch_isend_irecv(ops).pop().wait()

        torch.xpu.synchronize(device=self.local_device)

        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(2)

        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
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
                # the individual ops inside the coalescing group the individual op metadata,
                # but not the timing info coming from the actual coalesced kernel
                profiling_name = (
                    "xccl:recv 0<-1" if self.rank == 0 else "xccl:send 1->0"
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
                t["entries"][coalesced_op]["profiling_name"], "xccl:coalesced"
            )
            self.assertEqual(t["entries"][coalesced_op]["p2p_seq_id"], expected_seq)
            expected_seq += 1
            # TODO: Mark completed in PGXCCL so that the "state" field can be asserted here
            # self.assertEqual(t["entries"][coalesced_op]["state"], "completed")
            self.assertEqual(t["entries"][coalesced_op]["input_sizes"], [])
            self.assertEqual(t["entries"][coalesced_op]["output_sizes"], [])
            if timing_enabled:
                duration = t["entries"][coalesced_op]["duration_ms"]
                self.assertTrue(0.001 < duration < 10000, duration)
            else:
                self.assertTrue("duration_ms" not in t["entries"][coalesced_op])
            self.assertEqual(
                t["entries"][coalesced_op]["timeout_ms"], DEFAULT_PG_TIMEOUT
            )

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
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
        a destructed Work obj's xpu events
        """
        if timing_enabled:
            self.skipTest("XCCL timing is not consistent, skipping")

        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        num_repeats = 10
        ops_per_repeat = len(op_sizes)
        for _ in range(num_repeats):
            for input_sizes in op_sizes:
                tensor = torch.zeros(input_sizes).to(self.local_device)
                if self.rank == 0:
                    dist.recv(tensor, 1)
                elif self.rank == 1:
                    tensor *= 2
                    dist.send(tensor, 0)

        torch.xpu.synchronize(device=self.local_device)
        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        self.assertEqual(len(t["entries"]), num_repeats * (ops_per_repeat))
        expected_seq = 1
        expected_op_id = 1
        for seq in range(num_repeats * ops_per_repeat):
            input_sizes = op_sizes[seq % ops_per_repeat]
            profiling_name = "xccl:recv 0<-1" if self.rank == 0 else "xccl:send 1->0"
            self.assertEqual(t["entries"][seq]["profiling_name"], profiling_name)
            # we don't increment collective_seq_id for p2p ops.
            self.assertEqual(t["entries"][seq]["collective_seq_id"], 0)
            self.assertEqual(t["entries"][seq]["p2p_seq_id"], expected_seq)
            expected_seq += 1
            self.assertEqual(t["entries"][seq]["op_id"], expected_op_id)
            expected_op_id += 1
            self.assertEqual(t["entries"][seq]["input_sizes"], [input_sizes])
            self.assertEqual(t["entries"][seq]["output_sizes"], [input_sizes])
            # TODO: Mark completed in PGXCCL so that the "state" field can be asserted here
            # self.assertEqual(t["entries"][seq]["state"], "completed")

            if timing_enabled:
                duration = t["entries"][seq]["duration_ms"]
                self.assertTrue(0.001 < duration < 10000, duration)
            else:
                self.assertTrue("duration_ms" not in t["entries"][seq])

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("timing_enabled", [True, False])
    def test_allgather_uneven(self, timing_enabled):
        if self.rank == self.MAIN_PROCESS_RANK:
            return
        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()

        output_split_sizes = [i + 1 for i in range(self.world_size)]
        sum_len = sum(output_split_sizes)
        output_tensor = torch.zeros(sum_len, 2).to(self.rank)
        expected_tensor = torch.ones(sum_len, 2).to(self.rank)
        input_tensor = torch.ones(output_split_sizes[self.rank], 2).to(self.rank)

        dist.all_gather(
            list(torch.split(output_tensor, output_split_sizes)), input_tensor
        )
        torch.xpu.synchronize(device=self.rank)
        self.assertEqual(output_tensor, expected_tensor)
        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        self.assertEqual(len(t["entries"]), self.world_size + 1)
        for i in range(self.world_size):
            self.assertEqual(t["entries"][i]["profiling_name"], "xccl:_broadcast_oop")
            # collective_seq_id should be incremented once.
            self.assertEqual(t["entries"][i]["collective_seq_id"], 1)
            self.assertEqual(t["entries"][i]["input_sizes"], [[i + 1, 2]])
            self.assertEqual(
                t["entries"][i]["output_sizes"],
                [[i + 1, 2]],
            )
            self.assertEqual(t["entries"][i]["state"], "scheduled")
            # No event is recorded for individual ops
            self.assertTrue("time_discovered_completed_ns" in t["entries"][i])
        self.assertEqual(
            t["entries"][self.world_size]["profiling_name"], "xccl:ALLGATHER_coalesced"
        )

    # TODO(whc) test out other ops (And combinations of ops, if that's valid?)
    @requires_xccl()
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
        pg = self._create_process_group_xccl()
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

        torch.xpu.synchronize(device=self.rank)

        if timing_enabled:
            # wait for watchdog thread to process the queue of works
            time.sleep(1)

        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())

        self.assertEqual(
            len(t["entries"]), 1
        )  # one for the reduce_scatter_tensor_coalesced
        self.assertEqual(
            t["entries"][0]["profiling_name"], "xccl:reduce_scatter_tensor_coalesced"
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

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("timing_enabled", [True, False])
    def test_fr_record_reset_circular_buffer_full(self, timing_enabled):
        """
        Test that when the circular buffer in entries_ is full and we call reset,
        then fill the buffer with new entries, dump_entries returns only the new
        entries and not the old ones.
        """
        if self.rank == self.MAIN_PROCESS_RANK:
            return

        # Override buffer size to 10 for faster testing
        os.environ["TORCH_FR_BUFFER_SIZE"] = "10"

        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        self.set_thread_name("fr_test_thread")
        a = torch.full((3, 4), float(self.rank), device=device)

        # Fill the buffer completely with 10 entries
        for _ in range(10):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        time.sleep(1)

        # Verify buffer is full with 10 entries
        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        self.assertEqual(len(t["entries"]), 10)

        # Now reset the flight recorder
        torch._C._distributed_c10d._reset_fr_recording_xccl()

        # Add new entries after reset - fill the buffer completely again
        for _ in range(10):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        time.sleep(1)

        # Verify we get exactly 10 new entries, not 20
        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        self.assertEqual(len(t["entries"]), 10)

        # Verify all entries have the expected properties (from after reset)
        # After reset, record IDs should start from 0 again
        for i, entry in enumerate(t["entries"]):
            self.assertIn("profiling_name", entry)
            self.assertEqual(entry["profiling_name"], "xccl:all_reduce")
            self.assertIn("record_id", entry)
            # Record IDs should be sequential starting from 0 after reset
            self.assertEqual(entry["record_id"], i)

        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("timing_enabled", [True, False])
    def test_fr_record_reset_partial_overwrite(self, timing_enabled):
        """
        Test that when the circular buffer is full, we reset, and then add fewer
        entries than the buffer size, we only get the new entries.
        This tests that old entries at the end of the circular buffer are properly
        filtered out based on reset_epoch.
        """
        if self.rank == self.MAIN_PROCESS_RANK:
            return

        # Override buffer size to 10 for faster testing
        os.environ["TORCH_FR_BUFFER_SIZE"] = "10"

        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        self.set_thread_name("fr_test_thread")
        a = torch.full((3, 4), float(self.rank), device=device)

        # Fill the buffer completely
        for _ in range(10):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        time.sleep(1)

        # Reset the flight recorder
        torch._C._distributed_c10d._reset_fr_recording_xccl()

        # Add only 3 new entries (much less than buffer size)
        for _ in range(3):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        time.sleep(1)

        # Verify we only get the 3 new entries, not 10
        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        self.assertEqual(len(t["entries"]), 3)

        # Verify record IDs start from 0 after reset
        for i, entry in enumerate(t["entries"]):
            self.assertIn("record_id", entry)
            self.assertEqual(entry["record_id"], i)

        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("timing_enabled", [True, False])
    def test_fr_record_reset_wraparound(self, timing_enabled):
        """
        Test that when we reset in the middle of the circular buffer and then
        wrap around, dump_entries correctly returns only entries from the current
        epoch in the correct order.
        """
        if self.rank == self.MAIN_PROCESS_RANK:
            return

        # Override buffer size to 10 for faster testing
        os.environ["TORCH_FR_BUFFER_SIZE"] = "10"

        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        self.set_thread_name("fr_test_thread")
        a = torch.full((3, 4), float(self.rank), device=device)

        # Fill half the buffer
        for _ in range(5):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        time.sleep(1)

        # Reset at this point (reset happens at index 5)
        torch._C._distributed_c10d._reset_fr_recording_xccl()

        # Now add 8 entries, which will wrap around
        # (5->9 fills rest of buffer, then 0->2 wraps around)
        for _ in range(8):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        time.sleep(1)

        # Should get exactly 8 entries, properly ordered
        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        self.assertEqual(len(t["entries"]), 8)

        # Entries should be in chronological order
        # The dump_entries() method returns entries from next_ to end, then 0 to next_
        # After filtering old entries, we should have 8 entries in order
        # Verify record IDs start from 0 after reset (id_ is reset in reset_all())
        for i, entry in enumerate(t["entries"]):
            self.assertIn("profiling_name", entry)
            self.assertIn("record_id", entry)
            self.assertEqual(entry["record_id"], i)

        dist.destroy_process_group()

    @requires_xccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIACCELERATOR, "XCCL test requires 2+ XPUs")
    @parametrize("timing_enabled", [True, False])
    def test_fr_record_multiple_resets(self, timing_enabled):
        """
        Test multiple consecutive resets to ensure each reset properly increments
        the epoch and filters out entries from previous epochs.
        """
        if self.rank == self.MAIN_PROCESS_RANK:
            return

        # Override buffer size to 10 for faster testing
        os.environ["TORCH_FR_BUFFER_SIZE"] = "10"

        pg = self._create_process_group_xccl()
        if timing_enabled:
            pg._enable_collectives_timing()
        device = self.local_device
        self.set_thread_name("fr_test_thread")
        a = torch.full((3, 4), float(self.rank), device=device)

        # First batch: 2 entries
        for _ in range(2):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        time.sleep(1)

        # First reset
        torch._C._distributed_c10d._reset_fr_recording_xccl()

        # Second batch: 3 entries
        for _ in range(3):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        time.sleep(1)

        # Second reset
        torch._C._distributed_c10d._reset_fr_recording_xccl()

        # Third batch: 4 entries
        for _ in range(4):
            f = pg.allreduce(a)
        f.wait()
        torch.xpu.synchronize(device=device)
        time.sleep(1)

        # Should only see the last 4 entries
        t = pickle.loads(torch._C._distributed_c10d._dump_xccl_trace())
        self.assertEqual(len(t["entries"]), 4)

        # Verify record IDs start from 0 after the last reset
        for i, entry in enumerate(t["entries"]):
            self.assertIn("record_id", entry)
            self.assertEqual(entry["record_id"], i)

        dist.destroy_process_group()


def check_if_test_is_skipped(fn):
    def wrapper(self, *args, **kwargs):
        for skip in TEST_SKIPS.values():
            if self.processes[0].exitcode == skip.exit_code:
                return MultiProcessTestCase._check_return_codes(self, *args, **kwargs)
        return fn(self, *args, **kwargs)

    return wrapper


class XCCLTraceTestDumpOnTimeoutBase(XCCLTraceTestBase):
    timeout_sec = 1

    def _create_process_group_xccl(self):
        store = dist.FileStore(self.file_name, self.world_size)
        c10d.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            timeout=timedelta(seconds=XCCLTraceTestDumpOnTimeoutBase.timeout_sec),
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    @check_if_test_is_skipped
    def _check_return_codes(self, fn, elapsed_time):
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
class XCCLTraceTestDumpOnTimeout(XCCLTraceTestDumpOnTimeoutBase):
    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    @parametrize("timing_enabled", [True, False])
    def test_timeout_dumps(self, timing_enabled):
        # dump on heartbeatmonitor thread
        os.environ["TORCH_XCCL_COORD_CHECK_MILSEC"] = "1000"

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

        pg = self._create_process_group_xccl()
        if timing_enabled:
            # we force disabled timing in setup, since there is no 'disable' function
            pg._enable_collectives_timing()

        device = self.local_device
        with torch.xpu.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            if self.rank == 0:
                pg.allreduce(a).wait()

            # rank 0 will crash before it passes the sync, but rank1 will exit quickly and cleanly
            torch.xpu.synchronize(device=device)


instantiate_parametrized_tests(ProcessGroupXCCLGroupTest)
instantiate_parametrized_tests(XCCLTraceTestDumpOnTimeout)
instantiate_parametrized_tests(XCCLTraceTest)


@skip_but_pass_in_sandcastle
class XCCLTraceTestTimeoutDumpOnStuckRanks(XCCLTraceTestDumpOnTimeoutBase):
    @check_if_test_is_skipped
    def _check_return_codes(self, fn, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to abort and rank1 to exit cleanly in this test
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, -6)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_timeout_dumps_on_stuck_ranks(self):
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

        pg = self._create_process_group_xccl()
        device = self.local_device
        with torch.xpu.device(device):
            a = torch.full((3, 4), float(self.rank), device=device)

            pg.allreduce(a).wait()
            if self.rank == 0:
                pg.allreduce(a).wait()

            # rank 0 will get stuck, timeout and then signal a timeout to all ranks.
            torch.xpu.synchronize(device=device)

            if self.rank == 1:
                # Force rank 1 to idle so that it will eventually timeout as well after
                # getting the global signal to dump the debugging info.
                time.sleep(600)


@skip_but_pass_in_sandcastle
class XcclErrorDumpTest(XCCLTraceTestBase):
    def _wait_process(self, rank, timeout):
        try:
            self.processes[rank].join(timeout)
            return self.processes[rank].exitcode
        except TimeoutError:
            return None

    @check_if_test_is_skipped
    def _check_return_codes(self, fn, elapsed_time):
        # the base test infra assumes processes exit with matching return codes,
        # but we want rank0 to abort with exception and rank1 to exit with exit 1
        self.assertEqual(self.processes[0].exitcode, -6)
        self.assertEqual(self.processes[1].exitcode, 1)

    @requires_xccl()
    @skip_if_lt_x_gpu(2)
    def test_xccl_errors_dump(self):
        os.environ["TORCH_FR_BUFFER_SIZE"] = "1000"

        if self.rank == self.MAIN_PROCESS_RANK:
            # wait for both rank0 and 1 to crash before looking for dump
            self.assertEqual(self._wait_process(0, timeout=90), -6)
            self.assertEqual(self._wait_process(1, timeout=90), 1)
            # verify that the trace file exists for rank0
            self.assertTrue(os.path.exists(self._trace_name(rank=0)))
            return

        store = c10d.FileStore(self.file_name, self.world_size)
        process_group = c10d.ProcessGroupXCCL(
            store,
            self.rank,
            self.world_size,
            timeout=timedelta(seconds=10),
        )
        process_group.allreduce(torch.rand(10).xpu(self.rank))
        if self.rank == 0:
            work = process_group.allreduce(torch.rand(10).xpu(self.rank))
            # expect an error to be raised
            with self.assertRaisesRegex(dist.DistBackendError, ""):
                # Block the current stream on the XCCL stream
                work.wait()
                # Run some GPU operations
                torch.rand(10).xpu(self.rank)
        elif self.rank == 1:
            # Clean up structures (ex: files for FileStore before going down)
            del process_group
            sys.exit(1)


# tests that needs to be run with a larger world size
class ProcessGroupXCCLLargerScaleTest(MultiProcessTestCase):
    def _create_process_group_xccl(self, store, opts, device_id=None):
        # create xccl processgroup with opts
        c10d.init_process_group(
            "xccl",
            world_size=self.world_size,
            rank=self.rank,
            store=store,
            pg_options=opts,
            device_id=device_id,
        )
        pg = c10d.distributed_c10d._get_default_group()
        return pg

    def opts(self, high_priority_stream=False):
        opts = c10d.ProcessGroupXCCL.Options()
        opts.is_high_priority_stream = high_priority_stream
        return opts

    def setUp(self):
        super().setUp()
        # self.num_gpus = torch.xpu.device_count()
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
        return init_multigpu_helper(self.world_size, "xccl")

    @skip_if_lt_x_gpu(8)
    @skipIfXpu(msg="XCCL doesn't currently support comm split, skipping test")
    def test_comm_split_group_larger_scale(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank}")
        pg = self._create_process_group_xccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        tensor = torch.full((1,), self.rank).xpu(device)
        ng1 = c10d.split_group(pg, [[0, 1], [2, 3, 4, 5, 6, 7]])

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
            tensor2 = torch.full((1,), self.rank).xpu(device)
            dist.broadcast(tensor2, 7, group=ng2)
            self.assertEqual(tensor2, torch.full((1,), 7))
        else:
            self.assertEqual(ng2, None)
        # a barrier and a xpu sync before destroying all pgs.
        dist.barrier(pg)
        torch.xpu.synchronize()
        dist.destroy_process_group()

    @skip_if_lt_x_gpu(8)
    @skipIfXpu(msg="XCCL doesn't currently support comm split, skipping test")
    def test_comm_recursive_split_group(self):
        store = c10d.FileStore(self.file_name, self.world_size)
        device = torch.device(f"xpu:{self.rank}")
        pg = self._create_process_group_xccl(store, self.opts(), device_id=device)
        backend = pg._get_backend(torch.device(device))

        # split the default PG into 2 subgroups, each subgroup (ng1) has 4 ranks.
        tensor1 = torch.full((1,), self.rank).xpu(device)
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
        tensor2 = torch.full((1,), self.rank).xpu(device)
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

        # Test the case when the split changes the pg option of split group
        # while the parent pg option is not changed.
        new_pg = c10d.new_group([0, 1, 2, 3, 4, 5, 6, 7], device_id=device)
        backend_new_pg = new_pg._get_backend(torch.device(device))
        self.assertEqual(len(backend_new_pg.options.global_ranks_in_group), 8)
        c10d.split_group(new_pg, [[0, 2, 4, 6], [1, 3, 5, 7]])
        self.assertEqual(len(backend_new_pg.options.global_ranks_in_group), 8)
        # a barrier and a xpu sync before destroying all pgs.
        dist.barrier(pg)
        torch.xpu.synchronize()
        dist.destroy_process_group()


if __name__ == "__main__":
    assert not torch.xpu._initialized, (
        "test_distributed must not have initialized CUDA context on main process"
    )

    run_tests()
