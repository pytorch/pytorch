

import abc
import faulthandler
import itertools
import logging
import multiprocessing
import os
import queue
import subprocess
import sys
import tempfile
import threading
import time
import traceback
import types
import unittest
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial, reduce, wraps
from io import StringIO
from typing import Dict, NamedTuple, Optional, Union, List, Any, Callable, Tuple
from unittest.mock import patch

from torch._logging._internal import trace_log
import torch
import torch._dynamo.test_case
import torch.cuda.nccl
import torch.distributed as c10d
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory
import torch.nn as nn
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    find_free_port,
    IS_SANDCASTLE,
    retry_on_connect_failures,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_ROCM,
    TEST_WITH_TSAN,
    TestCase,
    run_tests,
    TEST_HPU,
)
from torch.testing._internal.distributed.multi_threaded_pg import (
    _install_threaded_pg,
    _uninstall_threaded_pg,
    ProcessLocalGroup,
)
import operator

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TestSkip(NamedTuple):
    exit_code: int
    message: str


TEST_SKIPS = {
    "backend_unavailable": TestSkip(
        72, "Skipped because distributed backend is not available."
    ),
    "small_worldsize": TestSkip(73, "Skipped due to small world size."),
    "odd_worldsize": TestSkip(87, "Skipped due to odd world size."),
    "no_cuda": TestSkip(74, "CUDA is not available."),
    "multi-gpu-1": TestSkip(75, "Need at least 1 CUDA device"),
    "multi-gpu-2": TestSkip(77, "Need at least 2 CUDA devices"),
    "multi-gpu-3": TestSkip(80, "Need at least 3 CUDA devices"),
    "multi-gpu-4": TestSkip(81, "Need at least 4 CUDA devices"),
    "multi-gpu-5": TestSkip(82, "Need at least 5 CUDA devices"),
    "multi-gpu-6": TestSkip(83, "Need at least 6 CUDA devices"),
    "multi-gpu-7": TestSkip(84, "Need at least 7 CUDA devices"),
    "multi-gpu-8": TestSkip(85, "Need at least 8 CUDA devices"),
    "nccl": TestSkip(76, "c10d not compiled with NCCL support"),
    "skipIfRocm": TestSkip(78, "Test skipped for ROCm"),
    "no_peer_access": TestSkip(79, "Test skipped because no GPU peer access"),
    "generic": TestSkip(
        86, "Test skipped at subprocess level, look at subprocess log for skip reason"
    ),
    "importerror": TestSkip(88, "Test skipped due to missing import"),
    "no_accelerator": TestSkip(89, "accelerator is not available."),
}


@dataclass
class DistTestCases:
    # Backends that do not support a specific collective
    skip_collective = {}
    skip_collective["allgather_coalesced"] = {"nccl", "mpi", "ucc"}
    skip_collective["reduce"] = set()
    skip_collective["sendrecv anysource"] = {"nccl", "ucc"}
    skip_collective["cpu barrier"] = {"nccl", "ucc"}

    # Sets showing that something is implemented
    backend_feature = {}
    backend_feature["gpu"] = {"nccl", "gloo", "ucc"}
    backend_feature["cuda"] = {"nccl", "gloo", "ucc"}
    backend_feature["ddp"] = {"nccl", "gloo", "ucc"}
    backend_feature["subgroup"] = {"nccl", "gloo", "ucc"}
    backend_feature["plugin"] = set()
    if TEST_HPU:
        backend_feature["hpu"] = {"hccl"}


def skip_if_no_gpu(func):
    """Skips if the world size exceeds the number of GPUs, ensuring that if the
    test is run, each rank has its own GPU via ``torch.cuda.device(rank)``."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            sys.exit(TEST_SKIPS["no_cuda"].exit_code)
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.device_count() < world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{world_size}"].exit_code)
        if TEST_HPU and torch.hpu.device_count < world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{world_size}"].exit_code)

        return func(*args, **kwargs)

    return wrapper


# TODO (kwen2501): what is the purpose of this decorator?  Tests with this
# decorator were always skipped. So they may be outdated already.
# Oct 2024: bumping the small-world criteria to < 8, as we are increasing the
# number of GPUs in CI from 2 to 4, and we need to continue skipping those tests
# to keep CI green. But this is just a temporary solution. We should clean up
# those tests somehow.
def skip_if_small_worldsize(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (os.environ["BACKEND"] != "mpi") and int(os.environ["WORLD_SIZE"]) < 8:
            sys.exit(TEST_SKIPS["small_worldsize"].exit_code)

        return func(*args, **kwargs)

    return wrapper


def skip_if_odd_worldsize(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (os.environ["BACKEND"] != "mpi") and int(os.environ["WORLD_SIZE"]) % 2 == 1:
            sys.exit(TEST_SKIPS["odd_worldsize"].exit_code)

        return func(*args, **kwargs)

    return wrapper


def require_n_gpus_for_nccl_backend(n, backend):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if backend == "nccl" and torch.cuda.device_count() < n:
                sys.exit(TEST_SKIPS[f"multi-gpu-{n}"].exit_code)
            else:
                return func(*args, **kwargs)

        return wrapper

    return decorator


def import_transformers_or_skip():
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            try:
                from transformers import (  # noqa: F401
                    AutoModelForMaskedLM,
                    BertConfig,
                )

                return func(*args, **kwargs)
            except ImportError:
                sys.exit(TEST_SKIPS["importerror"].exit_code)

        return wrapper

    return decorator


def at_least_x_gpu(x):
    return torch.cuda.is_available() and torch.cuda.device_count() >= x


def skip_if_lt_x_gpu(x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            if TEST_HPU and torch.hpu.device_count() >= x:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS[f"multi-gpu-{x}"].exit_code)

        return wrapper

    return decorator


# This decorator helps avoiding initializing cuda while testing other backends
def nccl_skip_if_lt_x_gpu(backend, x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if backend != "nccl":
                return func(*args, **kwargs)
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS[f"multi-gpu-{x}"].exit_code)

        return wrapper

    return decorator


def verify_ddp_error_logged(model_DDP, err_substr):
    # Verify error was logged in ddp_logging_data.
    ddp_logging_data = model_DDP._get_ddp_logging_data()
    assert "iteration" in ddp_logging_data
    assert "has_error" in ddp_logging_data
    assert "error" in ddp_logging_data
    logging_err = ddp_logging_data["error"]
    # Remove C++ stacktrace if needed.
    actual = (
        err_substr
        if err_substr.find("\nException raised from ") == -1
        else err_substr.split("\nException raised from ")[0]
    )
    assert (
        actual in logging_err
    ), f"Did not find expected {actual} in ddp logging data error: {logging_err}"


def with_nccl_blocking_wait(func):
    """
    Convenience decorator to set/unset TORCH_NCCL_BLOCKING_WAIT flag. Note that use of
    this decorator will override the setting of TORCH_NCCL_ASYNC_ERROR_HANDLING for
    the particular test. After the test, both TORCH_NCCL_BLOCKING_WAIT and
    TORCH_NCCL_ASYNC_ERROR_HANDLING will be restored to their original values.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save and unset TORCH_NCCL_ASYNC_ERROR_HANDLING
        try:
            cached_nccl_async_error_handling: Union[str, None] = os.environ[
                "TORCH_NCCL_ASYNC_ERROR_HANDLING"
            ]
            del os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"]
        except KeyError:
            # TORCH_NCCL_ASYNC_ERROR_HANDLING was unset
            cached_nccl_async_error_handling = None

        # Save val of TORCH_NCCL_BLOCKING_WAIT and set it.
        try:
            cached_nccl_blocking_wait: Union[str, None] = os.environ[
                "TORCH_NCCL_BLOCKING_WAIT"
            ]
        except KeyError:
            cached_nccl_blocking_wait = None
        finally:
            os.environ["TORCH_NCCL_BLOCKING_WAIT"] = "1"

        try:
            ret = func(*args, **kwargs)
            return ret
        finally:
            # restore old values.
            if cached_nccl_async_error_handling is not None:
                os.environ[
                    "TORCH_NCCL_ASYNC_ERROR_HANDLING"
                ] = cached_nccl_async_error_handling

            if cached_nccl_blocking_wait is not None:
                os.environ["TORCH_NCCL_BLOCKING_WAIT"] = cached_nccl_blocking_wait

    return wrapper


def with_dist_debug_levels(levels):
    """
    Runs a test for each distributed debug level specified in levels.
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            old_level = os.environ.get("TORCH_DISTRIBUTED_DEBUG", None)
            for level in levels:
                os.environ["TORCH_DISTRIBUTED_DEBUG"] = level
                c10d.set_debug_level_from_env()
                ret = func(*args, **kwargs)
                c10d.barrier()
                if old_level is not None:
                    os.environ["TORCH_DISTRIBUTED_DEBUG"] = old_level
            # Only returns test return for last test, but since these are
            # unittests the return value is not really used and earlier tests
            # would've raised had they failed.
            return ret

        return wrapper

    return decorator


def requires_gloo():
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_gloo_available(),
        "c10d was not compiled with the Gloo backend",
    )


def requires_nccl_version(version, msg):
    if not c10d.is_nccl_available():
        return skip_but_pass_in_sandcastle(
            "c10d was not compiled with the NCCL backend",
        )
    else:
        return skip_but_pass_in_sandcastle_if(
            torch.cuda.nccl.version() < version,
            f"Requires NCCL version greater than or equal to: {version}, found: {torch.cuda.nccl.version()}, reason: {msg}",
        )


def requires_nccl():
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_nccl_available(),
        "c10d was not compiled with the NCCL backend",
    )

def requires_ucc():
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_ucc_available(),
        "c10d was not compiled with the UCC backend",
    )

def requires_mpi():
    return skip_but_pass_in_sandcastle_if(
        not c10d.is_mpi_available(),
        "c10d was not compiled with the MPI backend",
    )


def requires_multicast_support():
    has_multicast_support = (
        torch.cuda.is_available()
        and _SymmetricMemory.has_multicast_support(DeviceType.CUDA, 0)
    )
    return skip_but_pass_in_sandcastle_if(
        not has_multicast_support,
        "multicast support is not available",
    )


def skip_if_rocm_multiprocess(func):
    """Skips a test for ROCm"""
    func.skip_if_rocm_multiprocess = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_ROCM:
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS["skipIfRocm"].exit_code)

    return wrapper


def skip_if_win32():
    return skip_but_pass_in_sandcastle_if(
        sys.platform == "win32",
        "This unit test case is not supported on Windows platform",
    )


def sm_is_or_higher_than(device: torch.device, major: int, minor: int) -> bool:
    """
    Returns True if the device's compute capability is (major, minor) or higher.
    Error out if the device is not a CUDA device.
    Returns False if device is a RoCM device.
    """
    if device.type != "cuda":
        raise ValueError("sm_is_or_later() is only supported for CUDA devices")

    if torch.version.hip is not None:
        # ROCm devices may have different compute capability codes
        return False

    return torch.cuda.get_device_capability(device) >= (major, minor)


@retry_on_connect_failures
def create_tcp_store(
    addr="localhost",
    world_size=1,
    is_master=True,
    timeout=timedelta(minutes=5),
    wait_for_workers=True,
    jit_class=False,
    use_libuv=True,
):
    """
    Creates a TCP store. Retries if the chosen port is already in use.
    """
    port = find_free_port()
    if jit_class:
        timeout_millisecond = int(timeout / timedelta(milliseconds=1))
        return torch.classes.dist_c10d.TCPStore(
            addr, port, world_size, is_master, timeout_millisecond
        )
    else:
        return c10d.TCPStore(
            addr, port, world_size, is_master, wait_for_workers=wait_for_workers, use_libuv=use_libuv
        )


if TEST_WITH_TSAN:
    # TSAN runs much slower.
    TIMEOUT_DEFAULT = 500
else:
    TIMEOUT_DEFAULT = int(os.getenv('DISTRIBUTED_TESTS_DEFAULT_TIMEOUT', '300'))
TIMEOUT_OVERRIDE = {"test_ddp_uneven_inputs": 400}


# https://github.com/pytorch/pytorch/issues/75665
if TEST_WITH_ROCM:
    TIMEOUT_OVERRIDE["test_join_kwargs"] = 200


def create_device(interface=None):
    if sys.platform == "win32" or interface is None:
        return c10d.ProcessGroupGloo.create_device(hostname="127.0.0.1")
    else:
        return c10d.ProcessGroupGloo.create_device(interface=interface)


def get_timeout(test_id) -> int:
    return TIMEOUT_OVERRIDE.get(test_id.split(".")[-1], TIMEOUT_DEFAULT)


@contextmanager
def captured_output():
    new_out, new_err = StringIO(), StringIO()
    old_out, old_err = sys.stdout, sys.stderr
    try:
        sys.stdout, sys.stderr = new_out, new_err
        yield sys.stdout, sys.stderr
    finally:
        sys.stdout, sys.stderr = old_out, old_err


def simple_sparse_reduce_tests(rank: int, world_size: int, num_inputs: int = 1):
    """
    Generate a number of basic test cases for sparse reduction.
    These cover tensors with a varying number of sparse dimensions and a varying
    number of dense dimensions. The only reduction operation we support is sum.
    """

    def generate(rank: int, world_size: int, sparse_dims: int = 1, dense_dims: int = 0):
        # First sparse dimension is [0..rank].
        # Subsequent dimensions are always 0, so we know there is
        # a non-empty intersection between any two sparse tensors.
        indices = torch.reshape(torch.arange(rank + 1), (1, rank + 1))
        shape = [world_size] + [2 for _ in range(dense_dims)]
        for _ in range(sparse_dims - 1):
            indices = torch.cat((indices, torch.zeros(1, rank + 1)))
            shape.append(world_size)
        values = torch.ones([rank + 1] + [2 for _ in range(dense_dims)])
        return torch.sparse_coo_tensor(indices, values, shape)

    def compute_sum(fn, world_size: int):
        return reduce(
            operator.add, [fn(rank, world_size) for rank in range(world_size)]
        )

    return [
        (
            [
                fn(num_inputs * rank + i, num_inputs * world_size)
                for i in range(num_inputs)
            ],
            [compute_sum(fn, num_inputs * world_size) for i in range(num_inputs)],
        )
        for fn in [
            partial(generate, sparse_dims=1),
            partial(generate, sparse_dims=2),
            partial(generate, sparse_dims=3),
            partial(generate, dense_dims=1),
            partial(generate, dense_dims=2),
            partial(generate, dense_dims=3),
        ]
    ]


# HELPER FOR MULTIGPU TESTS
def init_multigpu_helper(world_size: int, backend: str):
    """Multigpu tests are designed to simulate the multi nodes with multi
    GPUs on each node. Nccl backend requires equal #GPUs in each process.
    On a single node, all visible GPUs are evenly
    divided to subsets, each process only uses a subset.
    """
    nGPUs = torch.cuda.device_count()
    if TEST_HPU:
        nGPUs = torch.hpu.device_count()

    visible_devices = range(nGPUs)

    # If rank is less than or equal to number of available GPU's
    # then each rank can be mapped to corresponding GPU.
    nGPUs_per_process = 1
    if world_size > nGPUs:
        nGPUs_per_process = nGPUs // world_size
    rank_to_GPU = {
        i: list(visible_devices[i * nGPUs_per_process : (i + 1) * nGPUs_per_process])
        for i in range(world_size)
    }
    return rank_to_GPU


tmp_dir: Optional[tempfile.TemporaryDirectory] = None


def initialize_temp_directories(init_method: Optional[str] = None) -> None:
    global tmp_dir
    tmp_dir = tempfile.TemporaryDirectory()
    os.environ["TEMP_DIR"] = tmp_dir.name
    os.mkdir(os.path.join(tmp_dir.name, "barrier"))
    os.mkdir(os.path.join(tmp_dir.name, "test_dir"))
    init_dir_path = os.path.join(tmp_dir.name, "init_dir")
    os.mkdir(init_dir_path)
    # Set init method if specified.
    if init_method is not None:
        os.environ["INIT_METHOD"] = init_method
    else:
        os.environ["INIT_METHOD"] = FILE_SCHEMA + os.path.join(
            init_dir_path, "shared_init_file"
        )


def cleanup_temp_dir() -> None:
    if tmp_dir is not None:
        tmp_dir.cleanup()


# Most tests operate with this worldsize
DEFAULT_WORLD_SIZE = 4

# [How does MultiProcessTestCase work?]
# Each MultiProcessTestCase instance uses 1 + `world_size()` processes, by
# default `world_size()` returns 4. Let's take `test_rpc_spawn.py` as an
# example which inherits from this class. Its `Setup()` methods calls into
# `MultiProcessTestCase._spawn_processes()` which spawns `world_size()`
# subprocesses. During the spawn, the main process passes the test name to
# subprocesses, and the name is acquired from self.id(). The subprocesses
# then use the provided test function name to retrieve the function attribute
# from the test instance and run it. The main process simply waits for all
# subprocesses to join.


class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1
    # This exit code is used to indicate that the test code had an error and
    # exited abnormally. There are certain tests that might use sys.exit() to
    # simulate failures and in those cases, we can't have an exit code of 0,
    # but we still want to ensure we didn't run into any other errors.
    TEST_ERROR_EXIT_CODE = 10

    # do not early terminate for distributed tests.
    def _should_stop_test_suite(self) -> bool:
        return False

    # Many test cases init a process group but do not destroy it.  This property
    # determines whether this base test class should call
    # `destroy_process_group` on behalf of the test. Its value is customizable
    # by derived TestCase's but it is a pan-TestCase value (cannot be customized
    # for each test).
    @property
    def destroy_pg_upon_exit(self) -> bool:
        return True

    @property
    def world_size(self) -> int:
        return DEFAULT_WORLD_SIZE

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                fn()

        return types.MethodType(wrapper, self)

    # The main process spawns N subprocesses that run the test.
    # Constructor patches current instance test method to
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    def __init__(self, method_name: str = "runTest", methodName: str = "runTest") -> None:
        # methodName is the correct naming in unittest and testslide uses keyword arguments.
        # So we need to use both to 1) not break BC and, 2) support testslide.
        if methodName != "runTest":
            method_name = methodName
        super().__init__(method_name)
        try:
            fn = getattr(self, method_name)
            setattr(self, method_name, self.join_or_run(fn))
        except AttributeError as e:
            if methodName != 'runTest':
                # we allow instantiation with no explicit method name
                # but not an *incorrect* or missing method name
                raise ValueError(f"no such test method in {self.__class__}: {methodName}") from e

    def setUp(self) -> None:
        super().setUp()
        self.skip_return_code_checks = []  # type: ignore[var-annotated]
        self.processes = []  # type: ignore[var-annotated]
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name
        # pid to pipe consisting of error message from process.
        self.pid_to_pipe = {}  # type: ignore[var-annotated]

    def tearDown(self) -> None:
        super().tearDown()
        for p in self.processes:
            p.terminate()
        # Each Process instance holds a few open file descriptors. The unittest
        # runner creates a new TestCase instance for each test method and keeps
        # it alive until the end of the entire suite. We must thus reset the
        # processes to prevent an effective file descriptor leak.
        self.processes = []

    def _current_test_name(self) -> str:
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    def _start_processes(self, proc) -> None:
        self.processes = []
        for rank in range(int(self.world_size)):
            parent_conn, child_conn = torch.multiprocessing.Pipe()
            process = proc(
                target=self.__class__._run,
                name="process " + str(rank),
                args=(rank, self._current_test_name(), self.file_name, child_conn),
                kwargs={
                    "fake_pg": getattr(self, "fake_pg", False),
                }
            )
            process.start()
            logger.info("Started process %s with pid %s", rank, process.pid)
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process
        self._start_processes(proc)

    class Event(Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        logger.info("Starting event listener thread for rank %s", rank)
        while True:
            ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])

            if parent_pipe in ready_pipes:

                if parent_pipe.closed:
                    logger.info(
                        "Pipe closed for process %s, stopping event listener thread", rank
                    )
                    return

                event = parent_pipe.recv()
                logger.info("Received event %s on process %s", event, rank)

                if event == MultiProcessTestCase.Event.GET_TRACEBACK:
                    # Return traceback to the parent process.
                    with tempfile.NamedTemporaryFile(mode="r+") as tmp_file:
                        faulthandler.dump_traceback(tmp_file)
                        # Flush buffers and seek to read from the beginning
                        tmp_file.flush()
                        tmp_file.seek(0)
                        parent_pipe.send(tmp_file.read())

                        logger.info("Process %s sent traceback", rank)

            if signal_pipe in ready_pipes:
                return

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe, **kwargs) -> None:
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe)

    def run_test(self, test_name: str, parent_pipe) -> None:
        # Start event listener thread.
        signal_recv_pipe, signal_send_pipe = torch.multiprocessing.Pipe(duplex=False)
        event_listener_thread = threading.Thread(
            target=MultiProcessTestCase._event_listener,
            args=(parent_pipe, signal_recv_pipe, self.rank),
            daemon=True,
        )
        event_listener_thread.start()
        if sys.platform != "win32" and sys.platform != "darwin":
            # Register signal handler to dump stack traces on FATALs.
            # Windows and MacOS do not support the signal handlers.
            torch._C._set_print_stack_traces_on_fatal_signal(True)
        # Show full C++ stacktraces when a Python error originating from C++ is raised.
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retrieving a corresponding test and executing it.
        try:
            getattr(self, test_name)()
        except unittest.SkipTest as se:
            logger.info(
                "Process %s skipping test %s for following reason: %s", self.rank, test_name, str(se)
            )
            sys.exit(TEST_SKIPS["generic"].exit_code)
        except Exception:
            logger.error(
                "Caught exception: \n%s exiting "
                "process %s with exit code: %s",
                traceback.format_exc(), self.rank, MultiProcessTestCase.TEST_ERROR_EXIT_CODE
            )
            # Send error to parent process.
            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)

            assert event_listener_thread is not None
            event_listener_thread.join()
            # Close pipe after done with test.
            parent_pipe.close()

        if self.destroy_pg_upon_exit:
            try:
                # Some tests do destroy the pgs, and destroy can't be called twice.
                # This avoids spewing warnings about improperly shutting down.
                c10d.destroy_process_group()
            except (AssertionError, ValueError):
                pass

    def _get_timedout_process_traceback(self) -> None:
        pipes = []
        for i, process in enumerate(self.processes):
            if process.exitcode is None:
                pipe = self.pid_to_pipe[process.pid]
                try:
                    pipe.send(MultiProcessTestCase.Event.GET_TRACEBACK)
                    pipes.append((i, pipe))
                except ConnectionError as e:
                    logger.error(
                        "Encountered error while trying to get traceback for process %s: %s", i, e
                    )

        # Wait for results.
        for rank, pipe in pipes:
            try:
                # Wait for traceback
                if pipe.poll(5):
                    if pipe.closed:
                        logger.info(
                            "Pipe closed for process %s, cannot retrieve traceback", rank
                        )
                        continue

                    traceback = pipe.recv()
                    logger.error(
                        "Process %s timed out with traceback: \n\n%s", rank, traceback
                    )
                else:
                    logger.error(
                        "Could not retrieve traceback for timed out process: %s", rank
                    )
            except ConnectionError as e:
                logger.error(
                    "Encountered error while trying to get traceback for process %s: %s", rank, e
                )

    def _join_processes(self, fn) -> None:
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False
        try:
            while True:
                # check to see if any subprocess exited with an error early.
                for (i, p) in enumerate(self.processes):
                    # This is the exit code processes exit with if they
                    # encountered an exception.
                    if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                        print(
                            f"Process {i} terminated with exit code {p.exitcode}, terminating remaining processes."
                        )
                        active_children = torch.multiprocessing.active_children()
                        for ac in active_children:
                            ac.terminate()
                        subprocess_error = True
                        break
                if subprocess_error:
                    break
                # All processes have joined cleanly if they all a valid exitcode
                if all(p.exitcode is not None for p in self.processes):
                    break
                # Check if we should time out the test. If so, we terminate each process.
                elapsed = time.time() - start_time
                if elapsed > timeout:
                    self._get_timedout_process_traceback()
                    print(
                        f"Timing out after {timeout} seconds and killing subprocesses."
                    )
                    for p in self.processes:
                        p.terminate()
                    break
                # Sleep to avoid excessive busy polling.
                time.sleep(0.1)

            elapsed_time = time.time() - start_time

            if fn in self.skip_return_code_checks:
                self._check_no_test_errors(elapsed_time)
            else:
                self._check_return_codes(elapsed_time)
        finally:
            # Close all pipes
            for pipe in self.pid_to_pipe.values():
                pipe.close()

    def _check_no_test_errors(self, elapsed_time) -> None:
        """
        Checks that we didn't have any errors thrown in the child processes.
        """
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    f"Process {i} timed out after {elapsed_time} seconds"
                )
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time) -> None:
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        # If no processes are spawned, there is nothing to check.
        if not self.processes:
            logger.warning("Note: no subprocesses were spawned, test was likely skipped.")
            return

        first_process = self.processes[0]
        # first, we check if there are errors in actual processes
        # (via TEST_ERROR_EXIT CODE), and raise an exception for those.
        # the reason we do this is to attempt to raise a more helpful error
        # message than "Process x terminated/timed out"
        # TODO: we should pipe the exception of the failed subprocess here.
        # Currently, the actual exception is displayed as a logging output.
        errored_processes = [
            (i, p)
            for i, p in enumerate(self.processes)
            if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE
        ]
        if errored_processes:
            error = ""
            for i, process in errored_processes:
                # Get error from pipe.
                error_message = self.pid_to_pipe[process.pid].recv()
                error += (
                    f"Process {i} exited with error code {MultiProcessTestCase.TEST_ERROR_EXIT_CODE} "
                    f"and exception:\n{error_message}\n"
                )

            raise RuntimeError(error)
        # If no process exited uncleanly, we check for timeouts, and then ensure
        # each process exited cleanly.
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    f"Process {i} terminated or timed out after {elapsed_time} seconds"
                )
            self.assertEqual(
                p.exitcode,
                first_process.exitcode,
                msg=f"Expect process {i} exit code to match Process 0 exit code of {first_process.exitcode}, but got {p.exitcode}",
            )
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                if IS_SANDCASTLE:
                    # Don't use unittest.skip to skip the test on sandcastle
                    # since it creates tasks for skipped tests assuming there
                    # is some follow-up needed. Instead just "pass" the test
                    # with an appropriate message.
                    logger.info(
                        "Skipping %s on sandcastle for the following reason: %s", self.id(), skip.message
                    )
                    return
                else:
                    raise unittest.SkipTest(skip.message)
        self.assertEqual(
            first_process.exitcode,
            0,
            msg=f"Expected zero exit code but got {first_process.exitcode} for pid: {first_process.pid}",
        )

    @property
    def is_master(self) -> bool:
        return self.rank == 0

# Utility base class for distributed Multi Process Test cases
# This abstracts the PG creation and deletion, the backends are selected based
# on device type. The tests functions can be instantiated per device type using
# common_device_type.instantiate_device_type_tests
# other backends can add entry in backend() function
class DistributedTestBase(MultiProcessTestCase):

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    def tearDown(self):
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def backend(self, device) -> str:
        if "cuda" in device:
            return "nccl"
        elif "hpu" in device :   # intel gaudi
            return "hccl"
        else :
            return "gloo"

    def create_pg(self, device):
        num_visible_devices = torch.get_device_module(device).device_count()
        store = torch.distributed.FileStore(self.file_name, num_visible_devices)
        torch.distributed.init_process_group(
            backend=self.backend(device),
            world_size=self.world_size,
            rank=self.rank,
            store=store
        )
        if "nccl" in self.backend(device):
            torch.cuda.set_device(self.rank)
        return torch.distributed.distributed_c10d._get_default_group()

    def rank_to_device(self, device):
        num_visible_devices = torch.get_device_module(device).device_count()
        return {i: [i % num_visible_devices] for i in range(self.world_size)}

def run_subtests(
    cls_inst,
    subtest_config: Dict[str, List[Any]],
    test_fn: Callable,
    *test_args,
    **test_kwargs: Any,
):
    """
    Runs a test function given by ``test_fn`` as a subtest according to the
    configurations specified by ``subtest_config``. This amortizes the
    costly setup overhead (including process spawn and initializing the
    process group) over the subtests.

    Args:
        subtest_config (Dict[str, List[Any]]): A mapping from subtest
            keyword argument name to a list of its possible values.
        test_fn (Callable): A callable that runs the actual test.
        test_args: Positional arguments to pass to ``test_fn``.
        test_kwargs: Keyword arguments to pass to ``test_fn``.
    """
    # Convert the config mapping to a list to have a fixed order
    subtest_config_items: List[Tuple[str, List[Any]]] = list(subtest_config.items())
    subtest_config_keys: List[str] = [item[0] for item in subtest_config_items]
    subtest_config_values: List[List[Any]] = [item[1] for item in subtest_config_items]
    for values in itertools.product(*subtest_config_values):
        # Map keyword to chosen value
        subtest_kwargs = dict(zip(subtest_config_keys, values))
        with cls_inst.subTest(**subtest_kwargs):
            torch._dynamo.reset()
            test_fn(*test_args, **test_kwargs, **subtest_kwargs)
            torch._dynamo.reset()
        c10d.barrier()


# Cannot use functools.cache as it requires python 3.9
EFA_PROBE_RESULT = None


def has_efa() -> bool:
    """
    If shell command `fi_info -p efa -t FI_EP_RDM` returns exit code 0 then we assume that the machine has
    Libfabric EFA interfaces and EFA software components installed,
    see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html.
    """
    global EFA_PROBE_RESULT
    if EFA_PROBE_RESULT is not None:
        return EFA_PROBE_RESULT

    try:
        EFA_PROBE_RESULT = (
            subprocess.run(["fi_info", "-p", "efa", "-t", "FI_EP_RDM"], check=False).returncode == 0
        )
    except FileNotFoundError:
        EFA_PROBE_RESULT = False
    return EFA_PROBE_RESULT


def tp_transports():
    """
    If the machine has Libfabric EFA interfaces and EFA software components installed it may cause
    'RuntimeError: In operator() at tensorpipe/common/ibv.h:172 "": Operation not supported' if tensorpipe
    uses InfiniBand transport, so we exclude it from tensorpipe transports,
    see https://github.com/pytorch/pytorch/issues/73885 and https://github.com/pytorch/pytorch/issues/65022
    """
    return ["shm", "uv"] if has_efa() else None


def spawn_threads_and_init_comms(
    func=None, timeout=TIMEOUT_DEFAULT, world_size=DEFAULT_WORLD_SIZE
):
    """
    Wrapper to use with a test method
    """
    if func is None:
        return partial(
            spawn_threads_and_init_comms, timeout=timeout, world_size=world_size
        )


    def _run_test_method_with_multi_threads(world_size, callback):
        world = _install_threaded_pg()
        global_store = c10d.HashStore()

        def world_is_valid():
            return world == c10d.distributed_c10d._world

        def worker(rank, world_pg, store):
            c10d.init_process_group(
                backend="threaded", rank=rank, world_size=world_size, store=store
            )
            try:
                callback()
            except BaseException as ex:
                # Exceptions are handled in MultiThreadedTestCase
                MultiThreadedTestCase.exception_queue.put((rank, sys.exc_info()))
                ProcessLocalGroup.exception_handle(ex)  # trigger _terminate event and awaken worker threads
            finally:
                if world_is_valid():
                    c10d.destroy_process_group()

        threads = []
        for rank in range(world_size):
            t = threading.Thread(target=worker, args=(rank, world, global_store))
            t.start()
            threads.append(t)

        return threads


    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # TODO: get test name from kwargs
        torch._C._distributed_c10d._set_thread_isolation_mode(True)
        try:
            threads = _run_test_method_with_multi_threads(world_size, lambda: func(self, *args, **kwargs))
            # join and error handling
            MultiThreadedTestCase._join_threads(threads, func)
        finally:
            torch._C._distributed_c10d._set_thread_isolation_mode(False)

    return wrapper


class MultiThreadedTestCase(TestCase):
    """
    Test runner that runs all tests with the in-proc process group using
    multiple threads with the threaded process group.

    Each test spawns world_size threads and run the test method in each thread.

    Difference from regular MultiProcess test runner:
    Must explicitly defines SetUp and call self._spawn_threads() to run the tests.
    Cannot use setUp / tearDown (must use perThreadSetup / perThreadShutdown)
        to set up / tear down each thread when running each test.
    No global state possible
        How bad of a limitation is this?
    """
    exception_queue = queue.Queue()

    MAIN_THREAD_RANK = -1

    def join_or_run(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_THREAD_RANK:
                self._join_threads(self.threads, fn)
            else:
                fn()

        return types.MethodType(wrapper, self)

    def __init__(self, method_name: str = "runTest", methodName: str = "runTest") -> None:
        # methodName is the correct naming in unittest and testslide uses keyword arguments.
        # So we need to use both to 1) not break BC and, 2) support testslide.
        if methodName != "runTest":
            method_name = methodName
        super().__init__(method_name)
        try:
            fn = getattr(self, method_name)
            setattr(self, method_name, self.join_or_run(fn))
        except AttributeError as e:
            if methodName != 'runTest':
                # we allow instantiation with no explicit method name
                # but not an *incorrect* or missing method name
                raise ValueError(f"no such test method in {self.__class__}: {methodName}") from e

    def perThreadSetUp(self):
        # super().setUp()  # TestCase.setUp() calls torch.manual_seed()
        pass

    def perThreadTearDown(self):
        pass

    def setUp(self) -> None:
        """
        setUp only set up things in the main thread, if you want to configure things
        in the spawned threads, use perThreadSetUp
        """
        super().setUp()
        self.rank = self.MAIN_THREAD_RANK
        self.threads = []
        # Show full C++ stacktraces when a Python error originating from C++ is raised.
        os.environ["TORCH_SHOW_CPP_STACKTRACES"] = "1"

    def tearDown(self):
        """
        tearDown only set up things in the main thread, if you want to configure things
        in the spawned threads, use perThreadTearDown
        """
        super().tearDown()
        self.threads = []

    def _spawn_threads(self):
        """
        class method to spawn threads and run test, use this method in the SetUp of your TestCase
        """
        torch._C._distributed_c10d._set_thread_isolation_mode(True)
        test_name = self._current_test_name
        # for each test case, we need to create thread local world, and a global store
        world = _install_threaded_pg()
        self.__class__.global_store = c10d.HashStore()

        def world_is_valid():
            return world == c10d.distributed_c10d._world

        if not world_is_valid():
            raise RuntimeError("Invalid world")

        for rank in range(self.world_size):
            t = threading.Thread(target=self.__class__._run, args=(test_name, rank, self.world_size))
            t.start()
            self.threads.append(t)

    @classmethod
    def _run(cls, test_name, rank, world_size, **kwargs):
        self = cls(test_name)
        self.rank = rank

        # precision/rel_tol is a thread-local setting since it may be overridden per test, need to make
        # every thread have the same value. This would be relevant when we use op db tests, where it
        # needs those states to be set i.e. using instantiate_device_type_tests()
        # TODO: figure out a better way to do this
        if hasattr(self, "_tls"):
            self._tls = threading.local()
            self._tls.precision = TestCase._precision
            self._tls.rel_tol = TestCase._rel_tol

        self.run_test_with_threaded_pg(test_name, rank, world_size)

    def run_test_with_threaded_pg(self, test_name, rank, world_size):
        """
        Run the current test associated with `test_name` using the threaded process group.
        """
        c10d.init_process_group(
            backend="threaded", rank=rank, world_size=world_size, store=self.__class__.global_store
        )
        self.perThreadSetUp()

        try:
            getattr(self, test_name)()
        except BaseException as ex:
            self.exception_queue.put((rank, sys.exc_info()))
            ProcessLocalGroup.exception_handle(ex)  # trigger _terminate event and awaken worker threads
        finally:
            c10d.destroy_process_group()
            self.perThreadTearDown()


    @classmethod
    def _join_threads(cls, threads, fn):
        timeout = TIMEOUT_DEFAULT
        try:
            for idx, thread in enumerate(threads):
                thread.join(max(0, timeout))
                if thread.is_alive():
                    MultiThreadedTestCase.exception_queue.put(
                        (
                            idx,
                            (
                                TimeoutError,
                                TimeoutError(
                                    f"Rank failed to join in under {timeout} seconds"
                                ),
                                None,
                            ),
                        )
                    )
            ProcessLocalGroup.reset()
            failed_ranks = []
            while not cls.exception_queue.empty():
                failure = cls.exception_queue.get()
                failed_ranks.append(failure)
        finally:
            _uninstall_threaded_pg()
            torch._C._distributed_c10d._set_thread_isolation_mode(False)

        cls._check_return_codes(failed_ranks, timeout, fn)

    @classmethod
    def _check_return_codes(cls, failed_ranks, timeout, fn):
        # Print based on exceptions raised from threads
        #   SkipTest: print info for each thread
        #   TimeoutError: raise RuntimeError for any timed out thread
        #   Normal Exception: print error for each thread that raises exception
        #   and raise a RuntimeError
        error_msg = ""
        skip_code = -1
        for rank, exc_info in failed_ranks:
            exc = exc_info[1]
            if isinstance(exc, unittest.SkipTest):
                logger.info(
                    "Thread %s skipping test %s for following reason: %s", rank, fn, str(exc)
                )
                if skip_code < 0:
                    skip_code = TEST_SKIPS["generic"].exit_code
            elif isinstance(exc, TimeoutError):
                msg = f"Thread {rank} terminated or timed out after {timeout} seconds\n"
                logger.error(msg)
                raise RuntimeError(msg)
            elif isinstance(exc, Exception):
                msg = "".join(traceback.format_exception(*exc_info))
                logger.error(
                    "Caught exception: \n%s exiting thread %s", msg, rank
                )
                error_msg += (
                    f"Thread {rank} exited with exception:\n{msg}\n"
                )
            elif isinstance(exc, SystemExit):
                if type(exc.code) == int and skip_code < 0:
                    skip_code = exc.code

        # check exceptions
        if len(error_msg) > 0:
            raise RuntimeError(error_msg)
        # check skip
        if skip_code > 0:
            for skip in TEST_SKIPS.values():
                if skip_code == skip.exit_code:
                    if IS_SANDCASTLE:
                        # "pass" the test with an appropriate message.
                        logger.info(
                            "Skipping %s on sandcastle for the following reason: %s", fn, skip.message
                        )
                        return
                    else:
                        raise unittest.SkipTest(skip.message)

    @property
    def world_size(self) -> int:
        return DEFAULT_WORLD_SIZE

    @property
    def _current_test_name(self) -> str:
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    def assertEqualOnRank(self, x, y, msg=None, *, rank=0):
        """
        The reason why we have this util function instead of
        self.assertEqual is all threads are sharing one CPU RNG
        so the assertion result is only reliable on rank 0
        """
        if self.rank == rank:
            self.assertEqual(x, y, msg)

    def assertNotEqualOnRank(self, x, y, msg=None, *, rank=0):
        if self.rank == rank:
            self.assertNotEqual(x, y)


class SaveForwardInputsModule(nn.Module):
    def __init__(
        self,
        forward_inputs: Dict[nn.Module, torch.Tensor],
        cast_forward_inputs: bool,
    ) -> None:
        super().__init__()
        self.l = nn.Linear(100, 100)
        self.forward_inputs = forward_inputs
        self.cast_forward_inputs = cast_forward_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_inputs[self] = x
        return self.l(x.to(self.l.weight.dtype) if self.cast_forward_inputs else x)


class SaveForwardInputsModel(nn.Module):
    def __init__(
        self,
        forward_inputs: Dict[nn.Module, torch.Tensor],
        cast_forward_inputs: bool,
    ) -> None:
        super().__init__()
        self.c1 = SaveForwardInputsModule(forward_inputs, cast_forward_inputs)
        self.c2 = SaveForwardInputsModule(forward_inputs, cast_forward_inputs)
        self.forward_inputs = forward_inputs

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        self.forward_inputs[self] = x
        return self.c2(self.c1(x))

@contextmanager
def _dynamo_dist_per_rank_init(rank, world_size, init_pg=True, fake_pg=False):
    # To avoid multiple inheritance from _dynamo.test_case.TestCase and MultiProcessTestCase,
    # Just manually implement the most important part of the dynamo behavior to reset/clear.
    if not fake_pg:
        torch.cuda.set_device(rank)
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '6789'
    if init_pg:
        if fake_pg:
            store = torch.testing._internal.distributed.fake_pg.FakeStore()
            c10d.init_process_group(
                backend="fake",
                world_size=world_size,
                rank=rank,
                store=store,
            )
        else:
            c10d.init_process_group("nccl", rank=rank, world_size=world_size)
    torch._dynamo.reset()
    torch._dynamo.utils.counters.clear()
    try:
        yield
    finally:
        torch._dynamo.reset()
        torch._dynamo.utils.counters.clear()
        if init_pg:
            c10d.destroy_process_group()


class DynamoDistributedSingleProcTestCase(torch._dynamo.test_case.TestCase):
    """
    Test harness for single-process dynamo distributed tests,
    initializes dist process group.

    Prefer this for simple tests, as it's easier to debug.
    """

    @classmethod
    def setUpClass(cls):
        super().setUpClass()
        # _exit_stack is set up in TestCase
        cls._exit_stack.enter_context(
            patch.dict(
                os.environ,
                {
                    "MASTER_ADDR": "localhost",
                    "MASTER_PORT": "12355",
                },
            )
        )
        cls.rank = 0
        cls.device = f"cuda:{cls.rank}"
        cls.device_ids = None if "cuda" in cls.device else [cls.rank]
        c10d.init_process_group("nccl", rank=cls.rank, world_size=1)

    @classmethod
    def tearDownClass(cls):
        c10d.destroy_process_group()
        super().tearDownClass()


class DynamoDistributedMultiProcTestCase(MultiProcessTestCase):
    """
    Use this for tests that actually run on multiple GPUs.

    Decorate tests with @skip_if_lt_x_gpu(ngpu)

    Note: MultiProcTestCase spawns processes per test and is slow.
    Prefer MultiThreadedTestCase for most tests. Perhaps use this one
    sparingly for integration tests.
    """
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
    def world_size(self) -> int:
        return torch.cuda.device_count()

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe, **kwargs) -> None:
        trace_log.addHandler(logging.NullHandler())

        # The rest is copypasta from MultiProcessTestCase._run
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe)


class MultiProcContinousTest(TestCase):
    # Class variables:
    # number of test processes
    world_size: int = 2
    # rank of the current process
    rank: int = -1  # unset state
    # Rendezvous file
    rdvz_file: Optional[str] = None

    @classmethod
    @abc.abstractmethod
    def backend_str(cls) -> str:
        """
        ProcessGroup backend str.
        To be customized by sub test classes, e.g. "nccl".
        Here we raise error.
        """
        raise NotImplementedError("Please implement backend_str in your test class")

    @classmethod
    def opts(cls, high_priority_stream=False):
        """
        ProcessGroup init options.
        To be customized by sub test classes, e.g. ProcessGroupNCCLOpTest
        Here we return None.
        """
        return None

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Set up the process group.
        """
        super().setUpClass()
        if not 0 <= cls.rank < cls.world_size:
            raise RuntimeError(
                "Rank must be set and in the range of 0 to world_size. "
                f"World size: {cls.world_size} Rank: {cls.rank}"
            )
        if cls.rdvz_file:
            store = c10d.FileStore(cls.rdvz_file, cls.world_size)
        else:
            # torchrun takes care of rendezvous
            store = None
        opts = cls.opts()
        backend = cls.backend_str()
        print(f"Testing {backend=}")
        # create nccl processgroup with opts
        c10d.init_process_group(
            backend=backend,
            world_size=cls.world_size,
            rank=cls.rank,
            store=store,
            pg_options=opts,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()
        print(f"Rank {cls.rank} setup complete")

    @classmethod
    def tearDownClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, after all tests finish.
        Tear down the process group.
        """
        c10d.destroy_process_group()
        super().tearDownClass()
        # Clear up the rendezvous file
        if cls.rdvz_file:
            try:
                os.remove(cls.rdvz_file)
            except OSError:
                pass
        print(f"Rank {cls.rank} teardown complete")

    @classmethod
    def run_rank(
        cls,
        rank: int,
        world_size: int,
        rdvz_file: Optional[str] = None,
    ):
        """
        This is an entry point for each rank to run the tests in `MultiProcContinousTest`.
        In this entry point, we set the class variables for the test class.
        Then we run all tests.

        Note:
        - This helper only works for a subclass of `MultiProcContinousTest`.

        Example:
        - See `test_c10d_ops_nccl.py`.
        """
        # set class variables for the test class
        cls.rank = rank
        cls.world_size = world_size
        cls.rdvz_file = rdvz_file
        # Launch tests via `common_utils` infra
        run_tests()
