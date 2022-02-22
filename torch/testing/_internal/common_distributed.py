import faulthandler
import logging
import multiprocessing
import os
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
from functools import partial, reduce
from functools import wraps
from io import StringIO
from typing import NamedTuple, Optional, Union

import torch
import torch.cuda.nccl
import torch.distributed as c10d
from torch.testing._internal.common_utils import (
    TestCase,
    TEST_WITH_ROCM,
    TEST_WITH_TSAN,
    FILE_SCHEMA,
    find_free_port,
    retry_on_connect_failures,
    IS_SANDCASTLE,
    sandcastle_skip_if,
    sandcastle_skip,
)

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
}

@dataclass
class DistTestCases:
    # Backends that do not support a specific collective
    skip_collective = {}
    skip_collective["allgather_coalesced"] = {"nccl", "mpi"}
    skip_collective["reduce"] = set()
    skip_collective["sendrecv anysource"] = {"nccl"}
    skip_collective["cpu barrier"] = {"nccl"}

    # Sets showing that something is implemented
    backend_feature = {}
    backend_feature["gpu"] = {"nccl", "gloo"}
    backend_feature["cuda"] = {"nccl", "gloo"}
    backend_feature["ddp"] = {"nccl", "gloo"}
    backend_feature["subgroup"] = {"nccl", "gloo"}
    backend_feature["plugin"] = set()


def skip_if_no_gpu(func):
    """Nccl multigpu tests require at least 2 GPUS. Skip if this is not met"""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not torch.cuda.is_available():
            sys.exit(TEST_SKIPS["no_cuda"].exit_code)
        world_size = int(os.environ["WORLD_SIZE"])
        if torch.cuda.device_count() < world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{world_size}"].exit_code)

        return func(*args, **kwargs)

    return wrapper


def skip_if_small_worldsize(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        if (os.environ["BACKEND"] != "mpi") and int(os.environ["WORLD_SIZE"]) <= 2:
            sys.exit(TEST_SKIPS["small_worldsize"].exit_code)

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


def skip_if_lt_x_gpu(x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
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
        err_substr if err_substr.find("\nException raised from ") == -1
        else err_substr.split("\nException raised from ")[0]
    )
    assert actual in logging_err, f"Did not find expected {actual} in ddp logging data error: {logging_err}"


def with_nccl_blocking_wait(func):
    """
    Convenience decorator to set/unset NCCL_BLOCKING_WAIT flag. Note that use of
    this decorator will override the setting of NCCL_ASYNC_ERROR_HANDLING for
    the particular test. After the test, both NCCL_BLOCKING_WAIT and
    NCCL_ASYNC_ERROR_HANDLING will be restored to their original values.
    """

    @wraps(func)
    def wrapper(*args, **kwargs):
        # Save and unset NCCL_ASYNC_ERROR_HANDLING
        try:
            cached_nccl_async_error_handling: Union[str, None] = os.environ[
                "NCCL_ASYNC_ERROR_HANDLING"
            ]
            del os.environ["NCCL_ASYNC_ERROR_HANDLING"]
        except KeyError:
            # NCCL_ASYNC_ERROR_HANDLING was unset
            cached_nccl_async_error_handling = None

        # Save val of NCCL_BLOCKING_WAIT and set it.
        try:
            cached_nccl_blocking_wait: Union[str, None] = os.environ[
                "NCCL_BLOCKING_WAIT"
            ]
        except KeyError:
            cached_nccl_blocking_wait = None
        finally:
            os.environ["NCCL_BLOCKING_WAIT"] = "1"

        try:
            ret = func(*args, **kwargs)
            return ret
        finally:
            # restore old values.
            if cached_nccl_async_error_handling is not None:
                os.environ[
                    "NCCL_ASYNC_ERROR_HANDLING"
                ] = cached_nccl_async_error_handling

            if cached_nccl_blocking_wait is not None:
                os.environ["NCCL_BLOCKING_WAIT"] = cached_nccl_blocking_wait

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
    return sandcastle_skip_if(
        not c10d.is_gloo_available(),
        "c10d was not compiled with the Gloo backend",
    )


def requires_nccl_version(version, msg):
    if not c10d.is_nccl_available():
        return sandcastle_skip(
            "c10d was not compiled with the NCCL backend",
        )
    else:
        return sandcastle_skip_if(
            torch.cuda.nccl.version() < version,
            "Requires NCCL version greater than or equal to: {}, found: {}, reason: {}".format(
                version, torch.cuda.nccl.version(), msg
            ),
        )


def requires_nccl():
    return sandcastle_skip_if(
        not c10d.is_nccl_available(),
        "c10d was not compiled with the NCCL backend",
    )


def requires_mpi():
    return sandcastle_skip_if(
        not c10d.is_mpi_available(),
        "c10d was not compiled with the MPI backend",
    )


def skip_if_rocm(func):
    """Skips a test for ROCm"""
    func.skip_if_rocm = True

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_ROCM:
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS["skipIfRocm"].exit_code)

    return wrapper


def skip_if_win32():
    return sandcastle_skip_if(
        sys.platform == 'win32',
        "This unit test case is not supportted on Windows platform",
    )


@retry_on_connect_failures
def create_tcp_store(
    addr="localhost",
    world_size=1,
    is_master=True,
    timeout=timedelta(minutes=5),
    wait_for_workers=True,
    jit_class=False,
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
            addr, port, world_size, is_master, wait_for_workers=wait_for_workers
        )


if TEST_WITH_TSAN:
    # TSAN runs much slower.
    TIMEOUT_DEFAULT = 500
else:
    TIMEOUT_DEFAULT = 100
TIMEOUT_OVERRIDE = {"test_ddp_uneven_inputs": 400}


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
            lambda a, b: a + b, [fn(rank, world_size) for rank in range(world_size)]
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
    visible_devices = range(nGPUs)

    if backend == "nccl":
        # This is a hack for a known NCCL issue using multiprocess
        # in conjunction with multiple threads to manage different GPUs which
        # may cause ncclCommInitRank to fail.
        # http://docs.nvidia.com/deeplearning/sdk/nccl-release-notes/rel_2.1.4.html#rel_2.1.4
        # It slows down the performance of collective operations.
        # Without this setting NCCL might throw unhandled error.
        os.environ["NCCL_MAX_NRINGS"] = "1"

    # If rank is less than or equal to number of available GPU's
    # then each rank can be mapped to corresponding GPU.
    nGPUs_per_process = 1
    if world_size > nGPUs:
        nGPUs_per_process = nGPUs // world_size
    rank_to_GPU = {
        i: list(
            visible_devices[i * nGPUs_per_process : (i + 1) * nGPUs_per_process]
        )
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

    @property
    def world_size(self) -> int:
        return 4

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
    def __init__(self, method_name: str = "runTest") -> None:
        super().__init__(method_name)
        fn = getattr(self, method_name)
        setattr(self, method_name, self.join_or_run(fn))

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
            )
            process.start()
            logger.info(f"Started process {rank} with pid {process.pid}")
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        proc = torch.multiprocessing.get_context("spawn").Process
        self._start_processes(proc)

    class Event(Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        logger.info(f"Starting event listener thread for rank {rank}")
        while True:
            ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])

            if parent_pipe in ready_pipes:

                if parent_pipe.closed:
                    logger.info(
                        f"Pipe closed for process {rank}, stopping event listener thread"
                    )
                    return

                event = parent_pipe.recv()
                logger.info(f"Received event {event} on process {rank}")

                if event == MultiProcessTestCase.Event.GET_TRACEBACK:
                    # Return traceback to the parent process.
                    with tempfile.NamedTemporaryFile(mode="r+") as tmp_file:
                        faulthandler.dump_traceback(tmp_file)
                        # Flush buffers and seek to read from the beginning
                        tmp_file.flush()
                        tmp_file.seek(0)
                        parent_pipe.send(tmp_file.read())

                        logger.info(f"Process {rank} sent traceback")

            if signal_pipe in ready_pipes:
                return

    @classmethod
    def _run(cls, rank: int, test_name: str, file_name: str, parent_pipe) -> None:
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
                f"Process {self.rank} skipping test {test_name} for following reason: {str(se)}"
            )
            sys.exit(TEST_SKIPS["generic"].exit_code)
        except Exception as e:
            logger.error(
                f"Caught exception: \n{traceback.format_exc()} exiting "
                f"process {self.rank} with exit code: {MultiProcessTestCase.TEST_ERROR_EXIT_CODE}"
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
                        f"Encountered error while trying to get traceback for process {i}: {e}"
                    )

        # Wait for results.
        for rank, pipe in pipes:
            try:
                # Wait for traceback
                if pipe.poll(5):
                    if pipe.closed:
                        logger.info(
                            f"Pipe closed for process {rank}, cannot retrieve traceback"
                        )
                        continue

                    traceback = pipe.recv()
                    logger.error(
                        f"Process {rank} timed out with traceback: \n\n{traceback}"
                    )
                else:
                    logger.error(
                        f"Could not retrieve traceback for timed out process: {rank}"
                    )
            except ConnectionError as e:
                logger.error(
                    f"Encountered error while trying to get traceback for process {rank}: {e}"
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
                if all([p.exitcode is not None for p in self.processes]):
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
            for pid, pipe in self.pid_to_pipe.items():
                pipe.close()

    def _check_no_test_errors(self, elapsed_time) -> None:
        """
        Checks that we didn't have any errors thrown in the child processes.
        """
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    "Process {} timed out after {} seconds".format(i, elapsed_time)
                )
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time) -> None:
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
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
                    "Process {} exited with error code {} and exception:\n{}\n".format(
                        i, MultiProcessTestCase.TEST_ERROR_EXIT_CODE, error_message
                    )
                )

            raise RuntimeError(error)
        # If no process exited uncleanly, we check for timeouts, and then ensure
        # each process exited cleanly.
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError(
                    "Process {} terminated or timed out after {} seconds".format(
                        i, elapsed_time
                    )
                )
            self.assertEqual(
                p.exitcode,
                first_process.exitcode,
                msg="Expect process {} exit code to match Process 0 exit code of {}, but got {}".format(
                    i, first_process.exitcode, p.exitcode
                ),
            )
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                if IS_SANDCASTLE:
                    # Don't use unittest.skip to skip the test on sandcastle
                    # since it creates tasks for skipped tests assuming there
                    # is some follow-up needed. Instead just "pass" the test
                    # with an appropriate message.
                    logger.info(
                        f"Skipping {self.id()} on sandcastle for the following reason: {skip.message}"
                    )
                    return
                else:
                    raise unittest.SkipTest(skip.message)
        self.assertEqual(
            first_process.exitcode,
            0,
            msg="Expected zero exit code but got {} for pid: {}".format(first_process.exitcode, first_process.pid)
        )

    @property
    def is_master(self) -> bool:
        return self.rank == 0
