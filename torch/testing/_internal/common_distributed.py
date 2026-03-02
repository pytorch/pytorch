# mypy: ignore-errors

import faulthandler
import functools
import inspect
import itertools
import logging
import multiprocessing
import operator
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
from collections.abc import Callable
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from enum import Enum
from functools import partial, reduce, wraps
from io import StringIO
from typing import Any, NamedTuple
from unittest.mock import patch

import torch
import torch._dynamo.test_case
import torch.cuda.nccl
import torch.distributed as c10d
import torch.nn as nn
from torch._C._autograd import DeviceType
from torch._C._distributed_c10d import _SymmetricMemory
from torch._logging._internal import trace_log
from torch.testing._internal import common_utils
from torch.testing._internal.common_utils import (
    FILE_SCHEMA,
    find_free_port,
    IS_SANDCASTLE,
    LazyVal,
    retry_on_connect_failures,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_CUDA,
    TEST_HPU,
    TEST_WITH_ROCM,
    TEST_WITH_TSAN,
    TEST_XPU,
    TestCase,
)
from torch.testing._internal.distributed.multi_threaded_pg import (
    _install_threaded_pg,
    _uninstall_threaded_pg,
    ProcessLocalGroup,
)


logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

ACCELERATOR_DIST_BACKENDS = ["nccl", "xccl", "hccl"]
DDP_RANK_DEVICES = ["cuda", "xpu"]
HAS_ACCELERATOR = TEST_CUDA or TEST_HPU or TEST_XPU


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
    skip_collective["allgather_coalesced"] = {"nccl", "mpi", "ucc", "xccl"}
    skip_collective["reduce"] = set()
    skip_collective["sendrecv anysource"] = {"nccl", "ucc", "xccl"}
    skip_collective["cpu barrier"] = {"nccl", "ucc", "xccl"}

    # Sets showing that something is implemented
    backend_feature = {}
    backend_feature["gpu"] = {"nccl", "gloo", "ucc"}
    backend_feature["cuda"] = {"nccl", "gloo", "ucc"}
    backend_feature["ddp"] = {"nccl", "gloo", "ucc"}
    backend_feature["subgroup"] = {"nccl", "gloo", "ucc"}
    backend_feature["plugin"] = set()
    if TEST_HPU:
        backend_feature["hpu"] = {"hccl"}
    if TEST_XPU:
        backend_feature["xpu"] = {"xccl"}


def requires_ddp_rank(device):
    return device in DDP_RANK_DEVICES


def skip_if_no_gpu(func):
    """Skips if the world size exceeds the number of GPUs, ensuring that if the
    test is run, each rank has its own GPU via ``torch.cuda.device(rank)``."""

    @wraps(func)
    def wrapper(*args, **kwargs):
        if not (TEST_CUDA or TEST_HPU or TEST_XPU):
            sys.exit(TEST_SKIPS["no_cuda"].exit_code)
        world_size = int(os.environ["WORLD_SIZE"])
        if TEST_CUDA and torch.cuda.device_count() < world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{world_size}"].exit_code)
        if TEST_HPU and torch.hpu.device_count() < world_size:
            sys.exit(TEST_SKIPS[f"multi-gpu-{world_size}"].exit_code)
        if TEST_XPU and torch.xpu.device_count() < world_size:
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
                from transformers import AutoModelForMaskedLM, BertConfig  # noqa: F401

                return func(*args, **kwargs)
            except ImportError:
                sys.exit(TEST_SKIPS["importerror"].exit_code)

        return wrapper

    return decorator


def at_least_x_gpu(x):
    if TEST_CUDA and torch.cuda.device_count() >= x:
        return True
    if TEST_HPU and torch.hpu.device_count() >= x:
        return True
    if TEST_XPU and torch.xpu.device_count() >= x:
        return True
    return False


def _maybe_handle_skip_if_lt_x_gpu(args, msg) -> bool:
    _handle_test_skip = getattr(args[0], "_handle_test_skip", None)
    if len(args) == 0 or _handle_test_skip is None:
        return False
    _handle_test_skip(msg)
    return True


def skip_if_lt_x_gpu(x, *, allow_cpu=False):
    """Skip if fewer than x accelerators available.

    Args:
        x: Minimum number of accelerators required.
        allow_cpu: If True, run the test on CPU-only machines (no accelerators).
    """

    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            if TEST_HPU and torch.hpu.device_count() >= x:
                return func(*args, **kwargs)
            if TEST_XPU and torch.xpu.device_count() >= x:
                return func(*args, **kwargs)
            if allow_cpu and not (torch.cuda.is_available() or TEST_HPU or TEST_XPU):
                return func(*args, **kwargs)
            test_skip = TEST_SKIPS[f"multi-gpu-{x}"]
            if not _maybe_handle_skip_if_lt_x_gpu(args, test_skip.message):
                sys.exit(test_skip.exit_code)

        return wrapper

    return decorator


def requires_world_size(n: int):
    """
    Decorator to request a specific world size for a test. The test harness can
    read this attribute to set the number of ranks to spawn. If there are fewer
    than `n` CUDA devices available, the test should be skipped by the harness.

    Usage:
        @require_world_size(3)
        def test_something(self):
            ...
    """

    def decorator(func):
        func._required_world_size = n
        available = torch.cuda.device_count()
        return unittest.skipUnless(
            available >= n, f"requires {n} GPUs, found {available}"
        )(func)

    return decorator


def get_required_world_size(obj: Any, default: int) -> int:
    """
    Returns the requested world size for the currently running unittest method on `obj`
    if annotated via `@require_world_size(n)`, else returns `default`.
    """
    try:
        # Try MultiProcessTestCase helper first, then unittest fallback
        test_name = (
            obj._current_test_name()  # type: ignore[attr-defined]
            if hasattr(obj, "_current_test_name") and callable(obj._current_test_name)
            else obj._testMethodName
        )
        fn = getattr(obj, test_name)
        value = fn._required_world_size
        return int(value)
    except Exception:
        return default


# This decorator helps avoiding initializing cuda while testing other backends
def nccl_skip_if_lt_x_gpu(backend, x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if backend != "nccl":
                return func(*args, **kwargs)
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            test_skip = TEST_SKIPS[f"multi-gpu-{x}"]
            if not _maybe_handle_skip_if_lt_x_gpu(args, test_skip.message):
                sys.exit(test_skip.exit_code)

        return wrapper

    return decorator


def verify_ddp_error_logged(model_DDP, err_substr):
    # Verify error was logged in ddp_logging_data.
    ddp_logging_data = model_DDP._get_ddp_logging_data()
    if "iteration" not in ddp_logging_data:
        raise AssertionError("Expected 'iteration' in ddp_logging_data")
    if "has_error" not in ddp_logging_data:
        raise AssertionError("Expected 'has_error' in ddp_logging_data")
    if "error" not in ddp_logging_data:
        raise AssertionError("Expected 'error' in ddp_logging_data")
    logging_err = ddp_logging_data["error"]
    # Remove C++ stacktrace if needed.
    actual = (
        err_substr
        if err_substr.find("\nException raised from ") == -1
        else err_substr.split("\nException raised from ")[0]
    )
    if actual not in logging_err:
        raise AssertionError(
            f"Did not find expected {actual} in ddp logging data error: {logging_err}"
        )


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
            cached_nccl_async_error_handling: str | None = os.environ[
                "TORCH_NCCL_ASYNC_ERROR_HANDLING"
            ]
            del os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"]
        except KeyError:
            # TORCH_NCCL_ASYNC_ERROR_HANDLING was unset
            cached_nccl_async_error_handling = None

        # Save val of TORCH_NCCL_BLOCKING_WAIT and set it.
        try:
            cached_nccl_blocking_wait: str | None = os.environ[
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
                os.environ["TORCH_NCCL_ASYNC_ERROR_HANDLING"] = (
                    cached_nccl_async_error_handling
                )

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
    if not TEST_CUDA:
        return lambda f: f
    if not c10d.is_nccl_available():
        return skip_but_pass_in_sandcastle(
            "c10d was not compiled with the NCCL backend",
        )
    else:
        return skip_but_pass_in_sandcastle_if(
            torch.cuda.nccl.version() < version,
            f"Requires NCCL version greater than or equal to: {version}, found: {torch.cuda.nccl.version()}, reason: {msg}",
        )


def requires_nccl_shrink():
    """
    Require NCCL shrink support (NCCL available and version >= 2.27).
    """
    return requires_nccl_version((2, 27), "Need NCCL 2.27+ for shrink_group")


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


def requires_accelerator_dist_backend(backends=None):
    """
    Decorator to skip tests if no accelerator communication backend (NCCL, XCCL, HCCL) is available.

    Args:
        backends (Optional[List[str]]): Specific accelerator backends to check (e.g., ["nccl", "xccl", "hccl"]).
                                       If None, checks all supported accelerator backends (NCCL, XCCL, HCCL).

    Returns:
        callable: A decorator that skips the test if no specified accelerator backend is available.
    """
    if backends is None:
        backends = ACCELERATOR_DIST_BACKENDS

    backend_available = any(
        {
            "nccl": c10d.is_nccl_available,
            "xccl": c10d.is_xccl_available,
            "hccl": lambda: TEST_HPU,
        }.get(backend, lambda: False)()
        for backend in backends
    )

    return skip_but_pass_in_sandcastle_if(
        not backend_available,
        f"No accelerator communication backend available among {backends}",
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


def evaluate_platform_supports_symm_mem():
    if TEST_CUDA:
        if TEST_WITH_ROCM:
            arch_list = ["gfx942", "gfx950"]
            for arch in arch_list:
                if arch in torch.cuda.get_device_properties(0).gcnArchName:
                    return True
            return False
        else:
            return True
    else:
        return False


PLATFORM_SUPPORTS_SYMM_MEM: bool = LazyVal(
    lambda: evaluate_platform_supports_symm_mem()
)


def skip_if_rocm_multiprocess(func):
    """Skips a test for ROCm multiprocess UTs"""
    return unittest.skipIf(TEST_WITH_ROCM, TEST_SKIPS["skipIfRocm"].message)(func)


def skip_if_rocm_arch_multiprocess(arch: tuple[str, ...]):
    """Skips a test for given ROCm archs - multiprocess UTs"""

    def decorator(func):
        reason = None
        if TEST_WITH_ROCM:
            prop = torch.cuda.get_device_properties(0).gcnArchName.split(":")[0]
            if prop in arch:
                reason = f"skip_if_rocm_arch_multiprocess: test skipped on {arch}"

        return unittest.skipIf(reason is not None, reason)(func)

    return decorator


def skip_if_rocm_ver_lessthan_multiprocess(version=None):
    """Skips a test for ROCm based on ROCm ver - multiprocess UTs"""

    def decorator(func):
        reason = None
        if TEST_WITH_ROCM:
            rocm_version = str(torch.version.hip)
            rocm_version = rocm_version.split("-", maxsplit=1)[0]  # ignore git sha
            rocm_version_tuple = tuple(int(x) for x in rocm_version.split("."))
            if (
                rocm_version_tuple is None
                or version is None
                or rocm_version_tuple < tuple(version)
            ):
                reason = f"skip_if_rocm_ver_lessthan_multiprocess: ROCm {rocm_version_tuple} is available but {version} required"

        return unittest.skipIf(reason is not None, reason)(func)

    return decorator


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
    Returns True if device is a non-CUDA device.
    """
    if device.type != "cuda":
        return True

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
            addr,
            port,
            world_size,
            is_master,
            wait_for_workers=wait_for_workers,
            use_libuv=use_libuv,
        )


if TEST_WITH_TSAN:
    # TSAN runs much slower.
    TIMEOUT_DEFAULT = 500
else:
    TIMEOUT_DEFAULT = int(os.getenv("DISTRIBUTED_TESTS_DEFAULT_TIMEOUT", "300"))
TIMEOUT_OVERRIDE = {"test_ddp_uneven_inputs": 400}


# https://github.com/pytorch/pytorch/issues/75665
if TEST_WITH_ROCM:
    TIMEOUT_OVERRIDE["test_join_kwargs"] = 200


def create_device(interface=None, lazy_init: bool = False):
    if sys.platform == "win32" or interface is None:
        return c10d.ProcessGroupGloo.create_device(
            hostname="127.0.0.1", lazy_init=lazy_init
        )
    else:
        return c10d.ProcessGroupGloo.create_device(
            interface=interface, lazy_init=lazy_init
        )


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
    if TEST_XPU:
        nGPUs = torch.xpu.device_count()
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


tmp_dir: tempfile.TemporaryDirectory | None = None


def initialize_temp_directories(init_method: str | None = None) -> None:
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
if TEST_WITH_ROCM:
    DEFAULT_WORLD_SIZE = min(4, max(2, torch.cuda.device_count()))
else:
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
    def __init__(
        self, method_name: str = "runTest", methodName: str = "runTest"
    ) -> None:
        # methodName is the correct naming in unittest and testslide uses keyword arguments.
        # So we need to use both to 1) not break BC and, 2) support testslide.
        if methodName != "runTest":
            method_name = methodName
        super().__init__(method_name)
        try:
            fn = getattr(self, method_name)
            setattr(self, method_name, self.join_or_run(fn))
        except AttributeError as e:
            if methodName != "runTest":
                # we allow instantiation with no explicit method name
                # but not an *incorrect* or missing method name
                raise ValueError(
                    f"no such test method in {self.__class__}: {methodName}"
                ) from e

    def setUp(self) -> None:
        super().setUp()

        # Used for tests that are expected to return a non-0 exit code, such as
        # SIGABRT thrown by watchdog.
        self.special_return_code_checks: dict = {}

        # Used for tests that may return any exit code, which makes it hard to
        # check. This is rare, use with caution.
        self.skip_return_code_checks: list = []

        self.processes = []  # type: ignore[var-annotated]
        self.rank = self.MAIN_PROCESS_RANK
        with tempfile.NamedTemporaryFile(delete=False) as f:
            self.file_name = f.name
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
                args=(
                    rank,
                    self._current_test_name(),
                    self.file_name,
                    child_conn,
                ),
                kwargs={
                    "fake_pg": getattr(self, "fake_pg", False),
                },
            )
            process.start()
            logger.info("Started process %s with pid %s", rank, process.pid)
            self.pid_to_pipe[process.pid] = parent_conn
            self.processes.append(process)

    def _spawn_processes(self) -> None:
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            pass

        proc = torch.multiprocessing.get_context("spawn").Process
        self._start_processes(proc)

    class Event(Enum):
        GET_TRACEBACK = 1

    @staticmethod
    def _event_listener(parent_pipe, signal_pipe, rank: int):
        logger.debug("Starting event listener thread for rank %s", rank)
        while True:
            ready_pipes = multiprocessing.connection.wait([parent_pipe, signal_pipe])

            if parent_pipe in ready_pipes:
                if parent_pipe.closed:
                    logger.debug(
                        "Pipe closed for process %s, stopping event listener thread",
                        rank,
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
    def _run(
        cls, rank: int, test_name: str, file_name: str, parent_pipe, **kwargs
    ) -> None:
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
        common_utils.set_rng_seed()

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retrieving a corresponding test and executing it.
        try:
            getattr(self, test_name)()
        except unittest.SkipTest as se:
            logger.info(  # noqa: G200
                "Process %s skipping test %s for following reason: %s",
                self.rank,
                test_name,
                str(se),
            )
            sys.exit(TEST_SKIPS["generic"].exit_code)
        except Exception:
            logger.error(
                "Caught exception: \n%s exiting process %s with exit code: %s",
                traceback.format_exc(),
                self.rank,
                MultiProcessTestCase.TEST_ERROR_EXIT_CODE,
            )
            # Send error to parent process.
            parent_pipe.send(traceback.format_exc())
            sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        finally:
            if signal_send_pipe is not None:
                signal_send_pipe.send(None)

            if event_listener_thread is None:
                raise AssertionError("Expected event_listener_thread to not be None")
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
                except ConnectionError:
                    logger.exception(
                        "Encountered error while trying to get traceback for process %s",
                        i,
                    )

        # Wait for results.
        for rank, pipe in pipes:
            try:
                # Wait for traceback
                if pipe.poll(5):
                    if pipe.closed:
                        logger.info(
                            "Pipe closed for process %s, cannot retrieve traceback",
                            rank,
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
            except ConnectionError:
                logger.exception(
                    "Encountered error while trying to get traceback for process %s",
                    rank,
                )

    def _join_processes(self, fn) -> None:
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False
        try:
            while True:
                # check to see if any subprocess exited with an error early.
                for i, p in enumerate(self.processes):
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
            self._check_return_codes(fn, elapsed_time)
        finally:
            # Close all pipes
            for pipe in self.pid_to_pipe.values():
                pipe.close()

    def _check_return_codes(self, fn, elapsed_time) -> None:
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        # If no processes are spawned, there is nothing to check.
        if not self.processes:
            logger.warning(
                "Note: no subprocesses were spawned, test was likely skipped."
            )
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

        # Skip the test return code check
        if fn in self.skip_return_code_checks:
            return

        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                if IS_SANDCASTLE:
                    # Don't use unittest.skip to skip the test on sandcastle
                    # since it creates tasks for skipped tests assuming there
                    # is some follow-up needed. Instead just "pass" the test
                    # with an appropriate message.
                    logger.info(
                        "Skipping %s on sandcastle for the following reason: %s",
                        self.id(),
                        skip.message,
                    )
                    return
                else:
                    raise unittest.SkipTest(skip.message)

        # In most cases, we expect test to return exit code 0, standing for success.
        expected_return_code = 0
        # In some negative tests, we expect test to return non-zero exit code,
        # such as watchdog throwing SIGABRT.
        if fn in self.special_return_code_checks:
            expected_return_code = self.special_return_code_checks[fn]

        self.assertEqual(
            first_process.exitcode,
            expected_return_code,
            msg=f"Expected exit code {expected_return_code} but got {first_process.exitcode} for pid: {first_process.pid}",
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
        os.environ["WORLD_SIZE"] = str(self.world_size)
        self._spawn_processes()

    def tearDown(self):
        try:
            torch.distributed.destroy_process_group()
        except AssertionError:
            pass
        try:
            os.remove(self.file_name)
        except OSError:
            pass

    def backend(self, device) -> str:
        if "cuda" in device:
            return "nccl"
        elif "hpu" in device:  # intel gaudi
            return "hccl"
        elif "xpu" in device:
            return "xccl"
        else:
            return "gloo"

    def create_pg(self, device, world_size=None):
        if world_size is None:
            world_size = self.world_size
        num_visible_devices = torch.get_device_module(device).device_count()
        store = torch.distributed.FileStore(self.file_name, num_visible_devices)
        torch.distributed.init_process_group(
            backend=self.backend(device),
            world_size=world_size,
            rank=self.rank,
            store=store,
        )
        if "nccl" in self.backend(device) or "xccl" in self.backend(device):
            torch.accelerator.set_device_index(self.rank)
        return torch.distributed.distributed_c10d._get_default_group()

    def rank_to_device(self, device):
        num_visible_devices = torch.get_device_module(device).device_count()
        return {i: [i % num_visible_devices] for i in range(self.world_size)}


def run_subtests(
    cls_inst,
    subtest_config: dict[str, list[Any]],
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
    subtest_config_items: list[tuple[str, list[Any]]] = list(subtest_config.items())
    subtest_config_keys: list[str] = [item[0] for item in subtest_config_items]
    subtest_config_values: list[list[Any]] = [item[1] for item in subtest_config_items]
    for values in itertools.product(*subtest_config_values):
        # Map keyword to chosen value
        subtest_kwargs = dict(zip(subtest_config_keys, values, strict=True))
        with cls_inst.subTest(**subtest_kwargs):
            torch._dynamo.reset()
            test_fn(*test_args, **test_kwargs, **subtest_kwargs)
            torch._dynamo.reset()
        c10d.barrier()


@functools.cache
def has_efa() -> bool:
    """
    If shell command `fi_info -p efa -t FI_EP_RDM` returns exit code 0 then we assume that the machine has
    Libfabric EFA interfaces and EFA software components installed,
    see https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/efa-start.html.
    """

    try:
        return (
            subprocess.run(
                ["fi_info", "-p", "efa", "-t", "FI_EP_RDM"], check=False
            ).returncode
            == 0
        )
    except FileNotFoundError:
        pass
    return False


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
            except BaseException as ex:  # noqa: B036
                # Exceptions are handled in MultiThreadedTestCase
                MultiThreadedTestCase.exception_queue.put((rank, sys.exc_info()))
                ProcessLocalGroup.exception_handle(
                    ex
                )  # trigger _terminate event and awaken worker threads
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
            threads = _run_test_method_with_multi_threads(
                world_size, lambda: func(self, *args, **kwargs)
            )
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

    def __init__(
        self, method_name: str = "runTest", methodName: str = "runTest"
    ) -> None:
        # methodName is the correct naming in unittest and testslide uses keyword arguments.
        # So we need to use both to 1) not break BC and, 2) support testslide.
        if methodName != "runTest":
            method_name = methodName
        super().__init__(method_name)
        try:
            fn = getattr(self, method_name)
            setattr(self, method_name, self.join_or_run(fn))
        except AttributeError as e:
            if methodName != "runTest":
                # we allow instantiation with no explicit method name
                # but not an *incorrect* or missing method name
                raise ValueError(
                    f"no such test method in {self.__class__}: {methodName}"
                ) from e

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
            t = threading.Thread(
                target=self.__class__._run, args=(test_name, rank, self.world_size)
            )
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
            backend="threaded",
            rank=rank,
            world_size=world_size,
            store=self.__class__.global_store,
        )
        self.perThreadSetUp()

        try:
            getattr(self, test_name)()
        except BaseException as ex:  # noqa: B036
            self.exception_queue.put((rank, sys.exc_info()))
            ProcessLocalGroup.exception_handle(
                ex
            )  # trigger _terminate event and awaken worker threads
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
                    "Thread %s skipping test %s for following reason: %s",
                    rank,
                    fn,
                    str(exc),
                )
                if skip_code < 0:
                    skip_code = TEST_SKIPS["generic"].exit_code
            elif isinstance(exc, TimeoutError):
                msg = f"Thread {rank} terminated or timed out after {timeout} seconds\n"
                logger.error(msg)
                raise RuntimeError(msg)
            elif isinstance(exc, Exception):
                msg = "".join(traceback.format_exception(*exc_info))
                logger.error("Caught exception: \n%s exiting thread %s", msg, rank)
                error_msg += f"Thread {rank} exited with exception:\n{msg}\n"
            elif isinstance(exc, SystemExit):
                if type(exc.code) is int and skip_code < 0:
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
                            "Skipping %s on sandcastle for the following reason: %s",
                            fn,
                            skip.message,
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
        forward_inputs: dict[nn.Module, torch.Tensor],
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
        forward_inputs: dict[nn.Module, torch.Tensor],
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
def _dynamo_dist_per_rank_init(
    rank, world_size, backend=None, init_pg=True, fake_pg=False
):
    # To avoid multiple inheritance from _dynamo.test_case.TestCase and MultiProcessTestCase,
    # Just manually implement the most important part of the dynamo behavior to reset/clear.
    if not fake_pg:
        torch.accelerator.set_device_index(rank)

    device_type = (
        acc.type if (acc := torch.accelerator.current_accelerator()) else "cpu"
    )
    if backend is None:
        backend = c10d.get_default_backend_for_device(device_type)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "6789"
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
            c10d.init_process_group(backend=backend, rank=rank, world_size=world_size)
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
        device = torch.accelerator.current_accelerator().type
        cls.device = f"{device}:{cls.rank}"
        cls.device_ids = None if device in cls.device else [cls.rank]
        c10d.init_process_group(
            c10d.get_default_backend_for_device(device), rank=cls.rank, world_size=1
        )

    @classmethod
    def tearDownClass(cls):
        c10d.destroy_process_group()
        super().tearDownClass()


class DynamoDistributedMultiProcTestCase(DistributedTestBase):
    """
    Use this for tests that actually run on multiple GPUs.

    Decorate tests with @skip_if_lt_x_gpu(ngpu)

    Note: MultiProcTestCase spawns processes per test and is slow.
    Prefer MultiThreadedTestCase for most tests. Perhaps use this one
    sparingly for integration tests.
    """

    @property
    def world_size(self) -> int:
        return torch.accelerator.device_count()

    @classmethod
    def _run(
        cls, rank: int, test_name: str, file_name: str, parent_pipe, **kwargs
    ) -> None:
        trace_log.addHandler(logging.NullHandler())

        # The rest is copypasta from MultiProcessTestCase._run
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name
        self.run_test(test_name, parent_pipe)


class MultiProcContinuousTest(TestCase):
    # Class variables:
    MAIN_PROCESS_RANK = -1
    # number of test processes
    world_size: int = -2  # unset state
    # rank of the current process
    rank: int = -2  # unset state
    # Rendezvous file
    rdvz_file: str | None = None
    # timeout configured per class
    timeout: timedelta = timedelta(seconds=120)
    # Poison pill for rest of tests if one of them fails
    poison_pill: bool = False
    # Flag for lazy process spawning (to support instantiate_device_type_tests)
    _processes_spawned: bool = False

    @classmethod
    def backend_str(cls) -> str | None:
        """
        ProcessGroup backend str.
        To be customized by sub test classes, e.g. "nccl".
        Otherwise we return None -- lazily decided by tensor.
        """
        return None

    # Please override if you intend to test on specific device type
    @classmethod
    def device_type(cls) -> str:
        curr_device = torch.accelerator.current_accelerator()
        if curr_device is None:
            return "cpu"
        return curr_device.type

    @classmethod
    def opts(cls, high_priority_stream=False):
        """
        ProcessGroup init options.
        To be customized by sub test classes, e.g. ProcessGroupNCCLOpTest
        Here we return None.
        """
        return None

    @classmethod
    def _init_pg(cls, rank, world_size, rdvz_file):
        if rdvz_file is None:
            raise AssertionError("Expected rdvz_file to not be None")
        # rank should be local_rank for tests running on <= 8 gpus which is how all these tests are designed
        # and we expect LOCAL_RANK set by torchrun. Setting it lets init_device_mesh set the device without
        # issuing a warning
        os.environ["LOCAL_RANK"] = str(rank)
        store = c10d.FileStore(rdvz_file, world_size)
        # create nccl processgroup with opts
        c10d.init_process_group(
            backend=cls.backend_str(),
            world_size=world_size,
            rank=rank,
            store=store,
            pg_options=cls.opts(),
            timeout=cls.timeout,
        )
        cls.pg = c10d.distributed_c10d._get_default_group()

    @classmethod
    def _run_test_given_id(cls, test_id: str, **kwargs) -> None:
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        test_name = test_id.rsplit(".", maxsplit=1)[-1]
        # Get the test function from the test class
        self = cls(test_name)
        self.rank = cls.rank
        self.world_size = cls.world_size
        test_fn = getattr(self, test_name)

        # Ensure all the ranks use the same seed.
        common_utils.set_rng_seed()

        # Run the test function
        test_fn(**kwargs)

    @classmethod
    def _worker_loop(cls, rank, world_size, rdvz_file, task_queue, completion_queue):
        raised_exception = False
        # Sub tests are going to access these values, check first
        if not (0 <= rank < world_size):
            raise AssertionError(
                f"Expected 0 <= rank < world_size, got rank={rank}, world_size={world_size}"
            )
        # set class variables for the test class
        cls.rank = rank
        cls.world_size = world_size

        # Initialize the process group
        init_skip_reason = None
        try:
            cls._init_pg(rank, world_size, rdvz_file)
        except SystemExit as ex:
            exit_code = getattr(ex, "code", None)
            skip_entry = next(
                (v for v in TEST_SKIPS.values() if v.exit_code == exit_code),
                None,
            )
            if skip_entry:
                init_skip_reason = skip_entry.message
            else:
                raise

        # End of bootstrap
        logger.debug("Setup complete")

        # Loop forever, waiting for a test name to run
        while True:
            test_id = task_queue.get()
            logger.debug(f"Got test {test_id}")  # noqa: G004
            # None means exit
            if test_id is None:
                break

            # If init failed with a skip, respond with SkipTest for all tests
            if init_skip_reason is not None:
                completion_queue.put(unittest.SkipTest(init_skip_reason))
                continue

            # Run the test
            try:
                cls._run_test_given_id(test_id)
                completion_queue.put(test_id)
            except BaseException as ex:  # noqa: B036
                if isinstance(ex, SystemExit):
                    # Get exit code from the process
                    exit_code = getattr(ex, "code", None)

                    # Look up exit code in TEST_SKIPS to see if it is a valid skip
                    skip_entry = next(
                        (v for v in TEST_SKIPS.values() if v.exit_code == exit_code),
                        None,
                    )

                    # If we found an entry, we want to skip the test and the object back to the main process
                    if skip_entry:
                        completion_queue.put(unittest.SkipTest(skip_entry.message))
                        # Skip exception handling below, move to main thread for processing the skip
                        continue

                raised_exception = True
                # Send the exception and stack trace back to the dispatcher
                exc_info = sys.exc_info()
                tb_str = "".join(traceback.format_exception(*exc_info))
                # Create a new exception with the original exception and traceback
                enhanced_ex = RuntimeError(f"Exception in worker process:\n{tb_str}")
                enhanced_ex.__cause__ = ex
                completion_queue.put(enhanced_ex)

        # Termination
        logger.debug("Terminating ...")
        # Calling destroy_process_group when workers have exceptions
        # while others are doing collectives will cause a deadlock since
        # it waits for enqueued collectives to finish.
        # Only call this on a clean exit path
        if not raised_exception:
            c10d.destroy_process_group()

    @classmethod
    def _spawn_processes(cls, world_size) -> None:
        cls.processes = []
        cls.task_queues = []
        cls.completion_queues = []
        # Need a rendezvous file for `init_process_group` purpose.
        with tempfile.NamedTemporaryFile(delete=False) as f:
            cls.rdvz_file = f.name

        # CUDA multiprocessing requires spawn instead of fork, to make sure
        # child processes have their own memory space.
        try:
            torch.multiprocessing.set_start_method("spawn")
        except RuntimeError:
            # The start method has already been set
            pass

        for rank in range(int(world_size)):
            task_queue = torch.multiprocessing.Queue()
            completion_queue = torch.multiprocessing.Queue()
            process = torch.multiprocessing.Process(
                target=cls._worker_loop,
                name="process " + str(rank),
                daemon=True,  # so that child processes will exit if parent decides to terminate
                args=(rank, world_size, cls.rdvz_file, task_queue, completion_queue),
            )
            process.start()
            cls.processes.append(process)
            cls.task_queues.append(task_queue)
            cls.completion_queues.append(completion_queue)
            logger.debug("Started process %s with pid %s", rank, process.pid)  # noqa: UP031

    @classmethod
    def _get_world_size(cls, device_type: str) -> int:
        """
        Get world_size, handling both class variable and property definitions.
        Properties are instance-level and need special handling in class methods.
        """
        # Check if world_size is defined as a property (instance-level)
        world_size_attr = inspect.getattr_static(cls, "world_size", None)
        if isinstance(world_size_attr, property):
            # Create a temporary instance to evaluate the property
            # We use object.__new__ to avoid calling __init__ which may have side effects
            temp_instance = object.__new__(cls)
            world_size = world_size_attr.fget(temp_instance)
        else:
            world_size = cls.world_size

        # If world_size is not set (== -2), use device count
        if world_size == -2:
            world_size = torch.get_device_module(device_type).device_count()
            if world_size == 0:
                raise unittest.SkipTest(f"No {device_type} devices available")

        return world_size

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Note: Process spawning is deferred to setUp to support instantiate_device_type_tests,
        which calls setUpClass during class creation before any tests run.
        """
        super().setUpClass()

    @classmethod
    def _ensure_processes_spawned(cls):
        """
        Lazily spawn worker processes on first test run.
        This supports instantiate_device_type_tests which calls setUpClass during
        class creation (before any tests run), when spawning would be premature.
        """
        if cls._processes_spawned:
            return

        # Handle both method and string attribute for device_type
        # (instantiate_device_type_tests sets device_type as a string attribute,
        # making this compatible as a drop-in replacement for MultiProcessTestCase)
        device_type_attr = cls.device_type
        if callable(device_type_attr):
            device_type = device_type_attr()
        else:
            device_type = device_type_attr

        # Get world_size (handles both class variable and property)
        cls.world_size = cls._get_world_size(device_type)

        # Check if the specified backend is available before spawning processes
        backend = cls.backend_str() if callable(cls.backend_str) else cls.backend_str
        if backend is not None:
            backend_checks = {
                "nccl": c10d.is_nccl_available,
                "gloo": c10d.is_gloo_available,
                "mpi": c10d.is_mpi_available,
                "xccl": c10d.is_xccl_available,
            }
            check_fn = backend_checks.get(backend)
            if check_fn is not None and not check_fn():
                raise unittest.SkipTest(f"Backend '{backend}' is not available")

        logger.info(
            f"Testing class {cls.__name__} on {cls.world_size} {device_type}"  # noqa: G004
        )

        cls._spawn_processes(cls.world_size)
        cls._processes_spawned = True

    @classmethod
    def tearDownClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, after all tests finish.
        Tear down the process group if spawned.
        """
        # If processes were never spawned (e.g., all tests were skipped), nothing to tear down
        if not cls._processes_spawned:
            super().tearDownClass()
            return

        logger.debug(f"Joining {cls.world_size} workers")  # noqa: G004
        # Enqueue "None" to all workers to tell them to exit
        for task_queue in cls.task_queues:
            task_queue.put(None)

        # Wait for all workers to exit
        for process in cls.processes:
            process.join()

        # Clear up the rendezvous file
        try:
            os.remove(cls.rdvz_file)
        except OSError:
            pass

        logger.info(f"Class {cls.__name__} finished")  # noqa: G004
        super().tearDownClass()

    def setUp(self) -> None:
        """
        Test fixture. Run before each test.
        """
        super().setUp()

        # Ensure processes are spawned (lazy initialization for instantiate_device_type_tests)
        self.__class__._ensure_processes_spawned()

        # I am the dispatcher
        self.rank = self.MAIN_PROCESS_RANK

        # If this test class hits an exception in one test, skip the rest of tests
        if self.__class__.poison_pill:
            raise unittest.SkipTest(f"Previous test failed, skipping {self.id()}")

        # Enqueue "current test" to all workers
        for i, task_queue in enumerate(self.task_queues):
            logger.debug(f"Sending Rank {i}: {self.id()}")  # noqa: G004
            task_queue.put(self.id())

    def _worker_run_main_wait(self, fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                logger.debug(f"Waiting for workers to finish {self.id()}")  # noqa: G004
                # Wait for the workers to finish the test
                for i, completion_queue in enumerate(self.completion_queues):
                    rv = completion_queue.get()
                    if isinstance(rv, unittest.SkipTest):
                        raise rv
                    if isinstance(rv, BaseException):
                        # Hit an exception, re-raise it in the main process.
                        logger.warning(
                            f"Detected failure from Rank {i} in: {self.id()}, "  # noqa: G004
                            f"skipping rest of tests in Test class: {self.__class__.__name__}"  # noqa: G004
                        )
                        # Poison rest of tests (because ProcessGroup may be not
                        # reusable now)
                        self.__class__.poison_pill = True
                        raise rv

                    # Success
                    if rv != self.id():
                        raise AssertionError(
                            f"Expected rv == self.id(), got {rv} != {self.id()}"
                        )
                    logger.debug(
                        f"Main proc detected rank {i} finished {self.id()}"  # noqa: G004
                    )
            else:
                # Worker just runs the test
                fn()

        return types.MethodType(wrapper, self)

    # The main process spawns N subprocesses that run the test.
    # Constructor patches current instance test method to
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    def __init__(
        self, method_name: str = "runTest", methodName: str = "runTest"
    ) -> None:
        # methodName is the correct naming in unittest and testslide uses keyword arguments.
        # So we need to use both to 1) not break BC and, 2) support testslide.
        if methodName != "runTest":
            method_name = methodName
        super().__init__(method_name)
        try:
            fn = getattr(self, method_name)
            setattr(self, method_name, self._worker_run_main_wait(fn))
        except AttributeError as e:
            if methodName != "runTest":
                # we allow instantiation with no explicit method name
                # but not an *incorrect* or missing method name
                raise ValueError(
                    f"no such test method in {self.__class__}: {methodName}"
                ) from e
