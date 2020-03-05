from __future__ import absolute_import, division, print_function, unicode_literals

import sys
import tempfile
import time
import unittest
import logging
import six
import traceback

from collections import namedtuple
from functools import wraps

import torch
import torch.distributed as c10d

from functools import partial, reduce
from torch.testing._internal.common_utils import TestCase, TEST_WITH_ROCM

TestSkip = namedtuple('TestSkip', 'exit_code, message')


TEST_SKIPS = {
    "multi-gpu": TestSkip(75, "Need at least 2 CUDA devices"),
    "nccl": TestSkip(76, "c10d not compiled with NCCL support"),
    "known_issues": TestSkip(77, "Test skipped due to known issues"),
    "skipIfRocm": TestSkip(78, "Test skipped for ROCm")
}


def skip_if_not_multigpu(func):
    """Multi-GPU tests requires at least 2 GPUS. Skip if this is not met."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        if torch.cuda.is_available() and torch.cuda.device_count() >= 2:
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS['multi-gpu'].exit_code)

    return wrapper


def skip_if_lt_x_gpu(x):
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            if torch.cuda.is_available() and torch.cuda.device_count() >= x:
                return func(*args, **kwargs)
            sys.exit(TEST_SKIPS['multi-gpu'].exit_code)
        return wrapper

    return decorator


def skip_for_known_issues(func):
    """Skips a test due to known issues (for c10d)."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        sys.exit(TEST_SKIPS['known_issues'].exit_code)

    return wrapper


def requires_gloo():
    return unittest.skipUnless(
        c10d.is_gloo_available(),
        "c10d was not compiled with the Gloo backend",
    )

def requires_nccl_version(version, msg):
    if not c10d.is_nccl_available():
        return unittest.skip(
            "c10d was not compiled with the NCCL backend",
        )
    else:
        return unittest.skipIf(
            torch.cuda.nccl.version() < version,
            "Requires NCCL version greater than or equal to: {}, found: {}, reason: {}".format(
                version,
                torch.cuda.nccl.version(), msg),
        )

def requires_nccl():
    return unittest.skipUnless(
        c10d.is_nccl_available(),
        "c10d was not compiled with the NCCL backend",
    )


def requires_mpi():
    return unittest.skipUnless(
        c10d.is_mpi_available(),
        "c10d was not compiled with the MPI backend",
    )


def skip_if_rocm(func):
    """Skips a test for ROCm"""
    func.skip_if_rocm = True
    @wraps(func)
    def wrapper(*args, **kwargs):
        if not TEST_WITH_ROCM:
            return func(*args, **kwargs)
        sys.exit(TEST_SKIPS['skipIfRocm'].exit_code)

    return wrapper

TIMEOUT_DEFAULT = 100
TIMEOUT_OVERRIDE = {}


def get_timeout(test_id):
    return TIMEOUT_OVERRIDE.get(test_id.split('.')[-1], TIMEOUT_DEFAULT)


def simple_sparse_reduce_tests(rank, world_size, num_inputs=1):
    """
    Generate a number of basic test cases for sparse reduction.
    These cover tensors with a varying number of sparse dimensions and a varying
    number of dense dimensions. The only reduction operation we support is sum.
    """
    def generate(rank, world_size, sparse_dims=1, dense_dims=0):
        # First sparse dimension is [0..rank].
        # Subsequent dimensions are always 0, so we know there is
        # a non-empty intersection between any two sparse tensors.
        indices = [range(rank + 1)]
        shape = [world_size] + [2 for _ in range(dense_dims)]
        for _ in range(sparse_dims - 1):
            indices.append([0] * (rank + 1))
            shape.append(world_size)
        values = torch.ones([rank + 1] + [2 for _ in range(dense_dims)])
        return torch.sparse_coo_tensor(indices, values, shape)

    def compute_sum(fn, world_size):
        return reduce(lambda a, b: a + b, [fn(rank, world_size) for rank in range(world_size)])

    return [
        (
            [
                fn(num_inputs * rank + i, num_inputs * world_size)
                for i in range(num_inputs)
            ],
            [
                compute_sum(fn, num_inputs * world_size)
                for i in range(num_inputs)
            ],
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


class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1
    # This exit code is used to indicate that the test code had an error and
    # exited abnormally. There are certain tests that might use sys.exit() to
    # simulate failures and in those cases, we can't have an exit code of 0,
    # but we still want to ensure we didn't run into any other errors.
    TEST_ERROR_EXIT_CODE = 10

    @property
    def world_size(self):
        return 4

    @staticmethod
    def join_or_run(fn):
        @wraps(fn)
        def wrapper(self):
            if self.rank == self.MAIN_PROCESS_RANK:
                self._join_processes(fn)
            else:
                try:
                    fn(self)
                except Exception as e:
                    logging.error('Caught exception: \n{}exiting process with exit code: {}'
                                  .format(traceback.format_exc(), MultiProcessTestCase.TEST_ERROR_EXIT_CODE))
                    sys.exit(MultiProcessTestCase.TEST_ERROR_EXIT_CODE)
        return wrapper

    # The main process spawns N subprocesses that run the test.
    # This function patches overwrites every test function to either
    # assume the role of the main process and join its subprocesses,
    # or run the underlying test function.
    @classmethod
    def setUpClass(cls):
        for attr in dir(cls):
            if attr.startswith('test'):
                fn = getattr(cls, attr)
                setattr(cls, attr, cls.join_or_run(fn))

    def setUp(self):
        super(MultiProcessTestCase, self).setUp()
        self.skip_return_code_checks = []
        self.rank = self.MAIN_PROCESS_RANK
        self.file_name = tempfile.NamedTemporaryFile(delete=False).name

    def tearDown(self):
        super(MultiProcessTestCase, self).tearDown()
        for p in self.processes:
            p.terminate()

    def _current_test_name(self):
        # self.id() == e.g. '__main__.TestDistributed.TestAdditive.test_get_rank'
        return self.id().split(".")[-1]

    def _start_processes(self, proc):
        self.processes = []
        for rank in range(int(self.world_size)):
            process = proc(
                target=self.__class__._run,
                name='process ' + str(rank),
                args=(rank, self._current_test_name(), self.file_name))
            process.start()
            self.processes.append(process)

    def _fork_processes(self):
        if six.PY3:
            proc = torch.multiprocessing.get_context("fork").Process
        else:
            proc = torch.multiprocessing.Process
        self._start_processes(proc)

    def _spawn_processes(self):
        if six.PY3:
            proc = torch.multiprocessing.get_context("spawn").Process
        else:
            raise RuntimeError("Cannot use spawn start method with Python 2")
        self._start_processes(proc)

    @classmethod
    def _run(cls, rank, test_name, file_name):
        self = cls(test_name)
        self.rank = rank
        self.file_name = file_name

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retrieving a corresponding test and executing it.
        getattr(self, test_name)()
        # exit to avoid run teardown() for fork processes
        sys.exit(0)

    def _join_processes(self, fn):
        timeout = get_timeout(self.id())
        start_time = time.time()
        subprocess_error = False
        while True:
            # check to see if any subprocess exited with an error early.
            for p in self.processes:
                # This is the exited code processes exit with if they
                # encountered an exception.
                if p.exitcode == MultiProcessTestCase.TEST_ERROR_EXIT_CODE:
                    print("Some process exited badly, terminating rest.")
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
                print(
                    "Timing out after {} seconds and killing subprocesses.".format(
                        timeout
                    )
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

    def _check_no_test_errors(self, elapsed_time):
        """
        Checks that we didn't have any errors thrown in the child processes.
        """
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError('Process {} timed out after {} seconds'.format(i, elapsed_time))
            self.assertNotEqual(self.TEST_ERROR_EXIT_CODE, p.exitcode)

    def _check_return_codes(self, elapsed_time):
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
            error = "Processes {} exited with error code {}".format(
                " ".join([str(i) for (i, _) in errored_processes]),
                MultiProcessTestCase.TEST_ERROR_EXIT_CODE,
            )
            raise RuntimeError(error)
        # If no process exited uncleanly, we check for timeouts, and then ensure
        # each process exited cleanly.
        for i, p in enumerate(self.processes):
            if p.exitcode is None:
                raise RuntimeError('Process {} terminated or timed out after {} seconds'.format(i, elapsed_time))
            self.assertEqual(p.exitcode, first_process.exitcode)
        for skip in TEST_SKIPS.values():
            if first_process.exitcode == skip.exit_code:
                raise unittest.SkipTest(skip.message)
        self.assertEqual(first_process.exitcode, 0)

    @property
    def is_master(self):
        return self.rank == 0
