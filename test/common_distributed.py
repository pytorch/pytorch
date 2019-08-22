import multiprocessing
import sys
import tempfile
import time
import unittest

from collections import namedtuple
from functools import wraps

import torch
import torch.distributed as c10d

from common_utils import TestCase


TestSkip = namedtuple('TestSkip', 'exit_code, message')


TEST_SKIPS = {
    "multi-gpu": TestSkip(75, "Need at least 2 CUDA devices"),
    "nccl": TestSkip(76, "c10d not compiled with NCCL support"),
    "known_issues": TestSkip(77, "Test skipped due to known issues")
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


TIMEOUT_DEFAULT = 30
TIMEOUT_OVERRIDE = {}


def get_timeout(test_id):
    return TIMEOUT_OVERRIDE.get(test_id.split('.')[-1], TIMEOUT_DEFAULT)


class MultiProcessTestCase(TestCase):
    MAIN_PROCESS_RANK = -1

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
                fn(self)
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
        self.rank = self.MAIN_PROCESS_RANK
        self.file = tempfile.NamedTemporaryFile(delete=False)
        self.processes = [self._spawn_process(rank) for rank in range(int(self.world_size))]

    def tearDown(self):
        super(MultiProcessTestCase, self).tearDown()
        for p in self.processes:
            p.terminate()

    def _spawn_process(self, rank):
        name = 'process ' + str(rank)
        process = multiprocessing.Process(target=self._run, name=name, args=(rank,))
        process.start()
        return process

    def _run(self, rank):
        self.rank = rank

        # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
        # We're retreiving a corresponding test and executing it.
        getattr(self, self.id().split(".")[2])()
        sys.exit(0)

    def _join_processes(self, fn):
        timeout = get_timeout(self.id())
        start_time = time.time()
        for p in self.processes:
            p.join(timeout)
        elapsed_time = time.time() - start_time
        self._check_return_codes(elapsed_time)

    def _check_return_codes(self, elapsed_time):
        """
        Checks that the return codes of all spawned processes match, and skips
        tests if they returned a return code indicating a skipping condition.
        """
        first_process = self.processes[0]
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
