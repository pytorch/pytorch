from __future__ import absolute_import, division, print_function, unicode_literals

import os
import sys
import unittest

import torch.distributed as dist
from torch._utils_internal import (
    TEST_MASTER_ADDR as MASTER_ADDR,
    TEST_MASTER_PORT as MASTER_PORT,
)
from torch.testing._internal.common_distributed import (
    TEST_SKIPS,
    MultiProcessTestCase,
    initialize_temp_directories,
    cleanup_temp_dir,
)
from torch.testing._internal.common_utils import run_tests, TEST_WITH_ASAN, NO_MULTIPROCESSING_SPAWN
from torch.testing._internal.distributed.distributed_test import Barrier, _DistTestBase

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

BACKEND = os.environ["BACKEND"]
INIT_METHOD = os.getenv("INIT_METHOD", "env://")


if BACKEND == "gloo" or BACKEND == "nccl":
    WORLD_SIZE = os.environ["WORLD_SIZE"]

    @unittest.skipIf(
        TEST_WITH_ASAN, "Skip ASAN as torch + multiprocessing spawn have known issues"
    )
    @unittest.skipIf(
        NO_MULTIPROCESSING_SPAWN, "Spawn not available, skipping tests."
    )
    class TestDistBackendWithSpawn(MultiProcessTestCase, _DistTestBase):
        @property
        def init_method(self):
            return "file://{file_name}".format(file_name=self.file_name)

        # Needed since MultiProcessTestCase assumes a world_size of 4, but we
        # run these tests under other various world_sizes.
        @property
        def world_size(self):
            return os.environ["WORLD_SIZE"]

        @classmethod
        def setUpClass(cls):
            os.environ["MASTER_ADDR"] = str(MASTER_ADDR)
            os.environ["MASTER_PORT"] = str(MASTER_PORT)
            os.environ["WORLD_SIZE"] = str(WORLD_SIZE)
            super().setUpClass()

        def setUp(self):
            super().setUp()
            # initialize Barrier.
            initialize_temp_directories()
            Barrier.init()
            self._spawn_processes()

        def tearDown(self):
            cleanup_temp_dir()
            super(MultiProcessTestCase, self).tearDown()
            super(TestDistBackendWithSpawn, self).tearDown()

        @classmethod
        def _run(cls, rank, test_name, file_name):
            self = cls(test_name)
            self.rank = rank
            self.file_name = file_name
            try:
                dist.init_process_group(
                    init_method=self.init_method,
                    backend=BACKEND,
                    world_size=int(self.world_size),
                    rank=self.rank,
                )
            except RuntimeError as e:
                if "recompile" in e.args[0]:
                    sys.exit(TEST_SKIPS["backend_unavailable"].exit_code)

                raise

            # Execute barrier prior to running test to ensure that every process
            # has finished initialization and that the following test
            # immediately exiting due to a skip doesn't cause flakiness.
            self._barrier()

            # self.id() == e.g. '__main__.TestDistributed.test_get_rank'
            # We're retreiving a corresponding test and executing it.
            getattr(self, test_name)()
            self._barrier()
            dist.destroy_process_group()
            sys.exit(0)


if __name__ == "__main__":
    run_tests()
