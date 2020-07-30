from __future__ import absolute_import, division, print_function, unicode_literals
import os
import sys
import tempfile
from functools import wraps
import torch
import torch.cuda
import torch.distributed as dist
from torch.testing._internal.common_utils import TestCase, find_free_port, run_tests
from torch.distributed.distributed_c10d import _get_default_group
from torch._utils_internal import TEST_MASTER_ADDR as MASTER_ADDR
from torch._utils_internal import TEST_MASTER_PORT as MASTER_PORT
from torch.testing._internal.distributed.distributed_test import _DistTestBase, Barrier
from torch.testing._internal.common_distributed import (
    TEST_SKIPS,
    MultiProcessTestCase,
    initialize_temp_directories,
    cleanup_temp_dir,
)

CPP_EXTENSIONS_WARNING = """
Ninja (https://ninja-build.org) must be available to run C++ extensions tests,
but it could not be found. Install ninja with `pip install ninja`
or `conda install ninja`.
"""

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

BACKEND = os.environ["BACKEND"]
INIT_METHOD = os.getenv("INIT_METHOD", "env://")


def skip_if_no_ninja(func):

    @wraps(func)
    def wrapper(*args, **kwargs):
        try:
            import torch.utils.cpp_extension
            torch.utils.cpp_extension.verify_ninja_availability()
        except RuntimeError:
            print(CPP_EXTENSIONS_WARNING)
            return 0

        return func(*args, **kwargs)

    return wrapper


if BACKEND == "gloo" or BACKEND == "nccl":
    WORLD_SIZE = os.environ["WORLD_SIZE"]

    class TestDistBackend(MultiProcessTestCase, _DistTestBase):

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
            self._fork_processes()

        def tearDown(self):
            cleanup_temp_dir()
            super(MultiProcessTestCase, self).tearDown()
            super(TestDistBackend, self).tearDown()

        @classmethod
        def _run(cls, rank, test_name, file_name):
            self = cls(test_name)
            self.rank = rank
            self.file_name = file_name
            try:
                dist.init_process_group(
                    init_method=self.init_method,
                    backend=BACKEND,
                    world_size=int(WORLD_SIZE),
                    rank=self.rank
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


elif BACKEND == "mpi":
    WORLD_SIZE = os.environ["WORLD_SIZE"]
    dist.init_process_group(init_method=INIT_METHOD, backend="mpi")

    class TestMPI(TestCase, _DistTestBase):
        pass

elif BACKEND == "test":
    class TestBackendDynamicLoad(TestCase):
        def setUp(self):
            super(TestBackendDynamicLoad, self).setUp()

        def _load_test_backend(self):
            temp_dir = tempfile.mkdtemp()
            src = "{}/../cpp_extensions/cpp_c10d_extension.cpp".format(os.path.abspath(os.path.dirname(__file__)))
            extension = torch.utils.cpp_extension.load(
                name="torch_test",
                sources=[src],
                build_directory=temp_dir
            )

        @skip_if_no_ninja
        def test_backend_apis(self):
            self._load_test_backend()

            os.environ['WORLD_SIZE'] = '1'
            os.environ['MASTER_ADDR'] = '127.0.0.1'
            os.environ['MASTER_PORT'] = str(find_free_port())
            os.environ['RANK'] = '0'

            dist.init_process_group(backend='test', init_method='env://', world_size=1, rank=0)
            self.assertEqual(dist.get_rank(), 0)
            self.assertEqual(dist.get_world_size(), 1)

            process_group = _get_default_group()
            work = process_group.allreduce([torch.rand(1), torch.rand(1)])
            self.assertTrue(work.wait())
            self.assertTrue(work.is_completed())
            self.assertTrue(work.is_success())

            work = process_group.broadcast([torch.rand(1)])
            self.assertTrue(work.wait())
            self.assertTrue(work.is_completed())
            self.assertTrue(work.is_success())

            dist.destroy_process_group()

if __name__ == "__main__":
    assert (
        not torch.cuda._initialized
    ), "test_distributed must not have initialized CUDA context on main process"

    run_tests()
