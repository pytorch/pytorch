import os
import sys
import tempfile
from functools import wraps
import torch
import torch.cuda
import torch.distributed as dist

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_utils import TestCase, find_free_port, run_tests
from torch.distributed.distributed_c10d import _get_default_group
from torch.testing._internal.distributed.distributed_test import (
    DistributedTest, TestDistBackend
)

torch.backends.cuda.matmul.allow_tf32 = False

CPP_EXTENSIONS_WARNING = """
Ninja (https://ninja-build.org) must be available to run C++ extensions tests,
but it could not be found. Install ninja with `pip install ninja`
or `conda install ninja`.
"""

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

    class TestDistBackendWithFork(TestDistBackend, DistributedTest._DistTestBase):

        def setUp(self):
            super().setUp()
            self._fork_processes()
            torch.backends.cudnn.flags(allow_tf32=False).__enter__()


elif BACKEND == "mpi":
    WORLD_SIZE = os.environ["WORLD_SIZE"]
    dist.init_process_group(init_method=INIT_METHOD, backend="mpi")

    class TestMPIWithFork(TestCase, DistributedTest._DistTestBase):
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
