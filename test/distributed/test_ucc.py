# Owner(s): ["oncall: distributed"]

import os
import sys

import torch
import torch.distributed as dist

torch.backends.cuda.matmul.allow_tf32 = False

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from torch.testing._internal.common_distributed import DistTestCases
from torch.testing._internal.common_utils import (
    run_tests,
    sandcastle_skip_if,
)

# Sets showing that a collective isn't implemented
DistTestCases.skip_collective["allgather_coalesced"] = {"nccl", "mpi", "ucc"}
DistTestCases.skip_collective["gather"] = {"nccl", "ucc"}
DistTestCases.skip_collective["scatter"] = {"nccl", "ucc"}
DistTestCases.skip_collective["reduce"] = {"ucc"}
DistTestCases.skip_collective["sendrecv anysource"] = {"nccl", "ucc"}
DistTestCases.skip_collective["cpu barrier"] = {"nccl", "ucc"}

# Sets showing that something is implemented
DistTestCases.backend_feature["gpu"] = {"nccl", "gloo", "ucc"}
DistTestCases.backend_feature["cuda"] = {"nccl", "gloo", "ucc"}
DistTestCases.backend_feature["ddp"] = {"nccl", "gloo", "ucc"}
DistTestCases.backend_feature["subgroup"] = {"nccl", "gloo", "ucc"}
DistTestCases.backend_feature["plugin"] = {"ucc"}

os.environ["MASTER_ADDR"] = "localhost"

if "MASTER_PORT" not in os.environ:
    try:
        from caffe2.torch.fb.common.utils import get_free_port

        os.environ["MASTER_PORT"] = str(get_free_port())
    except ImportError:
        os.environ["MASTER_PORT"] = "12375"

os.environ["INIT_METHOD"] = "tcp://localhost:" + os.environ["MASTER_PORT"]

if "UCX_TLS" not in os.environ:
    os.environ["UCX_TLS"] = "sm,tcp"

try:
    import torch_ucc  # noqa: F401
except ImportError:
    try:
        from ucc_plugin import initialize_ucc_plugin
    except ImportError:
        raise RuntimeError("Unable to import initialize_ucc_plugin")
    else:
        initialize_ucc_plugin("ucc")

BACKEND = os.environ["BACKEND"]

# We have to import this after we change the values in DistTestCases
from torch.testing._internal.distributed.distributed_test import (
    TestDistBackend,
    DistributedTest,
)


class TestDistBackendWithSpawn(TestDistBackend, DistributedTest._DistTestBase):
    port_num = str(os.environ["MASTER_PORT"])

    def setUp(self):
        super().setUp()
        self._spawn_processes()
        torch.backends.cudnn.flags(allow_tf32=False).__enter__()

    # UCC does not support File Store today. Need to always use TCP Store.
    @property
    def init_method(self):
        return "tcp://localhost:" + self.port_num

    @sandcastle_skip_if(
        BACKEND == "ucc", "This test fails on UCC, so we are not running it today"
    )
    def test_isend_autograd_profiler(self):
        raise Exception("This test fails with UCC, not running it")

    @sandcastle_skip_if(
        BACKEND == "ucc", "This test fails on UCC, so we are not running it today"
    )
    def test_all_gather_object_subgroup(self):
        raise Exception("This test fails with UCC, not running it")

    @sandcastle_skip_if(
        BACKEND == "ucc", "This test fails on UCC, so we are not running it today"
    )
    def test_ddp_logging_data_cpu(self):
        raise Exception("This test fails with UCC, not running it")


if __name__ == "__main__":
    run_tests()
