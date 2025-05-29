import sys
from datetime import timedelta
from unittest import TestCase, skipIf, skipUnless
import logging

import torch
from torch.distributed import TCPStore
import torch.distributed as dist

from torch.testing._internal.common_utils import run_tests
from torch.distributed.checkpoint._pg_transport import PGTransport

from torch.testing._internal.common_cuda import TEST_MULTIGPU
from torch.testing._internal.common_distributed import (
    MultiProcContinousTest,
    requires_nccl,
)
from torch.testing._internal.common_utils import (
    check_leaked_tensors,
    instantiate_parametrized_tests,
    parametrize,
    skip_but_pass_in_sandcastle_if,
)


logger = logging.getLogger(__name__)

d_hid = 512
batch_size = 256

torch.manual_seed(0)


class PgTransportGloo(MultiProcContinousTest):
    @classmethod
    def backend_str(cls) -> str:
        # Testing with NCCL backend
        return "gloo"

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Set up the device.
        """
        super().setUpClass()
        dev_id = cls.rank % torch.cuda.device_count()
        cls.device = torch.device(f"cuda:{dev_id}")

    @requires_nccl()
    @skip_but_pass_in_sandcastle_if(not TEST_MULTIGPU, "NCCL test requires 2+ GPUs")
    def test_pg_transport_gloo(self) -> None:
        print("in test")
        device: torch.device = torch.device("cpu")
        dist.barrier()
        print("fin")

class PgTransportNCCL(MultiProcContinousTest):
    @classmethod
    def backend_str(cls) -> str:
        # Testing with NCCL backend
        return "nccl"

    @classmethod
    def setUpClass(cls):
        """
        Class-scope test fixture. Run once for entire test class, before any test starts.
        Set up the device.
        """
        super().setUpClass()
        dev_id = cls.rank % torch.cuda.device_count()
        cls.device = torch.device(f"cuda:{dev_id}")

    def test_pg_transport_nccl(self) -> None:
        pass


if __name__ == "__main__":
    run_tests()
