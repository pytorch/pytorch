# Owner(s): ["oncall: distributed"]


from test_c10d_spawn import TestDistributedNNFunctions

import torch
import torch.distributed as c10d
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle,
    TEST_WITH_DEV_DBG_ASAN,
)


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsUcc(TestDistributedNNFunctions):
        BACKEND = "ucc"

        def setUp(self):
            if not c10d.is_ucc_available():
                self.skipTest("c10d was not compiled with the UCC backend")
            if not torch.cuda.is_available() or torch.cuda.device_count() < 2:
                self.skipTest("Need at least 2 CUDA GPUs")
            super().setUp()

        @skip_but_pass_in_sandcastle(
            "runs into illegal memory access on first assertEqual check when run locally"
        )
        def test_all_gather(self):
            super().test_all_gather()


if __name__ == "__main__":
    run_tests()
