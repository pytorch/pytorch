# Owner(s): ["oncall: distributed"]


from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

import torch.distributed as c10d
from torch.testing._internal.common_distributed import requires_ucc, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    run_tests,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


NO_UCC = not hasattr(c10d, "ProcessGroupUCC")

# Fails on Python-3.9, see https://github.com/pytorch/pytorch/issues/51619


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsUcc(TestDistributedNNFunctions):
        # Test Common Ops First.
        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_broadcast(self):
            self._test_broadcast("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce(self):
            self._test_reduce("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_allreduce(self):
            self._test_allreduce("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        @skip_but_pass_in_sandcastle(
            "runs into illegal memory access on first assertEqual check when run locally"
        )
        def test_all_gather(self):
            self._test_all_gather("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all(self):
            self._test_all_to_all("ucc")

        @requires_ucc()
        @skip_if_lt_x_gpu(2)
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all_single(self):
            self._test_all_to_all_single("ucc")


if __name__ == "__main__":
    run_tests()
