# Owner(s): ["oncall: distributed"]


import sys

import torch
import torch.distributed as c10d


if not c10d.is_available():
    print("c10d not available, skipping tests", file=sys.stderr)
    sys.exit(0)

from test_c10d_spawn import _torch_dist_nn_available, TestDistributedNNFunctions

from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import requires_ucc, skip_if_lt_x_gpu
from torch.testing._internal.common_utils import (
    decorateIf,
    HardwareClassification,
    run_tests,
    skip_but_pass_in_sandcastle,
    skip_but_pass_in_sandcastle_if,
    TEST_WITH_DEV_DBG_ASAN,
)


# Fails on Python-3.9, see https://github.com/pytorch/pytorch/issues/51619


# Skip dev-asan as torch + multiprocessing spawn have known issues
if not TEST_WITH_DEV_DBG_ASAN:

    class TestDistributedNNFunctionsUcc(TestDistributedNNFunctions):
        hw_classification = HardwareClassification.ACCELERATOR

        def setUp(self):
            if self.device_type not in c10d.Backend.backend_capability["ucc"]:
                self.skipTest(f"UCC does not support {self.device_type}")
            if (
                self.device_type != "cpu"
                and len(self.get_all_devices()) < self.world_size
            ):
                self.skipTest(f"test requires {self.world_size}+ devices")
            super().setUp()

        @requires_ucc()
        @decorateIf(
            skip_if_lt_x_gpu(2),
            lambda params: torch.device(params["device"]).type != "cpu",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_broadcast(self, device):
            self._test_broadcast("ucc", device)

        @requires_ucc()
        @decorateIf(
            skip_if_lt_x_gpu(2),
            lambda params: torch.device(params["device"]).type != "cpu",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_reduce(self, device):
            self._test_reduce("ucc", device)

        @requires_ucc()
        @decorateIf(
            skip_if_lt_x_gpu(2),
            lambda params: torch.device(params["device"]).type != "cpu",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_allreduce(self, device):
            self._test_allreduce("ucc", device)

        @requires_ucc()
        @decorateIf(
            skip_if_lt_x_gpu(2),
            lambda params: torch.device(params["device"]).type != "cpu",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        @skip_but_pass_in_sandcastle(
            "runs into illegal memory access on first assertEqual check when run locally"
        )
        def test_all_gather(self, device):
            self._test_all_gather("ucc", device)

        @requires_ucc()
        @decorateIf(
            skip_if_lt_x_gpu(2),
            lambda params: torch.device(params["device"]).type != "cpu",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all(self, device):
            self._test_all_to_all("ucc", device)

        @requires_ucc()
        @decorateIf(
            skip_if_lt_x_gpu(2),
            lambda params: torch.device(params["device"]).type != "cpu",
        )
        @skip_but_pass_in_sandcastle_if(
            not _torch_dist_nn_available, "torch.distributed.nn is not available"
        )
        def test_all_to_all_single(self, device):
            self._test_all_to_all_single("ucc", device)

    instantiate_device_type_tests(
        TestDistributedNNFunctionsUcc,
        globals(),
    )


if __name__ == "__main__":
    run_tests()
