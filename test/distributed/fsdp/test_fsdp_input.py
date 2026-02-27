# Owner(s): ["oncall: distributed"]
import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module
from torch.optim import SGD
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestContinuous
from torch.testing._internal.common_utils import (
    parametrize,
    run_tests,
    subtest,
    TEST_WITH_DEV_DBG_ASAN,
)


if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)
if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestInput(FSDPTestContinuous):
    @property
    def world_size(self):
        return 1

    @skip_if_lt_x_gpu(1)
    @parametrize("input_cls", [subtest(dict, name="dict"), subtest(list, name="list")])
    def test_input_type(self, device, input_cls):
        """Test FSDP with input being a list or a dict, only single GPU."""

        class Model(Module):
            def __init__(self):
                super().__init__()
                self.layer = Linear(4, 4)

            def forward(self, input):
                if isinstance(input, list):
                    input = input[0]
                else:
                    if not isinstance(input, dict):
                        raise AssertionError(
                            f"Expected dict, got {type(input)}: {input}"
                        )
                    input = input["in"]
                return self.layer(input)

        fsdp_kwargs = {
            "device_id": device,
        }
        model = FSDP(Model().to(device), **fsdp_kwargs)
        optim = SGD(model.parameters(), lr=0.1)
        for _ in range(5):
            in_data = torch.rand(64, 4).to(device)
            in_data.requires_grad = True
            if input_cls is list:
                in_data = [in_data]
            else:
                self.assertTrue(input_cls is dict)
                in_data = {"in": in_data}
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()


devices = ("cuda", "hpu", "xpu")
instantiate_device_type_tests(TestInput, globals(), only_for=devices, allow_xpu=True)
if __name__ == "__main__":
    run_tests()
