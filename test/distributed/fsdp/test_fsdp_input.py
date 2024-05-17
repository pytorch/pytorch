# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module
from torch.optim import SGD
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
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


class TestInput(FSDPTest):
    @property
    def world_size(self):
        return 1

    @skip_if_lt_x_gpu(1)
    @parametrize("input_cls", [subtest(dict, name="dict"), subtest(list, name="list")])
    def test_input_type(self, input_cls):
        """Test FSDP with input being a list or a dict, only single GPU."""

        class Model(Module):
            def __init__(self):
                super().__init__()
                self.layer = Linear(4, 4)

            def forward(self, input):
                if isinstance(input, list):
                    input = input[0]
                else:
                    assert isinstance(input, dict), input
                    input = input["in"]
                return self.layer(input)

        model = FSDP(Model()).cuda()
        optim = SGD(model.parameters(), lr=0.1)

        for _ in range(5):
            in_data = torch.rand(64, 4).cuda()
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


instantiate_parametrized_tests(TestInput)

if __name__ == "__main__":
    run_tests()
