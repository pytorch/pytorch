# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module, Sequential
from torch.optim import SGD
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN

if not dist.is_available():
    print("Distributed not available, skipping tests", file=sys.stderr)
    sys.exit(0)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class InnerModel(Module):
    def __init__(self):
        super().__init__()
        self.layers = Sequential(FSDP(Linear(5, 5)))

    def forward(self, x):
        return self.layers(x)


class TestMultipleWrapping(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_multiple_wrapping(self):
        """
        This test simulates wrapping the module after training to run inference.
        This is required in cases where later in a session, the model is wrapped again in FSDP but
        contains nested FSDP wrappers within the module.
        """
        inner_model = InnerModel()
        model = FSDP(inner_model).cuda()
        optim = SGD(model.parameters(), lr=0.1)

        for i in range(3):
            input = torch.rand((1, 5), dtype=torch.float).cuda()
            input.requires_grad = True
            output = model(input)
            output.sum().backward()
            optim.step()
            optim.zero_grad()
        input = torch.rand((1, 5), dtype=torch.float).cuda()
        output = model(input)

        # second time to rewrap the inner model
        rewrapped_model = FSDP(inner_model).cuda()
        rewrapped_output = rewrapped_model(input)

        self.assertEqual(output, rewrapped_output)


if __name__ == "__main__":
    run_tests()
