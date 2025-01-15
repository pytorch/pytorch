# Owner(s): ["oncall: distributed"]
import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module, Sequential
from torch.optim import SGD
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype
from torch.testing._internal.common_utils import run_tests, TEST_WITH_DEV_DBG_ASAN


device_type = torch.device(get_devtype())

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
    def __init__(self, device):
        super().__init__()
        self.layers = Sequential(FSDP(Linear(5, 5), device_id=device_type.type))

    def forward(self, x):
        return self.layers(x)


class TestMultipleWrapping(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_multiple_wrapping(self, device):
        """
        This test simulates wrapping the module after training to run inference.
        This is required in cases where later in a session, the model is wrapped again in FSDP but
        contains nested FSDP wrappers within the module.
        """
        inner_model = InnerModel(device)
        model = FSDP(inner_model).to(device_type.type)
        optim = SGD(model.parameters(), lr=0.1)
        for _ in range(3):
            input = torch.rand((1, 5), dtype=torch.float).to(device_type.type)
            input.requires_grad = True
            output = model(input)
            output.sum().backward()
            optim.step()
            optim.zero_grad()
        input = torch.rand((1, 5), dtype=torch.float).to(device_type.type)
        output = model(input)
        # second time to rewrap the inner model
        # rewrapped_model = FSDP(inner_model, device_id=device)
        rewrapped_model = FSDP(inner_model).to(device_type.type)
        rewrapped_output = rewrapped_model(input)
        self.assertEqual(output, rewrapped_output)


devices = ("cuda", "hpu")
instantiate_device_type_tests(TestMultipleWrapping, globals(), only_for=devices)
if __name__ == "__main__":
    run_tests()
