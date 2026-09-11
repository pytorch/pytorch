# Owner(s): ["oncall: distributed"]
import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module, Sequential
from torch.optim import SGD
from torch.testing._internal.common_device_type import (
    Capability,
    instantiate_device_type_tests,
    requires_capabilities,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTestContinuous
from torch.testing._internal.common_utils import (
    HardwareClassification,
    run_tests,
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


class InnerModel(Module):
    def __init__(self, device):
        super().__init__()
        device_type = torch.device(device).type
        self.layers = Sequential(FSDP(Linear(5, 5), device_id=device_type))

    def forward(self, x):
        return self.layers(x)


class TestMultipleWrapping(FSDPTestContinuous):
    hw_classification = HardwareClassification.ACCELERATOR

    @requires_capabilities(Capability.distributed.backend, Capability.distributed.fsdp)
    @skip_if_lt_x_gpu(2)
    def test_multiple_wrapping(self, device):
        """
        This test simulates wrapping the module after training to run inference.
        This is required in cases where later in a session, the model is wrapped again in FSDP but
        contains nested FSDP wrappers within the module.
        """
        device_type = torch.device(device).type
        inner_model = InnerModel(device)
        model = FSDP(inner_model).to(device_type)
        optim = SGD(model.parameters(), lr=0.1)
        for _ in range(3):
            input = torch.rand((1, 5), dtype=torch.float).to(device_type)
            input.requires_grad = True
            output = model(input)
            output.sum().backward()
            optim.step()
            optim.zero_grad()
        input = torch.rand((1, 5), dtype=torch.float).to(device_type)
        output = model(input)
        # second time to rewrap the inner model
        # rewrapped_model = FSDP(inner_model, device_id=device)
        rewrapped_model = FSDP(inner_model).to(device_type)
        rewrapped_output = rewrapped_model(input)
        self.assertEqual(output, rewrapped_output)


instantiate_device_type_tests(
    TestMultipleWrapping,
    globals(),
    except_for=("cpu",),
    allow_xpu=True,
)
if __name__ == "__main__":
    run_tests()
