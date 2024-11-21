# Owner(s): ["oncall: distributed"]
import sys

import torch
from torch import distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module
from torch.nn.parallel import DistributedDataParallel
from torch.optim import SGD
from torch.testing._internal.common_device_type import instantiate_device_type_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest, get_devtype, get_full_params
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


class Model(Module):
    def __init__(self, wrap_fsdp):
        super().__init__()
        # keep everything deterministic for model initialization
        torch.manual_seed(0)
        self.inner = Linear(4, 4)
        if wrap_fsdp:
            self.inner = FSDP(self.inner)
        self.outer = Linear(4, 5)

    def forward(self, x):
        # Forward twice.
        i = self.inner(x)
        j = self.inner(x)
        return self.outer(i + j)


class TestMultiForward(FSDPTest):
    def _dist_train(self, wrap_fsdp):
        # keep everything deterministic for input data
        torch.manual_seed(0)
        model = Model(wrap_fsdp).to(device_type.type)
        if wrap_fsdp:
            model = FSDP(model, device_id=device_type.type)
        else:
            model = DistributedDataParallel(model, device_ids=[device_type.type])
        optim = SGD(model.parameters(), lr=0.1)
        in_data = torch.rand(64, 4).to(device_type.type)
        in_data.requires_grad = True
        for _ in range(3):
            out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()
        if wrap_fsdp:
            return get_full_params(model)
        return list(model.parameters())

    @skip_if_lt_x_gpu(2)
    def test_multi_forward(self):
        # DDP
        ddp_state = self._dist_train(wrap_fsdp=False)
        # FSDP
        fsdp_state = self._dist_train(wrap_fsdp=True)
        self.assertEqual(ddp_state, fsdp_state)


devices = ("cpu", "hpu")
instantiate_device_type_tests(TestMultiForward, globals(), only_for=devices)
if __name__ == "__main__":
    run_tests()
