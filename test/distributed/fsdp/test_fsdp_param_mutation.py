# Owner(s): ["oncall: distributed"]

import sys

import torch
from torch import distributed as dist
from torch.distributed._fsdp import FullyShardedDataParallel as FSDP
from torch.nn import Linear, Module
from torch.optim import SGD
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import (
    FSDPTest,
)
from torch.testing._internal.common_utils import TEST_WITH_DEV_DBG_ASAN, run_tests


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
    def __init__(self):
        super().__init__()
        # keep everything deterministic for model initialization
        torch.manual_seed(0)
        self.inner = FSDP(Linear(4, 4))
        self.outer = Linear(4, 5)

    def forward(self, x):
        y = self.inner(x)
        return self.outer(y)


# FSDP does not support the case when param is mutated outside FSDP,
# as the mutated param may point to a non_shard tensor or an invalid
# shard, FSDP can not guranttee this case work as expected.
class TestParamMutation(FSDPTest):
    @skip_if_lt_x_gpu(2)
    def test_param_mutation(self):
        # keep everything deterministic for input data
        torch.manual_seed(0)

        model = Model().cuda()
        model = FSDP(model)
        optim = SGD(model.parameters(), lr=0.1)

        in_data = torch.rand(64, 4).cuda()
        in_data.requires_grad = True
        for i in range(3):
            if i > 0:
                with self.assertRaisesRegex(
                    AssertionError,
                    "Parameter storage is changed after first iteration outside FSDP, this use case is not supported",
                ):
                    out = model(in_data)
                return
            else:
                out = model(in_data)
            out.sum().backward()
            optim.step()
            optim.zero_grad()
            # Param is mutated outside FSDP after first iteration
            model.half()


if __name__ == "__main__":
    run_tests()
