# Owner(s): ["oncall: distributed"]

import itertools
from functools import partial
from typing import cast, Dict, List, Optional, Tuple

import torch
import torch.fx as fx
import torch.nn as nn
from torch._functorch.aot_autograd import aot_module, make_boxed_func
from torch.distributed._functional_collectives import all_reduce
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)


class BoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln = nn.Sequential(
            nn.Linear(20, 20),
            nn.Softmax(),
        )

    def forward(self, input):
        return self.ln(input)


class NestedBoringModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.ln1 = torch.nn.Linear(20, 20)
        self.ln2 = torch.nn.Linear(20, 20)
        self.inner = BoringModel()

    def forward(self, input):
        return self.inner(self.ln2(self.ln1(input)))

counter = 1

def distribute(model: nn.Module) -> nn.Module:
    def fwd_compiler(gm: fx.GraphModule, inps: List[torch.Tensor]):
        return make_boxed_func(gm)

    def bwd_compiler(gm: fx.GraphModule, inps: List[torch.Tensor]):
        all_reduce(inps[0], "sum", [0, 1])
        return make_boxed_func(gm)

    return aot_module(model, fwd_compiler, bwd_compiler)


class IterGraphModuleMultiGPUTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_with_comm_fusion_cat(self) -> None:
        num_iters = 5
        model = NestedBoringModel().to("cuda")
        compile_m = distribute(model)

        for _ in range(1):
            batch = torch.randn(128, 20, device="cuda")
            output = compile_m(batch)
            output.sum().backward()

if __name__ == "__main__":
    run_tests()