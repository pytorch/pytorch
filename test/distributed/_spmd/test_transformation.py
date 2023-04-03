# Owner(s): ["oncall: distributed"]

from copy import deepcopy
from functools import wraps

import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed._spmd.api import compile
from torch.distributed._spmd.gm_transformation import GraphModuleTransformation
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms as base_with_comms,
)


def with_comms(func):
    @base_with_comms
    @wraps(func)
    def wrapper(self, *args, **kwargs):
        # make sure we set different random seeds for each rank
        # otherwise we dont need DDP / SPMD
        # (we would have the same parameters and inputs everywhere)
        torch.manual_seed(torch.distributed.get_rank())
        return func(self, *args, **kwargs)

    return wrapper


class DummyModel(nn.Module):
    def __init__(self, layers: int, dim: int):
        super().__init__()
        self.mod = nn.Sequential(*((nn.Linear(dim, dim), nn.ReLU()) * layers))

    def forward(self, x):
        return self.mod(x)


class TransformationTest(DTensorTestBase):
    @property
    def world_size(self):
        return 2

    def _init(self, batch_size, layers, dim):
        torch.manual_seed(0)
        model = DummyModel(layers, dim).cuda()
        ddp_model = DDP(deepcopy(model), device_ids=[dist.get_rank()])
        optim = torch.optim.Adam(
            model.parameters(), lr=0.01, foreach=True, capturable=True
        )
        ddp_optim = torch.optim.Adam(
            ddp_model.parameters(), lr=0.01, foreach=True, capturable=True
        )
        batch = torch.randn(batch_size, dim).cuda()

        # materialize optimizer states
        out = model(batch)
        out.sum().backward()
        optim.step()
        optim.zero_grad()

        ddp_out = ddp_model(batch)
        ddp_out.sum().backward()
        ddp_optim.step()
        ddp_optim.zero_grad()

        self.assertEqual(ddp_out, out)
        return model, optim, ddp_model, ddp_optim

    def _test_tran_step_with_ddp_without_optim_step(
        self, train_step, num_iters, batch_size, layers, dim
    ):
        def _ddp_train_step(model, optim, batch):
            model(batch).sum().backward()
            return [p.grad for p in model.parameters()]

        model, optim, ddp_model, ddp_optim = self._init(batch_size, layers, dim)
        for _ in range(num_iters):
            batch = torch.randn(batch_size, dim).cuda()
            out = train_step(model, optim, batch)
            ddp_out = _ddp_train_step(ddp_model, ddp_optim, batch)
            for g1, g2 in zip(out, ddp_out):
                self.assertEqual(g1 / self.world_size, g2)
            optim.zero_grad()
            ddp_optim.zero_grad()

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_basic_transformation(self):
        batch_size = 100
        layers = 10
        dim = 100
        num_iters = 5

        @compile(gm_transformation=GraphModuleTransformation(num_iters=num_iters))
        def train_step(model, optim, batch):
            model(batch).sum().backward()
            return [p.grad for p in model.parameters()]

        self._test_tran_step_with_ddp_without_optim_step(
            train_step, num_iters, batch_size, layers, dim
        )

    @skip_if_lt_x_gpu(2)
    @with_comms
    def test_inductor(self):
        batch_size = 100
        layers = 10
        dim = 100
        num_iters = 5

        @compile(
            gm_transformation=GraphModuleTransformation(
                num_iters=num_iters, enable_inductor=True
            )
        )
        def train_step(model, optim, batch):
            model(batch).sum().backward()
            return [p.grad for p in model.parameters()]

        model, optim, _, _ = self._init(batch_size, layers, dim)
        for _ in range(num_iters):
            batch = torch.randn(batch_size, dim).cuda()
            out = train_step(model, optim, batch)

            # TODO: using _test_tran_step_with_ddp_without_optim_step()
            # does not work because some mismatched accuracy. Need to
            # debug the issue.


if __name__ == "__main__":
    run_tests()
