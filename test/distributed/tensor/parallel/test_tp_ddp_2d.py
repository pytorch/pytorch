# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist

import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.distributed._tensor import DeviceMesh
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu

from torch.testing._internal.common_utils import run_tests

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# Tensor-Parallel degree
TP_DEGREE = 2
LR = 3e-5


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.net1 = torch.nn.Linear(5, 8)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(8, 4)
        self.net3 = torch.nn.Linear(4, 12)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x


def init_model(model_parallel_size=TP_DEGREE):
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()

    model = SimpleModel().cuda(rank)
    twod_model = model.copy().detach()
    model = DDP(model, process_group=dp_pg)

    # 2-D mesh is [dp, tp]
    twod_mesh = DeviceMesh(
        device_type="cuda",
        mesh=torch.arange(0, world_size).view(model_parallel_size, -1),
    )

    dp_pg = twod_mesh.get_dim_groups()[0]

    # Create Input
    twod_model = parallelize_module(twod_model, twod_mesh, PairwiseParallel(), tp_mesh_dim=1)
    twod_model = DDP(twod_model, process_group=dp_pg)

    return model, twod_model, dp_pg


class Test2dParallelIntegration(DTensorTestBase):
    def _check_module(self, m1, m2, check_grad=False, rank0_only_params=None):
        rank0_only_params = [] if rank0_only_params is None else rank0_only_params
        named_parameters = dict(m1.named_parameters())
        for name, param_m2 in m2.named_parameters():
            if self.rank != 0 and name in rank0_only_params:
                continue
            self.assertTrue(name in named_parameters)
            param_m1 = named_parameters[name]
            if check_grad:
                param_m2 = param_m2.grad
                param_m1 = param_m1.grad
            if isinstance(param_m2, DTensor):
                replicate = [Replicate()]
                param_m2 = param_m2.redistribute(
                    device_mesh=param_m2.device_mesh, placements=replicate
                ).to_local()
            self.assertEqual(param_m2, param_m1)

    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_ddp_integration_functionality(self) -> None:
        model, twod_model, dp_pg = init_model()
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)
        twod_optim = torch.optim.Adam(twod_model.parameters(), lr=0.0001)

        # Create Input
        input_seed = dist.get_rank(dp_pg)
        torch.manual_seed(input_seed + 1)
        input = torch.rand(4, 5).cuda(self.rank)

        output = model(input)
        twod_output = twod_model(input)
        self.assertEqual(output, twod_output)
        
        output.sum().backward()
        twod_output.sum().backward()
        optim.step()
        twod_optim.step()

        torch.manual_seed(input_seed + 1004)
        input = torch.rand(16, 5).cuda(self.rank)

        output = model(input)
        twod_output = twod_model(input)
        self.assertEqual(output, twod_output)


if __name__ == "__main__":
    run_tests()
