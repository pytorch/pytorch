# Owner(s): ["oncall: distributed"]

import functools
from typing import Any


import torch
import torch.nn.functional as F
import torch.distributed as dist
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed._tensor import (
    distribute_tensor,
    DeviceMesh,
    DTensor as DT,
    Shard,
    Replicate,
)

import torch.distributed.distributed_c10d as distributed_c10d

from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.distributed._tensor.parallel.fsdp import is_available

from torch.testing._internal.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# Tensor-Parallel degree
TP_DEGREE = 2
LR = 3e-5

OPS_NOT_SHARD = [
    "net3.weight",
    "net3.bias",
]

SHARD_PARAMS = [
    "net1.weight",
    "net1.bias",
    "net2.weight",
]


class SimpleModel(torch.nn.Module):
    def __init__(self):
        super(SimpleModel, self).__init__()
        self.net1 = torch.nn.Linear(5, 8)
        self.relu = torch.nn.ReLU()
        self.net2 = torch.nn.Linear(8, 4)
        self.net3 = torch.nn.Linear(4, 12)

    def forward(self, x):
        x = F.relu(self.net1(x))
        x = F.relu(self.net2(x))
        x = F.relu(self.net3(x))
        return x


def _aggregate_local_tensor(module: torch.nn.Module) -> torch.nn.Module:
    def hook_func(_module, _input, output):
        if isinstance(output, DT):
            replica_placement = [Replicate()]
            return output.redistribute(
                output.device_mesh, replica_placement
            ).to_local()

    module.register_forward_hook(hook_func)
    return module


def _replicate_input_tensor(
    module: torch.nn.Module, device_mesh, replica_placement
) -> torch.nn.Module:
    def hook_func(_, input):
        if not isinstance(input[0], DT):
            return DT.from_local(
                input[0], device_mesh, replica_placement, run_check=False
            )

    module.register_forward_pre_hook(hook_func)
    return module


def _gradient_hook(param, grad):
    param._local_tensor.grad = grad._local_tensor


def shard_module(m, pg):
    start_idx = distributed_c10d.get_global_rank(pg, 0)
    device_mesh = DeviceMesh(
        "cuda", list(range(start_idx, start_idx + pg.size())), dim_groups=[pg]
    )
    col_wise_sharding = [Shard(0)]
    row_wise_sharding = [Shard(1)]
    replicate = [Replicate()]
    m.net1.weight = torch.nn.Parameter(
        distribute_tensor(m.net1.weight, device_mesh, col_wise_sharding),
    )
    m.net2.weight = torch.nn.Parameter(
        distribute_tensor(m.net2.weight, device_mesh, row_wise_sharding)
    )
    m.net1.bias = torch.nn.Parameter(
        distribute_tensor(m.net1.bias, device_mesh, col_wise_sharding)
    )
    m.net2.bias = torch.nn.Parameter(
        distribute_tensor(m.net2.bias, device_mesh, replicate)
    )
    m = _replicate_input_tensor(m, device_mesh, replicate)
    m.net2 = _aggregate_local_tensor(m.net2)
    m.net1.weight.register_hook(
        functools.partial(_gradient_hook, m.net1.weight)
    )
    m.net2.weight.register_hook(
        functools.partial(_gradient_hook, m.net2.weight)
    )
    m.net1.bias.register_hook(functools.partial(_gradient_hook, m.net1.bias))
    m.net2.bias.register_hook(functools.partial(_gradient_hook, m.net2.bias))


def _shard_wrap_module(module, module_shard, fsdp_wrap, tp_pg, fsdp_pg):
    if module_shard:
        # Fetch the module sharding planner.
        shard_module(module, tp_pg)

    if fsdp_wrap and module_shard:
        return FSDP(module, process_group=fsdp_pg)
    if fsdp_wrap:
        return FSDP(module, process_group=distributed_c10d._get_default_group())
    return module


def init_model(model_parallel_size=TP_DEGREE):
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    world_size = dist.get_world_size()

    model = SimpleModel().cuda(rank)

    # 2-D mesh is [dp, tp]
    twod_mesh = DeviceMesh(
        device_type="cuda",
        mesh=torch.arange(0, world_size).view(model_parallel_size, -1),
    )

    fsdp_pg = twod_mesh.get_dim_groups()[0]
    tp_pg = twod_mesh.get_dim_groups()[1]

    # Create Input
    model = _shard_wrap_module(model, True, True, tp_pg, fsdp_pg)
    return model, tp_pg, fsdp_pg


def is_nested_tensor(val: Any) -> bool:
    if isinstance(val, ShardedTensor):
        if len(val.local_shards()) == 0:
            return False
        if isinstance(val.local_shards()[0].tensor, ShardedTensor):
            return True
        if isinstance(val.local_shards()[0].tensor, DT):
            raise ValueError("Cannot handle DT nested insided ST")
    # Safety valve for when this eventually happen
    elif isinstance(val, DT) and isinstance(
        val._local_tensor, (DT, ShardedTensor)
    ):
        raise ValueError("Cannot handle nested DT")
    return False


class Test2dParallelIntegration(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(4)
    def test_2d_fsdp_integration(self) -> None:
        if not is_available():
            self.skipTest("FSDP 2d parallel integration not available")

        model_tp, tp_pg, dp_pg = init_model()

        with FSDP.state_dict_type(model_tp, StateDictType.SHARDED_STATE_DICT):
            state_dict = model_tp.state_dict()
            # TODO once 2D is out, validate the nesting
            self.assertTrue(is_nested_tensor(state_dict["net1.weight"]))
            self.assertFalse(is_nested_tensor(state_dict["net3.bias"]))

        optim = torch.optim.Adam(model_tp.parameters(), lr=0.0001)

        # Create Input
        input_seed = self.rank
        torch.manual_seed(input_seed + 1)
        input = torch.rand(4, 5).cuda(self.rank)

        model_tp(input).sum().backward()
        optim.step()

        optim_state = FSDP.sharded_optim_state_dict(model_tp, optim)
        # TODO once 2D is out, validate the nesting
        self.assertTrue(
            is_nested_tensor(optim_state["state"]["net1.weight"]["exp_avg"])
        )
        self.assertFalse(
            is_nested_tensor(optim_state["state"]["net3.bias"]["exp_avg"])
        )


if __name__ == "__main__":
    run_tests()
