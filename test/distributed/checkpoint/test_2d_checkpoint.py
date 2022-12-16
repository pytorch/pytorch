# Owner(s): ["oncall: distributed"]

from typing import Any

import torch
import torch.distributed as dist
import torch.distributed.checkpoint as dist_cp

import torch.distributed.distributed_c10d as distributed_c10d
import torch.nn.functional as F
from torch.distributed._shard.sharded_tensor.api import ShardedTensor
from torch.distributed._tensor import DeviceMesh, DTensor as DT, Replicate
from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
from torch.distributed.tensor.parallel import (
    PairwiseParallel,
    parallelize_module,
)
from torch.distributed.tensor.parallel.fsdp import is_available
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu


from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

# Tensor-Parallel degree
TP_DEGREE = 2
LR = 3e-5


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


def _distribute_and_fsdp_wrap_module(
    module, module_shard, mesh_2d, fsdp_pg, use_orig_params, fsdp_nested
):
    if module_shard:
        module = parallelize_module(
            module, mesh_2d, PairwiseParallel(), tp_mesh_dim=1
        )
    pg = fsdp_pg if module_shard else distributed_c10d._get_default_group()

    if fsdp_nested:
        module.net1 = FSDP(
            module.net1, process_group=pg, use_orig_params=use_orig_params
        )
        module.net2 = FSDP(
            module.net2, process_group=pg, use_orig_params=use_orig_params
        )
    return FSDP(module, process_group=pg, use_orig_params=use_orig_params)


def init_model(
    model_parallel_size=TP_DEGREE, use_orig_params=False, fsdp_nested=False
):
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

    # Create Input
    model = _distribute_and_fsdp_wrap_module(
        model, True, twod_mesh, fsdp_pg, use_orig_params, fsdp_nested
    )

    return model, fsdp_pg


class Test2dCheckpoint(DTensorTestBase):



    @with_comms
    @skip_if_lt_x_gpu(4)
    @with_temp_dir
    def test_2d_fsdp_integration_functionality(self) -> None:
        if not is_available():
            self.skipTest("FSDP 2d parallel integration not available")

        CHECKPOINT_DIR = self.temp_dir

        model = init_model()[0]
        optim = torch.optim.Adam(model.parameters(), lr=0.0001)

        # Create Input
        input_seed = self.rank
        torch.manual_seed(input_seed + 1)
        input = torch.rand(4, 5).cuda(self.rank)

        model(input).sum().backward()
        optim.step()

        with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
            state_dict = {
                "model": model.state_dict(),
            }

            dist_cp.save_state_dict(
                state_dict=state_dict,
                storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
                planner=DefaultSavePlanner(
                    flatten_state_dict=True,
                    flatten_sharded_tensors=True,
                    dedup_replicated_tensors=True,
                ),
            )

        # now load the model and ensure the values are the same
        model_2 = init_model()[0]
        optim_2 = torch.optim.Adam(model_2.parameters(), lr=0.1)

        input_seed = self.rank
        torch.manual_seed(input_seed + 10)
        input = torch.rand(4, 5).cuda(self.rank)

        model_2(input).sum().backward()
        optim_2.step()

        with FSDP.summon_full_params(model):
            if dist.get_rank() == 0:
                print(f"model: {model.net1.weight}")

        with FSDP.summon_full_params(model_2):
            if dist.get_rank() == 0:
                print(f"model: {model_2.net1.weight}")

        with FSDP.summon_full_params(model):
            with FSDP.summon_full_params(model_2):
                for n_p1, n_p2 in zip(model.named_parameters(), model_2.named_parameters()):
                    print(f"rank{dist.get_rank()}, n_p1: {n_p1}, n_p2:{n_p2}")
                    self.assertNotEqual(n_p1, n_p2)
                    break


if __name__ == "__main__":
    run_tests()
