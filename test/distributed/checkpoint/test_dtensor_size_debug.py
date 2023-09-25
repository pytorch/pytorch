# Owner(s): ["oncall: distributed"]
import torch
import torch.distributed.checkpoint as dist_cp
from torch.distributed._shard.sharded_tensor import ShardedTensor

from torch.distributed._tensor import DTensor, init_device_mesh, Replicate, distribute_tensor, Shard
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp._shard_utils import _all_gather_sharded_tensor
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType

from torch.distributed.tensor.parallel import PairwiseParallel, parallelize_module
from torch.testing._internal.common_utils import run_tests

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    MLPModule,
    skip_if_lt_x_gpu,
    with_comms,
)
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir


class MLPModule(torch.nn.Module):
    def __init__(self, device):
        super().__init__()
        torch.manual_seed(5)
        self.net1 = torch.nn.Linear(2, 10, device=device)

    def forward(self, x):
        return self.net1(x)

    def reset_parameters(self):
        self.net1.reset_parameters()


class DTensorSizeDebug(DTensorTestBase):
    @with_comms
    @skip_if_lt_x_gpu(2)
    @with_temp_dir
    def test_fsdp_to_tp(self):
        CHECKPOINT_DIR = self.temp_dir

        mesh_shape = (self.world_size,)
        device_mesh = init_device_mesh(self.device_type, mesh_shape)
        model = MLPModule(self.device_type).cuda(self.rank)
        # create a FSDP wrapped model
        fsdp_model = FSDP(model, use_orig_params=True, device_mesh=device_mesh)

        FSDP.set_state_dict_type(
            fsdp_model,
            StateDictType.SHARDED_STATE_DICT,
        )
        fsdp_state_dict = fsdp_model.state_dict()
        # the weight shape should be [10, 2] but it was [12, 2] for rank 0, rank1, and rank2
        # and [4, 2] for rank[3].
        # The size is wrong because the internal implmentation of dtensor state_dict is using DTensor.from_local API.
        # https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/_shard_utils.py#L197
        for k,v in fsdp_state_dict.items():
            if "weight" in k:
                print(f"rank:{self.rank}, state_dict weight:{v}, weight.shape:{v.shape}\n")


        # the size of the dtensor is correct when we call distribute_tensor
        global_tensor = torch.randn(10, 2)
        dtensor = distribute_tensor(global_tensor, device_mesh, (Shard(0),))
        print(f"rank:{self.rank}, dtensor:{dtensor}, dtensor.shape:{dtensor.shape}")


        # seems the issue is that when we use the from_local API, then the size of DTensor is incorrect.
        if self.rank != 3:
            from_local_dtensor = DTensor.from_local(torch.randn(3, 2), device_mesh, (Shard(0),))
        else:
            from_local_dtensor = DTensor.from_local(torch.randn(1, 2), device_mesh, (Shard(0),))
        print(f"rank:{self.rank}, from_local_dtensor:{from_local_dtensor}, from_local_dtensor.shape:{from_local_dtensor.shape}")


if __name__ == "__main__":
    run_tests()
