# Owner(s): ["oncall: distributed"]

import torch
import torch.distributed as dist

from torch.distributed._tensor import DTensor
from torch.distributed._shard.sharded_tensor import ShardedTensor
from torch.distributed._tensor.placement_types import Shard
from torch.distributed.distributed_c10d import _get_default_group
from torch.distributed.fsdp._shard_utils import (
    _create_chunk_dtensor,
    _create_chunk_sharded_tensor,
    _gather_dtensor_state_dict,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_fsdp import FSDPTest
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

import torch

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType
import torch.distributed.checkpoint as dist_cp
import torch.distributed as dist

from torch.distributed.checkpoint.default_planner import (
    DefaultSavePlanner,
    DefaultLoadPlanner,
)

from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)
from torch.testing._internal.common_distributed import skip_if_lt_x_gpu
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed.checkpoint_utils import with_temp_dir

from torch.distributed.fsdp.api import (
    BackwardPrefetch,
    CPUOffload,
    FullOptimStateDictConfig,
    FullStateDictConfig,
    LocalOptimStateDictConfig,
    LocalStateDictConfig,
    MixedPrecision,
    OptimStateDictConfig,
    ShardedOptimStateDictConfig,
    ShardedStateDictConfig,
    ShardingStrategy,
    StateDictConfig,
    StateDictSettings,
    StateDictType,
)


def p0(line):
    if dist.get_rank() == 0:
        print(line)

class TestShardUtilsDistributedDTensor(DTensorTestBase):
    @property
    def world_size(self):
        return 4
    

    @with_comms
    @with_temp_dir
    def test_dtensor_sharded_state_dict(self):
        CHECKPOINT_DIR = self.temp_dir

        model = FSDP(torch.nn.Linear(4, 4, device="meta"))
        optim = torch.optim.Adam(model.parameters(), lr=0.1)
        model(torch.rand(4, 4, device=dist.get_rank())).sum().backward()
        optim.step()


        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(use_dtensor=True, offload_to_cpu=False),
            optim_state_dict_config=ShardedOptimStateDictConfig(use_dtensor=True, offload_to_cpu=False)
        )
        param_state_dict = model.state_dict()
        optim_state_dict = FSDP.optim_state_dict(model, optim)
        # print(f"param_state_dict:{param_state_dict}")
        p0(f"optim_state_dict:{optim_state_dict}")
        for k,v in optim_state_dict['state']['weight'].items():
            if isinstance(v, DTensor):
                print(f"k:{k}, v:{v.to_local()}")

        FSDP.set_state_dict_type(
            model,
            StateDictType.SHARDED_STATE_DICT,
            state_dict_config=ShardedStateDictConfig(use_dtensor=True, offload_to_cpu=False),
            optim_state_dict_config=ShardedOptimStateDictConfig(use_dtensor=False, offload_to_cpu=False)
        )
        optim_state_dict = FSDP.optim_state_dict(model, optim)
        p0(f"optim_state_dict:{optim_state_dict}")
        for k,v in optim_state_dict['state']['weight'].items():
            if isinstance(v, ShardedTensor):
                print(f"k:{k}, v:{v.local_shards()[0].tensor}")

        



        # dist_cp.save_state_dict(
        #     state_dict=state_dict,
        #     storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
        #     planner=DefaultSavePlanner(),
        # )

        # model_2 = FSDP(torch.nn.Linear(8, 8, device="meta"))

        # with FSDP.summon_full_params(model):
        #     with FSDP.summon_full_params(model_2):
        #         self.assertNotEqual(model.weight, model_2.weight)
        #         self.assertNotEqual(model.bias, model_2.bias)

        # # now load the model and ensure the values are the same
        # with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
        #     state_dict = {
        #         "model": model_2.state_dict(),
        #     }

        #     dist_cp.load_state_dict(
        #         state_dict=state_dict,
        #         storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
        #         planner=DefaultLoadPlanner(),
        #     )
        #     model_2.load_state_dict(state_dict["model"])

        # with FSDP.summon_full_params(model):
        #     with FSDP.summon_full_params(model_2):
        #         self.assertEqual(model.weight, model_2.weight)
        #         self.assertEqual(model.bias, model_2.bias)


if __name__ == "__main__":
    run_tests()
