# Copyright (c) Meta Platforms, Inc. and affiliates

"""
The following example demonstrates how to use Pytorch Distributed Checkpoint
to save a FSDP model. This is the current recommended way to checkpoint FSDP.
torch.save() and torch.load() is not recommended when checkpointing sharded models.
"""

import os
import shutil

import torch
import torch.distributed as dist
from torch.distributed.distributed_c10d import _get_default_group
import torch.distributed.checkpoint as dist_cp
import torch.multiprocessing as mp

from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp.fully_sharded_data_parallel import StateDictType, ShardingStrategy
from torch.distributed.checkpoint.optimizer import (
    load_sharded_optimizer_state_dict,
)

CHECKPOINT_DIR = f"/scratch/{os.environ['LOGNAME']}/checkpoint"


class TinyModel(torch.nn.Module):
    def __init__(self):
        super(TinyModel, self).__init__()

        self.linear1 = torch.nn.Linear(100, 200)
        self.activation = torch.nn.ReLU()
        self.linear2 = torch.nn.Linear(200, 10)
        self.softmax = torch.nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.softmax(x)
        return x




def main(rank, world_size):
    # Initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank % torch.cuda.device_count()) 

    default_pg = dist.distributed_c10d._get_default_group()
    print(f"rank:{dist.get_rank()}, default_pg:{default_pg}")

    if dist.get_rank() % 2 ==0:
        save_dist_ckpt(state_dict, shard_group_pg)

    # # Model creation
    # model = TinyModel()
    # model_sharding_strategy = ShardingStrategy.HYBRID_SHARD
    # mp_policy = None
    # wrapping_policy = None
    # model = FSDP(
    #     model,
    #     auto_wrap_policy=wrapping_policy,
    #     mixed_precision=mp_policy,
    #     sharding_strategy=model_sharding_strategy,
    #     device_id=rank,
    #     limit_all_gathers=True,
    #     use_orig_params=True,
    # )

    # optimizer = torch.optim.AdamW(
    #     model.parameters(), weight_decay=0.1, lr=0.001, betas=(0.9, 0.999)
    # )

    # # Save the model to CHECKPOINT_DIR
    # with FSDP.state_dict_type(model_1, StateDictType.SHARDED_STATE_DICT):
    #     state_dict = {
    #         "model": model_1.state_dict(),
    #         "optim": FSDP.optim_state_dict(model_1, optim_1),
    #     }

    #     dist_cp.save_state_dict(
    #         state_dict=state_dict,
    #         storage_writer=dist_cp.FileSystemWriter(CHECKPOINT_DIR),
    #     )

    # # Create a second model
    # model_2, optim_2 = init_model()

    # # Print the model parameters for both models.
    # # Before loading, the parameters should be different.
    # print_params("Before loading", model_1, model_2, optim_1, optim_2)

    # # Load model_2 with parameters saved in CHECKPOINT_DIR
    # with FSDP.state_dict_type(model_2, StateDictType.SHARDED_STATE_DICT):
    #     state_dict = {
    #         "model": model_2.state_dict(),
    #         # cannot load the optimizer state_dict together with the model state_dict
    #     }

    #     dist_cp.load_state_dict(
    #         state_dict=state_dict,
    #         storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
    #     )
    #     model_2.load_state_dict(state_dict["model"])

    #     optim_state = load_sharded_optimizer_state_dict(
    #         model_state_dict=state_dict["model"],
    #         optimizer_key="optim",
    #         storage_reader=dist_cp.FileSystemReader(CHECKPOINT_DIR),
    #     )

    #     flattened_osd = FSDP.optim_state_dict_to_load(
    #         model_2, optim_2, optim_state["optim"]
    #     )
    #     optim_2.load_state_dict(flattened_osd)

    # # Print the model parameters for both models.
    # # After loading, the parameters should be the same.
    # print_params("After loading", model_1, model_2, optim_1, optim_2)

    # # Shut down world pg
    # dist.destroy_process_group()


if __name__ == "__main__":
    
    world_size = int(os.getenv("WORLD_SIZE", 8))
    rank = int(os.getenv("RANK", -1))
    # parser.add_argument('--world_size', type=int, default=int(os.getenv("WORLD_SIZE", 4)))
    # parser.add_argument('--rank', type=int, default=int(os.getenv("RANK", -1)))

    # # Set up world pg
    # os.environ["MASTER_ADDR"] = "localhost"
    # os.environ["MASTER_PORT"] = "12355"

    print(f"Running hsdp checkpoint example on {world_size} devices.")
    main(rank, world_size)
    # shutil.rmtree(CHECKPOINT_DIR, ignore_errors=True)
    # mp.spawn(
    #     run_hsdp_checkpoint_example,
    #     args=(world_size,),
    #     nprocs=world_size,
    #     join=True,
    # )
