import torch
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import StateDictType
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp.wrap import (
    enable_wrap,
    transformer_auto_wrap_policy,
    wrap,
)
import os


from torch.distributed._shard.checkpoint import (
    FileSystemWriter,
    SavePlan,
    save_state_dict,
)

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


shard_group = 4
replicate_group = 2
for r in range(8):
    a = r % shard_group
    b = r // shard_group
    print(
        f"r:{r}, a:{a}, b:{b}, a%replicate_group.size():{a % replicate_group}, do_save:{a % replicate_group == b}"
    )


# save on ranks such that every replicate group is covered
def do_save(
    self, rank, local_rank, shard_group, replicate_group, is_dist=False
):
    if not is_dist:
        return rank == local_rank
    else:
        # shard_group size -- 4, replicate_group size -- 2
        a = rank % shard_group.size()
        b = rank // shard_group.size()
        return True if a % replicate_group.size() == b else False


def write_dcp(self, state_dict, process_group, save_name, rank):
    os.makedirs(save_name, exist_ok=True)
    writer = FileSystemWriter(save_name, single_file_per_rank=True)

    if state_dict is not None:
        print(f"Writing state dict on rank={rank}")
        save_state_dict(
            state_dict=state_dict,
            storage_writer=writer,
            process_group=process_group,
            # planner=DefaultSavePlanner(),
        )
        print(f"Finished writing state dict on rank={rank}")


def save_dcp(
    self,
    model_state,
    optimizer_state,
    process_group=None,
    dist_hsdp_ckp_save=False,
    replicate_group=None,
    **kwargs,
):
    if self.do_save(
        rank,
        local_rank,
        shard_group=process_group,
        replicate_group=replicate_group,
        is_dist=dist_hsdp_ckp_save,
    ):
        state_dict = {
            "model_state": model_state,
            "optimizer_state": optimizer_state,
        }
        self.write_dcp(state_dict, process_group, save_name, rank)


model_sharding_strategy = ShardingStrategy.HYBRID_SHARD
mp_policy = None
wrapping_policy = None




model = TinyModel()

model = FSDP(
    model,
    auto_wrap_policy=wrapping_policy,
    mixed_precision=mp_policy,
    sharding_strategy=model_sharding_strategy,
    device_id=local_rank,
    limit_all_gathers=True,
    use_orig_params=True,
)

optimizer = torch.optim.AdamW(
    model.parameters(), weight_decay=0.1, lr=0.001, betas=(0.9, 0.999)
)

shard_group = model.process_group
replicate_group = model._inter_node_state.process_group

# with FSDP.state_dict_type(model, StateDictType.SHARDED_STATE_DICT):
#     model_state = model.state_dict()
#     optim_state = FSDP.sharded_optim_state_dict(
#         model, optimizer, group=shard_group
#     )

# save_dcp(
#     model_state,
#     optim_state,
#     process_group=shard_group,
#     dist_hsdp_ckp_save=True,
#     replicate_group=replicate_group,
# )
