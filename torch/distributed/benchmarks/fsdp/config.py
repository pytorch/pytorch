from dataclasses import dataclass

import torch
from torch.distributed.fsdp import BackwardPrefetch, MixedPrecision, ShardingStrategy


@dataclass
class pytorch_fsdp_config:
    # FULL_SHARD, SHARD_GRAD_OP, NO_SHARD, HYBRID_SHARD, _HYBRID_SHARD_ZERO2
    sharding_strategy = ShardingStrategy.FULL_SHARD
    backward_prefetch = BackwardPrefetch.BACKWARD_PRE
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16, reduce_dtype=torch.bfloat16
    )
    limit_all_gathers = True
    use_orig_params = False

    def __str__(self):
        return f"pytorch_fsdp_config(\n" \
               f"    sharding_strategy: {self.sharding_strategy}\n" \
               f"    backward_prefetch: {self.backward_prefetch}\n" \
               f"    mixed_precision: {self.mixed_precision}\n" \
               f"    limit_all_gathers: {self.limit_all_gathers}\n" \
               f"    use_orig_params: {self.use_orig_params}\n" \
               f" )\n" 


@dataclass
class train_default_config:
    # seed
    seed: int = 2023
    # how many mini batches to time with
    total_steps_to_warm_up: int = 3
    total_steps_to_run: int = 10

    run_profiler: bool = False
    profile_folder: str = "/tmp/fsdp_profile_tracing"

    # activation checkpointing
    activation_checkpointing: bool = True

    # init the GPT large model using meta device
    use_meta_device_init = False
