import copy
import gc
import os
from datetime import timedelta

from logging import getLogger

import torch
import torch.nn as nn
from torch.distributed import init_process_group
from torch.distributed.checkpoint.state_dict import (
    get_model_state_dict,
    StateDictOptions,
)
from torch.distributed.fsdp import fully_shard, MixedPrecisionPolicy
from torch.testing._internal.common_fsdp import FSDPTest, MLP
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    ModelArgs,
    Transformer,
    TransformerBlock,
)
from torchao.float8.config import CastConfig, Float8LinearConfig, ScalingType
from torchao.float8.float8_linear_utils import convert_to_float8_training

logger = getLogger()


def rank0_print(*args, **kwargs):
    if torch.distributed.get_rank() == 0:  # default PG
        logger.info(*args, **kwargs)


class TestFullyShardAllGatherComm(FSDPTest):
    def test_all_gather_comm(self):
        dim = 1024
        model = nn.Sequential(*[MLP(dim) for _ in range(3)])
        ref_model_fp8 = copy.deepcopy(model)
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        model = fully_shard(model, mp_policy=mp_policy)
        optim = torch.optim.Adam(model.parameters())
        enable_fsdp_float8_all_gather = True
        scaling_type_weight = ScalingType.DYNAMIC
        float8_linear_config = Float8LinearConfig(
            enable_fsdp_float8_all_gather=enable_fsdp_float8_all_gather,
            cast_config_weight=CastConfig(scaling_type=scaling_type_weight),
        )
        ref_model_fp8 = convert_to_float8_training(
            ref_model_fp8,
            config=float8_linear_config,
        )
        ref_model_fp8 = fully_shard(ref_model_fp8)
        ref_optim_fp8 = torch.optim.Adam(ref_model_fp8.parameters(), lr=1e-2)
        with torch.profiler.profile(
            activities=[
                torch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            schedule=torch.profiler.schedule(wait=10, warmup=10, active=30, repeat=1),
            on_trace_ready=torch.profiler.tensorboard_trace_handler("execution_trace"),
        ) as prof:
            for _ in range(30):
                inp = torch.randn(
                    (dim, dim), device=torch.device("cuda"), dtype=torch.bfloat16
                )
                rank0_print("dist.all_gather_into_tensor(bf16)")
                optim.zero_grad()
                loss = model(inp).sum()
                loss.backward()
                optim.step()

                rank0_print("dist.all_gather_into_tensor(fp8)")
                ref_optim_fp8.zero_grad()
                ref_loss = ref_model_fp8(inp).sum()
                ref_loss.backward()
                ref_optim_fp8.step()
                prof.step()


if __name__ == "__main__":
    init_process_group(backend="nccl", timeout=timedelta(hours=24))
    run_tests()
