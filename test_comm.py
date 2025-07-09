import copy
import gc
import os
from datetime import timedelta

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


class TestFullyShardMemory(FSDPTest):
    # @property
    # def world_size(self) -> int:
    #    return 8
    def test_fully_shard_del_memory(self):
        dim = 1024
        model = nn.Sequential(*[MLP(dim) for _ in range(3)])
        ref_model_fp8 = copy.deepcopy(model)
        ref_model_bf16 = copy.deepcopy(model).to(torch.bfloat16)
        mp_policy = MixedPrecisionPolicy(param_dtype=torch.bfloat16)
        model = fully_shard(model, mp_policy=mp_policy)
        optim = torch.optim.Adam(model.parameters())
        inp = torch.randn((dim, dim), device=torch.device("cuda"), dtype=torch.bfloat16)
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
        ref_model_bf16 = fully_shard(ref_model_bf16)
        ref_optim_bf16 = torch.optim.Adam(ref_model_bf16.parameters(), lr=1e-2)
        with torch.profiler.profile(
            activities=[
                # orch.profiler.ProfilerActivity.CPU,
                torch.profiler.ProfilerActivity.CUDA,
            ],
            record_shapes=True,
            on_trace_ready=torch.profiler.tensorboard_trace_handler("execution_trace"),
        ) as prof:
            for round in range(20):
                if torch.distributed.get_rank() == 0:
                    print("dist.all_gather_into_tensor(bf16), with mp_policy(bf16)")
                optim.zero_grad()
                loss = model(inp).sum()
                loss.backward()
                optim.step()

                if torch.distributed.get_rank() == 0:
                    print("dist.all_gather_into_tensor(fp8)  ")
                ref_optim_fp8.zero_grad()
                ref_loss = ref_model_fp8(inp).sum()
                ref_loss.backward()
                ref_optim_fp8.step()


if __name__ == "__main__":
    init_process_group(backend="nccl", timeout=timedelta(hours=24))
    run_tests()
