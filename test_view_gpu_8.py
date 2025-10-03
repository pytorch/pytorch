# torchrun --nproc_per_node=8 test_view_gpu_8.py

import torch
import torch.nn as nn
from torch.distributed.tensor import (
    distribute_module,
    distribute_tensor,
    DTensor,
    init_device_mesh,
    Replicate,
    Shard,
)
from torch.testing._internal.common_utils import run_tests
from torch.testing._internal.distributed._tensor.common_dtensor import (
    DTensorTestBase,
    with_comms,
)

import os
import torch
import torch.distributed as dist

def main():
    dist.init_process_group(backend="nccl")
    gpu_id = int(os.environ["LOCAL_RANK"])
    device = f"cuda:{gpu_id}"
    torch.cuda.set_device(device)
    mesh_shape = (1, 2, 2, 2)
    mesh_dim_names = ("dp_replicate", "dp_shard", "cp", "tp")
    mesh_4d = init_device_mesh(
        "cuda", mesh_shape, mesh_dim_names=mesh_dim_names,
    )
    input_dim, output_dim = 4, 6
    global_linear = nn.Linear(input_dim, output_dim, bias=True)
    weight_sharding = (Replicate(), Replicate(), Replicate(), Shard(1))
    bias_sharding = (Replicate(), Replicate(), Replicate(), Replicate())
    def _partition_linear_fn(name, module, device_mesh):
        module.register_parameter(
            "weight",
            nn.Parameter(
                distribute_tensor(
                    module.weight,
                    device_mesh,
                    weight_sharding,
                )
            ),
        )
        if getattr(module, "bias", None) is not None:
            # The Linear module has bias
            module.register_parameter(
                "bias",
                nn.Parameter(
                    distribute_tensor(
                        module.bias,
                        device_mesh,
                        bias_sharding,
                    )
                ),
            )
    distributed_linear = distribute_module(global_linear, mesh_4d, partition_fn=_partition_linear_fn)
    batch_size, seq_len = 8, 8
    global_input = torch.randn([batch_size, seq_len, input_dim])
    input_sharding = (Replicate(), Shard(0), Shard(1), Shard(2))
    distributed_input = distribute_tensor(global_input, mesh_4d, input_sharding)
    print("results: ", distributed_linear(distributed_input))


if __name__ == "__main__":
    main()
