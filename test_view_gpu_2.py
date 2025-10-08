# torchrun --nproc_per_node=2 test_view_gpu_2.py

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
    mesh_shape = (2, )
    mesh = init_device_mesh("cuda", mesh_shape)
    input_dim, batch_size, seq_len = 2, 4, 2
    global_input = torch.arange(input_dim * batch_size * seq_len).float().view(input_dim, batch_size, seq_len)
    input_sharding = (Shard(1), )
    distributed_input = distribute_tensor(global_input, mesh, input_sharding)
    shard = distributed_input.view(batch_size * seq_len, input_dim)
    print(f"rank: {torch.distributed.get_rank()} shard: {shard}\n", flush=True)


if __name__ == "__main__":
    main()
