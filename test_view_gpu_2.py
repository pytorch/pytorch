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
from torch.nn import functional as F
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
    batch_size, seq_len, dim = 2, 4, 2
    global_inps = torch.arange(batch_size * seq_len * dim, device="cuda").float().view(batch_size, seq_len, dim)
    inps = distribute_tensor(global_inps, mesh, (Shard(1), ))
    global_weight = torch.eye(dim, device="cuda")
    weight = distribute_tensor(global_weight, mesh, (Shard(0), ))
    out = F.linear(inps, weight)
    print(f"rank: {torch.distributed.get_rank()} out: {out}")

if __name__ == "__main__":
    main()
