import torch
import torch.distributed as dist
import torch.nn as nn
from torch.distributed.tensor.parallel import parallelize_module, ColwiseParallel, RowwiseParallel
from torch.distributed._tensor import init_device_mesh
from torch._decomp import get_decompositions

# Initialize distributed
if not dist.is_initialized():
    dist.init_process_group(backend="nccl")

device_mesh = init_device_mesh("cuda", (dist.get_world_size(),))
rank = dist.get_rank()

print(f"[Rank {rank}] PyTorch version: {torch.__version__}")

import torch.utils._pytree
import torch.distributed.tensor._dtensor_spec
torch.utils._pytree.register_constant(
    torch.distributed.tensor._dtensor_spec.DTensorSpec
)


class ToyModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.in_proj = nn.Linear(10, 320)
        self.out_proj = nn.Linear(320, 160)

    def forward(self, x):
        return self.out_proj(torch.relu(self.in_proj(x)))


model = ToyModel().to("cuda")
parallelize_module(model.in_proj, device_mesh, ColwiseParallel())
parallelize_module(model.out_proj, device_mesh, RowwiseParallel())

inp = torch.rand(2, 10, device="cuda")

print(f"[Rank {rank}] Model parallelized successfully")

# Export the model - this works now with DTensorSpec registration
print(f"[Rank {rank}] Exporting model...")
exported_program = torch.export.export(model, (inp,), strict=False)
print(f"[Rank {rank}] Export succeeded")

# Now try to run decompositions - THIS FAILS
print(f"\n[Rank {rank}] Attempting to run decompositions on exported DTensor model...")

try:
    # Get standard decompositions
    decomp_table = get_decompositions([
        torch.ops.aten.embedding_dense_backward,
        torch.ops.aten.native_layer_norm_backward,
        torch.ops.aten.slice_backward,
        torch.ops.aten.select_backward,
        torch.ops.aten.norm.ScalarOpt_dim,
        torch.ops.aten.native_group_norm_backward,
        torch.ops.aten.upsample_bilinear2d.vec,
        torch.ops.aten.split.Tensor,
        torch.ops.aten.split_with_sizes,
    ])
    
    decomposed = exported_program.run_decompositions(decomp_table)
    print(decomposed)
    
except AssertionError as e:
    print(f"[Rank {rank}]  Decomposition failed with AssertionError:")
    print(f"    {e}")

except Exception as e:
    print(f"[Rank {rank}] Decomposition failed: {type(e).__name__}: {e}")

dist.destroy_process_group()
