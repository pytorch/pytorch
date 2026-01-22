"""
Meta Device Pattern Demo
Run with: torchrun --nproc_per_node=4 meta_device_demo.py
"""
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.tensor import DTensor, Shard, Replicate

def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if torch.cuda.is_available():
        return torch.cuda.memory_allocated() / 1024 / 1024
    return 0

def main():
    dist.init_process_group("nccl")
    rank = dist.get_rank()

    # Create a 2D mesh: (fsdp=2, tp=2) on 4 GPUs
    mesh = init_device_mesh("cuda", (2, 2), mesh_dim_names=("fsdp", "tp"))

    print(f"\n{'='*60}")
    print(f"RANK {rank}: Starting demo")
    print(f"{'='*60}")

    # =========================================================================
    # STEP 1: Build on meta device (NO memory allocated)
    # =========================================================================
    print(f"\n[STEP 1] Building on meta device...")

    with torch.device("meta"):
        # Imagine this is a 1B parameter weight matrix
        weight = torch.empty(32000, 8192, dtype=torch.float32)  # ~1GB if real

    print(f"  weight.shape  = {weight.shape}")
    print(f"  weight.device = {weight.device}")
    print(f"  weight.dtype  = {weight.dtype}")
    print(f"  GPU memory    = {get_memory_mb():.2f} MB  ← ZERO!")

    # Try to access data - will fail!
    try:
        _ = weight[0, 0].item()
    except NotImplementedError as e:
        print(f"  Accessing data fails: {type(e).__name__}")

    # =========================================================================
    # STEP 2: Define sharding placement (still no memory)
    # =========================================================================
    print(f"\n[STEP 2] Setting up sharding spec...")

    # Shard dim 0 on TP, dim 1 on FSDP
    # weight[32000, 8192] → each GPU gets [16000, 4096]
    placements = [Shard(1), Shard(0)]  # (fsdp shards dim1, tp shards dim0)

    print(f"  Placements: fsdp=Shard(1), tp=Shard(0)")
    print(f"  Full shape: [32000, 8192]")
    print(f"  Each GPU gets: [32000/2, 8192/2] = [16000, 4096]")
    print(f"  GPU memory = {get_memory_mb():.2f} MB  ← STILL ZERO!")

    # =========================================================================
    # STEP 3: Materialize as DTensor (NOW memory is allocated)
    # =========================================================================
    print(f"\n[STEP 3] Materializing as DTensor...")

    # Create local shard with actual data
    local_shape = (32000 // 2, 8192 // 2)  # [16000, 4096]
    local_tensor = torch.randn(local_shape, device="cuda", dtype=torch.float32)

    # Wrap as DTensor
    dt = DTensor.from_local(local_tensor, mesh, placements)

    print(f"  dt.shape (global) = {dt.shape}")
    print(f"  dt._local_tensor.shape = {dt._local_tensor.shape}")
    print(f"  dt.device = {dt.device}")
    print(f"  GPU memory = {get_memory_mb():.2f} MB  ← Only local shard!")

    # =========================================================================
    # STEP 4: Show what each rank has
    # =========================================================================
    print(f"\n[STEP 4] What each rank actually stores:")

    # Get mesh coordinates
    fsdp_idx = mesh.get_local_rank("fsdp")
    tp_idx = mesh.get_local_rank("tp")

    print(f"  Mesh coords: (fsdp={fsdp_idx}, tp={tp_idx})")
    print(f"  I own rows [{tp_idx*16000}:{(tp_idx+1)*16000}]")
    print(f"  I own cols [{fsdp_idx*4096}:{(fsdp_idx+1)*4096}]")

    # =========================================================================
    # STEP 5: Demonstrate all-gather (what FSDP does in forward)
    # =========================================================================
    print(f"\n[STEP 5] Simulating FSDP all-gather for forward pass...")

    # Redistribute to gather along FSDP dimension
    gathered = dt.redistribute(mesh, [Replicate(), Shard(0)])

    print(f"  After all-gather on FSDP dim:")
    print(f"  gathered.shape (global) = {gathered.shape}")
    print(f"  gathered._local_tensor.shape = {gathered._local_tensor.shape}")
    print(f"  GPU memory = {get_memory_mb():.2f} MB  ← Increased for gathered params")

    # =========================================================================
    # SUMMARY
    # =========================================================================
    print(f"\n{'='*60}")
    print(f"RANK {rank} SUMMARY:")
    print(f"  Full tensor:  32000 x 8192 = 262M params = 1048 MB")
    print(f"  My shard:     16000 x 4096 = 65.5M params = 262 MB")
    print(f"  Memory saved: 75% per GPU!")
    print(f"{'='*60}\n")

    dist.destroy_process_group()

if __name__ == "__main__":
    main()
