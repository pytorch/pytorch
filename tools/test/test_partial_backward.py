from torch.utils._debug_mode import DebugMode
import torch
import torch.distributed as dist
from torch.distributed._tensor import DeviceMesh, DTensor, Replicate, Partial
import os

def setup_distributed():
    """Initializes the distributed process group."""
    if not dist.is_available() or not torch.cuda.is_available() or torch.cuda.device_count() < 2:
        print("This example requires NCCL backend and 2+ GPUs.")
        return None

    if not dist.is_initialized():
        # Use environment variables set by torchrun
        dist.init_process_group("nccl")

    world_size = dist.get_world_size()
    rank = dist.get_rank()
    torch.cuda.set_device(rank)
    
    # Create a 1D device mesh
    mesh = DeviceMesh("cuda", torch.arange(world_size))
    return mesh, rank, world_size

def print_on_rank0(msg):
    """Helper to print only on rank 0."""
    if dist.get_rank() == 0:
        print(msg)

def run_example():
    mesh, rank, world_size = setup_distributed()
    if mesh is None:
        return

    if world_size != 2:
        print_on_rank0("This example is hard-coded for 2 ranks. Please run with --nproc-per-node=2")
        dist.destroy_process_group()
        return

    # --- 1. Create a Partial Operand 'p' ---
    # We must construct it from local data.
    # Let's define the global tensor 'p' to be [4.0, 6.0]
    # To do this, we'll make:
    #   - Rank 0's local data: [1.0, 2.0]
    #   - Rank 1's local data: [3.0, 4.0]
    # When summed (all-reduced), [1, 2] + [3, 4] = [4, 6]

    if rank == 0:
        local_p_data = torch.tensor([1.0, 2.0], device="cuda", requires_grad=True)
    else: # rank 1
        local_p_data = torch.tensor([3.0, 4.0], device="cuda", requires_grad=True)

    # Tell DTensor to treat this local data as a [Partial] sum
    p_placements = [Partial()]
    p = DTensor.from_local(local_p_data, mesh, p_placements)
    p.retain_grad()
    p.variable = "p"

    # --- 2. Create a Replicated Operand 'r' ---
    # Global 'r' will be [2.0, 2.0]
    r_placements = [Replicate()]
    r =DTensor.from_local(torch.tensor([2.0, 2.0], requires_grad=True), mesh, r_placements)
    r.retain_grad()
    r.variable = "r"

    print_on_rank0("\n--- Initial Setup ---")
    print(f"[Rank {rank}] Local 'p' data: {p.to_local()}")
    print(f"[Rank {rank}] Local 'r' data: {r.to_local()}")


    # --- 3. Operation involving the Partial Operand 'p' ---
    # output = p * r
    # Sharding: Partial * Replicate -> Partial
    # Rank 0: [1, 2] * [2, 2] = [2, 4] (Partial)
    # Rank 1: [3, 4] * [2, 2] = [6, 8] (Partial)
    # Global 'output' = [2, 4] + [6, 8] = [8, 12]
    output = p * r
    
    print_on_rank0("\n--- Forward Operation ---")
    print(f"[Rank {rank}] 'output' (Partial) local data: {output.to_local()}")

    # --- 4. Loss Calculation ---
    # .sum() on a Partial tensor will trigger an all-reduce
    # Global 'output' is [8, 12]
    # Global 'loss' = 8 + 12 = 20.0
    loss = output.sum()
    
    print_on_rank0(f"\n--- Loss Calculation ---")
    print_on_rank0(f"Global Loss (replicated): {loss.to_local()} placements: {loss.placements}") # loss is a scalar, replicated on all ranks

    # --- 5. Gradient Calculation ---
    print_on_rank0("\n--- Gradient Calculation (loss.backward()) ---")
    with DebugMode(record_torchfunction=True, record_faketensor=True, record_realtensor=True, record_tensor_attributes=["variable"], record_nn_module=True) as debug_mode:
        loss.backward()
    print_on_rank0(f"-------- DebugMode: loss.backward() -------- \n {debug_mode.debug_string()}")

    # --- 6. Inspect Gradients ---
    
    # Verification math:
    # loss = sum(p * r)
    #
    # d(loss)/dp = r
    #   The global gradient for 'p' is 'r', which is [2.0, 2.0].
    #   Since 'p' was [Partial], 'p.grad' will be [Replicate].
    #   Both ranks should have [2.0, 2.0] as the local gradient.
    #
    # d(loss)/dr = p
    #   The global gradient for 'r' is 'p', which is [4.0, 6.0].
    #   Since 'r' was [Replicate], 'r.grad' will be [Partial].
    #   Rank 0's local grad: [1.0, 2.0] (which is p's local data)
    #   Rank 1's local grad: [3.0, 4.0] (which is p's local data)
    #   Sum of local grads = [4.0, 6.0] (Correct!)

    print_on_rank0("\n--- Inspecting Gradients ---")
    
    # Check p.grad
    print(f"[Rank {rank}] p.grad sharding: {p.grad.placements}")
    print(f"[Rank {rank}] p.grad local data: {p.grad.to_local()}  <-- Expected [Replicate] of [2., 2.]")

    # Check r.grad
    print(f"[Rank {rank}] r.grad sharding: {r.grad.placements}")
    print(f"[Rank {rank}] r.grad local data: {r.grad.to_local()}  <-- Expected [Partial] (Rank0=[1., 2.], Rank1=[3., 4.])")

    dist.destroy_process_group()

if __name__ == "__main__":
    run_example()
