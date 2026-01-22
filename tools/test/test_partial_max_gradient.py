"""
Example to understand how Partial("max") gradients work in DTensor.

Run with: torchrun --nproc_per_node=2 tools/test/test_partial_max_gradient.py

KEY FINDING: The gradient flows to ALL ranks, not just the rank with the max value!
This is because the all-reduce(max) operation doesn't track which rank "won".
"""

import torch
import torch.distributed as dist
from torch.distributed.tensor import DTensor, Replicate, Shard, Partial, distribute_tensor, init_device_mesh


def main():
    dist.init_process_group(backend="gloo")
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    mesh = init_device_mesh("cpu", (world_size,))

    print(f"\n{'='*60}")
    print(f"Rank {rank}: Partial('max') gradient example")
    print(f"{'='*60}\n")

    # ==========================================================================
    # Example 1: Partial("max") gradient - WHO GETS THE GRADIENT?
    # ==========================================================================
    print(f"Rank {rank}: === Example 1: Partial('max') gradient ===")

    # Rank 0 has 5.0, Rank 1 has 3.0
    # The max is 5.0 (on rank 0)
    if rank == 0:
        local_val = torch.tensor([5.0], requires_grad=True)
    else:
        local_val = torch.tensor([3.0], requires_grad=True)

    print(f"Rank {rank}: local_val = {local_val.item()}")

    # Create Partial("max") DTensor - each rank has a candidate
    x_partial = DTensor.from_local(local_val, mesh, [Partial("max")])

    # Reduce to Replicate - this does all-reduce(max)
    x_full = x_partial.redistribute(mesh, [Replicate()])
    print(f"Rank {rank}: after all-reduce(max) = {x_full.to_local().item()}")

    # Backward
    loss = x_full.to_local().sum()
    loss.backward()

    print(f"Rank {rank}: gradient = {local_val.grad.item()}")
    print(f"Rank {rank}: ^ BOTH ranks get grad=1.0, even though only rank 0 had the max!")

    # ==========================================================================
    # Example 2: Compare with Partial("sum")
    # ==========================================================================
    print(f"\nRank {rank}: === Example 2: Compare with Partial('sum') ===")

    if rank == 0:
        local_sum = torch.tensor([2.0], requires_grad=True)
    else:
        local_sum = torch.tensor([3.0], requires_grad=True)

    print(f"Rank {rank}: local_val = {local_sum.item()}")

    x_partial_sum = DTensor.from_local(local_sum, mesh, [Partial("sum")])
    x_full_sum = x_partial_sum.redistribute(mesh, [Replicate()])
    print(f"Rank {rank}: after all-reduce(sum) = {x_full_sum.to_local().item()}")

    loss_sum = x_full_sum.to_local().sum()
    loss_sum.backward()

    print(f"Rank {rank}: gradient = {local_sum.grad.item()}")
    print(f"Rank {rank}: ^ Both ranks get grad=1.0 (correct for sum: d(a+b)/da = 1)")

    # ==========================================================================
    # Example 3: What SHOULD happen for true max gradient?
    # ==========================================================================
    print(f"\nRank {rank}: === Example 3: What SHOULD happen for true max? ===")
    print(f"Rank {rank}: For y = max(x0=5, x1=3) = 5:")
    print(f"Rank {rank}:   dy/dx0 = 1 (x0 is the max)")
    print(f"Rank {rank}:   dy/dx1 = 0 (x1 is not the max)")
    print(f"Rank {rank}: ")
    print(f"Rank {rank}: But DTensor gives grad=1 to BOTH ranks!")
    print(f"Rank {rank}: This is because all-reduce(max) doesn't track the winner.")

    # ==========================================================================
    # Example 4: Local max (within a rank) DOES work correctly
    # ==========================================================================
    print(f"\nRank {rank}: === Example 4: Local max works correctly ===")

    local_tensor = torch.tensor([1.0, 5.0, 3.0], requires_grad=True)
    local_max = local_tensor.max()
    local_max.backward()

    print(f"Rank {rank}: tensor = [1, 5, 3], max = {local_max.item()}")
    print(f"Rank {rank}: gradient = {local_tensor.grad.tolist()}")
    print(f"Rank {rank}: ^ Only index 1 (the max) gets gradient=1, others get 0")

    print(f"\nRank {rank}: CONCLUSION: Partial('max') autograd is broken for true max gradient!")
    print(f"Rank {rank}: It works for the forward pass but not backward.")

    dist.destroy_process_group()


if __name__ == "__main__":
    main()
