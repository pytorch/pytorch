#!/usr/bin/env python3
"""
PyTorch DTensor Demonstration Script

This script showcases the key features of PyTorch DTensor including:
1. Device mesh setup
2. Creating distributed tensors with different placement strategies
3. DTensor operations and redistribution
4. Distributed module parallelism
5. Collective operations

Usage:
    torchrun --nproc_per_node=4 --nnodes=1 dtensor_demo.py
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributed import init_process_group, destroy_process_group
from torch.distributed.device_mesh import init_device_mesh, DeviceMesh
from torch.distributed.tensor import (
    DTensor,
    distribute_tensor,
    distribute_module,
    Shard,
    Replicate,
    Partial,
)


def setup_distributed():
    """Initialize distributed process group."""
    if not torch.distributed.is_available():
        raise RuntimeError("Distributed not available")
    
    # Initialize process group
    init_process_group(backend="nccl" if torch.cuda.is_available() else "gloo")
    
    # Get rank and world size
    rank = int(os.environ["RANK"])
    world_size = int(os.environ["WORLD_SIZE"])
    
    print(f"Rank {rank}/{world_size} initialized")
    return rank, world_size


def demonstrate_device_mesh(device_type="cuda"):
    """Demonstrate DeviceMesh creation and usage."""
    print(f"\n=== Device Mesh Demo ===")
    
    rank, world_size = setup_distributed()
    
    # Create 1D device mesh
    mesh_1d = init_device_mesh(device_type, (world_size,))
    print(f"1D Mesh shape: {mesh_1d.mesh.shape}")
    print(f"Current coordinate: {mesh_1d.get_coordinate()}")
    
    # Create 2D device mesh (if we have enough devices)
    if world_size >= 4:
        mesh_2d = init_device_mesh(device_type, (2, world_size // 2))
        print(f"2D Mesh shape: {mesh_2d.mesh.shape}")
        print(f"Current 2D coordinate: {mesh_2d.get_coordinate()}")
        return mesh_1d, mesh_2d
    
    return mesh_1d, None


def demonstrate_dtensor_creation(mesh, device_type="cuda"):
    """Demonstrate different ways to create DTensors."""
    print(f"\n=== DTensor Creation Demo ===")
    
    # Method 1: distribute_tensor - convert existing tensor to DTensor
    global_tensor = torch.randn(8, 4, device=device_type)
    print(f"Original tensor shape: {global_tensor.shape}")
    
    # Shard along dimension 0
    sharded_tensor = distribute_tensor(global_tensor, mesh, [Shard(0)])
    print(f"Sharded DTensor global shape: {sharded_tensor.shape}")
    print(f"Sharded DTensor local shape: {sharded_tensor.to_local().shape}")
    print(f"Sharded DTensor placements: {sharded_tensor.placements}")
    
    # Replicate across all devices
    replicated_tensor = distribute_tensor(global_tensor, mesh, [Replicate()])
    print(f"Replicated DTensor global shape: {replicated_tensor.shape}")
    print(f"Replicated DTensor local shape: {replicated_tensor.to_local().shape}")
    print(f"Replicated DTensor placements: {replicated_tensor.placements}")
    
    # Method 2: from_local - create DTensor from local tensors
    local_tensor = torch.randn(2, 4, device=device_type)
    dtensor_from_local = DTensor.from_local(local_tensor, mesh, [Shard(0)])
    print(f"DTensor from local global shape: {dtensor_from_local.shape}")
    
    # Method 3: Direct DTensor factory functions
    dtensor_zeros = torch.distributed.tensor.zeros(
        8, 4, device_mesh=mesh, placements=[Shard(0)]
    )
    print(f"DTensor zeros shape: {dtensor_zeros.shape}")
    
    dtensor_ones = torch.distributed.tensor.ones(
        8, 4, device_mesh=mesh, placements=[Replicate()]
    )
    print(f"DTensor ones shape: {dtensor_ones.shape}")
    
    return sharded_tensor, replicated_tensor


def demonstrate_dtensor_operations(dt1, dt2):
    """Demonstrate DTensor operations."""
    print(f"\n=== DTensor Operations Demo ===")
    
    # Element-wise operations
    result_add = dt1 + dt2
    print(f"Addition result shape: {result_add.shape}")
    print(f"Addition result placements: {result_add.placements}")
    
    result_mul = dt1 * dt2
    print(f"Multiplication result shape: {result_mul.shape}")
    
    # Matrix operations
    if dt1.shape[1] == dt2.shape[0]:  # Can do matmul
        result_mm = torch.mm(dt1, dt2.t())
        print(f"Matrix multiply result shape: {result_mm.shape}")
    
    # Reduction operations
    result_sum = dt1.sum()
    print(f"Sum result: {result_sum}")
    print(f"Sum result type: {type(result_sum)}")
    
    result_mean = dt1.mean(dim=0)
    print(f"Mean along dim 0 shape: {result_mean.shape}")
    
    return result_add


def demonstrate_redistribution(dtensor):
    """Demonstrate DTensor redistribution."""
    print(f"\n=== DTensor Redistribution Demo ===")
    
    print(f"Original placement: {dtensor.placements}")
    print(f"Original local shape: {dtensor.to_local().shape}")
    
    # Redistribute from Shard to Replicate
    if dtensor.placements[0].is_shard():
        replicated = dtensor.redistribute(placements=[Replicate()])
        print(f"After redistribute to Replicate:")
        print(f"  Placement: {replicated.placements}")
        print(f"  Local shape: {replicated.to_local().shape}")
        
        # Redistribute back to Shard on different dimension
        if replicated.ndim > 1:
            resharded = replicated.redistribute(placements=[Shard(1)])
            print(f"After redistribute to Shard(1):")
            print(f"  Placement: {resharded.placements}")
            print(f"  Local shape: {resharded.to_local().shape}")
            
        return replicated
    
    return dtensor


def demonstrate_2d_sharding(mesh_2d, device_type="cuda"):
    """Demonstrate 2D tensor sharding if 2D mesh is available."""
    if mesh_2d is None:
        return None
        
    print(f"\n=== 2D DTensor Sharding Demo ===")
    
    # Create a larger tensor for 2D sharding
    global_tensor = torch.randn(8, 8, device=device_type)
    
    # Shard along both dimensions
    dtensor_2d = distribute_tensor(global_tensor, mesh_2d, [Shard(0), Shard(1)])
    print(f"2D sharded tensor global shape: {dtensor_2d.shape}")
    print(f"2D sharded tensor local shape: {dtensor_2d.to_local().shape}")
    print(f"2D sharded tensor placements: {dtensor_2d.placements}")
    
    # Mixed sharding: shard one dimension, replicate another
    dtensor_mixed = distribute_tensor(global_tensor, mesh_2d, [Shard(0), Replicate()])
    print(f"Mixed sharding global shape: {dtensor_mixed.shape}")
    print(f"Mixed sharding local shape: {dtensor_mixed.to_local().shape}")
    print(f"Mixed sharding placements: {dtensor_mixed.placements}")
    
    return dtensor_2d


class SimpleModel(nn.Module):
    """Simple neural network for distributed training demo."""
    
    def __init__(self, input_size=10, hidden_size=20, output_size=5):
        super().__init__()
        self.linear1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.linear2 = nn.Linear(hidden_size, output_size)
        
    def forward(self, x):
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return x


def demonstrate_distributed_module(mesh, device_type="cuda"):
    """Demonstrate distributed module parallelism."""
    print(f"\n=== Distributed Module Demo ===")
    
    # Create model
    model = SimpleModel().to(device_type)
    
    # Define partition function for tensor parallelism
    def partition_fn(name, module, mesh):
        if name == "linear1":
            # Shard weight along output dimension (column-wise)
            if hasattr(module, 'weight'):
                module.weight = nn.Parameter(
                    distribute_tensor(module.weight.data, mesh, [Shard(0)])
                )
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = nn.Parameter(
                    distribute_tensor(module.bias.data, mesh, [Shard(0)])
                )
        elif name == "linear2":
            # Shard weight along input dimension (row-wise)  
            if hasattr(module, 'weight'):
                module.weight = nn.Parameter(
                    distribute_tensor(module.weight.data, mesh, [Shard(1)])
                )
            # Keep bias replicated for row-wise sharding
            if hasattr(module, 'bias') and module.bias is not None:
                module.bias = nn.Parameter(
                    distribute_tensor(module.bias.data, mesh, [Replicate()])
                )
    
    # Distribute the module
    distributed_model = distribute_module(model, mesh, partition_fn)
    
    print(f"Distributed model created")
    for name, param in distributed_model.named_parameters():
        print(f"  {name}: shape={param.shape}, placements={param.placements}")
    
    # Test forward pass with distributed input
    batch_size = 4
    input_tensor = torch.randn(batch_size, 10, device=device_type)
    input_dtensor = distribute_tensor(input_tensor, mesh, [Replicate()])
    
    output = distributed_model(input_dtensor)
    print(f"Forward pass output shape: {output.shape}")
    print(f"Forward pass output placements: {output.placements}")
    
    return distributed_model


def demonstrate_gradient_flow(model, mesh, device_type="cuda"):
    """Demonstrate gradient computation with DTensors."""
    print(f"\n=== Gradient Flow Demo ===")
    
    # Create input and target
    batch_size = 4
    input_tensor = torch.randn(batch_size, 10, device=device_type)
    target_tensor = torch.randn(batch_size, 5, device=device_type)
    
    # Convert to DTensors
    input_dt = distribute_tensor(input_tensor, mesh, [Replicate()])
    target_dt = distribute_tensor(target_tensor, mesh, [Replicate()])
    
    # Forward pass
    output_dt = model(input_dt)
    
    # Compute loss
    loss = F.mse_loss(output_dt, target_dt)
    print(f"Loss: {loss}")
    print(f"Loss type: {type(loss)}")
    
    # Backward pass
    loss.backward()
    
    # Check gradients
    print("Gradients computed:")
    for name, param in model.named_parameters():
        if param.grad is not None:
            print(f"  {name}: grad_shape={param.grad.shape}, grad_placements={param.grad.placements}")
        else:
            print(f"  {name}: No gradient")


def demonstrate_collective_operations(mesh, device_type="cuda"):
    """Demonstrate collective operations through DTensor."""
    print(f"\n=== Collective Operations Demo ===")
    
    rank = torch.distributed.get_rank()
    
    # Create rank-specific data
    local_data = torch.full((2, 3), float(rank), device=device_type)
    print(f"Rank {rank} local data:\n{local_data}")
    
    # Create DTensor with Partial placement (pending reduction)
    partial_tensor = DTensor.from_local(local_data, mesh, [Partial(reduce_op="sum")])
    print(f"Partial tensor placement: {partial_tensor.placements}")
    
    # Redistribute to replicated (triggers allreduce)
    reduced_tensor = partial_tensor.redistribute(placements=[Replicate()])
    print(f"After allreduce (sum):\n{reduced_tensor.to_local()}")
    
    # Demonstrate full_tensor (allgather)
    sharded_data = torch.full((1, 3), float(rank), device=device_type)
    sharded_tensor = DTensor.from_local(sharded_data, mesh, [Shard(0)])
    
    full_tensor = sharded_tensor.full_tensor()
    print(f"Full tensor after allgather:\n{full_tensor}")


def main():
    """Main demonstration function."""
    print("PyTorch DTensor Demonstration")
    print("=" * 40)
    
    device_type = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device_type}")
    
    try:
        # Setup
        mesh_1d, mesh_2d = demonstrate_device_mesh(device_type)
        
        # Basic DTensor operations
        dt_shard, dt_replicate = demonstrate_dtensor_creation(mesh_1d, device_type)
        result_dt = demonstrate_dtensor_operations(dt_shard, dt_replicate)
        demonstrate_redistribution(dt_shard)
        
        # 2D operations if available
        demonstrate_2d_sharding(mesh_2d, device_type)
        
        # Module parallelism
        dist_model = demonstrate_distributed_module(mesh_1d, device_type)
        demonstrate_gradient_flow(dist_model, mesh_1d, device_type)
        
        # Collective operations
        demonstrate_collective_operations(mesh_1d, device_type)
        
    except Exception as e:
        print(f"Error in demonstration: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        # Cleanup
        if torch.distributed.is_initialized():
            destroy_process_group()
        print("\nDemonstration completed!")


if __name__ == "__main__":
    main()