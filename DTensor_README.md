# PyTorch DTensor Demonstration

This directory contains a comprehensive demonstration of PyTorch DTensor capabilities, showcasing distributed tensor operations and parallelism strategies.

## Files

- **`dtensor_demo.py`** - Main demonstration script showing DTensor features
- **`run_dtensor_demo.sh`** - Launch helper script with various configuration options
- **`DTensor_README.md`** - This documentation file

## Quick Start

The fastest way to run the demo:

```bash
# Recommended: 4 processes to enable all features including 2D mesh
torchrun --nproc_per_node=4 --nnodes=1 dtensor_demo.py

# Minimum: 2 processes for basic distributed operations
torchrun --nproc_per_node=2 --nnodes=1 dtensor_demo.py
```

## Features Demonstrated

### 1. Device Mesh Setup
- **1D Device Mesh**: Linear arrangement of devices
- **2D Device Mesh**: Grid arrangement for advanced sharding patterns
- **Device Coordination**: How ranks map to mesh coordinates

### 2. DTensor Creation Methods
- **`distribute_tensor()`**: Convert existing tensors to DTensors
- **`DTensor.from_local()`**: Create DTensors from local tensor shards
- **Factory Functions**: Direct creation with `zeros()`, `ones()`, etc.

### 3. Placement Strategies
- **`Shard(dim)`**: Partition tensor along specified dimension
- **`Replicate()`**: Full copies across all devices
- **`Partial(reduce_op)`**: Pending reduction operations

### 4. DTensor Operations
- **Element-wise Operations**: Addition, multiplication, etc.
- **Matrix Operations**: Matrix multiplication with distributed tensors
- **Reduction Operations**: Sum, mean across distributed dimensions
- **Automatic Layout Propagation**: How operations affect tensor layouts

### 5. Redistribution
- **Layout Changes**: Convert between Shard, Replicate, and Partial
- **Collective Communications**: Automatic allgather, allreduce, reduce_scatter
- **Multi-dimensional Resharding**: Change sharding dimensions

### 6. 2D Sharding (4+ processes required)
- **Multi-dimensional Partitioning**: Shard along multiple tensor dimensions
- **Mixed Layouts**: Combine sharding and replication across mesh dimensions
- **Advanced Parallelism**: Patterns used in large-scale models

### 7. Distributed Module Parallelism
- **Tensor Parallelism**: Distribute model parameters across devices
- **Custom Partitioning**: User-defined sharding strategies
- **Gradient Handling**: Automatic gradient distribution and collection

### 8. Collective Operations
- **AllReduce**: Sum partial results across devices
- **AllGather**: Collect sharded tensors into full tensors
- **Reduce-Scatter**: Combine and redistribute operations

## Launch Options

### Single Node Configurations

```bash
# Basic demo (2 processes)
torchrun --nproc_per_node=2 --nnodes=1 dtensor_demo.py

# Full demo with 2D mesh (4 processes) - Recommended
torchrun --nproc_per_node=4 --nnodes=1 dtensor_demo.py

# Large scale demo (8 processes)
torchrun --nproc_per_node=8 --nnodes=1 dtensor_demo.py
```

### CPU-Only Mode

```bash
CUDA_VISIBLE_DEVICES= torchrun --nproc_per_node=2 --nnodes=1 dtensor_demo.py
```

### With Debug Logging

```bash
TORCH_LOGS=+dtensor torchrun --nproc_per_node=4 --nnodes=1 dtensor_demo.py
```

### Multi-Node Setup

```bash
# Node 0
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=0 \
         --master_addr=<node0_ip> --master_port=29500 dtensor_demo.py

# Node 1
torchrun --nproc_per_node=4 --nnodes=2 --node_rank=1 \
         --master_addr=<node0_ip> --master_port=29500 dtensor_demo.py
```

## Prerequisites

- **PyTorch**: Built with distributed support
- **CUDA**: Optional, demo works on CPU
- **Multiple Processes**: At least 2 for distributed operations
- **Network**: For multi-node setups

## Key Concepts Explained

### Device Mesh
A logical grid of devices that defines how tensors are distributed. Each device has coordinates in this mesh.

### Placements
Describe how a tensor is laid out across the device mesh:
- **Shard(0)**: Split tensor along dimension 0
- **Replicate()**: Full copy on each device
- **Partial(sum)**: Each device has partial result, needs reduction

### Redistribution
The process of changing tensor layout, which may involve:
- **AllGather**: Shard → Replicate
- **Chunk**: Replicate → Shard
- **AllReduce**: Partial → Replicate
- **Reduce-Scatter**: Partial → Shard
- **All-to-All**: Shard(dim1) → Shard(dim2)

### Automatic Operation Handling
DTensor automatically:
1. Validates operation compatibility
2. Performs necessary redistributions
3. Executes operations on local shards
4. Determines output layout

## Expected Output

The demo will show output from each rank including:
- Device mesh information
- Tensor shapes (global vs local)
- Placement strategies
- Operation results
- Communication patterns
- Model parameter distribution

## Troubleshooting

### Common Issues
1. **NCCL Errors**: Check CUDA installation and device visibility
2. **Hanging**: Ensure all processes are launched consistently
3. **Memory Issues**: Reduce tensor sizes for limited GPU memory
4. **Import Errors**: Verify PyTorch distributed installation

### Debug Tips
- Use `TORCH_LOGS=+dtensor` for detailed logging
- Start with CPU-only mode to isolate GPU issues
- Check process coordination with simple collective operations

## Learning Path

1. **Start Simple**: Run with 2 processes to understand basics
2. **Explore Placements**: Try different Shard/Replicate combinations
3. **Scale Up**: Use 4+ processes for 2D mesh features
4. **Advanced Patterns**: Experiment with custom partitioning functions
5. **Real Applications**: Apply patterns to actual model training

This demonstration provides a foundation for understanding distributed tensor operations in PyTorch, applicable to large-scale model training and inference scenarios.
