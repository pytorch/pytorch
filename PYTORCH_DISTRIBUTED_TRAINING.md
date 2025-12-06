# PyTorch Distributed Training Guide

## Overview

Distributed training is essential for scaling neural network training across multiple GPUs and machines. This guide covers PyTorch's distributed training capabilities.

## Prerequisites

- PyTorch 1.9+ (recommended 2.0+)
- NCCL 2.x or higher for GPU-to-GPU communication
- Python 3.8+
- MPI for multi-node training (optional)

## Distributed Data Parallel (DDP)

### Basic Setup

```python
import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.distributed as dist

def setup(rank, world_size):
    dist.init_process_group(
        backend='nccl',
        rank=rank,
        world_size=world_size
    )

def cleanup():
    dist.destroy_process_group()
```

### Training Loop

```python
def train(rank, world_size, model, train_loader):
    setup(rank, world_size)
    
    model = model.to(rank)
    ddp_model = DDP(model, device_ids=[rank])
    
    optimizer = torch.optim.Adam(ddp_model.parameters())
    criterion = nn.CrossEntropyLoss()
    
    for epoch in range(num_epochs):
        train_loader.sampler.set_epoch(epoch)
        for data, target in train_loader:
            data, target = data.to(rank), target.to(rank)
            
            optimizer.zero_grad()
            output = ddp_model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
    
    cleanup()
```

## Key Concepts

1. **Process Group**: Coordinates all processes
2. **Rank**: Unique identifier for each process
3. **World Size**: Total number of processes
4. **Backend**: Communication protocol (nccl, gloo, mpi)

## Launch Command

```bash
torchrun --nproc_per_node=<num_gpus> train.py
```

## Performance Optimization Tips

- Use gradient accumulation for larger effective batch sizes
- Increase batch size to maximize GPU utilization
- Use mixed precision training (torch.cuda.amp)
- Profile communication overhead with torch.distributed.profiler

## Common Issues

- **Port conflicts**: Ensure MASTER_PORT is available
- **Timeout**: Increase timeout for slow networks
- **Deadlocks**: Verify all processes reach collective operations
