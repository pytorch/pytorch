# Autotuning for Asymmetric Collective Operations

**Status**: Future Work / Advanced Use Case  
**Date**: 2024-11  
**Audience**: PyTorch Distributed Team

---

## Table of Contents

1. [Overview](#overview)
2. [What are Asymmetric Operations?](#what-are-asymmetric-operations)
3. [Problem Statement](#problem-statement)
4. [Solution Design](#solution-design)
5. [Implementation Examples](#implementation-examples)
6. [Comparison with Heterogeneous Configs](#comparison-with-heterogeneous-configs)

---

## Overview

This document describes how to handle **asymmetric collective operations** where different ranks execute **completely different operations** (not just different configs of the same operation).

**Important Distinction**:
- **Heterogeneous Configs** (previous doc): Same operation, different configs available
- **Asymmetric Operations** (this doc): Different operations on different ranks

**Example Scenarios**:
- Point-to-point: Rank 0 does `send()`, Rank 1 does `recv()`
- Pipeline Parallelism: Different ranks execute different model stages
- Leader/Worker: Rank 0 coordinates, other ranks compute
- Custom distributed algorithms: Each rank has unique role

---

## What are Asymmetric Operations?

### Scenario 1: Point-to-Point Communication (Send/Recv)

```python
# Rank 0 executes SEND operation
@torch.library.custom_op("mylib::send_data", mutates_args=())
def send_data(x: torch.Tensor, dst: int) -> torch.Tensor:
    torch.ops._c10d_functional.send(x, dst, "default")
    return x

# Rank 1 executes RECV operation (completely different!)
@torch.library.custom_op("mylib::recv_data", mutates_args=())
def recv_data(shape: tuple, src: int) -> torch.Tensor:
    result = torch.empty(shape, device='cuda')
    torch.ops._c10d_functional.recv(result, src, "default")
    return result

# Different ranks run DIFFERENT functions!
if rank == 0:
    result = send_data(tensor, dst=1)  # Send operation
elif rank == 1:
    result = recv_data(shape, src=0)   # Recv operation
```

### Scenario 2: Pipeline Parallelism

```python
# Rank 0: First stage
class Stage0(nn.Module):
    def forward(self, x):
        x = self.layers_0_to_10(x)
        send_to_next_stage(x, dst=1)  # Send to Rank 1
        return x

# Rank 1: Middle stage
class Stage1(nn.Module):
    def forward(self):
        x = recv_from_prev_stage(src=0)  # Recv from Rank 0
        x = self.layers_11_to_20(x)
        send_to_next_stage(x, dst=2)     # Send to Rank 2
        return x

# Rank 2: Last stage
class Stage2(nn.Module):
    def forward(self):
        x = recv_from_prev_stage(src=1)  # Recv from Rank 1
        output = self.layers_21_to_30(x)
        return output  # No send needed

# Each rank runs COMPLETELY different code path!
```

### Scenario 3: Hierarchical All-Reduce (Leader-based)

```python
# Ranks 0, 4, 8, 12 are LEADERS - execute leader operations
@torch.library.custom_op("mylib::leader_reduce", mutates_args=())
def leader_reduce(x: torch.Tensor) -> torch.Tensor:
    # Step 1: Local reduce within group
    local_result = intra_group_reduce(x)
    # Step 2: Leader-only inter-group reduce
    leader_result = inter_leader_reduce(local_result)
    # Step 3: Broadcast back to group
    broadcast_to_group(leader_result)
    return leader_result

# Ranks 1-3, 5-7, 9-11, 13-15 are WORKERS - execute worker operations
@torch.library.custom_op("mylib::worker_reduce", mutates_args=())
def worker_reduce(x: torch.Tensor) -> torch.Tensor:
    # Step 1: Local reduce within group (same as leader)
    local_result = intra_group_reduce(x)
    # Step 2: WAIT (don't participate in inter-leader reduce)
    # Step 3: Receive broadcast from leader
    final_result = receive_from_leader()
    return final_result

# Different ranks execute different operations!
if rank % 4 == 0:
    result = leader_reduce(x)
else:
    result = worker_reduce(x)
```

---

## Problem Statement

### Key Challenges

1. **Different Operations**: Ranks are autotuning completely different functions
2. **Paired Execution**: Operations must be paired (send needs recv, leader needs worker)
3. **Coordinated Selection**: Config choices must be compatible across ranks
4. **Synchronization**: All ranks must still hit barriers at the same points

### Example Problem

```python
# Rank 0 has these SEND configs to autotune:
send_configs = [
    CustomOpConfig(send_direct),           # Direct send
    CustomOpConfig(send_chunked, size=1MB), # Chunked send
]

# Rank 1 has these RECV configs to autotune:
recv_configs = [
    CustomOpConfig(recv_direct),           # Direct recv
    CustomOpConfig(recv_chunked, size=1MB), # Chunked recv
]

# PROBLEM: If Rank 0 selects "send_chunked", 
# Rank 1 MUST select "recv_chunked" (matching implementation)!
# Otherwise communication will fail!
```

---

## Solution Design

### Approach 1: Role-Based Autotuning with Paired Configs

Key idea: Group configs into "compatible sets" and ensure matched selection.

```python
class PairedOpConfig:
    """Config for paired operations (e.g., send/recv)."""
    
    def __init__(
        self,
        config_id: str,  # Unique identifier for this config pair
        send_impl: Optional[Callable] = None,
        recv_impl: Optional[Callable] = None,
        **params
    ):
        self.config_id = config_id
        self.send_impl = send_impl
        self.recv_impl = recv_impl
        self.params = params

# Define paired configs
paired_configs = [
    PairedOpConfig(
        config_id="direct",
        send_impl=send_direct,
        recv_impl=recv_direct,
    ),
    PairedOpConfig(
        config_id="chunked_1mb",
        send_impl=send_chunked,
        recv_impl=recv_chunked,
        chunk_size=1024*1024,
    ),
]

def register_paired_op_autotuning(
    send_op,
    recv_op,
    paired_configs: list[PairedOpConfig],
    role_fn: Callable[[int], str],  # rank -> "sender" or "receiver"
):
    """
    Register autotuning for paired operations.
    
    Args:
        send_op: Send operation
        recv_op: Receive operation
        paired_configs: List of compatible config pairs
        role_fn: Function determining each rank's role
    """
    
    def autotuning_lowering(*args, **kwargs):
        rank = dist.get_rank()
        role = role_fn(rank)
        
        # Step 1: Exchange role information
        roles = [None] * dist.get_world_size()
        dist.all_gather_object(roles, role)
        
        # Find sender and receiver ranks
        senders = [r for r, role in enumerate(roles) if role == "sender"]
        receivers = [r for r, role in enumerate(roles) if role == "receiver"]
        
        # Step 2: Benchmark paired configs
        # Key: Both sender and receiver must benchmark at the same time!
        timings = {}
        
        for i, paired_cfg in enumerate(paired_configs):
            dist.barrier()  # CRITICAL: All ranks sync
            
            if role == "sender":
                # Sender benchmarks send implementation
                impl = paired_cfg.send_impl
                timing = benchmark_send(impl, paired_cfg.params, args, kwargs)
            elif role == "receiver":
                # Receiver benchmarks recv implementation
                impl = paired_cfg.recv_impl
                timing = benchmark_recv(impl, paired_cfg.params, args, kwargs)
            else:
                # Other ranks wait
                timing = 0.0
            
            # Collect MAX timing (bottleneck)
            timing_tensor = torch.tensor([timing], device=f'cuda:{rank}')
            dist.all_reduce(timing_tensor, op=dist.ReduceOp.MAX)
            timings[i] = timing_tensor.item()
        
        # Step 3: All ranks select the same paired config
        best_idx = min(timings, key=timings.get)
        best_paired_cfg = paired_configs[best_idx]
        
        # Step 4: Apply the appropriate implementation for this rank's role
        if role == "sender":
            return apply_send_config(best_paired_cfg, args, kwargs)
        elif role == "receiver":
            return apply_recv_config(best_paired_cfg, args, kwargs)
        
    # Register lowerings for both operations
    lowerings[send_op._opoverload] = autotuning_lowering
    lowerings[recv_op._opoverload] = autotuning_lowering
```

### Approach 2: Coordinator-Based Autotuning

Key idea: One rank acts as coordinator to orchestrate autotuning.

```python
def register_asymmetric_autotuning(
    operations: dict[str, torch._ops.OpOverload],  # role -> operation
    configs: dict[str, list[CustomOpConfig]],      # role -> configs
    role_fn: Callable[[int], str],                 # rank -> role
):
    """
    Register autotuning for asymmetric operations with coordinator.
    
    Args:
        operations: Map from role name to operation
        configs: Map from role name to list of configs
        role_fn: Function determining each rank's role
    """
    
    def autotuning_lowering(*args, **kwargs):
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        role = role_fn(rank)
        
        # Step 1: Coordinator (Rank 0) collects role info
        if rank == 0:
            roles = [None] * world_size
            dist.all_gather_object(roles, role)
            
            # Group ranks by role
            role_groups = {}
            for r, role_name in enumerate(roles):
                if role_name not in role_groups:
                    role_groups[role_name] = []
                role_groups[role_name].append(r)
        else:
            dist.all_gather_object([None] * world_size, role)
            role_groups = None
        
        # Broadcast role groups to all ranks
        role_groups = broadcast_from_rank0(role_groups)
        
        # Step 2: Each role group benchmarks its configs
        my_configs = configs[role]
        num_configs = max(len(cfg_list) for cfg_list in configs.values())
        
        timings = {}
        for i in range(num_configs):
            dist.barrier()  # All ranks sync
            
            if i < len(my_configs):
                cfg = my_configs[i]
                # Benchmark this config
                timing = benchmark_config(cfg, args, kwargs)
            else:
                # This role doesn't have config i
                timing = 0.0
            
            # Collect timings
            timing_tensor = torch.tensor([timing], device=f'cuda:{rank}')
            dist.all_reduce(timing_tensor, op=dist.ReduceOp.MAX)
            timings[i] = timing_tensor.item()
        
        # Step 3: Coordinator selects best config index
        if rank == 0:
            best_idx = min(timings, key=timings.get)
        else:
            best_idx = None
        
        # Broadcast decision to all ranks
        best_idx = broadcast_from_rank0(best_idx)
        
        # Step 4: Each rank applies its role-specific config
        if best_idx < len(my_configs):
            return apply_config(my_configs[best_idx], args, kwargs)
        else:
            return apply_fallback(operations[role], args, kwargs)
    
    # Register for all operations
    for op in operations.values():
        lowerings[op._opoverload] = autotuning_lowering
```

---

## Implementation Examples

### Example 1: Send/Recv Autotuning

```python
# Define send operation
@torch.library.custom_op("mylib::send", mutates_args=())
def my_send(x: torch.Tensor, dst: int) -> torch.Tensor:
    torch.ops._c10d_functional.send(x, dst)
    return x

# Define recv operation
@torch.library.custom_op("mylib::recv", mutates_args=())
def my_recv(shape: tuple, src: int) -> torch.Tensor:
    result = torch.empty(shape, device='cuda')
    torch.ops._c10d_functional.recv(result, src)
    return result

# Send implementations
def send_direct(x, dst):
    torch.ops._c10d_functional.send(x, dst)
    return x

def send_chunked(x, dst, chunk_size):
    chunks = x.split(chunk_size)
    for chunk in chunks:
        torch.ops._c10d_functional.send(chunk, dst)
    return x

# Recv implementations
def recv_direct(shape, src):
    result = torch.empty(shape, device='cuda')
    torch.ops._c10d_functional.recv(result, src)
    return result

def recv_chunked(shape, src, chunk_size):
    # Receive in chunks and concatenate
    chunks = []
    total_size = torch.prod(torch.tensor(shape)).item()
    num_chunks = (total_size + chunk_size - 1) // chunk_size
    
    for i in range(num_chunks):
        chunk = torch.empty(chunk_size, device='cuda')
        torch.ops._c10d_functional.recv(chunk, src)
        chunks.append(chunk)
    
    result = torch.cat(chunks).reshape(shape)
    return result

# Define paired configs
paired_configs = [
    PairedOpConfig(
        config_id="direct",
        send_impl=send_direct,
        recv_impl=recv_direct,
    ),
    PairedOpConfig(
        config_id="chunked",
        send_impl=send_chunked,
        recv_impl=recv_chunked,
        chunk_size=1024*1024,
    ),
]

# Define role function
def determine_role(rank):
    if rank == 0:
        return "sender"
    elif rank == 1:
        return "receiver"
    else:
        return "idle"

# Register paired autotuning
register_paired_op_autotuning(
    send_op=my_send,
    recv_op=my_recv,
    paired_configs=paired_configs,
    role_fn=determine_role,
)

# Usage
if rank == 0:
    # Sender
    x = torch.randn(1024, 1024, device='cuda:0')
    my_send(x, dst=1)
elif rank == 1:
    # Receiver
    result = my_recv(shape=(1024, 1024), src=0)
```

### Example 2: Pipeline Parallelism Autotuning

```python
# Define stage transfer operations
@torch.library.custom_op("mylib::stage_send", mutates_args=())
def stage_send(x: torch.Tensor, dst_stage: int) -> torch.Tensor:
    torch.ops._c10d_functional.send(x, dst_stage)
    return x

@torch.library.custom_op("mylib::stage_recv", mutates_args=())
def stage_recv(shape: tuple, src_stage: int) -> torch.Tensor:
    result = torch.empty(shape, device='cuda')
    torch.ops._c10d_functional.recv(result, src_stage)
    return result

# Stage implementations with different configs
def stage0_forward(x, config):
    """Stage 0: Input processing"""
    x = process_input_layer(x, config)
    stage_send(x, dst_stage=1)
    return x

def stage1_forward(config):
    """Stage 1: Middle layers"""
    x = stage_recv(shape=input_shape, src_stage=0)
    x = process_middle_layers(x, config)
    stage_send(x, dst_stage=2)
    return x

def stage2_forward(config):
    """Stage 2: Output layers"""
    x = stage_recv(shape=middle_shape, src_stage=1)
    output = process_output_layers(x, config)
    return output

# Define configs for each stage
stage_configs = {
    "stage0": [
        CustomOpConfig(fast_input_processing),
        CustomOpConfig(accurate_input_processing),
    ],
    "stage1": [
        CustomOpConfig(fast_middle_processing),
        CustomOpConfig(memory_efficient_middle_processing),
    ],
    "stage2": [
        CustomOpConfig(fast_output_processing),
        CustomOpConfig(accurate_output_processing),
    ],
}

# Define role based on rank
def get_stage_role(rank):
    num_stages = 3
    return f"stage{rank % num_stages}"

# Register
register_asymmetric_autotuning(
    operations={
        "stage0": stage0_forward,
        "stage1": stage1_forward,
        "stage2": stage2_forward,
    },
    configs=stage_configs,
    role_fn=get_stage_role,
)
```

### Example 3: Leader/Worker Autotuning

```python
# Leader operation
@torch.library.custom_op("mylib::leader_allreduce", mutates_args=())
def leader_allreduce(x: torch.Tensor, group_id: int) -> torch.Tensor:
    # Intra-group reduce
    group_result = group_reduce(x, group_id)
    # Inter-leader reduce
    leader_result = leader_reduce(group_result)
    # Broadcast to group
    broadcast_to_workers(leader_result, group_id)
    return leader_result

# Worker operation
@torch.library.custom_op("mylib::worker_allreduce", mutates_args=())
def worker_allreduce(x: torch.Tensor, group_id: int) -> torch.Tensor:
    # Intra-group reduce
    group_reduce(x, group_id)
    # Wait for leader
    result = receive_from_leader(group_id)
    return result

# Configs
leader_configs = [
    CustomOpConfig(leader_allreduce_nccl),
    CustomOpConfig(leader_allreduce_optimized),
]

worker_configs = [
    CustomOpConfig(worker_allreduce_nccl),
    CustomOpConfig(worker_allreduce_optimized),
]

# Role function
def get_leader_worker_role(rank):
    group_size = 4
    if rank % group_size == 0:
        return "leader"
    else:
        return "worker"

# Register
register_asymmetric_autotuning(
    operations={
        "leader": leader_allreduce,
        "worker": worker_allreduce,
    },
    configs={
        "leader": leader_configs,
        "worker": worker_configs,
    },
    role_fn=get_leader_worker_role,
)
```

---

## Comparison with Heterogeneous Configs

### Heterogeneous Configs (Previous Doc)

```python
# Same operation, different configs available
# All ranks execute: my_allreduce(x)

# Rank 0, 1 (H100): Have config A, B, C
# Rank 2, 3 (A100): Have config A, B only

# Solution: Exchange configs, benchmark all, select best
```

**Characteristics**:
- ✅ Same operation on all ranks
- ✅ Same function signature
- ✅ Only config availability differs
- ✅ Simple to synchronize

### Asymmetric Operations (This Doc)

```python
# Different operations on different ranks
# Rank 0: send(x, dst=1)
# Rank 1: recv(shape, src=0)

# Completely different functions!
```

**Characteristics**:
- ❌ Different operations
- ❌ Different function signatures
- ❌ Must coordinate config selection
- ❌ More complex synchronization

### Summary Table

| Aspect | Heterogeneous Configs | Asymmetric Operations |
|--------|----------------------|----------------------|
| Operation | Same | Different |
| Function Signature | Same | Different |
| Config Exchange | Yes | Paired/Coordinated |
| Selection | Independent | Must Match |
| Complexity | Low | High |
| Use Cases | Mixed GPUs | Send/Recv, Pipeline PP |
| vLLM Need? | No | No |
| V1 Scope? | No | No |

---

## Key Takeaways

1. **Asymmetric operations are fundamentally different** from heterogeneous configs
2. **Config pairing/coordination is critical** - selections must be compatible
3. **Synchronization is more complex** - different operations, same barriers
4. **Use cases are specialized** - Send/Recv, Pipeline PP, Leader/Worker patterns

## When to Use This Approach

**Use asymmetric operation autotuning when**:
- Different ranks execute fundamentally different operations
- Operations must be paired (send/recv, push/pull)
- Pipeline parallelism with different stages
- Custom distributed algorithms with specialized roles

**Don't use when**:
- All ranks run the same operation (use standard V1 approach)
- Only hardware/config differs (use heterogeneous configs approach)
- vLLM or similar symmetric workloads

---

## Conclusion

Asymmetric operation autotuning is a **complex, advanced feature** that requires:
- Careful coordination of config selection across ranks
- Paired/matched implementations
- Complex synchronization logic

**For V1**: This is out of scope. Focus on homogeneous, symmetric operations.

**For Future**: If needed, implement paired/coordinator-based approaches described here.

---

## References

- [HETEROGENEOUS_AUTOTUNING_DESIGN.md](./HETEROGENEOUS_AUTOTUNING_DESIGN.md) - Heterogeneous configs (different from this!)
- [MASTER_GUIDE.md](./MASTER_GUIDE.md) - V1 Implementation Guide
- [collective_benchmarking.py](../torch/_inductor/runtime/collective_benchmarking.py) - Core benchmarking utilities
