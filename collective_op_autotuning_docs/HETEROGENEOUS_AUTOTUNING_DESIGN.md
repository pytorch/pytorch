# Heterogeneous Autotuning for Collective Operations

**Status**: Future Work / V2+ Enhancement  
**Date**: 2024-11  
**Audience**: PyTorch Distributed Team

---

## Table of Contents

1. [Overview](#overview)
2. [Motivation](#motivation)
3. [Problem Statement](#problem-statement)
4. [Proposed Solution](#proposed-solution)
5. [Detailed Design](#detailed-design)
6. [Implementation Example](#implementation-example)
7. [Limitations and Trade-offs](#limitations-and-trade-offs)
8. [Future Work](#future-work)

---

## Overview

This document describes a potential enhancement to the collective operation autotuning system to support **heterogeneous environments** where different ranks may have different implementation capabilities.

**Important Note**: This feature is **NOT required for vLLM** or the V1 implementation. vLLM's Tensor Parallelism operates in homogeneous environments where all ranks run identical code. This document serves as a reference for potential future enhancements.

---

## Motivation

### Current Assumption (V1)

The current collective autotuning system assumes:
- All ranks run **identical code**
- All ranks have the **same set of configs** available
- All ranks select the **same optimal config**

This works perfectly for:
- ✅ vLLM Tensor Parallelism
- ✅ Standard distributed training (FSDP, DDP)
- ✅ Homogeneous GPU clusters

### Potential Future Need

In some advanced scenarios, different ranks might have different capabilities:

1. **Heterogeneous Hardware**
   - Mix of H100, A100, and V100 GPUs
   - Different CUDA compute capabilities
   - Hardware-specific optimizations (e.g., H100's Transformer Engine)

2. **Different Software Stacks**
   - Different CUDA versions across nodes
   - Different kernel implementations available
   - Platform-specific optimizations (e.g., AWS vs on-prem)

3. **Specialized Roles**
   - Leader ranks with additional optimizations
   - Edge nodes with different networking hardware
   - Heterogeneous clusters with different interconnects

---

## Problem Statement

### Scenario

Consider a 4-GPU setup:
- **Rank 0, 1**: H100 GPUs with specialized all-reduce kernels
- **Rank 2, 3**: A100 GPUs without H100-specific optimizations

```python
# Rank 0, 1 (H100) have these configs:
configs_h100 = [
    CustomOpConfig(allreduce_nccl),           # Standard NCCL
    CustomOpConfig(allreduce_cuda_graph),     # CUDA Graph optimized
    CustomOpConfig(allreduce_h100_optimized), # H100-specific ⭐
]

# Rank 2, 3 (A100) only have these configs:
configs_a100 = [
    CustomOpConfig(allreduce_nccl),           # Standard NCCL
    CustomOpConfig(allreduce_cuda_graph),     # CUDA Graph optimized
    # NO H100-specific optimization available
]
```

### Challenges

1. **Config Mismatch**: Different ranks have different numbers of configs
2. **Synchronization**: All ranks must hit barriers at the same points
3. **Selection**: How do we choose a config when not all ranks can use it?
4. **Fallback**: What happens when the "best" config isn't available on some ranks?

---

## Proposed Solution

### High-Level Approach

1. **Capability Declaration**: Each rank declares which configs it supports
2. **Config Exchange**: Ranks share their available configs with all other ranks
3. **Unified Config List**: Create a superset of all configs from all ranks
4. **Synchronized Benchmarking**: All ranks benchmark together, unavailable configs use fallback/skip
5. **Unified Selection**: All ranks select the same "best" config based on max timing
6. **Conditional Execution**: Ranks use the selected config if available, otherwise fallback

### Key Principles

- ✅ All ranks must synchronize at barriers (no hanging)
- ✅ All ranks select the same "best" config (consistency)
- ✅ Timing uses MAX across ranks (collective ops are bound by slowest rank)
- ✅ Ranks without the "best" config use a fallback implementation

---

## Detailed Design

### Step 1: Capability Check Function

Define a function that determines which configs each rank can use:

```python
def check_h100_availability(rank: int) -> list[bool]:
    """
    Check which configs are available on this rank.
    
    Args:
        rank: Current rank ID
        
    Returns:
        Boolean list indicating availability of each config
    """
    gpu_name = torch.cuda.get_device_name(rank)
    is_h100 = "H100" in gpu_name
    
    return [
        True,      # config0: allreduce_nccl - available on all GPUs
        True,      # config1: allreduce_cuda_graph - available on all GPUs
        is_h100,   # config2: allreduce_h100_optimized - H100 only
    ]

# Example results:
# Rank 0 (H100): [True, True, True]
# Rank 1 (H100): [True, True, True]
# Rank 2 (A100): [True, True, False]
# Rank 3 (A100): [True, True, False]
```

### Step 2: Config Exchange

Use `all_gather_object` to share config information across all ranks:

```python
def gather_configs_from_all_ranks(
    my_configs: list[CustomOpConfig],
    rank: int,
    world_size: int
) -> list[list[dict]]:
    """
    Gather config metadata from all ranks.
    
    Returns:
        List of config metadata lists, one per rank
    """
    # Serialize configs to transferable format
    my_config_metadata = [
        {
            'name': cfg.decomposition.__name__ if cfg.decomposition else 'default',
            'params': cfg.params,
        }
        for cfg in my_configs
    ]
    
    # Gather from all ranks
    all_configs_metadata = [None] * world_size
    dist.all_gather_object(all_configs_metadata, my_config_metadata)
    
    return all_configs_metadata

# After execution, all ranks have:
# all_configs_metadata = [
#   [config0_meta, config1_meta, config2_meta],  # Rank 0's configs
#   [config0_meta, config1_meta, config2_meta],  # Rank 1's configs
#   [config0_meta, config1_meta],                # Rank 2's configs
#   [config0_meta, config1_meta],                # Rank 3's configs
# ]
```

### Step 3: Merge Configs

Create a unified config list and track availability:

```python
def merge_configs(
    all_rank_configs: list[list[dict]],
    rank: int,
    world_size: int
) -> tuple[list[dict], dict[int, list[bool]]]:
    """
    Merge configs from all ranks into a unified list.
    
    Returns:
        - unified_configs: List of unique configs
        - config_availability: Dict mapping rank to boolean list
    """
    unified_configs = []
    config_map = {}  # For deduplication
    config_availability = {r: [] for r in range(world_size)}
    
    # Collect all unique configs
    for src_rank, rank_configs in enumerate(all_rank_configs):
        for cfg_meta in rank_configs:
            # Create unique key for config
            cfg_key = f"{cfg_meta['name']}_{cfg_meta['params']}"
            
            if cfg_key not in config_map:
                # New config found
                config_map[cfg_key] = len(unified_configs)
                unified_configs.append(cfg_meta)
                
                # Initialize availability (all False by default)
                for r in range(world_size):
                    config_availability[r].append(False)
            
            # Mark this config as available on src_rank
            cfg_idx = config_map[cfg_key]
            config_availability[src_rank][cfg_idx] = True
    
    return unified_configs, config_availability

# Result:
# unified_configs = [
#   {'name': 'allreduce_nccl', ...},        # idx=0
#   {'name': 'allreduce_cuda_graph', ...},  # idx=1
#   {'name': 'allreduce_h100', ...},        # idx=2
# ]
#
# config_availability = {
#   0: [True, True, True],   # Rank 0 has all 3
#   1: [True, True, True],   # Rank 1 has all 3
#   2: [True, True, False],  # Rank 2 has only first 2
#   3: [True, True, False],  # Rank 3 has only first 2
# }
```

### Step 4: Synchronized Benchmarking

Benchmark all configs with proper synchronization:

```python
def benchmark_unified_configs(
    unified_configs: list[dict],
    config_availability: dict[int, list[bool]],
    rank: int,
    args: tuple,
    kwargs: dict
) -> dict[int, float]:
    """
    Benchmark all configs with cross-rank synchronization.
    
    Key: All ranks must participate in all barriers, even if they
    don't have certain configs.
    """
    timings = {}
    world_size = dist.get_world_size()
    
    for i, cfg_meta in enumerate(unified_configs):
        # CRITICAL: All ranks must hit this barrier
        dist.barrier()
        
        # Check if this rank has this config
        if config_availability[rank][i]:
            # Rank has config, benchmark it normally
            timing = actually_benchmark_config(cfg_meta, args, kwargs)
            # e.g., Rank 0 benchmarks config2 → 8.7ms
        else:
            # Rank doesn't have config, send 0 (won't affect MAX)
            timing = 0.0
            # e.g., Rank 2 doesn't have config2 → 0.0ms
        
        # Collect timings across all ranks using MAX
        # Why MAX? Collective ops are bottlenecked by slowest rank
        timing_tensor = torch.tensor(
            [timing], 
            dtype=torch.float32, 
            device=f'cuda:{rank}'
        )
        dist.all_reduce(timing_tensor, op=dist.ReduceOp.MAX)
        
        # All ranks now have the same timing (MAX among ranks with config)
        timings[i] = timing_tensor.item()
        
        # Example for config2:
        # Rank 0: 8.7ms → all_reduce(MAX) → 8.7ms
        # Rank 1: 8.5ms → all_reduce(MAX) → 8.7ms
        # Rank 2: 0.0ms → all_reduce(MAX) → 8.7ms (doesn't affect MAX)
        # Rank 3: 0.0ms → all_reduce(MAX) → 8.7ms (doesn't affect MAX)
    
    return timings

# Result (all ranks have identical timing dict):
# timings = {
#   0: 10.5,  # allreduce_nccl (MAX across all 4 ranks)
#   1: 12.3,  # allreduce_cuda_graph (MAX across all 4 ranks)
#   2: 8.7,   # allreduce_h100_optimized (MAX across Rank 0,1 only)
# }
```

### Step 5: Unified Selection

All ranks select the same best config:

```python
# All ranks execute identical code
best_idx = min(timings, key=timings.get)  # best_idx = 2 (8.7ms)
best_config = unified_configs[best_idx]

print(f"Rank {rank}: Selected config {best_idx} ({best_config['name']})")
# All ranks print: "Selected config 2 (allreduce_h100_optimized)"
```

### Step 6: Conditional Execution

Apply the selected config with fallback handling:

```python
if config_availability[rank][best_idx]:
    # This rank has the best config, use it
    result = apply_config(best_config, args, kwargs)
    # Rank 0, 1: Use H100-optimized implementation
else:
    # This rank doesn't have the best config, use fallback
    result = apply_fallback(custom_op, args, kwargs)
    # Rank 2, 3: Use fallback (e.g., second-best config or default)
```

---

## Implementation Example

### Complete Registration Function

```python
def register_heterogeneous_autotuning(
    custom_op,
    configs: list[CustomOpConfig],
    capability_check_fn: Optional[Callable[[int], list[bool]]] = None,
):
    """
    Register autotuning with heterogeneous config support.
    
    Args:
        custom_op: Custom operation to register
        configs: Full list of possible configs
        capability_check_fn: Function mapping rank → config availability
    """
    
    @functools.wraps(custom_op)
    def autotuning_lowering(*args, **kwargs):
        if not dist.is_initialized():
            # Non-distributed fallback
            return normal_autotune(custom_op, configs, args, kwargs)
        
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        
        # Step 1: Determine available configs for this rank
        if capability_check_fn:
            available_flags = capability_check_fn(rank)
            my_configs = [cfg for cfg, avail in zip(configs, available_flags) if avail]
        else:
            my_configs = configs
        
        # Step 2: Exchange config info
        all_rank_configs = gather_configs_from_all_ranks(my_configs, rank, world_size)
        
        # Step 3: Create unified config list
        unified_configs, config_availability = merge_configs(
            all_rank_configs, rank, world_size
        )
        
        # Step 4: Synchronized benchmarking
        timings = benchmark_unified_configs(
            unified_configs,
            config_availability,
            rank,
            args,
            kwargs
        )
        
        # Step 5: All ranks select same best config
        best_idx = min(timings, key=timings.get)
        best_config = unified_configs[best_idx]
        
        # Step 6: Apply with fallback
        if config_availability[rank][best_idx]:
            return apply_config(best_config, args, kwargs)
        else:
            return apply_fallback(custom_op, args, kwargs)
    
    lowerings[custom_op._opoverload] = autotuning_lowering
```

### Usage Example

```python
# Define configs
configs = [
    CustomOpConfig(allreduce_nccl),
    CustomOpConfig(allreduce_cuda_graph),
    CustomOpConfig(allreduce_h100_optimized),  # H100-only
]

# Define capability check
def check_capabilities(rank):
    gpu_name = torch.cuda.get_device_name(rank)
    is_h100 = "H100" in gpu_name
    return [True, True, is_h100]

# Register with heterogeneous support
register_heterogeneous_autotuning(
    my_allreduce,
    configs=configs,
    capability_check_fn=check_capabilities,
)
```

---

## Limitations and Trade-offs

### Advantages

- ✅ Supports heterogeneous hardware environments
- ✅ Maintains synchronization across all ranks
- ✅ Maximizes performance by selecting best available config
- ✅ Graceful fallback for ranks without optimal config

### Disadvantages

- ❌ Increased complexity compared to homogeneous case
- ❌ Potential performance degradation if fallback is used frequently
- ❌ Requires careful design of fallback strategies
- ❌ More difficult to debug and reason about

### When to Use

**Use heterogeneous autotuning when**:
- Running on mixed GPU types (H100 + A100 + V100)
- Different nodes have different software versions
- Specialized optimizations available on subset of ranks

**Don't use when**:
- All ranks have identical hardware (most production cases)
- vLLM or similar homogeneous workloads
- V1 implementation (out of scope)

---

## Future Work

### Potential Enhancements

1. **Smart Fallback Selection**
   - Instead of generic fallback, select next-best available config
   - Cache fallback selections to avoid repeated benchmarking

2. **Capability Auto-Detection**
   - Automatically detect GPU capabilities
   - Query CUDA driver for supported features
   - No manual `capability_check_fn` needed

3. **Performance Modeling**
   - Predict performance on ranks without config
   - Make smarter selection decisions based on availability

4. **Dynamic Rank Grouping**
   - Group ranks by capability
   - Use different configs for different groups
   - More complex but potentially higher performance

### Integration with V2

If V2 (MultiTemplateBuffer-based) is implemented, this heterogeneous support can be integrated:

```python
class HeterogeneousCollectiveMultiTemplateBuffer(CollectiveMultiTemplateBuffer):
    """
    MultiTemplateBuffer with heterogeneous config support.
    """
    
    def __init__(
        self,
        choices: list[ChoiceCaller],
        config_availability: dict[int, list[bool]],
        **kwargs
    ):
        super().__init__(choices, **kwargs)
        self.config_availability = config_availability
    
    def finalize_choice(self):
        """Finalize with heterogeneous awareness."""
        # Similar logic but integrated with scheduler
        pass
```

---

## Conclusion

This document describes a comprehensive approach to supporting heterogeneous environments in collective operation autotuning. While **not required for current use cases** (vLLM, standard distributed training), this design provides a foundation for future enhancements when heterogeneous clusters become more common.

**For V1 Implementation**: Focus on homogeneous case only. This document serves as reference for potential V2+ features.

**Key Takeaway**: The core principle is maintaining synchronization while allowing different ranks to have different capabilities through careful config exchange, unified benchmarking, and smart fallback strategies.

---

## References

- [MASTER_GUIDE.md](./MASTER_GUIDE.md) - V1 Implementation Guide
- [collective_benchmarking.py](../torch/_inductor/runtime/collective_benchmarking.py) - Core benchmarking utilities
- [custom_op.py](../torch/_inductor/kernel/custom_op.py) - Custom op autotuning framework
