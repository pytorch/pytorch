# MST Implementation: Introducing DStorage 

Implementation: https://github.com/pytorch/pytorch/pull/174267


DStorage does 3 things:
1. Consolidating all parameters into a unified byte buffer with proper dtype alignment (bf16, fp32)
2. Enabling batched all-gather and reduce-scatter operations
3. Supporting flexible sharding strategies: 1) Shard(i):per-parameter sharding, 2) Owned: parameter-boundary ownership, mixed placements)

## Core Components

```
┌─────────────────────────────────────────────────────────────────┐
│                         DStorage                                 │
├─────────────────────────────────────────────────────────────────┤
│  byte_storage: torch.Tensor (uint8)   # Sharded parameter data  │
│  param_infos: dict[str, ParamInfo]    # Per-param metadata      │
│  mesh: DeviceMesh                     # Sharding topology       │
│  state: ShardedState                  # SHARDED | UNSHARDED     │
├─────────────────────────────────────────────────────────────────┤
│  unshard() → all-gather + swap to unsharded params              │
│  reshard() → reduce-scatter grads + restore sharded DTensors    │
└─────────────────────────────────────────────────────────────────┘
```

## Memory Layout

### Byte Buffer Structure

Parameters are laid out sequentially with proper alignment:

```
┌────────────────────────────────────────────────────────────────┐
│ Sharded Byte Storage (per-rank)                                │
├─────────┬─────────────┬─────────┬─────────────┬───────────────┤
│ param1  │   padding   │ param2  │   padding   │    param3     │
│ (fp32)  │ (alignment) │ (fp16)  │ (alignment) │    (bf16)     │
└─────────┴─────────────┴─────────┴─────────────┴───────────────┘
     ↑                       ↑                        ↑
     │                       │                        │
  byte_offset=0        byte_offset=N          byte_offset=M
```

### Region Layout (Mixed Placements)

When using mixed Shard/Owned placements, buffer is organized by region:

```
┌──────────────────────────────────────────────────────────────┐
│                    Byte Storage                               │
├───────────────────────────────┬──────────────────────────────┤
│     Shard Region              │      Owned Region            │
│  (params with Shard placement)│  (params with Owned placement)│
│  [batched all-gather]         │  [variable-size all-gather]  │
└───────────────────────────────┴──────────────────────────────┘
      shard_region_start       shard_region_end
                               owned_region_start    owned_region_end
```

### ParamInfo Structure

Each parameter is described by:

| Field | Description |
|-------|-------------|
| `fqn` | Fully qualified name (e.g., `layers.0.fc1.weight`) |
| `global_shape` | Original unsharded shape |
| `local_shape` | Shape on current rank after sharding |
| `dtype` | Parameter dtype (supports mixed dtypes) |
| `placements` | Sharding specification (`Shard(dim)` or `Owned(rank)`) |
| `byte_offset` | Offset into sharded byte buffer |
| `unsharded_byte_offset` | Offset into unsharded buffer |
| `owner_rank` | For `Owned` placement: which rank owns the full parameter |

### Placement Types

#### `Shard(dim)`
Standard DTensor sharding - parameter is split along `dim` across all ranks.

```
World size = 4
Parameter shape = (100, 256)
Shard(0) → Each rank gets (25, 256) local shard
```

Supports uneven sharding where `dim % world_size != 0` using ceiling-based chunk sizes.

#### `Owned(owner_rank)`
Parameter-boundary sharding - entire parameter lives on one rank.

```
World size = 4
Owned(2) → Rank 2 has full (100, 256), other ranks have empty tensor
```

Useful for:
- Small parameters where sharding overhead exceeds benefit
- Parameters that don't divide evenly
- Memory-balanced distribution via greedy bin-packing

## Lifecycle

### Forward Pass

```
                    ┌──────────────┐
                    │   SHARDED    │
                    │  (DTensors)  │
                    └──────┬───────┘
                           │
              forward_pre_hook: unshard()
                           │
                    ┌──────▼───────┐
                    │  UNSHARDED   │  ← all-gather byte buffer
                    │ (full params)│    swap DTensors → unsharded params
                    └──────┬───────┘
                           │
                    forward computation
                           │
              forward_post_hook: register backward hooks
                           │
                    ┌──────▼───────┐
                    │   OUTPUT     │
                    └──────────────┘
```

### Backward Pass

```
                    ┌──────────────┐
                    │  grad_output │
                    └──────┬───────┘
                           │
               pre_backward: (params still unsharded)
                           │
                   backward computation
                           │
              post_backward_callback: reshard()
                           │
                    ┌──────▼───────┐
                    │   SHARDED    │  ← reduce-scatter gradients
                    │  (DTensors)  │    restore sharded params
                    └──────────────┘
```

## API

### `fully_shard_flat(module, mesh, **options) → DStorage`

Main entry point for applying flat-storage FSDP sharding.

```python
from torch.distributed.fsdp._fully_shard import fully_shard_flat, get_dstorage

# Basic usage
mesh = init_device_mesh("cuda", (world_size,))
storage = fully_shard_flat(model, mesh)

# Nested wrapping (inner-first)
for layer in model.layers:
    fully_shard_flat(layer, mesh)
storage = fully_shard_flat(model, mesh)  # Only wraps root params
```

#### Parameters

| Parameter | Type | Default | Description |
|-----------|------|---------|-------------|
| `module` | `nn.Module` | required | Module to shard |
| `mesh` | `DeviceMesh` | required | 1D device mesh for sharding |
| `placements` | `tuple[Placement, ...]` | `(Shard(0),)` | Default sharding placement |
| `reshard_after_forward` | `bool` | `True` | Reshard after forward for memory savings |
| `register_hooks` | `bool` | `True` | Auto-register forward/backward hooks |
| `shard_strategy` | `str` | `"per_param"` | `"per_param"` or `"param_boundary"` |
| `shard_placement_fn` | `Callable` | `None` | Per-parameter placement function |

### `get_dstorage(module) → DStorage | None`

Retrieve the DStorage associated with a module.

```python
storage = get_dstorage(model.layers[0])
if storage:
    print(f"Layer has {len(storage.param_infos)} managed params")
```

### DStorage Methods

| Method | Description |
|--------|-------------|
| `unshard()` | All-gather byte buffer, swap to unsharded params |
| `reshard()` | Reduce-scatter gradients, restore sharded DTensors |
| `get_local_view(fqn)` | Get typed tensor view from sharded storage |
| `get_unsharded_view(fqn)` | Get typed tensor view from unsharded storage |
| `all_gather()` | Perform all-gather, return unsharded byte buffer |

## Sharding Strategies

### 1. Per-Parameter Sharding (Default)

Each parameter is sharded across all ranks along dimension 0.

```python
storage = fully_shard_flat(model, mesh)
# or explicitly:
storage = fully_shard_flat(model, mesh, shard_strategy="per_param")
```

### 2. Parameter-Boundary Sharding

Each parameter is assigned to exactly one rank using greedy bin-packing for memory balance.

```python
storage = fully_shard_flat(model, mesh, shard_strategy="param_boundary")
```

### 3. Mixed Placement (Custom)

Per-parameter control via placement function:

```python
from torch.distributed.fsdp._fully_shard import Owned
from torch.distributed.tensor import Shard

def placement_fn(fqn: str, param: nn.Parameter):
    if "embed" in fqn:
        return Owned(0)      # Embeddings owned by rank 0
    elif param.numel() < 1024:
        return Owned(1)      # Small params owned by rank 1
    else:
        return Shard(0)      # Large params sharded

storage = fully_shard_flat(model, mesh, shard_placement_fn=placement_fn)
```



## Collective Operations

### All-Gather (Unshard)

**For Shard params:** Single batched `all_gather_into_tensor` for the entire Shard region:
```python
# Each rank contributes same-sized chunk
gathered = all_gather_into_tensor(shard_region)
# Redistribute to unsharded buffer per-param
```

**For Owned params:** Variable-size all-gather:
```python
# Gather sizes first, then variable-size all-gather
all_gather(output_tensors, my_owned_region)
```

### Reduce-Scatter (Reshard)

**For Shard params:** Batched reduce-scatter with padding for uneven sharding:
```python
# Pad gradients for even chunking
chunk_cat(grads, num_chunks=world_size)
reduce_scatter_tensor(output, input, op=AVG)
```

**For Owned params:** Point-to-point reduce to owner:
```python
reduce(grad, dst=owner_rank, op=AVG)
```

## Nested Wrapping

DStorage supports hierarchical module wrapping for granular memory control:

```python
# Wrap inner modules first
for i, layer in enumerate(model.layers):
    layer_storage = fully_shard_flat(layer, mesh)
    # Each layer has its own DStorage

# Wrap root module (excludes already-wrapped layer params)
root_storage = fully_shard_flat(model, mesh)
# root_storage only manages embed, norm, lm_head, etc.
```

The `_get_managed_named_params` function automatically excludes parameters from submodules that already have DStorage.

## Hook-Based Scheduling

Forward hooks are registered automatically:

```python
# Pre-forward: unshard parameters
module.register_forward_pre_hook(_pre_forward)

# Post-forward: register backward hooks on outputs
module.register_forward_hook(_post_forward)

# Post-backward: reshard and reduce-scatter (via autograd callback)
Variable._execution_engine.queue_callback(_post_backward)
```

## Usage Example

```python
import torch
import torch.distributed as dist
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp._fully_shard import fully_shard_flat, get_dstorage

# Initialize distributed
dist.init_process_group(backend="nccl")
rank = dist.get_rank()
world_size = dist.get_world_size()
torch.cuda.set_device(rank)

# Create mesh and model
mesh = init_device_mesh("cuda", (world_size,))
model = MyTransformer().cuda()

# Apply nested FSDP sharding
for layer in model.layers:
    fully_shard_flat(layer, mesh)
root_storage = fully_shard_flat(model, mesh)

# Training loop (hooks handle unshard/reshard automatically)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

for batch in dataloader:
    optimizer.zero_grad()
    loss = model(batch).sum()
    loss.backward()
    torch.cuda.synchronize()  # Wait for post-backward callback
    optimizer.step()
```

## Comparison with FSDP2

| Feature | DStorage | FSDP2 |
|---------|----------|-------|
| Storage | Unified byte buffer | Per-FSDPParam flat storage |
| Collectives | Batched per-DStorage | Batched per-FSDPState |
| Mixed dtypes | Single buffer with alignment | Separate buffers |
| Placement | Shard + Owned | Shard only |
| Complexity | Simpler, single abstraction | More features (prefetch, etc.) |

## Future Work

- Multi-dimensional mesh support (HSDP)
- Gradient prefetching for overlapped communication
- Integration with activation checkpointing
- CPU offloading support
- Compile-time optimization with `torch.compile`

## References

- [PyTorch FSDP2 Design](https://github.com/pytorch/pytorch/tree/main/torch/distributed/fsdp)
- [DTensor Specification](https://github.com/pytorch/pytorch/tree/main/torch/distributed/tensor)
