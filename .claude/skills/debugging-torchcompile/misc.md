# Miscellaneous Debugging Tips

Small tips and tricks that don't fit neatly into the other categories. When a topic here grows large, it should be moved to its own file.

## Sequence Numbers (seq_nr) for Debugging

Sequence numbers help correlate forward and backward operations when debugging autograd issues within compiled graphs.

### What are Sequence Numbers?
Every autograd node gets a unique `seq_nr` that links forward ops to their backward counterparts. This is useful for:
- Mapping backward errors to forward ops
- Understanding execution order
- Debugging gradient computation issues

### Viewing Sequence Numbers

```python
import torch
from torch._logging import trace_structured

# Enable logging that shows seq_nr
# TORCH_LOGS="+aot_graphs" python script.py

# In traces, look for _seq_nr annotations on nodes
```

### Using seq_nr in Debugging

When you see an error in the backward pass:
1. Find the `seq_nr` in the error message or logs
2. Search for the same `seq_nr` in the forward graph
3. The matching node is where the problematic op originated

## Quick Disable for Bisection

When debugging, quickly disable torch.compile to check if an issue is compile-related:

```python
import torch

# Disable at module level
torch._dynamo.config.disable = True

# Or use environment variable
# TORCHDYNAMO_DISABLE=1 python script.py
```

## Checking Compilation Status

```python
import torch

def check_if_compiled(fn, args):
    """Check if a function actually got compiled."""
    import torch._dynamo

    # Clear cache first
    torch._dynamo.reset()

    # Compile and run
    compiled = torch.compile(fn)
    result = compiled(*args)

    # Check compilation stats
    stats = torch._dynamo.utils.counters
    print(f"Frames compiled: {stats['stats']['calls_captured']}")
    print(f"Graph breaks: {stats['stats']['graph_breaks']}")

    return result
```

## Deterministic Compilation

For reproducible debugging:

```python
import torch
import random
import numpy as np

# Set all seeds
torch.manual_seed(42)
random.seed(42)
np.random.seed(42)

# Disable non-deterministic autotuning
import torch._inductor.config as config
config.max_autotune = False
config.triton.autotune_cublasLt = False
```

## Memory Debugging

```python
import torch

# Track memory usage during compilation
torch.cuda.memory._record_memory_history()

compiled_fn = torch.compile(fn)
result = compiled_fn(x)

# Get memory snapshot
snapshot = torch.cuda.memory._snapshot()
torch.cuda.memory._save_segment_usage(snapshot, "memory_usage.svg")
torch.cuda.memory._save_memory_usage(snapshot, "memory_timeline.html")
```

## Handling "Silent" Failures

Sometimes torch.compile silently falls back to eager mode. Catch these:

```python
import torch._dynamo

# Make compilation failures loud
torch._dynamo.config.suppress_errors = False

# Or require full compilation
@torch.compile(fullgraph=True)  # Fail if any graph breaks
def fn(x):
    return x * 2
```

## Testing Compiled vs Eager

```python
import torch

def compare_outputs(fn, *args, **kwargs):
    """Compare compiled vs eager outputs."""
    # Eager
    eager_out = fn(*args, **kwargs)

    # Compiled
    compiled_fn = torch.compile(fn)
    compiled_out = compiled_fn(*args, **kwargs)

    # Compare
    if isinstance(eager_out, torch.Tensor):
        torch.testing.assert_close(eager_out, compiled_out)
    elif isinstance(eager_out, (tuple, list)):
        for e, c in zip(eager_out, compiled_out):
            torch.testing.assert_close(e, c)

    print("✓ Outputs match")
    return eager_out, compiled_out
```

## Backend Selection

```python
import torch

# Use specific backend
@torch.compile(backend="inductor")  # Default
def fn1(x): return x * 2

@torch.compile(backend="eager")  # Just trace, no codegen
def fn2(x): return x * 2

@torch.compile(backend="aot_eager")  # AOT Autograd + eager
def fn3(x): return x * 2

# List available backends
from torch._dynamo import list_backends
print(list_backends())
```

## Kernel Debugging with Breakpoints

```python
import torch._inductor.config as config

# Generate debuggable code
config.debug = True
config.triton.debug_sync_kernel = True

# For CUDA debugging
# cuda-gdb --args python script.py
```

## Common Pitfalls

### Mutable Default Arguments
```python
# Bad: mutable default causes recompilation
def fn(x, cache=[]):
    cache.append(x)
    return x

# Good: use None
def fn(x, cache=None):
    if cache is None:
        cache = []
    cache.append(x)
    return x
```

### Global State
```python
# Bad: global state causes guard failures
SCALE = 1.0

def fn(x):
    return x * SCALE  # Recompiles if SCALE changes

# Good: pass as argument
def fn(x, scale):
    return x * scale
```

### Type Changes
```python
# Bad: type changes cause recompilation
def fn(x, flag):
    if flag:  # bool vs int causes recompile
        return x * 2
    return x

# Good: consistent types
def fn(x, flag: bool):
    if flag:
        return x * 2
    return x
```

## Printing in Compiled Code

```python
# graph_break: print() causes graph break by default

# Option 1: Use torch._dynamo.comptime for compile-time prints
from torch._dynamo import comptime

@torch.compile
def fn(x):
    comptime.print("Compiling!")  # Only printed during compilation
    return x * 2

# Option 2: Use trace for runtime debug that doesn't break the graph
# (requires special setup)
```

## Finding What Triggered Recompilation

```python
import torch._dynamo

# Get detailed guard info
# TORCH_LOGS="+guards" python script.py

# Or programmatically
def debug_recompilations(fn, args_list):
    compiled = torch.compile(fn)

    for i, args in enumerate(args_list):
        print(f"--- Call {i} ---")
        torch._dynamo.reset()  # Clear to see fresh compilation
        result = compiled(*args)

    # Check counters
    print(torch._dynamo.utils.counters)
```
