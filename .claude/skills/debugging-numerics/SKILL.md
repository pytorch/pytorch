---
name: debugging-numerics
description: Debug numerical issues in PyTorch including accuracy problems, NaN/Inf values, non-deterministic results, and divergence between eager and compiled modes. Use when investigating wrong outputs, comparing tensor values between runs, or when the user mentions numerical accuracy, reproducibility, determinism, or tensor hash comparison.
---

# Debugging Numerical Issues

Debug accuracy problems, NaN/Inf values, non-deterministic behavior, and numerical divergence in PyTorch.

## When to Use

- Model produces wrong outputs or NaN/Inf
- Different results between eager and compiled modes
- Non-deterministic results across runs
- Need to compare tensors between two execution paths
- Debugging gradient issues or autograd problems

## DebugMode for Operation Tracing

`DebugMode` intercepts PyTorch operations and logs them with optional tensor hashing for comparing runs.

### Basic Usage

```python
from torch.utils._debug_mode import DebugMode

def run_model():
    x = torch.randn(8, 8)
    return torch.mm(torch.relu(x), x.T)

with DebugMode() as dm:
    out = run_model()

print(dm.debug_string())
```

### With Tensor Hashing for Numerical Comparison

```python
with (
    DebugMode(
        record_stack_trace=True,
        record_ids=True,
        record_output=True,
    ) as dm,
    DebugMode.log_tensor_hashes(
        hash_fn=["norm"],  # or "hash_tensor"
        hash_inputs=True,
    ),
):
    result = model(x)

print(dm.debug_string(show_stack_trace=True))
```

### Comparing Eager vs Compiled

```python
def run_model(model, data, *, compile_with=None):
    if compile_with is not None:
        model = torch.compile(model, backend=compile_with)
    with DebugMode(record_output=True) as dm, DebugMode.log_tensor_hashes(
        hash_inputs=True,
    ):
        out = model(*data)
    return dm, out

# Compare runs
dm_eager, _ = run_model(model, inputs)
dm_compiled, _ = run_model(model, inputs, compile_with="inductor")

print("Eager mode:")
print(dm_eager.debug_string())
print("Compiled mode:")
print(dm_compiled.debug_string())

# Look for different tensor hashes to find divergence point
```

### Logging Triton Kernels

DebugMode also captures Inductor-generated Triton kernels:

```python
x = torch.randn(3, 3, device="cuda")

with (
    DebugMode(record_output=True) as dm,
    DebugMode.log_tensor_hashes(hash_inputs=True)
):
    torch.compile(fn)(x)

print(dm.debug_string())  # Shows [triton] prefixed kernels
```

## Reproducibility and Determinism

### Setting Seeds for Reproducibility

```python
import torch
import random
import numpy as np

def set_seed(seed=42):
    torch.manual_seed(seed)
    random.seed(seed)
    np.random.seed(seed)

    # For CUDA
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)

set_seed(42)
```

### Enabling Deterministic Mode

```python
import torch

# Force deterministic algorithms (may be slower)
torch.use_deterministic_algorithms(True)

# For cuDNN
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

### Debugging Non-Deterministic Operations

```python
# This will raise an error for non-deterministic ops
torch.use_deterministic_algorithms(True)

try:
    # Some ops like index_add_ on CUDA are non-deterministic
    result = tensor.index_add_(0, indices, source)
except RuntimeError as e:
    print(f"Non-deterministic operation: {e}")
```

### Fill Uninitialized Memory

```python
# Prevent undefined behavior from uninitialized tensors
torch.utils.deterministic.fill_uninitialized_memory = True
```

### DataLoader Reproducibility

```python
def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

g = torch.Generator()
g.manual_seed(0)

loader = DataLoader(
    dataset,
    batch_size=32,
    num_workers=4,
    worker_init_fn=seed_worker,
    generator=g,
)
```

## Debugging NaN/Inf Values

### Detecting NaN/Inf

```python
def check_tensor(name, tensor):
    if torch.isnan(tensor).any():
        print(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        print(f"Inf detected in {name}")

# Use hooks to check intermediate values
def forward_hook(module, input, output):
    check_tensor(module.__class__.__name__, output)

for module in model.modules():
    module.register_forward_hook(forward_hook)
```

### Anomaly Detection

```python
# Enable anomaly detection for autograd
torch.autograd.set_detect_anomaly(True)

# This will give detailed error messages for NaN gradients
with torch.autograd.detect_anomaly():
    output = model(input)
    loss = criterion(output, target)
    loss.backward()  # Will raise if NaN in backward
```

### Common NaN/Inf Causes

1. **Division by zero**: Add epsilon to denominators
   ```python
   result = x / (y + 1e-8)
   ```

2. **Log of zero/negative**: Clamp inputs
   ```python
   result = torch.log(x.clamp(min=1e-8))
   ```

3. **Exploding gradients**: Use gradient clipping
   ```python
   torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
   ```

4. **Overflow in exp**: Clamp before exp
   ```python
   result = torch.exp(x.clamp(max=88))  # float32 overflow ~88
   ```

## Comparing Tensors

```python
import torch

def compare_tensors(t1, t2, name="tensor", rtol=1e-5, atol=1e-8):
    """Compare two tensors and report differences."""
    if t1.shape != t2.shape:
        print(f"{name}: Shape mismatch {t1.shape} vs {t2.shape}")
        return False

    if not torch.allclose(t1, t2, rtol=rtol, atol=atol):
        diff = (t1 - t2).abs()
        print(f"{name}: Max diff = {diff.max().item():.2e}")
        print(f"{name}: Mean diff = {diff.mean().item():.2e}")
        print(f"{name}: Locations of max diff: {(diff == diff.max()).nonzero()}")
        return False

    print(f"{name}: Match within tolerance")
    return True

# Use assert_close for automatic error messages
torch.testing.assert_close(actual, expected, rtol=1e-5, atol=1e-8)
```

## Accuracy Minifier

When you have accuracy issues with torch.compile, use the minifier:

```bash
# Find smallest graph reproducing accuracy issue
TORCHDYNAMO_REPRO_AFTER="aot" TORCHDYNAMO_REPRO_LEVEL=4 python script.py
```

Repro levels:
- 1: Dump graph on error
- 2: Dump graph and minify
- 3: Minify and check for crashes
- 4: Minify and check for accuracy issues

## Environment Variables

```bash
# Enable deterministic algorithms
CUBLAS_WORKSPACE_CONFIG=:4096:8  # Required for some CUDA ops

# Disable cuDNN benchmark for determinism
CUDNN_DETERMINISTIC=1
```

## CUDA Memory Debugging

For memory-related numerical issues (e.g., OOM causing partial computation), use memory snapshots:

### Recording Memory Snapshots

```python
# Enable memory history with tracebacks
torch.cuda.memory._record_memory_history()

# Run your code
run_model()

# Dump snapshot for visualization
torch.cuda.memory._dump_snapshot("memory_snapshot.pickle")
```

### Visualizing Memory

1. Open https://pytorch.org/memory_viz
2. Drag and drop your pickle file
3. Use "Active Memory Timeline" to see tensor lifetimes
4. Use "Allocator State History" to see allocation events

### Identifying Memory Leaks

```python
# Check allocated memory
print(f"Allocated: {torch.cuda.memory_allocated() / 1024**2:.1f} MB")
print(f"Reserved: {torch.cuda.memory_reserved() / 1024**2:.1f} MB")

# Empty cache to return memory to CUDA
torch.cuda.empty_cache()

# Check for non-PyTorch allocations
device_idx = 0
print(f"Total device memory used: {torch.cuda.device_memory_used(device_idx) / 1024**2:.1f} MB")
```

### Common Memory Issues

1. **Accumulating gradients**: Use `optimizer.zero_grad()` or `loss.backward()` properly
2. **Keeping tensor references**: Use `del tensor` and `torch.cuda.empty_cache()`
3. **History accumulation**: Use `.item()` or `.detach()` when tracking losses
