# Triton Kernel Recording in DebugMode

## Overview

The enhanced DebugMode now supports recording Triton kernel outputs and logging tensor hashes **with proper synchronization**. This ensures that tensor data is fully computed before being recorded.

## Key Features

### 1. Record Triton Outputs
Clone and save tensor outputs after kernel execution:

```python
from torch.utils._debug_mode import DebugMode

with DebugMode() as debug_mode:
    with DebugMode.record_outputs():
        result = my_compiled_model(input_tensor)

# Access recorded outputs
for call in debug_mode.operators:
    if hasattr(call, 'record') and call.record:
        outputs = call.record.get('output')  # List of cloned tensors
```

### 2. Log Tensor Hashes
Compute and log tensor hashes for debugging/comparison:

```python
from torch.utils._debug_mode import DebugMode

with DebugMode() as debug_mode:
    with DebugMode.log_tensor_hashes():
        result = my_compiled_model(input_tensor)

# View hashes in debug string
print(debug_mode.debug_string())
# Output includes: [triton] kernel_name(...)  # {'hash': [0.123, 0.456, ...]}
```

### 3. Custom Hash Function
Provide your own hashing logic:

```python
def my_hash_fn(tensor):
    """Custom hash: sum of absolute values"""
    return tensor.abs().sum().item()

with DebugMode() as debug_mode:
    with DebugMode.log_tensor_hashes(hash_fn=my_hash_fn):
        result = my_compiled_model(input_tensor)
```

### 4. Hash Both Inputs and Outputs
```python
with DebugMode() as debug_mode:
    with DebugMode.log_tensor_hashes(hash_inputs=True):
        result = my_compiled_model(input_tensor)

# Logs will contain both 'hash' (outputs) and 'input_hash' (inputs)
```

## Synchronization Behavior

**Important:** The implementation automatically synchronizes the GPU only when necessary:

- ✅ **Synchronizes** when `record_outputs()` or `log_tensor_hashes()` is active
- ✅ **No sync overhead** for just recording kernel metadata (name, shapes, dtypes)
- ✅ **Async-safe** - respects CUDA stream semantics

## Complete Example

```python
import torch
from torch.utils._debug_mode import DebugMode

@torch.compile
def my_kernel(x, y):
    return x + y * 2

x = torch.randn(1024, device='cuda')
y = torch.randn(1024, device='cuda')

# Example 1: Just record call metadata (no sync needed)
with DebugMode() as debug_mode:
    result = my_kernel(x, y)

print(debug_mode.debug_string())
# Shows: [triton] triton_kernel_name(in_ptr0=..., out_ptr0=...)

# Example 2: Record outputs with automatic sync
with DebugMode() as debug_mode:
    with DebugMode.record_outputs():
        result = my_kernel(x, y)

# Access cloned outputs
for call in debug_mode.operators:
    if hasattr(call, 'record') and 'output' in call.record:
        print(f"Kernel: {call.kernel_name}")
        print(f"Outputs: {call.record['output']}")

# Example 3: Log hashes for debugging
with DebugMode() as debug_mode:
    with DebugMode.log_tensor_hashes(hash_inputs=True):
        result1 = my_kernel(x, y)
        result2 = my_kernel(x, y)  # Should have same hashes

print(debug_mode.debug_string())
```

## Implementation Details

### How Synchronization Works

In `CachingAutotuner.run()` (triton_heuristics.py:1393-1407):

```python
result = launcher(*args, **kwargs, stream=stream)  # Async GPU launch

if debug_mode:
    # Check if we need to access tensor data
    if _RECORD_TRITON_OUTPUTS or _LOG_TRITON_HASHES:
        # Synchronize to ensure kernel completion
        device_interface.synchronize(device_interface.current_device())

    # Now safe to access tensor data
    debug_call.record_triton_output()
```

### What Gets Recorded

- **Without sync** (default): Kernel name, arg names, tensor shapes/dtypes/devices
- **With sync** (when using hooks): Actual tensor data, cloned outputs, computed hashes

## Performance Considerations

- **Synchronization overhead**: Only when explicitly enabled via `record_outputs()` or `log_tensor_hashes()`
- **Memory overhead**: Cloned tensors are stored in `call.record['output']`
- **Recommended**: Use these hooks sparingly, for debugging specific issues

## Backward Compatibility

All existing DebugMode functionality remains unchanged. The new Triton recording features are opt-in via context managers.
