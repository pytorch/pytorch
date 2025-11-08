# Enhanced DebugMode with Triton Kernel Tracking - Complete Guide

## New Features

### 1. Argument Name Tracking
Every Triton kernel call now records the names of its arguments:
- `arg_names`: tuple of argument names (e.g., `('in_ptr0', 'in_ptr1', 'out_ptr0', 'xnumel')`)
- Displayed in debug output for easy identification
- Essential for understanding which tensors are being passed

### 2. Automatic Input/Output Detection
The system automatically detects which arguments are modified (outputs):
- **Pre-execution hashing**: Captures hash of each tensor before kernel runs
- **Post-execution hashing**: Captures hash of each tensor after kernel runs
- **Comparison**: Identifies modified tensors by comparing hashes
- **Annotation**: Modified arguments marked with `[OUT]` in debug output

### 3. Hash Logging for Triton Kernels
The existing `log_tensor_hashes()` context manager now works with Triton kernels:
- Automatically enabled when used with `record_triton_kernels=True`
- Stores both pre and post-execution hashes
- Perfect for determinism testing

## Usage Examples

### Basic Argument Tracking

```python
from torch.utils._debug_mode import DebugMode, _TritonKernelCall

with DebugMode(record_triton_kernels=True, store_original_args=True) as debug_mode:
    result = compiled_function(x, y)

triton_calls = [op for op in debug_mode.operators if isinstance(op, _TritonKernelCall)]

for call in triton_calls:
    print(f"Kernel: {call.kernel_name}")
    print(f"Grid: {call.grid}")

    # Access argument names and values
    if call.arg_names:
        for i, (name, arg) in enumerate(zip(call.arg_names, call.args)):
            if isinstance(arg, torch.Tensor):
                print(f"  {name}: tensor {arg.shape} {arg.dtype}")
            else:
                print(f"  {name}: {arg}")
```

**Output:**
```
Kernel: triton_poi_fused_add_0
Grid: (64, 1, 1)
  in_ptr0: tensor torch.Size([128, 128]) torch.float32
  in_ptr1: tensor torch.Size([128, 128]) torch.float32
  out_ptr0: tensor torch.Size([128, 128]) torch.float32
  xnumel: 16384
```

### Input/Output Detection with Hashing

```python
from torch.utils._debug_mode import DebugMode, _TritonKernelCall

with DebugMode(record_triton_kernels=True, store_original_args=True) as debug_mode:
    with DebugMode.log_tensor_hashes():
        result = compiled_function(x, y)

triton_calls = [op for op in debug_mode.operators if isinstance(op, _TritonKernelCall)]

for call in triton_calls:
    print(f"Kernel: {call.kernel_name}")

    # Check which arguments were modified (outputs)
    if call.modified_args:
        print(f"Output arguments:")
        for idx in call.modified_args:
            name = call.arg_names[idx] if call.arg_names else f"arg{idx}"
            print(f"  [{idx}] {name}")

    # Access pre/post hashes
    if call.pre_hashes:
        print(f"Pre-execution hashes: {call.pre_hashes}")
    if call.post_hashes:
        print(f"Post-execution hashes: {call.post_hashes}")
```

**Output:**
```
Kernel: triton_poi_fused_add_0
Output arguments:
  [2] out_ptr0
Pre-execution hashes: {0: 13199.96, 1: 13125.97, 2: 0.0}
Post-execution hashes: {0: 13199.96, 1: 13125.97, 2: 18572.63}
```

### Rendered Output with Annotations

```python
# The render() method now shows argument names and [OUT] markers
for call in triton_calls:
    print(call.render([]))
```

**Output:**
```
[Triton] triton_poi_fused_add_0[grid=(128, 1, 1)](
    in_ptr0=t: f32[128, 128],
    in_ptr1=t: f32[128, 128],
    [OUT] out_ptr0=t: f32[128, 128],
    xnumel=16384
)  # {'outputs': ['out_ptr0'], 'pre_hashes': {...}, 'post_hashes': {...}}
```

### Debugging Non-Determinism

The killer feature - automatically find where non-determinism first appears:

```python
from torch.utils._debug_mode import DebugMode, _TritonKernelCall

# Run 1
with DebugMode(record_triton_kernels=True, store_original_args=True) as debug_mode1:
    with DebugMode.log_tensor_hashes():
        output1 = compiled_fn(inputs)
        grads1 = torch.autograd.grad(output1, inputs, grad_outputs)

triton_calls1 = [op for op in debug_mode1.operators if isinstance(op, _TritonKernelCall)]

# Run 2 (after reset)
torch._dynamo.reset()
with torch._inductor.utils.fresh_inductor_cache():
    with DebugMode(record_triton_kernels=True, store_original_args=True) as debug_mode2:
        with DebugMode.log_tensor_hashes():
            output2 = compiled_fn(inputs)
            grads2 = torch.autograd.grad(output2, inputs, grad_outputs)

triton_calls2 = [op for op in debug_mode2.operators if isinstance(op, _TritonKernelCall)]

# Compare kernel by kernel to find first divergence
for i, (call1, call2) in enumerate(zip(triton_calls1, triton_calls2)):
    # Compare kernel names and grids
    assert call1.kernel_name == call2.kernel_name
    assert call1.grid == call2.grid

    # Compare input hashes
    for arg_idx in call1.pre_hashes:
        hash1 = call1.pre_hashes[arg_idx]
        hash2 = call2.pre_hashes.get(arg_idx)

        if hash1 != hash2:
            arg_name = call1.arg_names[arg_idx]
            print(f"FIRST DIVERGENCE at kernel {i}: {call1.kernel_name}")
            print(f"  Input argument '{arg_name}' differs")
            print(f"  Hash 1: {hash1}, Hash 2: {hash2}")

            # This is the first place where inputs differ!
            # The non-determinism comes from an earlier kernel
            break

    # Compare output hashes
    for arg_idx in call1.post_hashes:
        hash1 = call1.post_hashes[arg_idx]
        hash2 = call2.post_hashes.get(arg_idx)

        if hash1 != hash2:
            arg_name = call1.arg_names[arg_idx]
            print(f"OUTPUT DIVERGENCE at kernel {i}: {call1.kernel_name}")
            print(f"  Output argument '{arg_name}' differs")
            print(f"  Hash 1: {hash1}, Hash 2: {hash2}")

            # This kernel produces different outputs despite same inputs
            # The kernel itself is non-deterministic!
            break
```

## Key Data Structures

### _TritonKernelCall Attributes

```python
class _TritonKernelCall:
    kernel_name: str              # Name of the Triton kernel
    grid: tuple                   # (grid_0, grid_1, grid_2)
    args: tuple                   # Actual argument values
    kwargs: dict                  # Keyword arguments
    arg_names: tuple[str, ...]    # Names of arguments

    # Hash tracking (populated when log_tensor_hashes is used)
    pre_hashes: dict[int, float]  # {arg_index: hash_before}
    post_hashes: dict[int, float] # {arg_index: hash_after}
    modified_args: set[int]       # Indices of args that changed (outputs)

    # Logging
    log: dict                     # Additional log information
    record: dict                  # Hook-provided records
```

## Implementation Details

### How It Works

1. **Launcher Code Generation** (`triton_heuristics.py`):
   - Generates launcher functions that wrap Triton kernel execution
   - Injects pre-execution hook call before `runner()`
   - Injects post-execution hook call after `runner()`
   - Passes argument names and values to hooks

2. **Hook System** (`_debug_mode.py`):
   - `_call_triton_kernel_hooks_pre()`: Called before kernel execution
     - Creates `_TritonKernelCall` object
     - Computes pre-execution hashes if hash logging enabled
     - Returns call object as token
   - `_call_triton_kernel_hooks_post()`: Called after kernel execution
     - Receives token from pre-hook
     - Computes post-execution hashes
     - Compares hashes to detect modified arguments
     - Updates call object with results

3. **Hash Computation**:
   - Uses `default_hash_fn()` - L1 norm in float64/complex128
   - Handles NaN/inf values gracefully
   - Computed inside `_DisablePythonDispatcher` context

## Performance Considerations

- **Minimal overhead when disabled**: Hooks check if list is empty first
- **Storage cost with `store_original_args=True`**: Keeps references to all tensor arguments
- **Hash computation cost**: Only when `log_tensor_hashes()` is active
- **Per-kernel overhead**: One function call pre and post execution

## Comparison with Previous Approach

| Feature | Before | After |
|---------|--------|-------|
| Visibility into Triton kernels | ❌ No | ✅ Yes |
| Argument names | ❌ No | ✅ Yes |
| Input/output detection | ❌ Manual | ✅ Automatic |
| Hash logging support | ❌ No | ✅ Yes |
| Determinism debugging | ⚠️ Complex | ✅ Simple |
| Integration | ⚠️ Requires code modification | ✅ Flag-based |

## Files

- `test_triton_debug_enhanced.py` - Complete test suite demonstrating all features
- `test_flex_determinism_enhanced.py` - Enhanced version of your determinism test
- `TRITON_DEBUG_MODE_ENHANCED.md` - This guide

## Common Patterns

### Pattern 1: Quick Output Check

```python
with DebugMode(record_triton_kernels=True) as dm:
    with DebugMode.log_tensor_hashes():
        result = compiled_fn(x)

for call in [op for op in dm.operators if isinstance(op, _TritonKernelCall)]:
    if call.modified_args:
        outputs = [call.arg_names[i] for i in call.modified_args]
        print(f"{call.kernel_name}: outputs {outputs}")
```

### Pattern 2: Find Non-Deterministic Kernel

```python
# Compare two runs and find first divergence
for i, (c1, c2) in enumerate(zip(calls1, calls2)):
    # Check inputs first
    for j in c1.pre_hashes:
        if c1.pre_hashes[j] != c2.pre_hashes[j]:
            print(f"Kernel {i}: Input {c1.arg_names[j]} differs")
            # Previous kernel was non-deterministic
            if i > 0:
                print(f"  -> Issue in kernel {i-1}: {calls1[i-1].kernel_name}")
            break

    # Check outputs
    for j in c1.post_hashes:
        if c1.post_hashes[j] != c2.post_hashes[j]:
            print(f"Kernel {i}: Output {c1.arg_names[j]} differs")
            print(f"  -> THIS kernel is non-deterministic: {c1.kernel_name}")
            break
```

### Pattern 3: Audit All Kernels

```python
with DebugMode(record_triton_kernels=True) as dm:
    with DebugMode.log_tensor_hashes():
        result = compiled_fn(x)

for call in [op for op in dm.operators if isinstance(op, _TritonKernelCall)]:
    n_inputs = len([i for i in call.pre_hashes if i not in call.modified_args])
    n_outputs = len(call.modified_args) if call.modified_args else 0
    print(f"{call.kernel_name}: {n_inputs} inputs, {n_outputs} outputs")
```

## Future Enhancements

Possible additions:
1. Capture kernel source code along with invocation
2. Add timing information per kernel
3. Support for filtering specific kernels
4. Automatic bisection to find problematic kernel
5. Integration with torch profiler
6. Visual diffing tool for hash mismatches
