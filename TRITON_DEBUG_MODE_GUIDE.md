# Enhanced DebugMode with Triton Kernel Tracking

## Overview

We've enhanced PyTorch's `DebugMode` to provide **kernel-level debugging** for Triton kernels executed by inductor. This allows you to track and compare Triton kernel invocations, which is crucial for debugging non-determinism in compiled code.

## What Was Changed

### 1. New Components in `torch/utils/_debug_mode.py`

- **`_TritonKernelCall` class**: Records Triton kernel invocations with:
  - Kernel name
  - Grid configuration (grid_0, grid_1, grid_2)
  - Input arguments (tensors and scalars)
  - Kwargs

- **`_call_triton_kernel_hooks()` function**: Global hook called before each Triton kernel execution

- **`record_triton_kernels` parameter**: New DebugMode initialization parameter to enable Triton kernel tracking

### 2. Modified Launcher Code in `torch/_inductor/runtime/triton_heuristics.py`

- Modified `_gen_launcher_code()` to inject hook calls into generated launcher functions
- Hook is called with kernel name, grid, and all arguments before kernel execution

## Usage

### Basic Usage

```python
from torch.utils._debug_mode import DebugMode, _TritonKernelCall

# Enable Triton kernel recording
with DebugMode(record_triton_kernels=True) as debug_mode:
    output = compiled_function(x, y)

# Extract Triton kernel calls
triton_calls = [op for op in debug_mode.operators if isinstance(op, _TritonKernelCall)]

for call in triton_calls:
    print(f"Kernel: {call.kernel_name}")
    print(f"Grid: {call.grid}")
    print(f"Args: {call.args}")
```

### Debugging Non-Determinism

```python
from torch.utils._debug_mode import DebugMode, _TritonKernelCall

# Run 1
with DebugMode(record_triton_kernels=True, store_original_args=True) as debug_mode1:
    out1 = compiled_fn(inputs)
    grad1 = torch.autograd.grad(out1, inputs, grad_outputs)

triton_calls1 = [op for op in debug_mode1.operators if isinstance(op, _TritonKernelCall)]

# Run 2 (after resetting dynamo and inductor cache)
torch._dynamo.reset()
with torch._inductor.utils.fresh_inductor_cache():
    with DebugMode(record_triton_kernels=True, store_original_args=True) as debug_mode2:
        out2 = compiled_fn(inputs)
        grad2 = torch.autograd.grad(out2, inputs, grad_outputs)

triton_calls2 = [op for op in debug_mode2.operators if isinstance(op, _TritonKernelCall)]

# Compare kernel calls to find where non-determinism starts
for i, (call1, call2) in enumerate(zip(triton_calls1, triton_calls2)):
    if call1.kernel_name != call2.kernel_name:
        print(f"Kernel {i}: Different kernels!")

    # Compare tensor arguments
    for j, (t1, t2) in enumerate(zip(call1.args, call2.args)):
        if isinstance(t1, torch.Tensor) and not torch.equal(t1, t2):
            print(f"Kernel {i} ({call1.kernel_name}), arg {j}: Input differs!")
            print(f"  Max diff: {(t1 - t2).abs().max()}")
            # This is where non-determinism first appears!
            break
```

### Integration with Your Test

For your `test_flex_determinism.py`, you can now use:

```python
from torch.utils._debug_mode import DebugMode, _TritonKernelCall

# In your test loop
for run in range(num_runs):
    torch._dynamo.reset()

    with torch._inductor.utils.fresh_inductor_cache():
        with DebugMode(record_triton_kernels=True, store_original_args=True) as debug_mode:
            compiled_fn = torch.compile(
                flex_forward, fullgraph=True, dynamic=dynamic, backend=backend, mode=mode
            )
            output = compiled_fn(q, k, v, block_mask)
            grad_q, grad_k, grad_v = torch.autograd.grad(
                output, inputs=[q, k, v], grad_outputs=[grad_out]
            )

    triton_calls = [op for op in debug_mode.operators if isinstance(op, _TritonKernelCall)]

    if run == 0:
        first_logs = triton_calls
    else:
        # Compare with first run
        for j, (log, first_log) in enumerate(zip(triton_calls, first_logs)):
            print(f"Comparing kernel {j}: {log.kernel_name}")
            for arg_idx, (t1, t2) in enumerate(zip(log.args, first_log.args)):
                if isinstance(t1, torch.Tensor):
                    if not torch.equal(t1, t2):
                        print(f"  Arg {arg_idx} differs!")
                        print(f"  Shape: {t1.shape}, Max diff: {(t1-t2).abs().max()}")
                        # Found the first divergence!
                        breakpoint()
```

## Benefits Over Previous Approach

1. **No need for commented-out code**: The functionality is now built-in and can be enabled/disabled via a flag

2. **Works with compiled code**: Previous DebugMode couldn't see inside inductor-generated code

3. **Precise kernel-level tracking**: Know exactly which kernel and which input first becomes non-deterministic

4. **Minimal overhead**: Hooks are only called when `record_triton_kernels=True`

5. **Clean API**: Integrates seamlessly with existing DebugMode features

## Performance Notes

- Enabling `record_triton_kernels=True` adds a small overhead (function call per kernel launch)
- Use `store_original_args=True` to keep tensor references for comparison (uses more memory)
- Without `store_original_args`, tensors are stringified immediately to save memory

## Examples

See:
- `test_triton_debug_mode.py` - Basic functionality test
- `test_determinism_debug.py` - Example of debugging non-determinism

## Future Enhancements

Possible extensions:
1. Capture kernel outputs as well as inputs
2. Add timing information per kernel
3. Support for recording kernel source code
4. Automatic diff reporting between runs
