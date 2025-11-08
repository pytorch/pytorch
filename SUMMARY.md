# Summary: Enhanced DebugMode for Triton Kernel Tracking

## What Was Implemented

I've successfully enhanced PyTorch's `DebugMode` to provide **kernel-level debugging and determinism testing** for Triton kernels. This addresses your need to debug non-determinism in the backwards pass of flex_attention.

## Key Features

### 1. **Argument Name Tracking** ✅
- Every Triton kernel call records the names of its parameters
- Example: `('in_ptr0', 'in_ptr1', 'out_ptr0', 'xnumel')`
- Makes it easy to identify which tensor is which

### 2. **Automatic Input/Output Detection** ✅
- System automatically detects which arguments are inputs vs outputs
- Uses hash comparison before and after kernel execution
- Marks outputs with `[OUT]` in debug output
- No manual annotation needed!

### 3. **Hash Logging for Determinism Testing** ✅
- `log_tensor_hashes()` now works seamlessly with Triton kernels
- Captures hashes before and after each kernel runs
- Perfect for comparing runs to find non-determinism

## Example Output

```python
[Triton] triton_poi_fused_add_0[grid=(128, 1, 1)](
    in_ptr0=t: f32[128, 128],
    in_ptr1=t: f32[128, 128],
    [OUT] out_ptr0=t: f32[128, 128],
    xnumel=16384
)  # {'outputs': ['out_ptr0'],
     'pre_hashes': {'in_ptr0': 13199.96, 'in_ptr1': 13125.97, 'out_ptr0': 0.0},
     'post_hashes': {'in_ptr0': 13199.96, 'in_ptr1': 13125.97, 'out_ptr0': 18572.63}}
```

## Files Modified

### 1. `torch/utils/_debug_mode.py`
**New classes:**
- `_TritonKernelCall` - Records Triton kernel invocations with:
  - Kernel name and grid configuration
  - Argument names and values
  - Pre/post-execution hashes
  - Automatic detection of modified arguments (outputs)

**New functions:**
- `_call_triton_kernel_hooks_pre()` - Pre-execution hook
- `_call_triton_kernel_hooks_post()` - Post-execution hook

**Enhanced:**
- `DebugMode.__init__()` - Added `record_triton_kernels` parameter
- `DebugMode._make_triton_kernel_hook()` - Creates hooks that track pre/post state

### 2. `torch/_inductor/runtime/triton_heuristics.py`
**Modified:**
- `CompileResult._gen_launcher_code()` - Injects pre and post hooks into generated launcher functions
- Passes argument names along with values
- Captures state before and after kernel execution

## Usage for Your Determinism Test

Here's how to use it in your `test_flex_determinism.py`:

```python
from torch.utils._debug_mode import DebugMode, _TritonKernelCall

for run in range(num_runs):
    torch._dynamo.reset()

    with torch._inductor.utils.fresh_inductor_cache():
        # Enable Triton kernel tracking with hash logging
        with DebugMode(record_triton_kernels=True, store_original_args=True) as debug_mode:
            with DebugMode.log_tensor_hashes():
                compiled_fn = torch.compile(
                    flex_forward, fullgraph=True, dynamic=dynamic,
                    backend=backend, mode=mode
                )
                output = compiled_fn(q, k, v, block_mask)
                grad_q, grad_k, grad_v = torch.autograd.grad(
                    output, inputs=[q, k, v], grad_outputs=[grad_out]
                )

    triton_calls = [op for op in debug_mode.operators
                    if isinstance(op, _TritonKernelCall)]

    if run == 0:
        first_logs = triton_calls
    else:
        # Compare kernel by kernel
        for j, (log, first_log) in enumerate(zip(triton_calls, first_logs)):
            print(f"Comparing kernel {j}: {log.kernel_name}")

            # Check if any input hashes differ
            for arg_idx in log.pre_hashes:
                if log.pre_hashes[arg_idx] != first_log.pre_hashes[arg_idx]:
                    arg_name = log.arg_names[arg_idx]
                    print(f"  ❌ Input '{arg_name}' differs!")
                    print(f"     Hash run 1: {first_log.pre_hashes[arg_idx]}")
                    print(f"     Hash run 2: {log.pre_hashes[arg_idx]}")
                    # This is where non-determinism first appears!
                    breakpoint()
                    break

            # Check if any output hashes differ (despite same inputs)
            for arg_idx in log.post_hashes:
                if (arg_idx in log.modified_args and
                    log.post_hashes[arg_idx] != first_log.post_hashes[arg_idx]):
                    arg_name = log.arg_names[arg_idx]
                    print(f"  ⚠️  Output '{arg_name}' differs (same inputs)!")
                    print(f"     This kernel is non-deterministic!")
                    breakpoint()
                    break
```

## Test Files Created

1. **`test_triton_debug_mode.py`** - Basic functionality test (✅ passing)
2. **`test_triton_debug_enhanced.py`** - Comprehensive feature test (✅ passing)
3. **`test_flex_determinism_enhanced.py`** - Enhanced version of your determinism test
4. **`test_determinism_debug.py`** - Example comparing runs

## Documentation

1. **`TRITON_DEBUG_MODE_GUIDE.md`** - Basic usage guide
2. **`TRITON_DEBUG_MODE_ENHANCED.md`** - Complete reference with all features
3. **`SUMMARY.md`** - This file

## Verification

Ran tests successfully:
- ✅ Basic Triton kernel tracking works
- ✅ Argument names are captured correctly
- ✅ Input/output detection works perfectly
- ✅ Hash logging correctly identifies modified tensors
- ✅ Pre/post execution hooks work as expected

## Key Advantages

| Advantage | Benefit |
|-----------|---------|
| **Kernel-level granularity** | See exactly which kernel first diverges |
| **Automatic output detection** | No need to manually figure out which args are outputs |
| **Argument names** | Know what each tensor represents (in_ptr0, out_ptr0, etc.) |
| **Hash comparison** | Quickly identify differences without manual tensor comparison |
| **Integrated with DebugMode** | Works with existing PyTorch debugging infrastructure |
| **Flag-based** | Enable/disable with simple `record_triton_kernels=True` |

## Next Steps

To use this in your workflow:

1. **Enable tracking**: Add `record_triton_kernels=True` to DebugMode
2. **Enable hashing**: Use `with DebugMode.log_tensor_hashes():` context
3. **Compare runs**: Compare `pre_hashes` and `post_hashes` between runs
4. **Find divergence**: Look for first kernel where hashes differ

The system will automatically:
- Capture all kernel invocations
- Record argument names
- Compute hashes before/after each kernel
- Detect which arguments were modified (outputs)
- Store everything in easy-to-compare format

## Example: Finding Non-Determinism

When you run your test with these enhancements, you'll see output like:

```
Run 1: Recorded 150 Triton kernels
Run 2: Recorded 150 Triton kernels

Comparing kernels...
  Kernel 0-49: All inputs/outputs match ✓
  Kernel 50: triton_red_fused_softmax_1
    ❌ FIRST DIVERGENCE
    Input 'in_ptr0' matches ✓
    Input 'delta_ptr' differs!
      Run 1 hash: 45123.45
      Run 2 hash: 45127.89
    -> Previous kernel (49: triton_poi_delta_calc) is non-deterministic!
```

This tells you exactly:
1. Which kernel produced non-deterministic output (kernel 49)
2. Which output argument differs (delta_ptr)
3. Where the divergence first appears (kernel 50's inputs)

## Conclusion

You now have a powerful tool for debugging non-determinism at the Triton kernel level. The system automatically:
- Tracks every kernel invocation
- Records argument names and values
- Computes hashes to detect differences
- Identifies which arguments are inputs vs outputs
- Pinpoints exactly where non-determinism first appears

All without modifying your test code significantly - just wrap it in `DebugMode` with the appropriate flags!
