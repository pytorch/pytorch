---
name: aoti-debug
description: Debug AOTInductor (AOTI) errors and crashes. Use when encountering AOTI segfaults, device mismatch errors, constant loading failures, or runtime errors from aot_compile, aot_load, aoti_compile_and_package, or aoti_load_package.
---

# AOTI Debugging Guide

This skill helps diagnose and fix common AOTInductor issues.

## Error Pattern Routing

**Check the error message and route to the appropriate sub-guide:**

### Triton Index Out of Bounds
If the error matches this pattern:
```
Assertion `index out of bounds: 0 <= tmpN < ksM` failed
```
**→ Follow the guide in `triton-index-out-of-bounds.md`**

### All Other Errors
Continue with the sections below.

---

## First Step: Always Check Device and Shape Matching

**For ANY AOTI error (segfault, exception, crash, wrong output), ALWAYS check these first:**

1. **Compile device == Load device**: The model must be loaded on the same device type it was compiled on
2. **Input devices match**: Runtime inputs must be on the same device as the compiled model
3. **Input shapes match**: Runtime input shapes must match the shapes used during compilation (or satisfy dynamic shape constraints)

```python
# During compilation - note the device and shapes
model = MyModel().eval()           # What device? CPU or .cuda()?
inp = torch.randn(2, 10)           # What device? What shape?
compiled_so = torch._inductor.aot_compile(model, (inp,))

# During loading - device type MUST match compilation
loaded = torch._export.aot_load(compiled_so, "???")  # Must match model/input device above

# During inference - device and shapes MUST match
out = loaded(inp.to("???"))  # Must match compile device, shape must match
```

**If any of these don't match, you will get errors ranging from segfaults to exceptions to wrong outputs.**

## Key Constraint: Device Type Matching

**AOTI requires compile and load to use the same device type.**

- If you compile on CUDA, you must load on CUDA (device index can differ)
- If you compile on CPU, you must load on CPU
- Cross-device loading (e.g., compile on GPU, load on CPU) is NOT supported

## Common Error Patterns

### 1. Device Mismatch Segfault

**Symptom**: Segfault, exception, or crash during `aot_load()` or model execution.

**Example error messages**:
- `The specified pointer resides on host memory and is not registered with any CUDA device`
- Crash during constant loading in AOTInductorModelBase
- `Expected out tensor to have device cuda:0, but got cpu instead`

**Cause**: Compile and load device types don't match (see "First Step" above).

**Solution**: Ensure compile and load use the same device type. If compiled on CPU, load on CPU. If compiled on CUDA, load on CUDA.

### 2. Input Device Mismatch at Runtime

**Symptom**: RuntimeError during model execution.

**Cause**: Input device doesn't match compile device (see "First Step" above).

**Better Debugging**: Run with `AOTI_RUNTIME_CHECK_INPUTS=1` for clearer errors. This flag validates all input properties including device type, dtype, sizes, and strides:
```bash
AOTI_RUNTIME_CHECK_INPUTS=1 python your_script.py
```

This produces actionable error messages like:
```
Error: input_handles[0]: unmatched device type, expected: 0(cpu), but got: 1(cuda)
```


## Debugging CUDA Illegal Memory Access (IMA) Errors

If you encounter CUDA illegal memory access errors, follow this systematic approach:

### Step 1: Sanity Checks

Before diving deep, try these debugging flags:

```bash
AOTI_RUNTIME_CHECK_INPUTS=1
TORCHINDUCTOR_NAN_ASSERTS=1
```

These flags take effect at compilation time (at codegen time):

- `AOTI_RUNTIME_CHECK_INPUTS=1` checks if inputs satisfy the same guards used during compilation
- `TORCHINDUCTOR_NAN_ASSERTS=1` adds codegen before and after each kernel to check for NaN

### Step 2: Pinpoint the CUDA IMA

CUDA IMA errors can be non-deterministic. Use these flags to trigger the error deterministically:

```bash
PYTORCH_NO_CUDA_MEMORY_CACHING=1
CUDA_LAUNCH_BLOCKING=1
```

These flags take effect at runtime:

- `PYTORCH_NO_CUDA_MEMORY_CACHING=1` disables PyTorch's Caching Allocator, which allocates bigger buffers than needed immediately. This is usually why CUDA IMA errors are non-deterministic.
- `CUDA_LAUNCH_BLOCKING=1` forces kernels to launch one at a time. Without this, you get "CUDA kernel errors might be asynchronously reported" warnings since kernels launch asynchronously.

### Step 3: Identify Problematic Kernels with Intermediate Value Debugger

Use the AOTI Intermediate Value Debugger to pinpoint the problematic kernel:

```bash
AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3
```

This prints kernels one by one at runtime. Together with previous flags, this shows which kernel was launched right before the error.

To inspect inputs to a specific kernel:

```bash
AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="triton_poi_fused_add_ge_logical_and_logical_or_lt_231,_add_position_embeddings_kernel_5" AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2
```

If inputs to the kernel are unexpected, inspect the kernel that produces the bad input.

## Additional Debugging Tools

### Logging and Tracing

- **tlparse / TORCH_TRACE**: Provides complete output codes and records guards used
- **TORCH_LOGS**: Use `TORCH_LOGS="+inductor,output_code"` to see more PT2 internal logs
- **TORCH_SHOW_CPP_STACKTRACES**: Set to `1` to see more stack traces

### Common Sources of Issues

- **Dynamic shapes**: Historically a source of many IMAs. Pay special attention when debugging dynamic shape scenarios.
- **Custom ops**: Especially when implemented in C++ with dynamic shapes. The meta function may need to be Symint'ified.

## API Notes

### Deprecated API
```python
torch._export.aot_compile()  # Deprecated
torch._export.aot_load()     # Deprecated
```

### Current API
```python
torch._inductor.aoti_compile_and_package()
torch._inductor.aoti_load_package()
```

The new API stores device metadata in the package, so `aoti_load_package()` automatically uses the correct device type. You can only change the device *index* (e.g., cuda:0 vs cuda:1), not the device *type*.

## Environment Variables Summary

| Variable | When | Purpose |
|----------|------|---------|
| `AOTI_RUNTIME_CHECK_INPUTS=1` | Compile time | Validate inputs match compilation guards |
| `TORCHINDUCTOR_NAN_ASSERTS=1` | Compile time | Check for NaN before/after kernels |
| `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | Runtime | Make IMA errors deterministic |
| `CUDA_LAUNCH_BLOCKING=1` | Runtime | Force synchronous kernel launches |
| `AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3` | Compile time | Print kernels at runtime |
| `TORCH_LOGS="+inductor,output_code"` | Runtime | See PT2 internal logs |
| `TORCH_SHOW_CPP_STACKTRACES=1` | Runtime | Show C++ stack traces |
