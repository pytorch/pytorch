# Debugging AOT Inductor

AOT Inductor (Ahead-Of-Time Inductor) compiles exported PyTorch models into shared libraries for deployment in non-Python environments. This guide covers debugging AOT Inductor issues, particularly CUDA illegal memory access (IMA) errors.

## When to Use

- Debugging `torch._inductor.aoti_compile_and_package` failures
- Debugging `torch._inductor.aoti_load_package` issues
- Investigating CUDA IMA errors in AOT compiled models
- Creating minimal reproductions for bug reports

## Quick Reference: Debugging Workflow

1. **Sanity checks** - Use basic debugging flags
2. **Make error deterministic** - Disable caching and async launches
3. **Identify problematic kernel** - Use intermediate value debugger
4. **Create minimal repro** - Use the AOT Inductor minifier

## Step 1: Sanity Checks

Before deep debugging, try these flags at compilation time:

```bash
# Check if inputs satisfy compilation guards
AOTI_RUNTIME_CHECK_INPUTS=1 python script.py

# Add NaN checks before and after each kernel
TORCHINDUCTOR_NAN_ASSERTS=1 python script.py
```

## Step 2: Make CUDA IMA Errors Deterministic

CUDA IMA errors are often non-deterministic due to PyTorch's caching allocator. Use these runtime flags:

```bash
# Disable PyTorch's caching allocator (makes IMA deterministic)
PYTORCH_NO_CUDA_MEMORY_CACHING=1

# Force synchronous kernel launches (see exact error location)
CUDA_LAUNCH_BLOCKING=1

# Combined
PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_LAUNCH_BLOCKING=1 python script.py
```

**Why this works:**
- The caching allocator allocates larger buffers than needed, masking out-of-bounds access
- Async kernel launches report errors at later API calls, not the actual error location

## Step 3: Identify Problematic Kernels

Use the Intermediate Value Debug Printer to find which kernel caused the error.

### Print kernel names only (fastest)

```bash
AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3 python script.py
```

This prints kernel names one by one at runtime. The last kernel printed before the error is the starting point for investigation.

### Print tensor values for specific kernels

```bash
AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="triton_poi_fused_add_0,kernel_name_2" \
  AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2 python script.py
```

This prints tensor values before and after the specified kernels.

### Save intermediate tensors to files

```bash
AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="triton_poi_fused_add_0" \
  AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=1 python script.py
```

Tensors are saved as `.pt` files in a temp directory. Load them for debugging:

```python
tensor = torch.load("before_launch_triton_poi_fused_add_0_buf1_cuda:0.pt", weights_only=True)
print(tensor)
```

### Debug Printer Modes

| Mode | Value | Description |
|------|-------|-------------|
| SAVE_ONLY | 1 | Save tensor values to .pt files |
| PRINT_ONLY | 2 | Print tensor stats before/after kernels |
| PRINT_KERNEL_NAME_ONLY | 3 | Print kernel names only (fastest) |

## Step 4: Use the AOT Inductor Minifier

The minifier creates a minimal reproduction of the error.

### Enable the minifier

```python
import torch._inductor.config as config
config.aot_inductor.dump_aoti_minifier = True

# Or via environment variable
# DUMP_AOTI_MINIFIER=1 python script.py
```

### Workflow

1. Run your script with the minifier enabled
2. Find `minifier_launcher.py` in the debug output directory
3. Run the minifier launcher to generate `repro.py`

```bash
# Step 1: Enable minifier and run
DUMP_AOTI_MINIFIER=1 python your_script.py
# Output: Writing minified repro to: /path/to/minifier/minifier_launcher.py

# Step 2: Run the minifier
python /path/to/minifier/minifier_launcher.py
# Output: Wrote minimal repro out to repro.py

# Step 3: Use repro.py for debugging or bug reports
python repro.py
```

### Configure output directory

```python
import torch._dynamo.config
torch._dynamo.config.debug_dir_root = "/path/to/debug/output"
```

### Minifier launcher options

The generated `minifier_launcher.py` supports two commands:

```python
# Run minifier to create minimal repro
run_repro(exported_program, command='minify', ...)

# Just compile/load/run without minifying
run_repro(exported_program, command='run', ...)
```

## Additional Debugging Tools

### Logging

```bash
# Inductor logs with output code
TORCH_LOGS="+inductor,output_code" python script.py

# Show C++ stack traces
TORCH_SHOW_CPP_STACKTRACES=1 python script.py

# Use tlparse for comprehensive reports
TORCH_TRACE=/path/to/trace python script.py
tlparse /path/to/trace -o report
```

### Guard inspection

Use `tlparse` to see the guards used during compilation, which helps debug runtime failures.

## Common Issues

### Dynamic Shapes

Dynamic shapes are a common source of CUDA IMA errors. Pay attention to:
- Shape constraints during export
- Guard failures at runtime
- Buffer size calculations in generated code

```python
# Export with explicit dynamic dimensions
batch_dim = torch.export.Dim("batch", min=1, max=1024)
exported = torch.export.export(model, inputs, dynamic_shapes={"x": {0: batch_dim}})
```

### Custom Ops

Custom ops, especially in C++, need proper SymInt handling for dynamic shapes:
- Ensure meta functions are SymInt-aware
- Test with both static and dynamic shapes

### Input Validation

Runtime input validation can catch mismatches:

```bash
# Validate inputs match export guards
AOTI_RUNTIME_CHECK_INPUTS=1 python script.py
```

## Environment Variables Summary

| Variable | When | Purpose |
|----------|------|---------|
| `AOTI_RUNTIME_CHECK_INPUTS=1` | Compile | Check inputs match guards |
| `TORCHINDUCTOR_NAN_ASSERTS=1` | Compile | Add NaN checks around kernels |
| `PYTORCH_NO_CUDA_MEMORY_CACHING=1` | Runtime | Deterministic CUDA IMA |
| `CUDA_LAUNCH_BLOCKING=1` | Runtime | Synchronous kernel launches |
| `AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=N` | Compile | Debug printer (1=save, 2=print, 3=names) |
| `AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="..."` | Compile | Filter kernels for debug printer |
| `DUMP_AOTI_MINIFIER=1` | Compile | Enable minifier |

## Example: Full Debugging Session

```bash
# 1. Make error deterministic
PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_LAUNCH_BLOCKING=1 python script.py

# 2. Find problematic kernel
PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_LAUNCH_BLOCKING=1 \
  AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3 python script.py
# Note: Last kernel printed is "triton_poi_fused_add_0"

# 3. Inspect that kernel's inputs/outputs
PYTORCH_NO_CUDA_MEMORY_CACHING=1 CUDA_LAUNCH_BLOCKING=1 \
  AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="triton_poi_fused_add_0" \
  AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2 python script.py

# 4. Create minimal repro
DUMP_AOTI_MINIFIER=1 python script.py
python /path/to/minifier_launcher.py
# Now use repro.py for bug report
```
