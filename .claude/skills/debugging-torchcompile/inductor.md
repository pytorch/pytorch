# Debugging Inductor

Inductor is the default backend for torch.compile that generates optimized Triton (GPU) or C++ (CPU) code.

## Common Issues

### Viewing Generated Code

```python
import torch._inductor.config as config

# Print generated Triton/C++ code
config.debug = True

# Or use environment variable
# TORCH_COMPILE_DEBUG=1 python script.py
```

The debug output directory contains:
- `fx_graph_readable.py` - The FX graph before Inductor
- `fx_graph_runnable.py` - Runnable version of the FX graph
- `output_code.py` - Generated Triton/C++ code

### Codegen Errors

When Inductor fails to generate valid code:

```python
import torch._inductor.config as config

# Get detailed error information
config.debug = True
config.verbose_progress = True

# Disable specific optimizations to isolate the issue
config.pattern_matcher = False  # Disable pattern matching
config.max_autotune = False     # Disable autotuning
```

### Performance Issues

#### Checking Fusion
```python
import torch._inductor.config as config

# Log fusion decisions
# TORCH_LOGS="+inductor" python script.py

# See what kernels are generated
config.debug = True  # Look at output_code.py
```

#### Autotuning
```python
import torch._inductor.config as config

# Enable max autotuning (slower compile, faster runtime)
config.max_autotune = True
config.max_autotune_gemm = True

# Cache autotuning results
config.autotune_local_cache = True
config.autotune_remote_cache = True  # If available
```

### Triton-Specific Issues

```python
import torch._inductor.config as config

# Fall back to non-Triton code
config.triton.cudagraphs = False

# Debug Triton kernel issues
config.triton.debug_sync_kernel = True
config.triton.debug_sync_graph = True

# Check for Triton version issues
import triton
print(triton.__version__)
```

### CPU Codegen Issues

```python
import torch._inductor.config as config

# Force C++ backend
config.cpp_wrapper = True

# Debug C++ code
config.debug = True  # Generated code in output_code.cpp

# Disable vectorization to isolate issues
config.cpp.simdlen = 1
```

## Useful Configurations

```python
import torch._inductor.config as config

# Debugging
config.debug = True                    # Write debug files
config.verbose_progress = True         # Log compilation progress
config.trace.enabled = True            # Enable tracing

# Optimization control
config.pattern_matcher = True          # Enable pattern matching (default)
config.max_autotune = False            # Max autotuning (slower compile)
config.freezing = False                # Freeze weights into graph

# Triton settings
config.triton.cudagraphs = True        # Use CUDA graphs
config.triton.unique_kernel_names = True  # Unique names for profiling

# CPU settings
config.cpp.threads = -1                # Auto-detect threads
config.cpp.simdlen = None              # Auto-detect SIMD width
```

## Comparing Against Eager

```python
import torch

def check_correctness(fn, args):
    # Eager result
    eager_result = fn(*args)

    # Compiled result
    compiled_fn = torch.compile(fn)
    compiled_result = compiled_fn(*args)

    # Compare
    torch.testing.assert_close(eager_result, compiled_result)
    print("Results match!")

# Or use the built-in minifier for bug reports
# TORCHDYNAMO_REPRO_AFTER="aot" python script.py
```

## Environment Variables

```bash
# Inductor debug logging
TORCH_LOGS="+inductor"

# Full compilation debug output
TORCH_COMPILE_DEBUG=1

# Specific Inductor logging
TORCH_LOGS="+output_code"    # Print generated code
TORCH_LOGS="+fusion"         # Log fusion decisions
TORCH_LOGS="+schedule"       # Log scheduling decisions

# Disable specific features
TORCHINDUCTOR_TRITON_CUDAGRAPHS=0
TORCHINDUCTOR_FREEZING=0
```

## Minifier for Bug Reports

When you hit a crash, use the minifier to create a minimal reproduction:

```bash
# Minify after AOT Autograd
TORCHDYNAMO_REPRO_AFTER="aot" python script.py

# Minify after Inductor
TORCHDYNAMO_REPRO_AFTER="inductor" python script.py
```

This creates a standalone reproduction script in `minifier_launcher.py`.

## Profiling Generated Code

```python
import torch
from torch.profiler import profile, ProfilerActivity

compiled_fn = torch.compile(fn)

with profile(activities=[ProfilerActivity.CUDA]) as prof:
    compiled_fn(x)

print(prof.key_averages().table(sort_by="cuda_time_total"))
```

For kernel-level profiling:
```python
import torch._inductor.config as config
config.triton.unique_kernel_names = True  # Makes kernels identifiable in profiler
```

## AOT Inductor Intermediate Value Debug Printer

For debugging CUDA IMA kernels or numerical discrepancies in AOT Inductor, use the intermediate value debug printer:

```bash
# Print intermediate tensor values before/after each kernel
AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2 python script.py

# Print kernel names only (useful for pinpointing problematic kernels)
AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=3 python script.py

# Save intermediate tensor values to .pt files
AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=1 python script.py

# Filter to specific kernels
AOT_INDUCTOR_FILTERED_KERNELS_TO_PRINT="aoti_torch_cuda_addmm_out" \
  AOT_INDUCTOR_DEBUG_INTERMEDIATE_VALUE_PRINTER=2 python script.py
```

Saved tensors can be loaded for further debugging:
```python
tensor = torch.load("before_launch_aoti_torch_cuda_addmm_out_buf1_cuda:0.pt", weights_only=True)
print(tensor)
```
