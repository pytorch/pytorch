# Common Debugging Tools

This document covers environment variables, config options, and logging that apply across the torch.compile stack.

## TORCH_LOGS Environment Variable

The primary way to enable logging across the stack:

```bash
# Basic component logging
TORCH_LOGS="+dynamo"           # Dynamo tracing
TORCH_LOGS="+inductor"         # Inductor codegen
TORCH_LOGS="+export"           # Export process

# Specific topics
TORCH_LOGS="+graph_breaks"     # Graph break reasons
TORCH_LOGS="+guards"           # Guard failures (recompilations)
TORCH_LOGS="+output_code"      # Generated Triton/C++ code
TORCH_LOGS="+fusion"           # Fusion decisions
TORCH_LOGS="+schedule"         # Scheduling decisions
TORCH_LOGS="+bytecode"         # Bytecode analysis (very verbose)
TORCH_LOGS="+aot_graphs"       # AOT Autograd graphs

# Combine multiple
TORCH_LOGS="+dynamo,+inductor,+graph_breaks"

# Set log levels
TORCH_LOGS="dynamo:INFO,inductor:DEBUG"

# Log to file
TORCH_LOGS_OUT="/path/to/logfile.txt"
```

## TORCH_COMPILE_DEBUG

Comprehensive debug mode that dumps intermediate representations:

```bash
TORCH_COMPILE_DEBUG=1 python script.py
```

Creates a debug directory with:
- `fx_graph_readable.py` - FX graph in readable format
- `fx_graph_runnable.py` - Runnable FX graph
- `output_code.py` - Generated Triton/C++ code
- Debug traces and logs

## Config Namespaces

### torch._dynamo.config
```python
import torch._dynamo.config as dynamo_config

# Debugging
dynamo_config.verbose = True
dynamo_config.suppress_errors = False
dynamo_config.log_level = logging.DEBUG

# Graph capture
dynamo_config.cache_size_limit = 8
dynamo_config.assume_static_by_default = True
dynamo_config.automatic_dynamic_shapes = True

# Inlining
dynamo_config.inline_inbuilt_nn_modules = True
```

### torch._inductor.config
```python
import torch._inductor.config as inductor_config

# Debugging
inductor_config.debug = True
inductor_config.verbose_progress = True
inductor_config.trace.enabled = True

# Codegen
inductor_config.triton.cudagraphs = True
inductor_config.cpp_wrapper = False
inductor_config.max_autotune = False

# Pattern matching
inductor_config.pattern_matcher = True
inductor_config.freezing = False
```

### torch._functorch.config
```python
import torch._functorch.config as functorch_config

# AOT Autograd settings
functorch_config.debug_assert = True
```

## Using Config Patches

For temporary config changes in tests or debugging:

```python
import torch._dynamo.config

# As context manager
with torch._dynamo.config.patch(verbose=True, suppress_errors=False):
    result = compiled_fn(x)

# As decorator
@torch._dynamo.config.patch(verbose=True)
def test_my_feature():
    pass
```

## Resetting State

```python
import torch._dynamo

# Clear all cached compilations
torch._dynamo.reset()

# Clear caches and start fresh
torch.compiler.reset()
```

## Useful Environment Variables

| Variable | Purpose |
|----------|---------|
| `TORCH_LOGS` | Enable component logging |
| `TORCH_COMPILE_DEBUG` | Full debug output |
| `TORCHDYNAMO_REPRO_AFTER` | Generate minified repro |
| `TORCH_LOGS_OUT` | Log output file |
| `TORCHINDUCTOR_TRITON_CUDAGRAPHS` | Enable/disable CUDA graphs |
| `TORCHINDUCTOR_FREEZING` | Enable/disable weight freezing |
| `TORCHDYNAMO_VERBOSE` | Verbose Dynamo output |
| `TORCHDYNAMO_CACHE_SIZE_LIMIT` | Max cached graphs |

## Logging Integration

```python
import logging

# Set up Python logging for torch components
logging.getLogger("torch._dynamo").setLevel(logging.DEBUG)
logging.getLogger("torch._inductor").setLevel(logging.DEBUG)

# Or use the torch logging config
import torch._logging
torch._logging.set_logs(dynamo=logging.DEBUG)
```

## Structured Tracing

For production debugging with tlparse:

```python
from torch._logging import trace_structured

# Log debug artifacts
trace_structured(
    "artifact",
    metadata_fn=lambda: {
        "name": "debug_info",
        "encoding": "string",
    },
    payload_fn=lambda: debug_content,
)

# Check if tracing is enabled
from torch._logging._internal import trace_log
if trace_log.handlers:
    print("Use tlparse to extract debug artifacts")
```

## Profiling Compilation

```python
import torch._dynamo
import time

# Time compilation
start = time.time()
compiled_fn = torch.compile(fn)
_ = compiled_fn(x)  # First call triggers compilation
compile_time = time.time() - start
print(f"Compilation took: {compile_time:.2f}s")

# Profile with torch.profiler
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
    compiled_fn(x)

print(prof.key_averages().table())
```

## Quick Reference: Debug Levels

| Level | What You Get |
|-------|--------------|
| Basic | `torch._dynamo.config.verbose = True` |
| Graph breaks | `TORCH_LOGS="+graph_breaks"` |
| Full trace | `TORCH_LOGS="+dynamo,+inductor"` |
| Generated code | `TORCH_COMPILE_DEBUG=1` |
| Everything | `TORCH_LOGS="+dynamo,+inductor,+graph_breaks,+guards,+output_code"` |

## tlparse / TORCH_TRACE

Generate structured compilation reports that can be visualized:

```bash
# Generate trace file
TORCH_TRACE=/path/to/trace_dir python script.py

# Parse the trace into a readable report
pip install tlparse
tlparse /path/to/trace_dir -o report_output
```

The report includes:
- Compilation timeline
- Graph breaks with explanations
- Guard failures and recompilations
- Generated code for each compiled region

Example report: https://web.mit.edu/~ezyang/Public/bhack-20240609-tlparse/index.html

### Extended Debug Information

For deeper debugging, use these environment variables:

```bash
# Debug specific guard creation
TORCHDYNAMO_EXTENDED_DEBUG_GUARD_ADDED="Ne(s0, 10)" python script.py

# Debug specific symbol creation
TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="u2" python script.py

# Enable C++ backtraces (slow but detailed)
TORCHDYNAMO_EXTENDED_DEBUG_CPP=1 python script.py
```

### Cold Start / Cache Debugging

```bash
# Disable all caches to measure cold start time
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python script.py

# Or programmatically
torch.compiler.config.force_disable_caches = True
```
