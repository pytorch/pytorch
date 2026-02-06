# Debugging Dynamo

Dynamo is the Python bytecode tracer that captures PyTorch operations into FX graphs.

## Common Issues

### Graph Breaks

Graph breaks occur when Dynamo encounters code it cannot trace. This splits the model into multiple subgraphs with Python execution between them.

#### Finding Graph Breaks
```python
import torch._dynamo

# Explain what's happening during compilation
explanation = torch._dynamo.explain(model)(example_input)
print(explanation)

# Or use logging
# TORCH_LOGS="+graph_breaks" python script.py
```

#### Common Causes of Graph Breaks
- Data-dependent control flow (`if tensor.item() > 0`)
- Unsupported Python built-ins
- Calls to non-PyTorch libraries (NumPy, etc.)
- Dynamic list/dict operations
- Print statements with tensor values

#### Fixing Graph Breaks
```python
# Bad: causes graph break
if x.sum() > 0:
    return x * 2

# Good: use torch.where for simple conditionals
return torch.where(x.sum() > 0, x * 2, x)

# Bad: graph break from .item()
threshold = x.mean().item()

# Good: keep as tensor
threshold = x.mean()
```

### Recompilations (Guard Failures)

Dynamo generates "guards" - conditions that must hold for a compiled graph to be reused. When guards fail, recompilation occurs.

#### Diagnosing Recompilations
```python
import torch._dynamo
torch._dynamo.config.cache_size_limit = 8  # Reduce to catch issues faster

# Log guard failures
# TORCH_LOGS="+guards" python script.py
```

#### Common Guard Failure Causes
- Changing input shapes (when not using dynamic shapes)
- Changing dtypes
- Different tensor strides/contiguity
- Changing Python scalar values captured in the graph

#### Fixing Recompilations
```python
# Enable dynamic shapes to avoid shape-based recompilations
@torch.compile(dynamic=True)
def fn(x):
    return x * 2

# Or mark specific dimensions as dynamic
from torch._dynamo import mark_dynamic
mark_dynamic(tensor, dim=0)  # Batch dimension is dynamic
```

### Debugging Tracing Errors

```python
import torch._dynamo

# Don't suppress errors - see the full traceback
torch._dynamo.config.suppress_errors = False

# Get verbose output
torch._dynamo.config.verbose = True

# Trace what Dynamo is doing
# TORCH_LOGS="+dynamo" python script.py
```

## Useful Configurations

```python
import torch._dynamo.config as config

# Debugging
config.verbose = True                    # Verbose logging
config.suppress_errors = False           # Show full errors
config.cache_size_limit = 8              # Max cached graphs per function

# Graph break control
config.assume_static_by_default = True   # Treat shapes as static by default
config.automatic_dynamic_shapes = False  # Disable auto dynamic shapes

# Performance tuning
config.inline_inbuilt_nn_modules = True  # Inline nn.Module forward calls
```

## Inspecting the FX Graph

```python
from torch.fx import symbolic_trace

@torch.compile(backend="eager")  # Use eager backend to just see the graph
def fn(x):
    return x * 2 + 1

# Or use export to get the graph
from torch.export import export
exported = export(fn, (example_input,))
print(exported.graph_module.graph)
```

## Environment Variables

```bash
# Detailed Dynamo logging
TORCH_LOGS="+dynamo"

# Graph break logging
TORCH_LOGS="+graph_breaks"

# Guard logging (for recompilation debugging)
TORCH_LOGS="+guards"

# Bytecode logging (very verbose)
TORCH_LOGS="+bytecode"

# Combine multiple
TORCH_LOGS="+dynamo,+graph_breaks,+guards"
```

## Skipping Compilation

```python
import torch._dynamo

# Skip a function entirely
@torch._dynamo.disable
def do_not_compile(x):
    # This will run in eager mode
    return complex_python_code(x)

# Or disable for a block
with torch._dynamo.disable():
    result = uncompilable_code()
```
