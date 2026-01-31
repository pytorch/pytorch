# Debugging Export

Export creates portable graph representations from PyTorch models, handling dynamic shapes and producing serializable artifacts.

## Common Issues

### Export Failures

```python
from torch.export import export

# Basic export
try:
    exported = export(model, (example_input,))
except Exception as e:
    print(f"Export failed: {e}")
```

#### Common Failure Causes
- Data-dependent control flow
- Unsupported operations
- Dynamic shape constraint violations
- In-place mutations on inputs

### Dynamic Shapes

#### Specifying Dynamic Shapes
```python
from torch.export import export, Dim

# Define dynamic dimensions
batch = Dim("batch", min=1, max=1024)
seq_len = Dim("seq_len", min=1, max=512)

# Use in export
dynamic_shapes = {
    "x": {0: batch, 1: seq_len},
}

exported = export(model, (example_input,), dynamic_shapes=dynamic_shapes)
```

#### Debugging Shape Constraints
```python
from torch.export import export

# Get verbose constraint information
try:
    exported = export(model, (example_input,))
except Exception as e:
    # Error messages often include suggested constraints
    print(e)
```

#### Common Shape Issues
```python
# Bad: hardcoded shape causes export failure with dynamic shapes
def forward(self, x):
    return x.view(32, -1)  # Assumes batch size is 32

# Good: use -1 for dynamic dimensions
def forward(self, x):
    return x.view(x.size(0), -1)  # Batch size is dynamic

# Bad: shape-dependent indexing
def forward(self, x):
    if x.size(0) > 10:
        return x[:10]

# Good: use torch.where or clamp
def forward(self, x):
    n = torch.clamp(torch.tensor(x.size(0)), max=10)
    return x[:n]
```

### Strict vs Non-Strict Mode

```python
from torch.export import export

# Strict mode (default) - requires pure functional code
exported = export(model, args, strict=True)

# Non-strict mode - more permissive but may miss issues
exported = export(model, args, strict=False)
```

#### When to Use Non-Strict
- Gradual migration from eager to export
- Models with minor mutations that don't affect correctness
- Debugging to isolate strict mode issues

### In-Place Operations

Export requires functional code. In-place operations on inputs are not allowed.

```python
# Bad: in-place on input
def forward(self, x):
    x.add_(1)  # In-place mutation
    return x

# Good: return new tensor
def forward(self, x):
    return x + 1
```

### Control Flow

Export has limited support for data-dependent control flow.

```python
from torch.export import export

# Bad: data-dependent if
def forward(self, x):
    if x.sum() > 0:  # Data-dependent
        return x * 2
    return x

# Option 1: Use torch.cond for simple cases
from torch._higher_order_ops.cond import cond

def forward(self, x):
    return cond(
        x.sum() > 0,
        lambda x: x * 2,
        lambda x: x,
        (x,)
    )

# Option 2: Use torch.where for element-wise
def forward(self, x):
    return torch.where(x.sum() > 0, x * 2, x)
```

### Debugging Exported Graphs

```python
from torch.export import export

exported = export(model, (example_input,))

# Print the graph
print(exported.graph_module.graph)

# Print readable format
exported.graph_module.print_readable()

# Get the graph signature (inputs/outputs)
print(exported.graph_signature)

# Run the exported model
output = exported.module()(example_input)
```

### Serialization Issues

```python
from torch.export import export, save, load

exported = export(model, (example_input,))

# Save
save(exported, "model.pt2")

# Load
loaded = load("model.pt2")

# Verify
torch.testing.assert_close(
    exported.module()(example_input),
    loaded.module()(example_input)
)
```

## Useful Configurations

```python
import torch._dynamo.config as dynamo_config

# Verbose export logging
dynamo_config.verbose = True

# Show constraints
dynamo_config.suppress_errors = False
```

## Environment Variables

```bash
# Export debugging
TORCH_LOGS="+export"

# Dynamo tracing (export uses Dynamo)
TORCH_LOGS="+dynamo"

# Combined
TORCH_LOGS="+export,+dynamo"
```

## Pre-Dispatch vs Post-Dispatch Export

```python
from torch.export import export

# Pre-dispatch (default): captures high-level ops
exported = export(model, args)

# The graph contains high-level ATen ops
print(exported.graph_module.graph)
```

Pre-dispatch is preferred for:
- Readability
- Portability across backends
- Optimizations that operate on high-level ops

## Custom Ops in Export

```python
from torch.library import Library, impl

# Register custom op for export
lib = Library("mylib", "FRAGMENT")
lib.define("custom_op(Tensor x) -> Tensor")

@impl(lib, "custom_op", "CompositeExplicitAutograd")
def custom_op_impl(x):
    return x * 2

# Now it can be exported
@torch.compile
def fn(x):
    return torch.ops.mylib.custom_op(x)
```

## Comparing Export with torch.compile

| Aspect | torch.compile | export |
|--------|---------------|--------|
| Use case | Runtime optimization | Deployment/serialization |
| Dynamic shapes | Automatic/optional | Explicit specification |
| Graph breaks | Allowed (falls back to eager) | Not allowed |
| Control flow | Python control flow OK | Must use higher-order ops |
| Result | JIT-compiled function | Portable graph module |
