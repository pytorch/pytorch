# Guard Filter Function (`guard_filter_fn`) Usage Guide

## Overview

The `guard_filter_fn` is an advanced feature in PyTorch's TorchDynamo compilation system that allows users to control which guards are included in compiled functions. Guards are runtime checks that ensure the compiled code is valid for the current execution context. By filtering guards, users can trade safety for performance by reducing guard overhead.

**WARNING**: This is an **unsafe** feature with no backward compatibility guarantees. Use with extreme caution and only when you fully understand the implications.

## What are Guards?

Guards are runtime checks that TorchDynamo inserts to ensure the compiled code remains valid. They check conditions such as:
- Tensor shapes and dtypes
- Object identities (ID_MATCH)
- Global variable values
- Module attributes
- Type information
- Dictionary versions

When a guard fails, TorchDynamo recompiles the function with the new inputs.

## The `guard_filter_fn` Callback

### Function Signature

```python
from torch._dynamo.types import GuardFilterEntry

def guard_filter_fn(guard_entries: list[GuardFilterEntry]) -> list[bool]:
    """
    Args:
        guard_entries: List of GuardFilterEntry objects containing information about each guard

    Returns:
        list[bool]: A list of boolean values, one for each guard entry.
                   True = keep the guard, False = skip the guard
    """
    pass
```

### GuardFilterEntry Structure

Each `GuardFilterEntry` (defined in `torch/_dynamo/types.py:42`) contains:

```python
@dataclasses.dataclass(frozen=True)
class GuardFilterEntry:
    name: str                           # Name of the guarded variable (e.g., "x", "self.weight")
    has_value: bool                     # Whether we have a concrete value for this guard
    value: object                       # The actual value (if has_value is True)
    guard_type: str                     # Type of guard (e.g., "TENSOR_MATCH", "ID_MATCH")
    derived_guard_types: tuple[str, ...] # Additional guard types derived from this guard
    is_global: bool                     # Whether this guard is on a global variable
    orig_guard: Guard                   # The original Guard object with full details
```

### Common Guard Types

- `TENSOR_MATCH`: Tensor shape, dtype, device checks
- `ID_MATCH`: Object identity checks
- `DICT_VERSION`: Dictionary version checks
- `WEAKREF_ALIVE`: Weak reference liveness checks
- `CLOSURE_MATCH`: Closure variable checks
- `EQUALS_MATCH`: Value equality checks

## How It Works

The guard filtering process (see `torch/_dynamo/guards.py:3600-3649`):

1. **Initial Guard Build**: All guards are built first to populate metadata
2. **Filter Execution**: Your `guard_filter_fn` is called with all guard entries
3. **Guard Selection**: Only guards where you returned `True` are kept
4. **Rebuild**: Guards are rebuilt with the filtered set

This two-phase approach is necessary because filtering decisions may depend on guard metadata that's only available after the initial build.

## Usage

### Basic Usage with torch.compile

```python
import torch

def my_guard_filter(guard_entries):
    # Example: Skip all ID_MATCH guards
    return [entry.guard_type != "ID_MATCH" for entry in guard_entries]

@torch.compile(
    fullgraph=True,
    options={"guard_filter_fn": my_guard_filter}
)
def my_function(x):
    return x + 1
```

### Accessing guard_filter_fn in torch._dynamo.optimize

The `guard_filter_fn` parameter is available in the internal `torch._dynamo.optimize()` API (see `torch/_dynamo/eval_frame.py:1388`):

```python
import torch._dynamo as dynamo

def my_filter(entries):
    return [not entry.is_global for entry in entries]

@dynamo.optimize(
    backend="inductor",
    guard_filter_fn=my_filter
)
def my_function(x, y):
    return x + y
```

## Built-in Helper Functions

PyTorch provides several pre-built filter functions in the `torch.compiler` module:

### 1. `skip_guard_on_inbuilt_nn_modules_unsafe`

Skips guards on built-in nn.Module classes (e.g., `torch.nn.Linear`, `torch.nn.LayerNorm`).

**Use case**: When you don't modify built-in module attributes during execution.

```python
import torch
import torch.compiler

class MyModel(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = torch.nn.Linear(10, 10)

    def forward(self, x):
        return self.linear(x)

model = MyModel()
opt_model = torch.compile(
    model,
    options={"guard_filter_fn": torch.compiler.skip_guard_on_inbuilt_nn_modules_unsafe}
)
```

**Location**: `torch/compiler/__init__.py:536`

### 2. `skip_guard_on_all_nn_modules_unsafe`

Skips guards on all nn.Module instances (both built-in and user-defined).

**Use case**: When you never modify any module attributes.

```python
opt_model = torch.compile(
    model,
    options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe}
)
```

**Location**: `torch/compiler/__init__.py:557`

### 3. `keep_tensor_guards_unsafe`

Keeps only tensor guards, filtering out everything else.

**Use case**: When you only care about tensor shape/dtype changes.

```python
opt_model = torch.compile(
    model,
    options={"guard_filter_fn": torch.compiler.keep_tensor_guards_unsafe}
)
```

**Location**: `torch/compiler/__init__.py:579`

### 4. `skip_guard_on_globals_unsafe`

Skips all guards on global variables.

**Use case**: When global variables remain constant during execution.

```python
opt_model = torch.compile(
    model,
    options={"guard_filter_fn": torch.compiler.skip_guard_on_globals_unsafe}
)
```

**Location**: `torch/compiler/__init__.py:606`

### 5. `skip_all_guards_unsafe`

Skips ALL guards - extremely dangerous!

**WARNING**: This removes all safety guarantees. Only use for benchmarking or when you're absolutely certain inputs never change.

```python
opt_model = torch.compile(
    model,
    options={"guard_filter_fn": torch.compiler.skip_all_guards_unsafe}
)
```

**Location**: `torch/compiler/__init__.py:621`

## Practical Examples

### Example 1: Filter by Guard Type

```python
def filter_by_guard_type(guard_entries):
    # Skip ID_MATCH guards but keep everything else
    return [entry.guard_type != "ID_MATCH" for entry in guard_entries]

@torch.compile(fullgraph=True, options={"guard_filter_fn": filter_by_guard_type})
def fn(x):
    return id(x)
```

**Test**: `test/dynamo/test_misc.py:13082` (`test_guard_filter_fn_by_id`)

### Example 2: Filter by Name and Value

```python
def filter_by_name_value(guard_entries):
    # Skip guards on variable 'y' when it's None
    return [
        not (entry.name == "y" and entry.value is None)
        for entry in guard_entries
    ]

@torch.compile(fullgraph=True, options={"guard_filter_fn": filter_by_name_value})
def fn(x, y):
    if y is not None:
        x += y
    return x
```

**Test**: `test/dynamo/test_misc.py:13115` (`test_guard_filter_fn_by_name_and_value`)

### Example 3: Filter Globals

```python
GLOBAL_CONSTANT = 42

def skip_globals(guard_entries):
    return [not entry.is_global for entry in guard_entries]

@torch.compile(fullgraph=True, options={"guard_filter_fn": skip_globals})
def fn(x):
    return x + GLOBAL_CONSTANT
```

**Test**: `test/dynamo/test_misc.py:13097` (`test_guard_filter_fn_by_is_global`)

### Example 4: Custom Complex Filter

```python
def custom_filter(guard_entries):
    results = []
    for entry in guard_entries:
        # Keep tensor guards
        if entry.guard_type == "TENSOR_MATCH":
            results.append(True)
        # Skip ID_MATCH guards
        elif entry.guard_type == "ID_MATCH":
            results.append(False)
        # Skip globals
        elif entry.is_global:
            results.append(False)
        # Keep everything else
        else:
            results.append(True)
    return results
```

## Integration with AOT Compilation

When using AOT (Ahead-of-Time) compilation, `guard_filter_fn` has a default behavior that filters out certain guard types unsuitable for serialization (see `torch/_dynamo/aot_compile.py:194-211`):

```python
def default_aot_guard_filter(guard_entries):
    return [
        not (
            g.is_global
            or g.guard_type in CheckFunctionManager.UNSUPPORTED_SERIALIZATION_GUARD_TYPES
        )
        for g in guard_entries
    ]
```

Unsupported guard types for serialization typically include:
- `ID_MATCH`
- `CLOSURE_MATCH`
- `WEAKREF_ALIVE`
- `DICT_VERSION`

## Special Behavior with `caching_precompile`

When `torch._dynamo.config.caching_precompile` is enabled, guard filtering has additional behavior (see `torch/_dynamo/guards.py:3569-3596`):

The system automatically filters out guards unsuitable for precompilation:
- `ID_MATCH`
- `CLOSURE_MATCH`
- `WEAKREF_ALIVE`
- `DICT_VERSION`

Your custom filter is applied first, then these additional filters are layered on top.

## Common Patterns

### Combining Multiple Filters

```python
def combined_filter(guard_entries):
    # First apply built-in filter
    keep_flags = torch.compiler.skip_guard_on_globals_unsafe(guard_entries)

    # Then apply custom logic
    for i, entry in enumerate(guard_entries):
        if keep_flags[i] and entry.guard_type == "ID_MATCH":
            keep_flags[i] = False

    return keep_flags
```

### Debugging: Log Filtered Guards

```python
def debug_filter(guard_entries):
    results = []
    for entry in guard_entries:
        keep = entry.guard_type != "ID_MATCH"
        if not keep:
            print(f"Filtering out: {entry.name} ({entry.guard_type})")
        results.append(keep)
    return results
```

## Performance Implications

Filtering guards can:
- **Reduce guard evaluation overhead** (faster compiled function calls)
- **Reduce memory usage** (fewer guard objects to maintain)
- **Reduce recompilations** (fewer guards to fail)

But at the cost of:
- **Correctness risks** (wrong results if assumptions violated)
- **Harder debugging** (failures may occur far from the root cause)

## Safety Considerations

### When is it Safe to Filter Guards?

1. **Tensor guards**: Generally safe to keep
2. **ID_MATCH on inputs**: Safe to skip if you don't rely on object identity
3. **Module attribute guards**: Safe to skip if modules are frozen
4. **Global guards**: Safe to skip if globals never change
5. **DICT_VERSION guards**: Risky to skip (dictionary mutations)

### When is it Unsafe?

1. **Dynamic code** that modifies objects during execution
2. **Training code** that updates module parameters
3. **Code with side effects** that depend on guard checks
4. **Multi-threaded environments** with shared state

## Testing Your Filter Function

Always test with recompilation detection:

```python
# Test that unwanted recompilation doesn't occur
import torch.compiler

@torch.compile(
    options={"guard_filter_fn": my_filter}
)
def fn(x):
    return x + 1

# First call compiles
fn(torch.randn(3, 2))

# Second call with different input
with torch.compiler.set_stance("fail_on_recompile"):
    fn(torch.randn(3, 2))  # Should not recompile if guards properly filtered
```

## Debugging

### Inspecting Guards

Use `guard_export_fn` to see what guards are being generated:

```python
def export_guards(guards_set):
    for guard in guards_set.guards:
        print(f"Guard: {guard.name}, Type: {guard.create_fn_name()}")

@torch.compile(
    options={
        "guard_export_fn": export_guards,
        "guard_filter_fn": my_filter
    }
)
def fn(x):
    return x + 1
```

### Inspecting Guard Failures

Use `guard_fail_fn` to understand which guards are failing:

```python
from torch._dynamo.types import GuardFail

def on_guard_fail(fail_info: GuardFail):
    print(f"Guard failed: {fail_info.reason}")
    print(f"Code: {fail_info.orig_code}")

@torch.compile(
    options={"guard_fail_fn": on_guard_fail}
)
def fn(x):
    return x + 1
```

## Related APIs

- **`guard_export_fn`**: Callback to export/inspect generated guards
- **`guard_fail_fn`**: Callback when a guard fails at runtime
- **`guard_filter_fn`**: Callback to filter which guards to keep (this document)

All three are part of the `Hooks` dataclass (`torch/_dynamo/hooks.py:22`).

## Source Code References

Key files for understanding guard filtering:

1. **Guard filtering logic**: `torch/_dynamo/guards.py:3600-3649`
2. **GuardFilterEntry definition**: `torch/_dynamo/types.py:42`
3. **API entry point**: `torch/_dynamo/eval_frame.py:1388`
4. **Helper functions**: `torch/compiler/__init__.py:536-635`
5. **Tests**: `test/dynamo/test_misc.py:13082-13237`
6. **AOT integration**: `torch/_dynamo/aot_compile.py:194-211`

## Summary

`guard_filter_fn` is a powerful but dangerous tool for optimizing TorchDynamo compilation:

- ✅ Use it to reduce guard overhead when you know your code patterns
- ✅ Use built-in helpers when they match your use case
- ✅ Test thoroughly with recompilation detection
- ⚠️ Understand each guard type you're filtering
- ⚠️ Document why guards are safe to skip in your use case
- ❌ Don't use in production without extensive testing
- ❌ Don't skip guards you don't understand
- ❌ Don't use `skip_all_guards_unsafe` except for benchmarking

Remember: **Every filtered guard is a potential correctness bug waiting to happen.**
