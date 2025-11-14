(dynamic_shapes_core_concepts)=
# Dynamic Shapes Core Concepts

This section described the core concepts of dynamic shapes in PyTorch. It is intended to be a
reference for engineers working on the PyTorch compiler stack and anyone who wants to understand
the inner workings of dynamic shapes.

## Symbolic integers
Symbolic integers (Symints) are used to represent variables that can span a range. For example:
```python
x = torch.randn(5, 5) # this tensor has a shape [5, 5]
torch._dynamo.decorators.mark_dynamic(x, 0)
x = torch.randn(5, 5) # this tensor has a shape [s0, 5]
y = torch.cat([x, x], dim=0) # this tensor has a shape [2*s0, 5]
```

However, `z = x * y` would throw an error since we know that pointwise operation like multiply must
operate on same sized tensors but we know statically `s0 != 2 * s0`. Astute readers may point out
that this is not true when `s0 == 0` and the reason why that doesn't matter here is described in
{ref}`zero-one-specialization`.

## Guards

In `torch.compile`, a guard is a mechanism that is used to ensure the validity of a compiled code graph.
By default, when you make a variable dynamic, it can range from `[-inf, inf]`. For example:

```python
def foo(x): return x / 2

This works for any dynamic x. But if your code is:

def foo(x)
    if x > 5:
        return x / 2
    return x / 3
```
If you call `foo(6)`, it returns `x / 2` and adds a guard `x > 5`. Calling `foo(4)` later will
require recompilation because the guard is broken.

## Runtime Asserts
You can use runtime asserts to provide hints when you know certain facts, like batch size being less than 100:

```python
def foo(batch_size):
    torch._check(batch_size < 100)
    if batch_size < 100:
        return do_something
    return do_something_else()
```

## "Hint" Value

A "hint value" in the context of `torch.compile` refers to the actual values known during the compilation process that help the JIT compiler make decisions about expressions. Hint values are particularly useful for handling dynamic shapes, as they provide concrete information that guides the compilation without requiring recompilation for varying dimensions.


## Dynamic Behavior Overview

PyTorch assumes static shapes by default. When a size change is detected, it attempts to
recompile with dynamic input, although this may fail if there are conditional branches
or missing support for dynamic shapes. To diagnose overspecialization, you can set
`TORCH_LOGS=dynamic` to view "eval" entries that indicate when and why guards are added.

If you anticipate a dimension will be dynamic, you can use `torch._dynamo.mark_dynamic(tensor, dim)`
to mark it in advance, specifying `min` and `max` values if known. Using `torch.compile(dynamic=False)`
disables automatic dynamic shapes, leading to recompilation for each unique size. Conversely,
`torch.compile(dynamic=True)` aims to use dynamic shapes as much as possible which is most useful
for small and may not be suitable for large models due to potential crashes or performance issues.

You can whitelist specific sources to be marked as dynamic using the `TORCH_COMPILE_DYNAMIC_SOURCES` environment variable or `torch.compiler.config.dynamic_sources`. This is particularly useful for large
models with graph breaks, as you can maintain dynamism across graph breaks since
source names stay consistent. You can also use this to mark integers as dynamic. The format is a comma-delimited list of source names, for example, `"L['x'], L['y']"`.
You can also use regexes, for example, `"L\['x.*'\], L\['y.*'\]")`.
This whitelist takes precedence over other flags like `dynamic=False` `force_nn_module_property_static_shapes`, and `force_parameter_static_shapes`.

Sometimes it can be cumbersome to find the right inputs to mark as dynamic. If
you're willing to take a performance hit for the first batch, one other affordable
option we have are the `eager_then_compile` stances which derive dynamism for you.
See {func}`torch.compiler.set_stance` for more details.


## Overall Architecture

Symbolic shapes workflow:

1. When compiling a frame in Dynamo, we allocate a `ShapeEnv` (attached to `FakeTensorMode`) to
track symbolic shapes.
2. We allocate symbolic sizes for tensors on entry, based on policy decisions.
3. We propagate symbolic sizes through operators, maintaining both FX IR for symbolic compute export
and Sympy expressions for reasoning.
4. We add guards based on conditionals during Dynamo tracing or Inductor optimization, induced from both Python and C++.
5. Guards can simplify symbolic variables. For instance, asserting `s0 == 4` allows replacing all occurrences of `s0` with `4`.
6. After tracing and optimizing, we install all guards with the compiled code, ensuring reusability only if all guards evaluate true.

## Internal API Class Hierarchy

### Python Classes

- **`SymInt`/`SymFloat`/`SymBool`**: User-visible classes that simulate their `int`/`float`/`bool` counterparts. Adding two `SymInts` produces a new `SymInt` that symbolically tracks the integer addition.

- **`SymNode`**: Internal structure (accessible via `symint.node`) that holds actual symbolic tracking information. `SymNode` is type-erased, making it convenient to represent mixed-type operations.

- **`ShapeEnv`**: Per-compile context state that tracks all free symbols and guards accumulated so far. Every `SymNode` records its `ShapeEnv` (but not vice versa; `SymNodes` are only used if they participate in a guard).

### C++ Equivalents

- **`c10::SymInt`/`SymFloat`/`SymBool`**: User-visible classes that simulate `int`/`float`/`bool`
- **`c10::SymNode`/`SymNodeImpl`**: Analogous to Python `SymNode`
- **No C++ `ShapeEnv`**: For debugging ease, the entire symbolic reasoning apparatus remains in Python

When writing code traceable with `make_fx`, it must handle `SymInt`/`SymFloat`/`SymBool` flowing through it.

## Value Ranges and Constraints

Symbolic variables maintain **value ranges** that specify the set of possible values. By default:
- Size-like unbacked `SymInts` have value range `[0, Inf]`
- Regular unbacked `SymInts` have value range `[-Inf, Inf]`

When assertions are made (e.g., `torch._check(x == y)`), the system:
1. Attempts to replace unbacked symbols with equivalent expressions
2. Refines value ranges based on the assertion
3. Remembers boolean expressions that are always true

Important files:

- C++ SymInt API: `c10/core/SymInt.h`, `SymFloat.h`, `SymBool.h`
- Python SymInt API: `torch/__init__.py` (look for `SymInt/SymFloat/SymBool`)
- C++ plumbing: `c10/core/SymNodeImpl.h`, `torch/csrc/utils/python_symnode.h`, `torch/csrc/jit/python/init.cpp`
- Python infrastructure: `torch/fx/experimental/symbolic_shapes.py`
- Other important files: `torch/_subclasses/fake_tensor.py`, `torch/_meta_registrations.py`, decomps, PrimTorch refs

```{seealso}
* {ref}`dynamic_shapes`
* {ref}`dynamic_shapes_troubleshooting`
```
