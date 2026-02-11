(troubleshooting_guardondatadependentsymnode_errors)=

# Troubleshooting GuardOnDataDependentSymNode Errors
When working with PyTorch models that have unbacked symbols which could be coming from data dependent ops like `item()`, `tolist()`, or `nonzero()`, or from manually marking some input sizes as dynamic using `torch._dynamo.decorators.mark_unbacked` you may encounter `GuardOnDataDependentSymNode` errors. This section explains what these errors are and how to fix them.

## Background:

**Backed dynamic shapes** emerged as a solution to the "endless recompilations" problem in PyTorch 2. When a function like `torch.ones(x)` was compiled with `x=10`, without dynamic shapes, Dynamo would insert a guard checking that "the input x is exactly 10" and generate a graph hard-coded for size 10. Calling with `x=20` would trigger another compilation, and so on.

To solve this, dynamic shapes can be used to stop hard-coding sizes and represent them symbolically. However, the compiler still needed to make branching decisions (e.g., `if x < 1024`), so we "backed" each dynamic shape with a hint; a concrete value from the example input used during compilation. The hint guides branch selection, and Dynamo adds guards ensuring the branch condition remains valid. These are called *backed* (or *guardable*) shapes because they are backed by a hint and can have guards constraining them.

**Unbacked dynamic shapes** arose from a different need: supporting data-dependent operations like `x.item()`. For such operations, the output value depends on tensor data and is unknown at compile time. Initially, these would trigger graph breaks, but this was problematic for export and performance. To keep data-dependent operations within the graph, we represent their outputs symbolically—but unlike backed shapes, we have no hint to resolve branching. These are called *unbacked* (or *guardless*) shapes. Over time, users have also deliberately chosen unbacked shapes for primary graph inputs to avoid branch-induced recompilations and compile graphs that work across all input shapes.

### Data-Dependent Errors

A key challenge with unbacked shapes is handling branches: without a hint, the compiler cannot determine which path to take, and the default behavior is to throw a `GuardOnDataDependentSymNode` error.

## Framework vs User Code Errors

Data-dependent errors (DDEs) can originate from two sources: **framework code** (PyTorch internals) and **user code** (your model). Historically, DDEs were a major pain point -especially for export users— because many common framework operations like reshaping, slicing, narrowing, selection, contiguity checks, and broadcasting checks would trigger these errors when encountering unbacked shapes.

**Framework code should no longer throw DDEs.** We have implemented explicit unbacked semantics throughout the PyTorch framework, addressing major code branches and eliminating the vast majority of framework-originated DDEs. Operations that previously failed—such as `view`, `narrow`, `select`, and various shape checks now handle unbacked shapes correctly by automatically selecting general code paths that work for all input values (by sometimes potentially deviating from eager semantics). This means you can now capture specialization-free graphs much more reliably without hitting framework DDEs.

If you encounter a DDE originating from PyTorch framework code (identifiable by the "Potential framework code culprit" in the error message pointing to files under `torch/`), this is likely a bug that should be reported, and fixed using the same methods explained later in this document.

Note that some operations are inherently not unbacked-friendly because they require knowing the exact value of a dynamic shape. The DDEs you may encounter will typically originate from **user code**—branches in your model that depend on data-dependent values.

The rest of this document explains how to deal with unbacked shapes in your code. The solutions generally fall into two categories:

1. **Avoid the DDE by rewriting your code to be resilient** — restructure your code so that it doesn't require branching on unbacked symbols, or use alternative APIs that handle unbacked shapes gracefully.
2. **Provide hints using `torch._check`** — when rewriting is not feasible, to teach the symbolic reasoning system facts about your unbacked `SymInts`.

## Common Error Pattern
The following output shows the common error pattern `GuardOnDataDependentSymNode` errors:

```sh
torch.fx.experimental.symbolic_shapes.GuardOnDataDependentSymNode: Could not guard on data-dependent expression Eq(u2, -1) (unhinted: Eq(u2, -1)).  (Size-like symbols: none)

Potential framework code culprit (scroll up for full backtrace):
  File "/data/users/ezyang/a/pytorch/torch/_prims_common/__init__.py", line 855, in infer_size
    if d == -1:

For more information, run with TORCH_LOGS="dynamic"
For extended logs when we create symbols, also add TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="u2"
If you suspect the guard was triggered from C++, add TORCHDYNAMO_EXTENDED_DEBUG_CPP=1
For more debugging help, see https://docs.google.com/document/d/1HSuTTVvYH1pTew89Rtpeu84Ht3nQEFTYhAX3Ypa_xJs/edit?usp=sharing
```

## Debugging Tools

Here is the list of some of the debugging tools available in PyTorch that you can use to troubleshoot these errors:

* `TORCH_LOGS="+dynamic"` - Shows detailed logs about symbolic operations
* `TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="u2"` - Provides extended logs for specific symbols
* `TORCHDYNAMO_EXTENDED_DEBUG_CPP=1` - Helps when guards are triggered from C++

## Error Variations

Here is the list of error variations that you might encounter:

| Error Variations | Description |
|------------------|-------------|
| "Could not guard on data-dependent expression" | Occurs when trying to extract a concrete boolean from expressions like u0 == 0 or u0 > 10 |
| "Could not extract specialized integer from data-dependent expression" | Occurs when trying to extract a concrete integer value. <br/> **Common causes:** <br/> - Control flow that depends on the integer (such as, looping `u0` times) <br/> - Overspecialization in code that could work symbolically |

## How to Diagnose Your Problem

### Step 1: Examine the Potential Culprit (Python Backtrace)

The exception provides a backtrace, which often indicates the problem.
Given that PT2 backtraces can be lengthy, the error message will also
suggest a potential framework culprit. For example:

```sh
Potential framework code culprit (scroll up for full backtrace):
  File "/data/users/ezyang/a/pytorch/torch/_prims_common/__init__.py", line 855, in infer_size
    if d == -1:
```
### Step 2: Examine the C++ Backtrace

If the framework code culprit is uninformative, the guard might be in C++. You can
force a C++ backtrace by running with `TORCHDYNAMO_EXTENDED_DEBUG_CPP=1`. This
provides a detailed C++ backtrace with Python, CPython, and C10/ATen/libtorch
frames interspersed. Look for symbols in the `at::` or `c10::` namespace that
resemble kernel-specific code, likely related to the kernel executed per the Python
backtrace. If using a non-debug build of PyTorch, inlining may cause missing
frames, requiring source code investigation to locate the issue.
For example, see https://github.com/pytorch/pytorch/pull/118579.

Here is an example C++ backtrace from a debugging session:

```
[2024-02-08 08:20:45,259] torch.fx.experimental.symbolic_shapes: [INFO]   File "../
__gen_aten__/out/RegisterCompositeImplicitAutograd.cpp", line 2025, in at::
(anonymous namespace)::(anonymous namespace)
::wrapper_CompositeImplicitAutograd_Tensor_narrow(at::Tensor const&, long,
at::Tensor const&, c10::SymInt) [2024-02-08 08:20:45,259] torch.fx.experimental.
symbolic_shapes: [INFO]   File "../aten/src/ATen/native/TensorShape.cpp", line 1410,
in at::native::narrow_tensor_symint(at::Tensor const&, long, at::Tensor const&,
c10::SymInt) [2024-02-08 08:20:45,259] torch.fx.experimental.symbolic_shapes:
[INFO]   File "../__gen_aten__/out/core/TensorMethods.cpp", line 52, in long
at::Tensor::item<long>() const [2024-02-08 08:20:45,259] torch.fx.experimental.
symbolic_shapes: [INFO]   File "../ATen/core/TensorBody.h", line 4274, in
at::Tensor::item() const
```

In this example, `at::native::narrow_tensor_symint` calls into `item<long>`, which
triggers the guard on a data-dependent `SymNode`.

**Consider the Following:**

* Does it make sense that this condition is triggering a guard on a
data-dependent symbol?
* If the equation involves two distinct symbols, should we know
they are actually equal?
* Is it possible to teach that piece of code how to handle inputs in
a generic way that works for all shapes?

Using `TORCH_LOGS=dynamic` and examining the user stack trace is crucial for
understanding how to fix the problem, as they guide you on how to modify the
user program.

```sh
[INFO] create_unbacked_symint u0 [-9223372036854775808, 9223372036854775807] (w.py:40 in custom_op_meta)
```

This log message indicates where (`w.py:40`) the unbacked `SymInt` was
allocated. An unbacked `SymInt` may be allocated multiple times, so track
their equalities:

```sh
[INFO] set_replacement u1 = u0 (trivial_lhs) ValueRanges(lower=0, upper=9223372036854775807, is_bool=False)
```

## Fixing the Error

Once you've identified the source of the error, ask yourself the following questions in order:

### Step 1: Can I rewrite my code to use a general path?

The best solution is to restructure your code so that it doesn't require branching on unbacked symbols at all. Ask yourself: **Is there a general code path that works for all shapes?**

For example, instead of:
```python
i = x.item()
if i > 4:
    return x * 2
else:
    return x + 3
```

Can you rewrite the logic to work without the branch? If the branch exists only for optimization or edge-case handling, consider designating a general path that handles all shapes.

#### Useful Utilities for Mindful Branching

PyTorch provides several utilities to express branching in a more dynamic shapes friendly manner:

**`statically_known_true(expr)`**: It:
- Never adds a new guard (no recompilation risk)
- Never fails on data dependency.

The API tries to evaluate the expression without adding guards. If it cannot, it returns `False`. Use this for short circuits that don't affect performance or optimizations that don't warrant recompilation.

```python
from torch.fx.experimental.symbolic_shapes import statically_known_true

# Instead of: if x.numel() > 10:
if statically_known_true(x.numel() > 10):
    # optimization path
    ...
else:
    # general path (taken when unknown)
    ...
```

**`guard_or_false(expr)` / `guard_or_true(expr)`**: These may add guards (if symbols are backed) but will never fail with data-dependent errors. If evaluation fails due to data dependency, they return `False` or `True` instead of hard failing. Use for performance optimizations that warrant recompilation:

```python
from torch.fx.experimental.symbolic_shapes import guard_or_false

# Instead of: if x == 0:
if guard_or_false(x == 0):
    return 1
else:
    torch._check(x != 0)  # runtime check for the general path
    return compute(x)
```

**`hint_int(expr, fallback=None)`**: Extracts a hint from a symbolic size and uses it in a branch. You can use it to evaluate the expression using the traced program's input shapes without guarding. Unlike `statically_known_true`, it picks the path that works for the input example instead of returning `False`. The optional `fallback` argument substitutes unbacked symbols; if not provided and the symbol is unbacked, it will raise an error.

```python
from torch.fx.experimental.symbolic_shapes import hint_int

# Use ONLY for optimizations, not correctness-critical branches
if hint_int(x.numel(), fallback=0) > 1024:
    # optimized path for large tensors
    ...
else:
    # general path
    ...
```

**Important:** These utilities should only be used for optimizations that do not require guarding (e.g., selecting a faster code path). Do not use them for correctness-critical branching, as the path chosen depends on the example input's values during tracing.


### Step 2: Do I know one path will always be taken?

If you cannot eliminate the branch, ask yourself: **For my specific model, do I know that one path will always be taken?**

If yes, you can use `torch._check` to inform the compiler which branch to take:

```python
i = x.item()
torch._check(i > 4)  # Assert that i > 4 is always true for your use case
if i > 4:
    return x * 2
else:
    return x + 3
```

By asserting `torch._check(i > 4)`, the symbolic reasoning system learns that `i > 4` is always `True`, allowing the branch to be resolved without error. The else branch becomes dead code from the compiler's perspective.

### torch._check(cond, msg_fn)

`torch._check` is a function used to assert conditions at runtime, particularly when dealing with symbolic integers (`SymInts`) in PyTorch.

**Example Usage:**

```python
torch._check(x.size(0) == y, lambda: f"size mismatch: {x.size(0)} != {y}")
```

The code above does the following:

* Creates a deferred runtime assertion instead of a compile-time guard
* Teaches the symbolic reasoning system facts about your unbacked SymInts
* Can eliminate unbacked symbols by replacing them with equivalent expressions
* Refines value ranges of symbols
* Remembers boolean expressions that are always true

Semantically, the function behaves like a conditional check:
```python
if not cond:
    raise RuntimeError(msg_fn())
```
But there are a number of key differences:

* The condition is always assumed true at compile time, even if it involves unbacked `SymInts`. The actual check is deferred to runtime, avoiding
compile-time errors. Instead of setting up a guard, we implement a
deferred runtime assertion to verify the condition at runtime. At compile
time, we assume the condition won't trigger an error, so we don't need
to determine if it evaluates to `True` or `False`.

* If you perform an equality test `u0 = RHS`, we try to replace all instances
of `u0` with RHS. We will ALWAYS do this if RHS has no unbacked symbols,
as removing unbacked symbols is beneficial—eliminating them prevents
the creation of a `GuardOnDataDependentSymNode`. Even if we are not able
to eliminate u0, we can refine its value range. The value range specifies
what the set of possible values for a variable are. By default, size-like
unbacked SymInts have a value range of `[0, Inf]`; if you assert it is
equal to an expression with a refined value range, say `[2, 20]`, then
`u0`'s value range will be updated to `[2, 20]`. We also have limited
support for propagating value ranges in reverse.

* If you perform a boolean test `f(u0)`, we will remember that this expression always evaluates to True, and if you evaluate an expression that contains this expression, we will substitute it with True. We also support some limited reasoning on logically equivalent statements. For example, if you `torch._check(u0 < 4)`, we will also know that `u0 >= 4` evaluates to `False`, and so performing a test like this in a normal non-check conditional will go through fine.

You can also use `torch._check` to assert constraints and refine value ranges. For example, `torch._check(u0 >= 0)` establishes that `u0` is non-negative, refining its value range to `[0, Inf]`. Similarly, `torch._check(x > 7)` constrains `x` to be greater than 7.

When unbacked symbols are passed to factory functions like `torch.empty`, they are automatically recognized as representing sizes.
### Step 3: Is it unfixable?

If both branches are genuinely needed at runtime (i.e., sometimes `i > 4` and sometimes `i <= 4`), then no `torch._check` can help—it is impossible to trace as is. In such cases, you may need to consider alternative approaches, such as using `torch.cond` or padding.

Another common unfixable pattern involves indexing a python list with a data-dependent value:

```python
return self.mlps[x.item()]
```

Here, `self.mlps` is a Python list or `ModuleList`, and the code branches on a data-dependent value. The simplest solution is to induce a graph break before the indexing operation.

## Some Common Fix Patterns

### Using `torch._check` for Sanity Checks in Model Code

If you have sanity checks in your model code that validate conditions, you can use `torch._check` instead of `if` statements. `torch._check` handles data dependency by deferring the checks to runtime, so they won't cause compile-time errors.

**Note:** For C++ code, use `TORCH_SYM_CHECK` which is the C++ equivalent of `torch._check`.

When combining conditions, use `sym_or`, `sym_and`, etc. to ensure expressions are not eagerly evaluated (which would trigger data-dependent errors):

```python
# Instead of:
# if x != y or x > y:
#     raise RuntimeError("...")

# Use:
from torch.fx.experimental.symbolic_shapes import sym_or
torch._check(sym_or(x != y, x > y), lambda: "Validation failed: expected x != y or x > y")
```

### `u0` is Actually Equal to `u1`, but We Don't Know It

Multiple unbacked `SymInts` can be known to be equal at compile time:

```python
i0 = x.sum().item()
i1 = x.sum().item()
return torch.randn(i0) + torch.randn(i1)
```

If there is a `torch._check(i0 == i1)` somewhere (in the example above, this
check would occur inside the shape-checking rule for addition), we will
automatically unify the two unbacked `SymInts` and recognize them as equal.
However, if such an assertion is missing, you may need to explicitly add an
assertion to achieve this unification. For an example, see
https://github.com/pytorch/pytorch/issues/111950).

```{note}
If we allocate an unbacked `SymInt` and
immediately set it equal to another, these instances are benign and not easily
eliminated entirely from the framework.
```

### `u0` is a Tensor

Another reason you might be overallocating unbacked `SymInts` is due to passing
around a `Tensor` and relying on its implicit conversion to an integer. Many
functions that accept an integer will also accept a `Tensor` and automatically
call `item()` on the integer argument. It's beneficial to examine
`TORCH_LOGS=dynamic` to determine whether the number of unbacked `SymInts` is
as expected or excessive. When this occurs, a new `SymInt` will be allocated at
the line where a PyTorch function is invoked.

This issue is less likely to cause problems now because the return value of
`t.item()` is memoized, ensuring that you consistently receive the same unbacked
`SymInt` if you call it multiple times.

### Overspecialization Issue

In non-strict export mode, consider the following code:

```python
u0 = x.sum().item()
return y[:u0]
```

This code will fail when trying to evaluate `u0` because, when a `SymInt` is
used directly inside a Python slice (without using Dynamo), Python forces the
integer to be specialized and fails if it is unbacked.

To resolve this, you can rewrite the program to avoid specialization.
For the example above, you can fix it by not using slices:

```python
u0 = x.sum().item()
return y.narrow(0, 0, u0)
```

For more details, see the related issue
https://github.com/pytorch/pytorch/issues/111950.

### Use Lengths Instead of Offsets

When working with variable sequence lengths, it's common to have tensors
representing either the lengths or offsets of the sequences. For example, given
`values = [[1, 2, 3], [4, 5], [6, 7, 8, 9]]`, you might have `lengths = [3, 2, 4]`
and `offsets = [0, 3, 5, 9]`. While these representations are interconvertible,
it's better to work with lengths when dealing with them as integers (by calling
`lengths.tolist()`), rather than offsets.

The reason is that when you perform a `torch.split()` on your `values` tensor, you
need to create tensors for each sub-sequence, such as tensors of sizes 3, 2, and 4.
If you have unbacked `SymInts` for sizes, they become `u0`, `u1`, and `u2`. You can
easily indicate that they are size-like, and you're done. However, if you have
unbacked `SymInts` for offsets, they become `u1 - u0`, `u2 - u1`, `u3 - u2`, which
complicates matters. These quantities cannot be conveniently marked as size-like,
leading to potential issues. Since it's relatively straightforward to write code
using either lengths or offsets, you should prefer using lengths.

```{seealso}
* {ref}`dynamic_shapes`
* {ref}`debugging-tlparse-torch-logs`
```
