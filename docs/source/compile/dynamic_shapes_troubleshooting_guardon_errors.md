(troubleshooting_guardondatadependentsymnode_errors)=

# Troubleshooting GuardOnDataDependentSymNode Errors

When working with PyTorch models that have data-dependent control flow (using functions
like `item()`, `tolist()`, or `nonzero())`, you may encounter `GuardOnDataDependentSymNode` errors.
This section explains what these errors are and how to fix them.

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

## Root Cause

These errors occur when PyTorch tries to convert a symbolic quantity (for example, `u2 == -1`)
into a concrete value (such as, `False`) to make branching decisions. In a typical scenario,
where data-dependent sizes are not involved, PyTorch can determine the concrete value at
compile time and install a guard to ensure the compilation result remains valid. However,
with data-dependent quantities, the true value is unknown at compile time, resulting in errors.

You can often rewrite your model, by adding `torch._check` or `torch._check_is_size` to
bypass these issues. This document aims to teach you how.

## Debugging Tools

Here is the list of some of the debugging tools available in PyTorch that you can use to troubleshoot these errors:

* `TORCH_LOGS="dynamic"` - Shows detailed logs about symbolic operations
* `TORCHDYNAMO_EXTENDED_DEBUG_CREATE_SYMBOL="u2"` - Provides extended logs for specific symbols
* `TORCHDYNAMO_EXTENDED_DEBUG_CPP=1` - Helps when guards are triggered from C++

## Error Variations

Here is a the list of error variations that you might encounter:

| Error Variations | Description |
|------------------|-------------|
| "Could not guard on data-dependent expression" | Occurs when trying to extract a concrete boolean from expressions like u0 == 0 or u0 > 10 |
| "Could not extract specialized integer from data-dependent expression" | Occurs when trying to extract a concrete integer value. <br/> **Common causes:** <br/> - Control flow that depends on the integer (such as, looping `u0` times) <br/> - Overspecialization in code that could work symbolically |

## How to Diagnose Your Problem

### Step 1: Examine the Potential Framework Culprit (Python Backtrace)

The exception provides a backtrace, which often indicates the problem.
Given that PT2 backtraces can be lengthy, the error message will also
suggest a potential framework culprit. For example:

```sh
Potential framework code culprit (scroll up for full backtrace):
  File "/data/users/ezyang/a/pytorch/torch/_prims_common/__init__.py", line 855, in infer_size
    if d == -1:
```

**Consider the Following:**

* Does it make sense that this condition is triggering a guard on a
data-dependent symbol?
* Should we know if the quantity in question is size-like?
(The exception lists size-like symbols; if a symbol is not listed,
it might be an arbitrary integer.)
* If the equation involves two distinct symbols, should we know
they are actually equal?
*  If all symbols are size-like but the equation involves 0 or 1,
are we missing a `guard_size_oblivious` wrapper? (Remember, for
`guard_size_oblivious` between two size tuples, use `sym_eq` instead
of regular equality.)

In the example above, testing if `d` (a data-dependent value) is `-1` suggests
that `d` should be non-negative if it were a size. This indicates a missing
`torch._check_is_size`. If `d` is already size-like but `numel() == 0` fails,
consider wrapping it in `guard_size_oblivious`.

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

### Step 2: Examine the C++ Backtrace

If the framework code culprit is uninformative, the guard might be in C++. You can
force a C++ backtrace by running with `TORCHDYNAMO_EXTENDED_DEBUG_CPP=1`. This
provides a detailed C++ backtrace with Python, CPython, and C10/ATen/libtorch
frames interspersed. Look for symbols in the `at::` or `c10::` namespace that
resemble kernel-specific code, likely related to the kernel executed per the Python
backtrace. If using a non-debug build of PyTorch, inlining may cause missing
frames, requiring source code investigation to locate the issue. For example, see https://github.com/pytorch/pytorch/pull/118579.

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

In this example, `at::native::narrow_tensor_symint` calls into `item`, which
triggers the guard on a data-dependent `SymNode`. You can modify the C++ code to
avoid specializing, or verify if you should be in this C++ code (e.g., `start` was
not expected to be a `Tensor`, and modifying this fixed the problem).

## Tools for Fixing Errors

There are a few important functions which you should use to troubleshoot this problem.

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

But there a number of key differences:

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
`u0`’s value range will be updated to `[2, 20]`. We also have limited
support for propagating value ranges in reverse.

* If you perform a boolean test `f(u0)`, we will remember that this expression always evaluates to True, and if you evaluate an expression that contains this expression, we will substitute it with True. We also support some limited reasoning on logically equivalent statements. For example, if you `torch._check(u0 < 4)`, we will also know that `u0 >= 4` evaluates to `False`, and so performing a test like this in a normal non-check conditional will go through fine.


### `torch._check_is_size(size)` and `guard_size_oblivious(cond)`

Example:
```python
u0 = y.item()
torch._check_is_size(u0)
```

**Semantic Equivalent:**

```python
if u0 < 0:
    raise RuntimeError("u0 is not a size")`
```
**Key Differences:**

Like `torch._check`, this test will always succeed at compile time, and it will establish that `u0 >= 0`. This refines the value range of `u0` to `[0, Inf]` instead of `[-Inf, Inf]`.

Marking `u0` as size-like is crucial. Size-like unbacked `SymInts` behave like
their regular counterparts, except when involved in a boolean expression
evaluated with `guard_size_oblivious`. In such cases, they are assumed not to equal zero or one, temporarily setting their value range to `[2, Inf]`. For instance, a conditional check like `u0 == 1` will evaluate to `False` when `u0` is size-like, instead of causing an error.

For example, `guard_size_oblivious(u0 == 1)` will always return `False` when `u0`
is size-like.

Marking unbacked symbols as size-like is essential in contexts where tensor
sizes are expected. PyTorch internals often check if sizes are zero or one to
handle special cases related to empty or single-element tensors. If you pass an
unbacked symbol to a factory function like `torch.empty`, it will automatically
be marked as size-like. However, some quantities, like arguments to `Tensor.view`,
cannot be inferred as size-like because `-1` is a valid argument. In such cases,
you need to explicitly use `torch._check_is_size` on an unbacked `SymInt` before
passing it to `view`.

In PyTorch framework code, if you need to test a size for zero or one, wrap the
test in `guard_size_oblivious` to assume that size-like unbacked `SymInts` will
not pass this test. Generally, most framework code has logic for the `>= 2`
case, which works for the `0/1` case. If using `guard_size_oblivious` in
PyTorch framework code resolves your issue, it's likely acceptable. However,
avoid using `guard_size_oblivious` in user code, especially if different
behavior is required for the `0/1` case at runtime, such as in a
hand-tracking application.

In C++, this can be done with `TORCH_GUARD_SIZE_OBLIVIOUS(u0.sym_eq(0))`, for example.

### torch._check_is_size(size, max=upper_bound) (New)

This function is semantically equivalent to `torch._check(size <= upper_bound)`.
However, under `guard_size_oblivious`, it assumes that `size < upper_bound`.
This functionality only works when the upper bound is an integer constant. If
`upper_bound` is a symbolic expression, normal semantics apply. There is
potential to extend this functionality to symbolic expressions with further
development.

For more details, see the related issue https://github.com/pytorch/pytorch/issues/120288.


### `torch._constrain_as_value` and `torch._constrain_as_size`

These APIs are more specialized and are effectively equivalent to
`torch._check` and `torch._check_is_size`, with the added capability
of adjusting the value range of a variable by specifying minimum and
maximum values. However, in recommendation models, these functions are
unlikely to resolve `GuardOnDataDependentSymNode` errors effectively.

While `constrain_as_value` might seem like a convenient way to ensure a
variable stays within the bounds of another tensor, it is often impractical.
This is because value ranges only support constant bounds, and it's common
for the tensor you want to index into to have a symbolic dimension (for
example, `s0`). Using its size as the maximum value for a value range
will force specialization, which is usually undesirable. Instead, if
necessary, manually handle range checks by using `torch._check()` on
appropriate expressions based on the errors you encounter.

## Common Fix Patterns

There are several common methods to resolve issues like this. Below,
we outline the most frequently used solutions.

### When It's Unfixable

In some cases, the issue is genuinely unfixable due to the nature of the code.
Consider the following example:

```python
i = x.item()
if i > 4:
  return x * 2
else:
  return x + 3
```

If the user code is branching on a data-dependent value, it is impossible to
trace as is. In such cases, you may need to consider alternative approaches,
such as using `torch.cond`.

Another common pattern involves indexing with a data-dependent value:

```python
return self.mlps[x.item()]
```

Here, `self.mlps` is a Python list or `ModuleList`, and the code branches on a data-dependent value. The simplest solution is to induce a graph break before the indexing operation.

### `u0` is a Size, but We Don’t Know It

Some guards fail on tests that essentially ask, "Is this a size?" but we don't know it is a size. These fall into two categories:

1. **Regular Tests:**

   These are tests like `u0 >= 0` or `u0 != -1` that are unconditionally true
   for sizes. Adding a `torch._check_is_size(...)` on the relevant size will
   assert that these tests are true. This is typically uncommon because if
   the test is for error checking, we can infer that the condition must be
   true, as an error would occur otherwise. An important exception is APIs
   that accept both sizes and `-1`; in such cases, the user must indicate that
   the input data-dependent quantity cannot be `-1`, as something unusual would
   happen otherwise. For an example, see
   https://github.com/pytorch/pytorch/pull/107788.

   Sometimes, you can refactor an error-checking API to split a logical
   disjunction of conditionals into separate conditionals. If you can do so
   to achieve a single `torch._check(x == y)` statement, it will enable
   the automatic generation of a deferred runtime assertion. For an example,
   see https://github.com/pytorch/pytorch/pull/110979.

2. **Edge Case Tests:**

   These are tests like `u0 == 0` or `u0 == 1`, which are not always true for
   sizes, but where our choice doesn’t really matter. These tests handle edge
   cases, such as dealing with an empty tensor or testing for broadcasting when
   we want to assume broadcasting is not occurring. To resolve these situations,
   two steps are needed:

   * First, the guard itself must be evaluated via `guard_size_oblivious`,
   which assumes that size-like integers cannot equal zero or one, with the
   promise that if they do, something reasonable will happen.
   * Second, the symbols themselves must be marked as size-like, either
   inferred because they were passed to tensor factory functions or explicitly
   specified with `torch._check_is_size(...)`. For examples of making guards
   size-oblivious, see https://github.com/pytorch/pytorch/pull/118579.

Sometimes, these tests can occur in C++. While there are corresponding
C++ APIs for these tests, it can be more challenging to localize the problem,
as you do not get a useful backtrace by default.

### `u0` is Actually Equal to `u1`, but We Don’t Know It

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
u0 = x.sum().item() return y[:u0]
```

This code will fail when trying to evaluate `u0` because, when a `SymInt` is
used directly inside a Python slice (without using Dynamo), Python forces the
integer to be specialized and fails if it is unbacked.

To resolve this, you can rewrite the program to avoid specialization.
For the example above, you can fix it by not using slices:

```python
u0 = x.sum().item() return y.narrow(0, 0, u0)
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
