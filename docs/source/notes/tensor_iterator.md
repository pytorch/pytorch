---
myst:
  html_meta:
    description: Developer guide to torch._tensor_iterator, the Python surface over ATen's TensorIterator build pipeline
    keywords: TensorIterator, TensorIteratorConfig, ATen, kernel dispatch, dtype promotion
---

(tensor-iterator)=

# TensorIterator (Python)

`torch._tensor_iterator` is a thin Python surface over ATen's
`at::TensorIterator` build pipeline. It is a developer tool: it lets you
inspect the result of a TensorIterator build (shape after coalesce/reorder,
strides, dtype/device inference, broadcast result) without leaving Python.

There is no `for_each` here -- this surface is build-only. Use it to debug
shape/dtype inference, validate custom-op contracts, or pattern-match the
post-build geometry from a dispatch decision (see
`torch._native/ops/scatter_add/cutedsl_impl.py` for an example).

## C++ fluent → Python kwargs

The C++ builder is a fluent `at::TensorIteratorConfig` whose setters return
`*this`:

```cpp
auto iter = at::TensorIteratorConfig()
    .add_output(out)
    .add_const_input(a)
    .add_const_input(b)
    .promote_inputs_to_common_dtype(true)
    .cast_common_dtype_to_outputs(true)
    .enforce_safe_casting_to_output(true)
    .build();
```

The Python equivalent passes operands and flags as keyword arguments to
`TensorIterator(...)`:

```python
from torch._tensor_iterator import TensorIterator

it = TensorIterator(
    outputs=[out],
    const_inputs=[a, b],
    promote_inputs_to_common_dtype=True,
    cast_common_dtype_to_outputs=True,
    enforce_safe_casting_to_output=True,
)
```

The mapping is mechanical:

| C++ setter                              | Python kwarg                          | Default |
|-----------------------------------------|---------------------------------------|---------|
| `add_output(t)`                         | `outputs=[t, ...]` (or `[None]`)      | `[]`    |
| `add_input(t)`                          | `inputs=[t, ...]`                     | `[]`    |
| `add_const_input(t)`                    | `const_inputs=[t, ...]`               | `[]`    |
| `check_all_same_dtype(b)`               | `check_all_same_dtype=b`              | `True`  |
| `check_all_same_device(b)`              | `check_all_same_device=b`             | `True`  |
| `promote_inputs_to_common_dtype(b)`     | `promote_inputs_to_common_dtype=b`    | `False` |
| `promote_integer_inputs_to_float(b)`    | `promote_integer_inputs_to_float=b`   | `False` |
| `cast_common_dtype_to_outputs(b)`       | `cast_common_dtype_to_outputs=b`      | `False` |
| `enforce_safe_casting_to_output(b)`     | `enforce_safe_casting_to_output=b`    | `False` |
| `enforce_linear_iteration(b)`           | `enforce_linear_iteration=b`          | `False` |
| `resize_outputs(b)`                     | `resize_outputs=b`                    | `True`  |
| `set_check_mem_overlap(b)`              | `check_mem_overlap=b`                 | `True`  |
| `allow_cpu_scalars(b)`                  | `allow_cpu_scalars=b`                 | `False` |
| `is_reduction(b)`                       | `is_reduction=b`                      | `False` |
| `declare_static_dtype(d)`               | `static_dtype=d`                      | `None`  |
| `declare_static_device(dev)`            | `static_device=dev`                   | `None`  |
| `declare_static_shape(s, squash)`       | `static_shape=s, squash_dims=squash`  | `None`  |

`outputs` accepts `None` placeholders for outputs the iterator should allocate
itself; `inputs` and `const_inputs` must be defined tensors.

## Factory shortcuts

The C++ named constructors at `aten/src/ATen/TensorIterator.cpp` (`binary_op`,
`unary_op`, `comparison_op`, `nullary_op`, `reduce_op`, `binary_float_op`,
`unary_float_op`) have direct Python equivalents that bake in the canonical
flag combinations:

```python
from torch._tensor_iterator import (
    binary_op,
    binary_float_op,
    comparison_op,
    nullary_op,
    reduce_op,
    unary_op,
    unary_float_op,
)

it = binary_op(None, a, b)             # auto-allocate output, promote+cast
it = comparison_op(None, a, b)         # output dtype forced to bool
it = unary_float_op(None, int_tensor)  # promotes int input to float
```

Each factory mirrors its C++ counterpart's flag set exactly; reach for them
when you'd reach for the C++ named constructor.

## Canonical-recipe caveats

The Python surface is a *canonical* projection of the C++ builder, not a
faithful replay of arbitrary fluent call sequences. Two consequences:

**Operand ordering is fixed at outputs → inputs → const_inputs.**
The C++ builder distinguishes `add_input(a); add_const_input(b)` from
`add_const_input(b); add_input(a)` -- `input(0)` refers to different operands.
The Python surface cannot express that distinction: every `inputs[i]` precedes
every `const_inputs[j]` in the registered operand list.

**Setters are applied as final state, not as a sequence of calls.**
Some C++ setters have order-dependent side effects -- e.g.
`promote_inputs_to_common_dtype(true)` also flips `check_all_same_dtype` to
`false`. The Python surface materializes the *final* boolean state of each
knob, so it can't reproduce a sequence where an intermediate setter observed a
since-overwritten value.

Every in-tree caller of `at::TensorIteratorConfig` fits the canonical-recipe
shape, so the lossiness is theoretical, not practical.

## Inspecting the result

After construction, the iterator is read-only. Useful properties and methods:

```python
it.ndim           # rank after coalesce/reorder
it.shape          # zero-copy memoryview of int64 dims
it.numel          # product of shape
it.ntensors       # total operands (outputs + inputs)
it.ninputs
it.noutputs
it.is_contiguous
it.is_trivial_1d
it.common_dtype   # inferred computation dtype, or None

it.tensor(i)              # operand at flat index i
it.input(i=0)             # input by input-index
it.output(i=0)            # output by output-index
it.dtype(i=0)             # per-operand dtype
it.device(i=0)            # per-operand device
it.strides(i)             # byte strides, zero-copy memoryview
it.element_strides(i)     # element strides (byte_stride // element_size)
```

`shape` and `strides(i)` return `memoryview` objects backed by the iterator's
own buffers. They are valid for the lifetime of the iterator; copy via
`tuple(it.shape)` if you need a snapshot.

## When to use this

* **Pre-dispatch layout analysis.** Build a TI on the same operands an aten
  kernel would, then pattern-match `it.ndim` / `it.strides(i)` to decide
  whether your custom kernel can handle the shape. The
  `_scatter_add_eligibility` helper in `torch/_native/ops/scatter_add/`
  is a worked example.
* **Debugging dtype/promotion surprises.** Construct a TI with the flags you
  think a kernel uses; `it.common_dtype` and `it.dtype(i)` show what the
  builder actually inferred.
* **Validating custom op contracts.** If your kernel claims to handle a
  certain shape/dtype combination, build a TI and assert on its post-build
  geometry.

## When *not* to use this

* You want to actually run a kernel. There is no `for_each` -- use the public
  `torch.*` op or write a C++ kernel.
* You need exact replay of an arbitrary `TensorIteratorConfig` call sequence
  (see caveats above).
