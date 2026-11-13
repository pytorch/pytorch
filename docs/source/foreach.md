# torch.foreach

```{eval-rst}
.. automodule:: torch.foreach
```

`torch.foreach` applies familiar PyTorch operations across lists of tensors. For
example, `torch.foreach.add(inputs, other)` is semantically equivalent to a
Python loop that applies {func}`torch.add` at every list position. When
available on an accelerator and certain conditions are met, a foreach operation will use a horizontally fused multi-tensor kernel to improve
runtime. On CUDA, common eligibility requirements include strided,
non-overlapping dense tensors on the same device, compatible dtypes, and
matching sizes and strides for corresponding tensors.

```python
inputs = [torch.ones(2), torch.ones(3)]
result = torch.foreach.add(inputs, 2)
# Equivalent to tuple(torch.add(tensor, 2) for tensor in inputs)

torch.foreach.mul_(inputs, 3)
# Mutates each tensor and returns `inputs`.
```

## API Coverage

A foreach API lifts an ordinary tensor operation over a list of inputs. This
creates a combinatorial space of possible signatures. Depending on the
operation, a tensor argument could be shared as a Tensor or supplied
elementwise as a TensorList, while a scalar argument could be a Scalar, a
ScalarList, a shared 0-D Tensor, or a packed 1-D CPU Tensor.

You can then imagine that one operation may take on various forms such as TensorList/TensorList, TensorList/Tensor, TensorList/ScalarList, TensorList/Scalar, etc. Operations with more parameters would have more combinations. The public foreach APIs support a subset of these combinations based on usage. If you would like to see an implementation of a missing combination, please file an [issue](https://github.com/pytorch/pytorch/issues/new/choose)!

Across the supported signatures, we maintain constraints that TensorList and ScalarList arguments must be non-empty, and corresponding tensor and scalar lists must have the same length.

Only signatures that explicitly list `Tensor` include it in the supported typed
surface. A 0-D Tensor that does not require gradients may sometimes be accepted
for a `Scalar` parameter through implicit scalar conversion. Converting an
accelerator Tensor this way reads its value on the host, which may be expensive. On CUDA, this synchronizes eager execution and is unsupported during CUDA graph capture, so it should not be relied upon as a Tensor overload.

The public foreach API also does not support `out=` variants and may have a higher memory footprint than looping through the non-foreach original API, as multiple intermediates can be alive simultaneously.

## Migrating from the private API

You may be familiar with the private spellings of foreach APIs, e.g., for {func}`torch.add`:

```python
torch._foreach_add(inputs, other)  # Private spelling
torch.foreach.add(inputs, other)   # Public beta spelling
```

The private spellings remain available with unchanged signatures for backward compatibility.
Public functions call the same ATen operators but improve API consistency in two ways:

1. All required operands are positional-only, and all optional parameters are keyword-only.
2. Parameter names align with the corresponding ordinary operation.

The primary tensor-list argument that the foreach API applies over is named `inputs`, and other
arguments retain the ordinary operation's logical name, even when the currently supported form
requires a list. This keeps signatures descriptive as operand forms evolve.

## Unary operations

```{eval-rst}
.. currentmodule:: torch.foreach
.. autosummary::
    :toctree: generated
    :nosignatures:

    abs
    abs_
    acos
    acos_
    asin
    asin_
    atan
    atan_
    ceil
    ceil_
    clone
    cos
    cos_
    cosh
    cosh_
    erf
    erf_
    erfc
    erfc_
    exp
    exp_
    expm1
    expm1_
    floor
    floor_
    frac
    frac_
    lgamma
    lgamma_
    log
    log_
    log10
    log10_
    log1p
    log1p_
    log2
    log2_
    neg
    neg_
    reciprocal
    reciprocal_
    round
    round_
    rsqrt
    rsqrt_
    sigmoid
    sigmoid_
    sign
    sign_
    sin
    sin_
    sinh
    sinh_
    sqrt
    sqrt_
    tan
    tan_
    tanh
    tanh_
    trunc
    trunc_
    zero_
```

## Binary operations

```{eval-rst}
.. currentmodule:: torch.foreach
.. autosummary::
    :toctree: generated
    :nosignatures:

    add
    add_
    sub
    sub_
    mul
    mul_
    div
    div_
    clamp_min
    clamp_min_
    clamp_max
    clamp_max_
    minimum
    minimum_
    maximum
    maximum_
    pow
    pow_
    copy_
```

## Pointwise operations

```{eval-rst}
.. currentmodule:: torch.foreach
.. autosummary::
    :toctree: generated
    :nosignatures:

    addcmul
    addcmul_
    addcdiv
    addcdiv_
    lerp
    lerp_
```

## Reductions and matrix operations

```{eval-rst}
.. currentmodule:: torch.foreach
.. autosummary::
    :toctree: generated
    :nosignatures:

    max
    norm
    mm
```
