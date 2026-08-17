# torch.foreach

```{eval-rst}
.. automodule:: torch.foreach
```

:::{warning}
`torch.foreach` is a beta API. Its signatures may change based on user feedback.
The existing private `torch._foreach_*` functions remain available for compatibility.
:::

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
creates a combinatorial set of possible signatures: each argument may be treated elementwise (as a List) or be shared (as a Tensor) by every operation. For example, a tensor argument may be a Tensor or a TensorList, while a scalar argument may be a single Scalar, a ScalarList, or even a 1D Tensor.

You can then imagine that one operation may take on various forms such as TensorList/TensorList, TensorList/Tensor, TensorList/ScalarList, TensorList/Scalar, etc. Operations with more parameters would have more combinations. The public foreach APIs support a subset of these combinations based on usage. If you would like to see an implementation of a missing combination, please file an [issue](https://github.com/pytorch/pytorch/issues/new/choose)!

Across the supported signatures, we maintain constraints that TensorList and ScalarList arguments must be non-empty, and corresponding tensor and scalar lists must have the same length.

The public foreach API does not support `out=` variants and has a higher memory footprint than the non-foreach original API.

## Public Migration

You may be familiar with the private spellings of foreach APIs, e.g., for {func}`torch.add`:

```python
torch._foreach_add(inputs, other)  # Private spelling
torch.foreach.add(inputs, other)   # Public beta spelling
```

The private spellings are from the days of old before we were ready to make them more visible and supported in public. The private APIs will maintain their functionality and signatures to support backwards compatibility. The new public APIs will share functionality--both these APIs route to the same ATen op--but we take the opportunity to rename some arguments for better consistency with their corresponding operation's arguments. We maintain the same argument name and pluralize those that are always lists, such as `inputs`, `tensor1s`, `tensor2s`, `ends`, `srcs`, and `mat2s`. 

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
