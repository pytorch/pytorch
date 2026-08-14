# torch.foreach

```{eval-rst}
.. automodule:: torch.foreach
```

:::{warning}
`torch.foreach` is a beta API. Its signatures may change based on user feedback.
The existing `torch._foreach_*` functions remain available for compatibility.
:::

`torch.foreach` applies familiar PyTorch operations across lists of tensors. For
example, `torch.foreach.add(inputs, other)` is semantically equivalent to a
Python loop that applies {func}`torch.add` at every list position. A foreach
implementation may use a multi-tensor or grouped kernel, but acceleration is
conditional and is not part of the API contract.

```python
inputs = [torch.ones(2), torch.ones(3)]
result = torch.foreach.add(inputs, 2)
# Equivalent to tuple(torch.add(tensor, 2) for tensor in inputs)

torch.foreach.mul_(inputs, 3)
# Mutates each tensor and returns the exact `inputs` container.
```

## Containers and return values

Tensor-list and scalar-list arguments accept either lists or tuples. Tensor
lists must be non-empty, and corresponding tensor and scalar lists must have
the same length.

- Functional operations return a tuple of tensors, regardless of whether the
  input container was a list or tuple.
- In-place operations mutate the tensors and return the exact input list or
  tuple object.
- The public API does not support `out=`. Existing dispatcher-only foreach out
  variants first allocate a functional result and then copy it into the supplied
  tensors, so they do not provide a direct-write memory-saving contract.

## Operand forms

An ordinary tensor or scalar argument can be lifted into several possible
foreach forms. The currently implemented forms are a usage-driven subset, not
the Cartesian product of every possible lifting.

| Ordinary argument | Foreach forms used by this API |
| --- | --- |
| Primary tensor | A tensor list, with one operation per position |
| Additional tensor | A corresponding tensor list; selected APIs also accept one shared 0-D scalar tensor |
| Scalar | One shared scalar, a scalar list, or selected tensor representations of scalars |

A shared Tensor accepted by `add`, `mul`, or `div` must be a 0-D scalar tensor;
it is not an arbitrary tensor broadcast across every input. `addcmul` and
`addcdiv` additionally accept a packed, contiguous 1-D CPU tensor containing one
scalar per list position.

### Binary coverage

| Operation | Shared scalar | Scalar list | Tensor list | Shared 0-D tensor | Additional behavior |
| --- | :---: | :---: | :---: | :---: | --- |
| `add` | Yes | Yes | Yes | Yes | `alpha` is supported only with a tensor list or shared 0-D tensor |
| `sub` | Yes | Yes | Yes | No | `alpha` is supported only with a tensor list |
| `mul` | Yes | Yes | Yes | Yes | |
| `div` | Yes | Yes | Yes | Yes | `rounding_mode` is not supported |
| `clamp_min`, `clamp_max` | Yes | Yes | Yes | No | |
| `minimum`, `maximum` | Yes | Yes | Yes | No | |
| `pow` | Yes | Yes | Yes | No | Also supports one shared scalar base with a tensor-list exponent |

The scalar forms of `minimum` and `maximum` are foreach extensions to the
ordinary APIs: they behave like applying `torch.clamp_max` and
`torch.clamp_min`, respectively. The tensor-list forms correspond directly to
`torch.minimum` and `torch.maximum`.

`lerp` accepts a tensor list for `ends` and either a shared scalar, scalar list,
or tensor list for `weight`. `addcmul` and `addcdiv` require tensor lists for
`tensor1s` and `tensor2s`; their keyword-only `value` may be a shared scalar,
scalar list, or packed scalar tensor.

Missing operand forms are intentional omissions from the current beta API. A
feature request should identify the operation, desired form, device, use case,
and whether acceleration or capturability is required.

## Differences from ordinary operations

Foreach names describe the corresponding ordinary operation but do not imply
that every ordinary option has been lifted:

- `div` does not support `rounding_mode`.
- `round` does not support `decimals`.
- `norm` reduces each entire tensor and does not support `dim` or `keepdim`.
- `max` reduces each entire tensor, has no dimensional form, and does not return
  indices. There is currently no matching foreach reduction named `min`.
- Only `copy_` and `zero_` are public; no functional `copy` or `zero` is exposed.
- `mm` is functional only.
- There is no public `powsum`; the existing private operation is an internal
  helper for norm implementations.
- No function accepts `out=`.

The reduction family is intentionally small: there is no all-element `min`,
`sum`, `mean`, or other general foreach reduction today.

The documented signatures are the complete supported subset. An unlisted option
or operand form is unsupported even when it could be expressed with a Python
loop.

## Execution and devices

Fast paths generally require tensor lists with compatible devices, dtypes,
layouts, sizes, and strides, plus non-overlapping dense tensors. Other supported
inputs fall back to per-tensor execution. Backend and dtype coverage can differ,
so code should rely on documented semantics rather than assuming that a call is
fused or launches one kernel.

The public Python functions continue to dispatch to private `aten::_foreach_*`
operators. Profiler events, compiler graphs, and some error messages may
therefore contain the private ATen spelling.

The public namespace is not registered as a TorchScript builtin. Existing
TorchScript code can continue to use the private compatibility functions.

## Migration

The public and compatibility spellings use the same native implementations:

```python
torch._foreach_add(inputs, other)  # Compatibility spelling
torch.foreach.add(inputs, other)   # Public beta spelling
```

Private parameter names are unchanged for backward compatibility. Public
parameters use the corresponding ordinary operation's semantic names and
pluralize arguments that are always lists, such as `inputs`, `tensor1s`,
`tensor2s`, `ends`, `srcs`, and `mat2s`.

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
