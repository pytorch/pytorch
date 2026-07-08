# Operator Coverage Limitations

Known gaps between what these operator definitions exercise and what
PyTorch allows. Each section is maintained by the diff that introduced
the operators it describes.

## Cross-Cutting

These limitations stem from the fuzzer's top-down graph construction
model and affect all or most operators.

- **Non-contiguous inputs** — Input strides are always copied from
  `output_spec.stride` or computed via `contiguous_stride()`.
  Non-contiguous inputs (transposed, sliced, expanded) are never
  independently generated.

- **In-place variants** — Codegen always assigns to a new variable.
  In-place forms (`.cos_()`, `inplace=True`) are never emitted.

- **`out=` parameter** — Pre-allocated output tensors are never passed.

- **Empty tensors** — Shape generators use `randint(1, 6)`, so
  size-0 dimensions are never produced.

- **Rank-0 broadcast** — `random_broadcast_shape` cannot produce a
  rank-0 (scalar) tensor input. The `while len(result) > 1` guard
  prevents dropping below rank 1.

- **Complex dtypes** — `FLOAT_DTYPES` contains no complex types.
  Particularly impacts `torch.angle`, `torch.real`, and any binary
  op that supports complex inputs.

- **Max rank** — `max_dims=3` caps tensor rank. 4-D (NCHW) and 5-D
  tensors common in ML workloads are never generated.

- **Single-input broadcast** — When broadcasting is applied, only one
  of N inputs is shrunk. Simultaneous broadcasting of multiple inputs
  (e.g., `(3,1) + (1,4)` → `(3,4)`) is never tested.

- **Broadcast stride** — Broadcast inputs always get contiguous strides.
  Non-contiguous broadcast inputs are valid but never generated.

- **Float8 dtypes** — `torch.float8_e4m3fn` etc. are excluded.

- **Half-precision edge cases** — No targeted generation of denormals,
  near-max values, or bf16 precision boundaries.

## Elementwise Math

- `torch.angle` and `torch.real` are no-ops on real-valued tensors;
  their interesting behavior requires complex dtypes (excluded above).

- `torch.clamp_max` / `torch.clamp_min` randomize scalar bounds but
  never test tensor bounds (`torch.clamp_max(input, max=other_tensor)`).

- `torch.heaviside` second arg is always a 0-d scalar tensor; never
  tests a broadcastable `values` tensor of higher rank.

- `torch.lerp` always uses 3 tensor inputs; the scalar weight variant
  `torch.lerp(start, end, weight=0.5)` is never tested.

- `torch.addcmul` integer inputs are valid in PyTorch but excluded by
  `requires_float = True`.

## Tensor Creation

- `_like` operators: the non-contiguous output path creates a
  contiguous intermediate via the `_like` call and `.copy_()` into
  `torch.empty_strided`. This does not test the op's native stride
  handling.

## Comparison

- `torch.eq` and `torch.ne` accept complex inputs in PyTorch, but
  the shared `ComparisonOperatorBase` excludes complex dtypes.

## WhereOperator

- Raw Python scalar dispatch is never tested — scalars are always
  wrapped in `torch.tensor(value, dtype=...)`. Using a raw scalar
  would trigger PyTorch type promotion that may not match
  `output_spec.dtype` (Python `float` is float64).

- The both-scalars form `torch.where(cond, scalar, scalar)` is never
  tested.

## Logical

- Both inputs always share the same randomly-chosen dtype. Mixed-dtype
  logical ops (e.g., `torch.logical_and(int_tensor, float_tensor)`)
  are never tested.

## Bitwise

- Both inputs always share the output dtype. Mixed integer-type
  bitwise ops with type promotion are never tested.

## Manipulation & Indexing

- `torch.conj_physical` never tests complex dtypes. On real-valued
  tensors it is a no-op; the actual conjugation logic is never
  exercised.

- `torch.fill` never uses a 0-d tensor fill value (always uses a
  scalar literal).

## Loss Functions

- `reduction` is always `'none'`. The `'mean'` and `'sum'` modes are
  never tested. This is structural: the top-down model needs a
  predictable output shape, and reduced-scalar outputs would require
  the loss operators to work like reduction operators with a
  separate "scalar mode" path.

- `F.cross_entropy`: `weight`, `ignore_index`, `label_smoothing`, and
  soft targets are never exercised. Only the 2-D `(N, C)` input form
  is tested; the multi-dimensional form `(N, C, d1, ...)` used in
  segmentation is excluded.

- `F.nll_loss`: `weight` and `ignore_index` are never exercised. Only
  the 2-D `(N, C)` input form is tested.

- `F.binary_cross_entropy_with_logits`: `pos_weight` is never
  exercised.

- `F.binary_cross_entropy`: `weight` is never exercised.

- `F.huber_loss` and `F.smooth_l1_loss`: hyperparameters are drawn
  from small discrete sets (`{0.1, 0.5, 1.0, 2.0}`); continuous
  draws and extreme values are never tested.

## Activations

- `F.prelu` never generates a scalar-broadcast weight `(1,)` for
  rank >= 2 inputs; always uses a per-channel weight of shape
  `(output_spec.size[1],)`.

- `F.softplus` `beta` is drawn from `[0.1, 5.0]`. The large-beta
  regime (where softplus degenerates to relu) is underexercised.

## Reductions

- **Tuple-dim reductions** — `dim` is always a single int. Multi-axis
  reductions like `torch.sum(x, dim=(0, 2))` are never tested.

- `torch.nansum` never receives inputs containing NaN values (the
  fuzzer's materializer produces standard random data). The
  distinguishing behavior — ignoring NaN — is never deliberately
  exercised.

- `torch.aminmax`, `torch.var_mean`, and `torch.count_nonzero` only
  cover the 0-D scalar (no-dim) form. The `dim`/`keepdim` forms are
  excluded (documented as intentional follow-up).

- Tuple-returning ops (`cummax`, `cummin`, `max`, `min`, `median`,
  `mode`, `aminmax`, `var_mean`) only extract one field (`.values`,
  `.min`, `.var`). The second return value (`.indices`, `.max`,
  `.mean`) is never tested.

- `torch.max` / `torch.min` 1-arg full-reduction form and 2-tensor
  element-wise form (`torch.max(a, b)`) are never tested.

- `torch.cumprod` `dtype=` kwarg is never exercised.

## Special Functions

- Chebyshev polynomial degree `n` is limited to `[0, 5]` and input
  `x` is not constrained to `[-1, 1]`. Higher degrees with
  unclamped `x` cause rapid numerical overflow in fp16/bf16.

- `torch.igammac` (upper incomplete gamma function) is absent.
