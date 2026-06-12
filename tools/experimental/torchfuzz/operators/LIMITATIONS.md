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
