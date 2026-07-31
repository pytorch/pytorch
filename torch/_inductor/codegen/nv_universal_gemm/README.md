# NVGEMM epilogue lowering

NVGEMM captures scheduler Loop IR in `loop_ir_epilogue_lowering.py`, lowers
supported expressions with `loop_ir_cutedsl_codegen.py`, and passes a
backend-neutral reduction contract through `gemm_epilogue.py`. Unsupported
programs must fall back without partially claiming their reduction producers.

## Backend parity

| Capability | Triton fusion | FlexGEMM / QuACK | NVGEMM CuTeDSL |
| --- | --- | --- | --- |
| Pointwise epilogue graph | General scheduler IR subject to Triton lowering | FX graph over the supported CuTeDSL op set | Loop IR over the supported CuTeDSL op set |
| Captured tensor inputs | General indexing subject to scheduler fusion | Tile, row, column, or scalar broadcast | Tile, row, or column broadcast |
| Same-shape auxiliary outputs | General scheduler outputs | Arbitrary tuple for `aten.mm` | Arbitrary tuple |
| Independent reductions in one kernel | Supported when scheduler and resource constraints permit | One local-reduction plan | One local-reduction plan |
| Reduction-fed full-shape consumers | General fused graph | One planned feed-main value | Primary plus one secondary callback; additional consumers fall back |
| Compressed reduction outputs | General scheduler outputs | At most one | At most one |
| Reduction geometry | General Triton reduction domains | Grouped GEMM M or N axis | Grouped GEMM M or N axis |
| Composite reductions | Expressible as ordinary Triton IR | Online softmax plus supported primitive reductions | Online softmax, variance, logsumexp, and supported primitive reductions |
| Unsupported expression | Triton scheduling or lowering fallback | FlexGEMM fallback | NVGEMM fallback |

NVGEMM is therefore not at Triton parity. It is close to FlexGEMM for the
shared single-reduction contract, but the two CuTeDSL backends are not exact
substitutes: FlexGEMM has FX-level ownership and scalar captured inputs, while
NVGEMM recognizes more composite Loop IR reductions and supports a second
reduction-fed consumer.

## Parity plan

1. **Sequence-based contracts.** Replace the singular `GemmReductionPlan` with
   a program containing reduction values, output stores, and consumer edges.
   Keep compatibility adapters for providers that accept only one reduction.
2. **Frontend-neutral reduction SSA.** Make FX and Loop IR lowering produce the
   same reduction-value graph. Each value records geometry, primitive combine
   semantics, source expression, finalizer expression, and dependent outputs.
3. **Tuple reduction state.** Teach generated source, init, combine, and
   finalize callbacks to operate on tuple state. Remove the reserved secondary
   reduction argument and bind consumer operands by reduction-value index.
4. **Provider ABI widening.** Pass tuples of compressed outputs and full-shape
   consumers through dense EFC and block-scaled providers. Include the complete
   structural program in specialization and cache keys.
5. **Scheduling by capability.** Ask each provider whether it can lower the
   complete program before claiming nodes. Keep geometry and resource limits in
   provider capability checks rather than frontend recognition.
6. **Parity coverage.** Add one-kernel and numerical tests for multiple
   independent reductions, three or more distinct reduction consumers, mixed
   compressed and full-shape outputs, tensor captures in reduction consumers,
   dynamic shapes, both grouped axes, and arbitrary output counts. Run each
   case against Triton fallback and both CuTeDSL frontends where applicable.
7. **Remove compatibility fields.** Once both providers consume the sequence
   ABI, delete `feed_output`, `secondary_feed_output`, and singular finalizer
   plumbing from the shared contract.

The sequence ABI is the main dependency: scheduler parity is unsafe until a
provider can accept the complete reduction program atomically.
