# NVGEMM epilogue lowering

NVGEMM captures scheduler Loop IR in `loop_ir_epilogue_lowering.py`, lowers
supported expressions with `loop_ir_cutedsl_codegen.py`, and passes a
backend-neutral reduction contract through `gemm_epilogue.py`. Unsupported
programs must fall back without partially claiming their reduction producers.

## Lowering flow

1. `NVGemmEpilogueCapture` normalizes fused scheduler nodes and interprets their
   Loop IR once.
2. `NVGemmEpilogueLowering` recognizes reduction regions and produces an
   `NVGemmEpilogueProgram`. The program owns captured nodes, reduction regions,
   the backend-neutral `GemmReductionPlan`, and tile constraints.
3. `NVUniversalGemmScheduling` applies fusion ordering, provider capability,
   and output-liveness policy. Pointwise lowering produces a
   `GemmEpiloguePlan` containing source, inputs, outputs, and name bindings.
4. Supported pointwise Loop IR becomes direct CuTeDSL source; unsupported
   expressions fall back to EVT through the same plan. Reduction callbacks
   become a `GemmReductionCompileConfig` shared by dense and block-scaled
   providers.
5. Scheduling and both providers validate the same reduction contract through
   `NVGemmReductionCapabilities` before the kernel claims its nodes.

## Backend parity

| Capability | Triton fusion | FlexGEMM / QuACK | NVGEMM CuTeDSL |
| --- | --- | --- | --- |
| Pointwise epilogue graph | General scheduler IR subject to Triton lowering | FX graph over the supported CuTeDSL op set | Loop IR over the supported CuTeDSL op set |
| Captured tensor inputs | General indexing subject to scheduler fusion | Tile, row, column, or scalar broadcast | Tile, row, or column broadcast; dense GEMM also supports scalar broadcast |
| Same-shape auxiliary outputs | General scheduler outputs | Arbitrary tuple for `aten.mm` | Arbitrary tuple |
| Independent reductions in one kernel | Supported when scheduler and resource constraints permit | One local-reduction plan | One local-reduction plan |
| Reduction-fed full-shape consumers | General fused graph | One planned feed-main value | Primary plus one secondary callback; additional consumers fall back |
| Compressed reduction outputs | General scheduler outputs | At most one | At most one |
| Reduction geometry | General Triton reduction domains | Grouped GEMM M or N axis | Grouped GEMM M or N axis |
| Composite reductions | Expressible as ordinary Triton IR | Online softmax plus supported primitive reductions | Online softmax, variance, logsumexp, and supported primitive reductions |
| Unsupported expression | Triton scheduling or lowering fallback | FlexGEMM fallback | NVGEMM fallback |

NVGEMM is therefore not at Triton parity. It is close to FlexGEMM for the
shared single-reduction contract, but the two CuTeDSL backends are not exact
substitutes: FlexGEMM has FX-level ownership and provider-wide scalar capture,
while NVGEMM recognizes more composite Loop IR reductions and supports a second
reduction-fed consumer.
