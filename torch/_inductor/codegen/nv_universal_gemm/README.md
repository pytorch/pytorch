# NVGEMM epilogue lowering

NVGEMM captures scheduler Loop IR in `loop_ir_epilogue_lowering.py`, lowers
supported expressions with `loop_ir_cutedsl_codegen.py`, and passes a
backend-neutral reduction contract through `gemm_epilogue.py`. Unsupported
programs must fall back without partially claiming their reduction producers.

## Lowering flow

1. For each proposed scheduler-node fusion, `NVGemmEpilogueCapture` normalizes
   the accumulated candidate and interprets its Loop IR.
2. `NVGemmEpilogueLowering` uses iteration ranges, reduction ranges, and access
   strides to recognize supported reduction regions and produces an
   `NVGemmEpilogueProgram`. The program owns captured nodes, reduction regions,
   the backend-neutral `GemmReductionPlan`, and tile constraints.
3. `NVUniversalGemmScheduling` applies fusion ordering, provider capability,
   and output-liveness policy. Pointwise lowering produces a
   `GemmEpiloguePlan` containing source, inputs, outputs, and name bindings.
4. Supported Loop IR becomes direct CuTeDSL source; unsupported expressions
   fall back to EVT through the same plan. Generated epilogues complete
   fragment-local reductions and return partial values for reductions that span
   fragments or threads. Dense and block-scaled kernels receive the shared
   geometry and generated combine/finalize callbacks needed to merge partials.
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
| Composite reductions | Expressible as ordinary Triton IR | Generated from supported reduction and pointwise FX nodes | Generated from supported reduction and pointwise Loop IR nodes |
| Unsupported expression | Triton scheduling or lowering fallback | FlexGEMM fallback | NVGEMM fallback |

NVGEMM represents each captured program through the structure of its Loop IR,
with its semantics carried by the generated epilogue and reduction callbacks.
