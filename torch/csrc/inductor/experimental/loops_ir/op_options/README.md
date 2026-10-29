# LoopsIR op options

Per-operation implementation options (registry, typed configs, fusion
contracts) plus three read-only plan passes over a core `KernelProgram`.
Separate pybind module `inductor_loops_ir_op_options`; it never re-registers
core types (load the core extension first).

## Passes (distinct, read-only; input program is never modified)

| Pass | C++ | Result |
| --- | --- | --- |
| decode_op_signatures | `decodeOpSignatures(const KernelProgram&, DeviceKind)` | `DecodedOps` |
| form_fused_op_regions | `formFusedOpRegions(const KernelProgram&, const DecodedOps&, const OptionRegistry&)` | `FusedOpRegions` |
| enumerate_op_candidates | `enumerateOpCandidates(const FusedOpRegions&, const DecodedOps&, const OptionRegistry&, const CodegenContext&)` | `CandidatePlan` |

Input is the post-`fuse_kernels` program (`run_pipeline(...).kernel_program`).
All passes use only inline core getters (no `kernel_ir.cpp` link dependency).

- **Decode** binds structured `CallArg` args/kwargs against the exact ATen
  schemas of `aten.mm.default`, `aten.addmm.default`, `aten.bmm.default`,
  `aten.sum.dim_IntList`, `aten.mean.dim`. Statuses: `Decoded`,
  `UnknownTarget` (untouched), `Unsupported` (typed reason: non-f32, explicit
  reduction dtype, `dim=[]`, bool Scalar, negative strides), `MalformedAbi`
  (schema/descriptor mismatch; no signature; fail closed). Layouts come from
  the extern's descriptors; `TensorBinding{root, root_offset}` composes view
  chain + descriptor offsets (backend `rel`). The original `ExternCall` is
  retained as `fallback`.
- **Regions**: prologue = single-region pointwise producer of an operand
  root; epilogue = single-region pointwise sole consumer of the result.
  Vetoes (typed `FusionVetoKind`): uses through any alias, output escape,
  extern between fused nodes, repeated operand, domain mismatch, non-affine
  or non-injective store, view_read, reductions. Each combination is checked
  against every implementing option's SDK contract (+ native plan pattern).
  Variant 0 is always unfused. `consumed_nodes`, `placement`, `produces`,
  `eliminated_buffers` describe the replacement; `shared_nodes` lists nodes
  claimed by more than one op (selection must keep consumed sets disjoint).
- **Candidates**: (variant x eligible option x manifest config), each
  registry-validated with its fused loops (`cache_key`). `candidates[0]` is
  the baseline: `native_aten_*` on variant 0 when eligible, else
  `OriginalExtern`. Nothing is selected. Strict plans contain only the native
  baseline; `native_aten_mm_bias_epilogue` and generated options are
  inductor-only (typed `NumericsFence` rejections recorded). The context's
  numerics must equal the program's.

## Executing a candidate (what the frozen plan must record)

Per candidate: `variant.signature`, `variant.fused_loops`, option id/config,
`input_bindings` (nullopt slot = computed by prologue, not an argument),
`extra_bindings`, `output_binding`. Tensor for a binding:
`as_strided(root, layout.sizes, layout.strides, root.storage_offset() + root_offset)`;
extras are their flat root buffers; the generated output pointer is the
output root (epilogue `store_offset` is inside the kernel). Python:
`codegen(option_id, config, signature, fused_loops, frozen_ctx)` then
`compile_generated_kernel(...).launch(*inputs, *extras, out)` or
`load_native_call(call)(*inputs, *extras)`.

## Numerics and ABI preconditions

- Generated addmm renders `AddmmEagerPlan` (C++, part of the cache key),
  derived from the signature and the frozen `addmm_lt_policy`, mirroring
  `Blas.cpp addmm_out_cuda_impl` (non-ROCm), as verified on live CUDA:
  - **LtBias** (beta == 1, contiguous bias squeezing to `[N]`, K > 1,
    N > 1): computes `alpha*(A@B) + bias`. A/B are read even at alpha = 0,
    so NaN/Inf propagate.
  - **Gemm**, alpha = ±0: A/B, prologues and the dot are skipped. The result
    is `bias` (beta == 1) or `beta*bias + 0`.
  - **BetaZero** (beta = ±0): the bias is ignored, so NaN/Inf do not
    propagate.
  - **EmptyK** (K == 0): `beta*bias`.

  ROCm (`AddmmLtPolicy.UNKNOWN`) is rejected for generated addmm. The env
  var `DISABLE_ADDMM_CUDA_LT` is frozen; eager caches it on first use, so it
  must not change afterwards. Remaining
  differences are rounding only (tile K order, separate alpha/beta multiplies
  vs vendor epilogue) behind the InductorOnly fence.
- `CodegenContext.device_index` (default 0) and `addmm_lt_policy` (default
  CudaDefault) are in the key; `FrozenCodegenContext` must agree with it.
- Aliasing. The registry checks names only: operand, extra, op_result and
  output names must be distinct. Physical proofs exist only where bindings
  exist:
  - `enumerate_op_candidates` gives a generated candidate only if the output
    root is a fresh node result (`TensorBinding.node_result`), shares no root
    with any operand or extra, and its store range fits the root.
  - `CompiledKernel.launch` rejects an output whose touched byte range
    intersects any argument in the same storage.
  - Native candidates return fresh eager tensors and are exempt.
  - A direct `codegen` call with arbitrary tensors has only the launch-time
    check.
- Offsets: prologue/epilogue offsets must be >= 0. Prologue `store_offset`
  and epilogue `op_result_offset` only fix in-tile identity maps, so they
  touch no memory. The epilogue `store_offset` writes memory: the output
  `KernelArg.required_elements` (= store_offset + extent) is enforced
  against the tensor's storage at launch. Decoded bindings are checked
  against their storage roots.

## Limits

Single fused consumer/producer level; one store per region; affine maps
only; core `Binary` has add/mul only (sub = add(x, mul(y, -1)); no div).
Epilogue extras that alias an op operand name are rejected by the SDK
collision check. Generated reduction: one dim. Static shapes, CUDA.

## Tests

`tests/op_options_cpu_test.cpp` (104 checks), `tests/op_passes_cpu_test.cpp`
(134 checks); Python: `test/inductor/test_loops_ir_op_options.py`,
`test/inductor/test_loops_ir_op_options_passes.py`.
