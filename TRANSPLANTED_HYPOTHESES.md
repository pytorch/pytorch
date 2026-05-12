# Transplanted Hypothesis Graph: tinygrad-experiments -> pytorch/pytorch

Sourced from 8 hypothesis graphs across 6 investigation tracks in
[kimjune01/tinygrad-experiments](https://github.com/kimjune01/tinygrad-experiments).

Each transplanted hypothesis maps a confirmed or strongly-evidenced tinygrad finding
to its pytorch/pytorch analogue. The question: which findings generalize, and where
in pytorch's codebase would the perturbation land?

---

## Concept Map: tinygrad -> pytorch

| tinygrad concept | pytorch equivalent | Notes |
|---|---|---|
| `realize` | `torch.compile` / `torch._dynamo` | Lazy graph -> concrete execution |
| BEAM search | Triton autotuner / `torch._inductor.autotune` | Black-box kernel optimization |
| `heuristic.py` | `torch._inductor.codegen.triton` | Kernel codegen heuristics |
| `PatternMatcher` / `graph_rewrite` | FX graph passes / `torch.fx.passes` | Pattern-based IR rewriting |
| `linearizer.py` | Triton codegen / `torch._inductor.scheduler` | Instruction scheduling within kernels |
| `schedule/rangeify.py` | `torch._inductor.ir.Loops` | Loop nest construction |
| `UOp` | FX `Node` / Triton IR | Intermediate representation |
| `PCONTIG` fusion | Inductor fusion passes | Multi-kernel -> single-kernel fusion |
| `MetalGraph` / `CUDAGraph` | `torch.cuda.CUDAGraph` | Graph-level kernel batching |
| GGUF dequant chains | `torch.ao.quantization` / bitsandbytes | Quantized weight handling |
| `tc.py` / tensor cores | `torch._inductor.codegen.cuda.cuda_kernel` | Tensor core code generation |
| `sz.py` line budget | N/A (pytorch has no line cap) | Architectural constraint unique to tinygrad |

---

## T1: Instruction Scheduling via Critical Path Length

**Source:** beam/HYPOTHESIS_GRAPH.md (H0.5), or/hypothesis-graph.md (H1, H1.1a)

**tinygrad finding:** Replacing the flat `{LOAD:-1, ALU:0, STORE:+1}` instruction priority
with CPL (Critical Path Length) priority produced a 23% speedup on matvec kernels.
A one-character change (`LOAD: -1 -> 0`) closed 20-50% of the gap on 9/11 benchmarks.
The OR investigation traced this to Hu's algorithm (1961) for precedence-constrained scheduling.

**pytorch analogue:** Triton's codegen relies on LLVM's instruction scheduler for
the final binary, but the Triton IR -> LLVM IR lowering determines instruction ordering
within basic blocks. `torch._inductor` generates Triton code with a specific load/compute
ordering baked in by the template.

**Perturbation:**
1. Find where Inductor's Triton codegen orders loads vs. compute in generated kernels.
   Start at `torch/_inductor/codegen/triton.py` and `triton_kernel.py`.
2. For matmul-shaped kernels, check whether loads are clustered before compute (tinygrad's
   original bug) or interleaved.
3. If clustered: test CPL-ordered interleaving on a matvec benchmark.

**Specific pytorch code paths:**
- `torch/_inductor/codegen/triton.py` — Triton kernel generation
- `torch/_inductor/scheduler.py` — decides kernel boundaries and operation ordering
- `torch/_inductor/ir.py` — `Loops`, `Reduction` nodes determine loop nest structure

**Predicted trajectory:** Convergent (low impact). Triton + LLVM already have scheduling passes.
But if Inductor's generated Triton code has a fixed load-first ordering that LLVM doesn't
override, the same 20%+ matvec improvement would apply. Worth checking.

**Fix branch:** `fix/inductor-cpl-scheduling` — modify Inductor's Triton codegen to
interleave loads with compute for memory-bound kernels.

---

## T2: Matvec Loop Ordering

**Source:** matvec/HYPOTHESIS_GRAPH.md (H1m, H5m)

**tinygrad finding:** The strongest single finding across all investigations.
For T=1 decode (matvec-shaped matmuls: 1xK @ KxN), the generated kernel walks
the weight matrix with stride 32768 bytes in the inner loop instead of stride 4.
This produces 8192x worse locality. PR #14630 showed a 2.76x CPU gap from this
alone, and a hand-written kernel closed to parity. On Metal, the same pattern
was confirmed via kernel dump. On CUDA (RTX 5000 Ada), the contiguous+prune fix
for lazy dequant chains gave 8.2x improvement, confirming the issue is cross-backend.

**pytorch analogue:** `torch._inductor` generates Triton kernels for matmul via
templates or autotuned configs. For T=1 matvec, the question is whether Inductor
falls back to a generic matmul template with wrong loop ordering, or dispatches
to a specialized matvec path.

**Perturbation:**
1. Generate a Triton kernel for `torch.mm(x, W)` where x is `(1, 4096)` and W is
   `(4096, 4096)` via `torch.compile`. Dump the generated Triton IR.
2. Check whether the inner loop iterates along the K (reduction) dimension with
   unit-stride access on W, or with strided access.
3. Compare to cuBLAS's matvec dispatch (which pytorch uses without `torch.compile`).

**Specific pytorch code paths:**
- `torch/_inductor/codegen/triton.py` — kernel generation for mm
- `torch/_inductor/kernel/mm.py` — matmul template selection
- `torch/_inductor/select_algorithm.py` — autotune between templates
- `aten/src/ATen/native/cuda/Blas.cu` — cuBLAS dispatch (the fallback)

**Predicted trajectory:** Likely convergent (pytorch already handles this).
cuBLAS dispatches matvec to `cublasSgemv` which has correct loop ordering.
Triton's autotuner likely picks a config with correct tiling. But worth verifying
that `torch.compile` doesn't produce a pathological kernel for this shape.

**Why this matters for pytorch:** LLM decode is 100% matvec. If `torch.compile`
produces suboptimal matvec kernels even 5% of the time, it affects every
production LLM deployment using compiled inference.

**Fix branch:** `fix/inductor-matvec-loop-order` — add a matvec-specific Triton
template to Inductor's template selection that guarantees unit-stride weight access.

---

## T3: Post-TC Optimization Axis Selection

**Source:** beam/HYPOTHESIS_GRAPH.md (H0.4, H0.5, H0.6a)

**tinygrad finding:** After enabling tensor cores, the heuristic applied UPCAST
to the wrong axis. For tall-skinny matmuls (16x4096 x 4096x4096), the heuristic
upcasted the M dimension (size 16) instead of the N dimension (size 4096).
BEAM discovered that UPCAST N + UNROLL K gives 43-59% speedup over the
heuristic's UPCAST M. Even square matmuls improved 51%. On AMD RDNA4 (gfx1201),
UNROLL(0,4) broke WMMA due to swizzle lane mapping incompatibility — the fix
required both-axis upcast + UNROLL(0,2) for gfx12.

**pytorch analogue:** Triton's tensor core code generation uses `tl.dot` which
maps to WMMA/MMA instructions. The tile sizes and unrolling factors are
autotuned, but the default configs may have the same axis bias.
`torch._inductor`'s CUDA codegen for tensor core ops (via cutlass or Triton)
has configurable tile sizes.

**Perturbation:**
1. Run `torch.compile` on a tall-skinny matmul `(16, 4096) @ (4096, 4096)` on
   both NVIDIA and AMD GPUs.
2. Dump the generated Triton kernel. Check which axis the tile configuration
   prioritizes for unrolling/upcast.
3. Compare to the optimal configuration from tinygrad's investigation:
   UPCAST N + UNROLL K for most backends; both-axis + UNROLL(0,2) for RDNA4.

**Specific pytorch code paths:**
- `torch/_inductor/kernel/mm.py` — matmul configs, `mm_configs()`
- `torch/_inductor/codegen/triton.py` — tile size selection
- `torch/_inductor/select_algorithm.py` — autotuning harness

**Predicted trajectory:** Likely convergent (autotuner handles this). But the
RDNA4-specific swizzle issue (H0.6a) is notable — pytorch's AMD support
may have the same lane-mapping incompatibility if the Triton compiler
doesn't handle RDNA4 WMMA operand layout correctly.

**Fix branch:** `fix/inductor-rdna4-wmma-unroll` — add RDNA4-specific
tile constraints to Inductor's mm autotuning configs.

**Issues to investigate:**
- Search pytorch issues for "RDNA4" or "gfx1201" WMMA failures
- Check `torch/_inductor/codegen/rocm/` for RDNA4-specific handling

---

## T4: Pattern Matcher Performance (graph_rewrite -> FX Passes)

**Source:** realize/HYPOTHESIS_GRAPH.md (H12, H18, H21, H27)

**tinygrad finding:** The deepest investigation (28 hypotheses). Key results:
- H12: Redundant op checks in compiled pattern matchers waste 3-4% (-3.2 to -4.0%).
  Each compiled matcher re-checks `uop.op` despite the dispatch table already filtering by op.
- H18: Merging all patterns per op into a "mega-matcher" (one function instead of N)
  eliminates N-1 function call overheads, producing -18% on micro-bench.
- H21: Automated mega-matcher generation: -15.2%.
- H22: But micro-bench gains don't propagate to end-to-end (-0% on real workloads).
  66% of rewrite calls hit 0-1 patterns. The bottleneck is traversal, not matching.
- H27: Cython transpile of the traversal loop: -7.3% end-to-end. The FIRST
  end-to-end signal in the entire investigation. The cost is CPython bytecode
  dispatch overhead in the driver loop (~300ns/node for dict/deque/tuple ops).

**pytorch analogue:** `torch.fx` graph passes traverse FX graphs and apply
pattern-based rewrites. `torch._inductor` has its own pass infrastructure.
The same structural question applies: is the bottleneck in matching or traversal?

**Perturbation:**
1. Profile `torch.compile`'s FX pass pipeline. Measure time in pattern matching
   vs. graph traversal for a representative model (ResNet-50, LLaMA).
2. Check whether FX passes have the same redundant-check pattern (dispatch by
   opcode, then re-check opcode inside the handler).
3. If traversal dominates: the Cython finding predicts that any hot-loop
   FX pass rewritten in C/C++ would save ~7% of compilation time.

**Specific pytorch code paths:**
- `torch/fx/passes/` — FX pass infrastructure
- `torch/_inductor/fx_passes/` — Inductor-specific passes
- `torch/fx/graph.py` — `Graph` traversal methods
- `torch/_inductor/pattern_matcher.py` — Inductor's pattern matching

**Predicted trajectory:** Convergent (pytorch passes are already in C++ for hot paths).
`torch._inductor.pattern_matcher` may have the redundant-check issue, but pytorch's
compilation pipeline is already heavily optimized. The tinygrad finding is most relevant
to pure-Python pass frameworks.

**Why this matters:** `torch.compile` latency is a known pain point. If 7% of
compilation time is pure CPython overhead in pass traversal (matching the tinygrad
finding), that's a concrete optimization target.

**Fix branch:** `perf/fx-pass-traversal-cython` — profile and potentially Cythonize
hot FX pass traversal loops.

---

## T5: Lazy Dequant Chain Regression

**Source:** pareto-frontier/HYPOTHESIS_GRAPH.md (H12, H13, H15-H18)

**tinygrad finding:** The single largest speedup in all investigations: 12.4x.
GGUF quantized models create lazy dequantization chains (130 UOps per weight tensor)
that fuse into the matmul kernel. The fused kernel re-executes the entire dequant
graph on every forward pass, producing 328-line kernels with 271 scalar byte loads
achieving 3 GB/s (2% of peak bandwidth). Fix: `.contiguous()` on weights (fusion barrier)
+ `prune=True` on JIT (onetime detection) -> 141 tok/s at 354 GB/s.

Key subtlety (H15): prune misclassifies cache/state kernels as onetime on the
prefill path. Fix: prune only on the decode (rollout) JIT, not prefill.

**pytorch analogue:** `torch.compile` with quantized models (bitsandbytes, GPTQ,
AWQ) faces the same question: does the compiled graph re-execute dequantization
on every forward pass, or does it cache the materialized weights?

**Perturbation:**
1. Load a quantized LLM (e.g., LLaMA-7B-GPTQ) with `torch.compile`.
2. Profile whether dequantization ops are re-executed on every `forward()` call
   or cached after first materialization.
3. Check `torch._inductor`'s constant folding: does it detect that weight
   dequant subgraphs have constant inputs and fold them?
4. If dequant is re-executed: test adding a `torch.compile` constant folding
   pass that materializes weight dequant results.

**Specific pytorch code paths:**
- `torch/_inductor/constant_folding.py` — constant folding pass
- `torch/_inductor/fx_passes/joint_graph.py` — joint forward/backward passes
- `torch/ao/quantization/` — quantization infrastructure
- `torch/_dynamo/variables/` — how dynamo traces quantized ops

**Predicted trajectory:** Divergent (likely a real issue). Bitsandbytes and
similar libraries create dequant graphs that `torch.compile` may not constant-fold
because the weights are marked as parameters (not constants). The tinygrad
finding suggests checking whether `torch._dynamo` treats quantized weight
buffers as compile-time constants or runtime inputs.

**Why this is high-priority:** Every quantized LLM deployment using `torch.compile`
may be hitting this. The 12.4x tinygrad speedup from a 2-line fix suggests
pytorch could have a similar-magnitude issue.

**Fix branch:** `fix/inductor-dequant-constant-fold` — ensure Inductor's constant
folding pass detects and materializes weight dequant subgraphs.

**Issues to investigate:**
- Search pytorch issues for "bitsandbytes torch.compile slow"
- Search for "dequantize" in `torch/_inductor/constant_folding.py`
- Check if `torch._dynamo` marks quantized weight buffers as `ConstantVariable`

---

## T6: Algebraic Fusion for Compound Reductions

**Source:** or/hypothesis-graph.md (H5), pareto-frontier/HYPOTHESIS_GRAPH.md (H6, H10)

**tinygrad finding:** Naive fusion (PCONTIG=99, concatenating serial-chain kernels)
is 1.8x SLOWER than 3 separate kernels for softmax. The problem is over-fusion:
register pressure blows up, occupancy drops. Three research groups independently
converged on algebraic decomposition as the fix:
- Flashlight: `exp` is a homomorphism (R,+) -> (R+,x), enabling online softmax
- RedFuser: separable decomposition F(x,d) = G(x)H(d) for O(1)-state fusion
- Neptune: algebraic correction terms when decomposition fails

The proof-manual analysis showed a clear kill-condition hierarchy:
homomorphism -> separable -> correction -> accept the split.

**pytorch analogue:** `torch._inductor`'s fusion passes decide which ops to fuse
into a single Triton kernel. The same over-fusion risk exists: fusing too aggressively
increases register pressure and hurts occupancy.

**Perturbation:**
1. Check whether Inductor's fusion pass has a register pressure model.
   Does it ever decide NOT to fuse because of predicted register pressure?
2. Generate a compiled softmax via `torch.compile(F.softmax)`. Is it one kernel
   or multiple? If one, does it use online softmax (algebraic decomposition)
   or naive max-subtract-exp-sum-divide (serial chain)?
3. Compare to FlashAttention's approach (which IS algebraic decomposition).

**Specific pytorch code paths:**
- `torch/_inductor/fx_passes/group_batch_fusion.py` — fusion decisions
- `torch/_inductor/scheduler.py` — `can_fuse()`, `score_fusion()`
- `torch/_inductor/codegen/triton.py` — register pressure estimation
- `torch/_inductor/lowering.py` — softmax lowering

**Predicted trajectory:** Convergent (pytorch already handles softmax well).
`F.softmax` is lowered to a specialized path, not naive fusion. FlashAttention
is integrated. But the general principle — "fusion decisions should account for
register pressure and algebraic structure" — applies to custom operators and
novel architectures that don't have hand-tuned paths.

**Why this matters:** As users write custom operations and expect `torch.compile`
to fuse them efficiently, the algebraic decomposition framework provides a
principled answer to "when should the compiler fuse?"

**Fix branch:** `feat/inductor-algebraic-fusion-cost-model` — add register pressure
and algebraic decomposability checks to Inductor's fusion scoring.

---

## T7: Theory-Guided Autotuning (Abduction Engine)

**Source:** beam/HYPOTHESIS_GRAPH.md (H0, H2-H5)

**tinygrad finding:** BEAM search (black-box autotuning) treats all transformations
as equally plausible. An abduction engine that forms theories about WHY a kernel
is slow and tests only relevant transformations beats the heuristic on 4/5
workloads with 52 trials vs BEAM's 193 actions, at 1.85x geometric mean speedup.

Key insight (H4): Cache the semantic THEORY ("tall-skinny matmuls benefit from
TC + UPCAST N + UNROLL K"), not the literal schedule. The theory transfers across
7 different matmul shapes while exact schedules fail on 3/6. The theory derived
from one measurement (gemm_1024) beats the heuristic 5-9x on ALL tested shapes
with zero additional measurements.

**pytorch analogue:** `torch._inductor.select_algorithm` autotuning picks between
kernel templates by timing each. Triton autotuner sweeps tile configs. Both are
black-box: time everything, pick the winner. Neither forms theories about WHY
one config is better.

**Perturbation:**
1. Profile Inductor's autotuning for a tall-skinny matmul (16x4096 x 4096x4096).
   How many configs does it try? Does it find tensor cores?
2. Implement a minimal theory cache: after autotuning one shape, store
   "tall-skinny matmul: tile config X is best because memory-bound" and
   apply it to subsequent similar shapes without re-tuning.
3. Measure: does the cached theory transfer? Does it reduce autotuning time
   for similar shapes?

**Specific pytorch code paths:**
- `torch/_inductor/select_algorithm.py` — `autotune_select_algorithm()`
- `torch/_inductor/kernel/mm.py` — `mm_configs()`, template list
- `torch/_inductor/autotune_process.py` — autotuning subprocess
- `torch/_inductor/runtime/triton_heuristics.py` — Triton config selection

**Predicted trajectory:** Divergent (real opportunity). pytorch's autotuner
re-tunes from scratch for every new shape. A theory cache that transfers
structural insights across shapes would reduce compilation time for models
with many similar-but-not-identical matmul shapes (e.g., MoE models with
different expert sizes, or attention heads at different sequence lengths).

**Fix branch:** `feat/inductor-theory-cache` — add a structural similarity
metric to Inductor's autotuning cache that enables theory transfer across shapes.

---

## T8: CPython JIT Impact on Compilation

**Source:** pareto-frontier/CPYTHON_JIT_HYPOTHESIS_GRAPH.md (H0-H14)

**tinygrad finding:** CPython 3.16's recording JIT provides 5-16% improvement
on pattern matching workloads (optimized builds), up from 0% on 3.14.
The bottleneck is the monomorphic call guard (`_GUARD_IP__PUSH_FRAME`)
which deopts on polymorphic call sites. The JIT traces the loop body as
native code but exits to tier 1 for every function call with a different callee.

The Cython transpile of the hot traversal loop proved 7.3% end-to-end improvement,
quantifying the CPython bytecode dispatch overhead. The gap is NOT from StackRef
conversion or generic dispatch (H14, killed) but from eliminating the entire
bytecode fetch/decode/jump loop.

**pytorch analogue:** pytorch's Python-side compilation pipeline (dynamo tracing,
FX pass execution, Inductor codegen) runs in CPython. The same CPython JIT
limitations apply to any hot Python loop in the compilation pipeline.

**Perturbation:**
1. Profile `torch.compile` on CPython 3.14 vs 3.16 (when available). Does the
   JIT help compilation speed?
2. Identify the hottest Python loops in the compilation pipeline. Are they
   amenable to Cythonization?
3. Check which FX pass loops have polymorphic call sites (different handler
   functions per node type).

**Specific pytorch code paths:**
- `torch/_dynamo/` — tracing infrastructure (Python-heavy)
- `torch/fx/interpreter.py` — `Interpreter.run()` loop
- `torch/_inductor/graph.py` — `GraphLowering` methods

**Predicted trajectory:** Convergent (pytorch already optimizes compilation
with C++ backends). The dynamo->inductor pipeline has C++ acceleration for
hot paths. But the Python-side graph manipulation (FX passes, shape propagation)
may still be bottlenecked by CPython overhead.

**Why this matters:** `torch.compile` latency is a top user complaint. If the
CPython JIT findings transfer (5-16% from JIT, 7% more from Cythonization),
that's a concrete 12-23% compilation speedup from runtime optimization alone.

---

## T9: Backend-Aware dtype Support Checks

**Source:** beam/HYPOTHESIS_GRAPH_PTX_BF16.md (H0-H6)

**tinygrad finding:** The `is_dtype_supported()` function checked the declared
renderer name (empty string) instead of the actually-selected renderer (PTXRenderer).
This caused a mismatch: bf16 was reported as supported, but the selected renderer
didn't support it, causing a KeyError. Root cause: silent renderer fallback
(CUDARenderer -> PTXRenderer when NVRTC is missing) + the support check reading
a stale/empty renderer name.

Shipped as PR #16108: `is_dtype_supported` now queries the resolved renderer.

**pytorch analogue:** pytorch's backend capability detection
(`torch.cuda.is_bf16_supported()`, device capability checks) may have similar
stale-state issues when the actual compute backend differs from the declared one
(e.g., running on a GPU that reports a capability it doesn't fully support,
or when CUDA falls back to a different code path).

**Perturbation:**
1. Check whether `torch.cuda.is_bf16_supported()` queries runtime device
   properties or compile-time constants.
2. Test on a GPU that has partial bf16 support (e.g., compute capability < 8.0
   with limited bf16 ops).
3. Check if there's a "silent fallback" pattern in pytorch's CUDA backend
   similar to tinygrad's CUDARenderer -> PTXRenderer fallback.

**Specific pytorch code paths:**
- `torch/cuda/__init__.py` — `is_bf16_supported()`
- `aten/src/ATen/cuda/CUDABlas.cpp` — cuBLAS dtype dispatch
- `torch/_inductor/codegen/cuda/` — CUDA codegen dtype handling

**Predicted trajectory:** Convergent (pytorch handles this). pytorch has
extensive device capability checking. But the "silent fallback" pattern is
worth checking — any place where pytorch silently switches backends without
updating capability flags could have the same class of bug.

---

## T10: Operations Research Formulations for Kernel Optimization

**Source:** or/hypothesis-graph.md (H1-H5, Proof Manual Validation)

**tinygrad finding:** Five compiler optimization problems map directly to
classical OR problems that the compiler community derived independently:

| Compiler problem | OR formulation | OR state of art |
|---|---|---|
| Instruction scheduling | RCPSP (Pritsker 1969) | Exact polynomial for special cases |
| Bank conflict avoidance | GF(2) assignment problem | XOR-swizzle is closed-form solution |
| Fusion decisions | Weighted hypergraph partitioning | METIS, Kernighan-Lin |
| Quantization bitwidth allocation | Integer linear programming | MxMoE (ICML 2025) |
| Autotuning search | Branch-and-bound | Constraint programming |

The proof manual validation showed that H1 (CPL scheduling) and H4 (fused dequant)
share a kill condition: greedy optimization crosses occupancy tier boundaries.
The fix is the same for both: APRP-aware resource budgeting.

**pytorch analogue:** `torch._inductor` makes all of these decisions:
- Instruction ordering within Triton kernels
- Shared memory layout for tiled operations
- Fusion decisions (which ops share a kernel)
- Quantization-aware compilation
- Autotuning search strategy

The OR lens predicts that Inductor's heuristic-based decisions could be improved
by formulating them as the corresponding OR problems.

**Perturbation:**
1. Map Inductor's fusion scoring (`scheduler.py:score_fusion()`) to hypergraph
   partitioning. Is the current heuristic equivalent to a known partitioning
   algorithm? Is it provably suboptimal for known cases?
2. Check whether Inductor's Triton codegen accounts for shared memory bank
   conflicts. If not, the XOR-swizzle finding applies directly.
3. Formulate Inductor's autotuning as a constraint satisfaction problem.
   Does the current grid search explore redundant configurations that
   constraint propagation would eliminate?

**Specific pytorch code paths:**
- `torch/_inductor/scheduler.py` — `score_fusion()`, `can_fuse()`
- `torch/_inductor/codegen/triton.py` — shared memory layout
- `torch/_inductor/select_algorithm.py` — autotuning grid
- `torch/_inductor/codegen/cuda/cuda_kernel.py` — CUDA codegen

**Predicted trajectory:** Divergent (real opportunity). Inductor's fusion
heuristics are known to be imperfect (users regularly report suboptimal
fusion decisions). Formulating them as OR problems provides a principled
improvement path and a way to prove optimality bounds.

**Fix branches (ordered by expected impact):**
1. `feat/inductor-fusion-as-partitioning` — reformulate fusion scoring
2. `feat/inductor-smem-xor-swizzle` — add bank conflict avoidance
3. `feat/inductor-autotune-constraint-prop` — reduce autotuning search space

---

## T11: Codebase Complexity as Selection Pressure

**Source:** linecount/HYPOTHESIS_GRAPH.md (H0-H4)

**tinygrad finding:** tinygrad's 24k-line hard cap creates selection pressure
where every new feature must offset with reductions. The cap has been raised
9 times in 2025 (15.5k -> 24k). The codebase is remarkably clean: zero dead
code in the top 10 files. The realistic line-reduction surface is ~65-70 lines
from two concrete refactors.

**pytorch analogue:** pytorch has no line cap. The codebase is ~3M lines.
The transplantable hypothesis is not the cap itself but the diagnostic:
"Is there dead code in pytorch's hottest files, and does complexity correlate
with bug density?"

**Perturbation:** Not a code change. A diagnostic question for /triage:
which files in `torch/_inductor/` are largest, and do they have unreferenced
functions? This would identify concrete refactoring targets.

**Not transplantable as a fix.** The line budget is an architectural constraint
unique to tinygrad. But the diagnostic method (exhaustive grep for unreferenced
symbols in the top N files) applies anywhere.

---

## Priority Ranking

Ranked by expected impact on pytorch/pytorch, accounting for whether pytorch
likely already handles the issue:

| Rank | Hypothesis | Expected impact | pytorch likely handles? |
|---|---|---|---|
| 1 | T5: Lazy dequant chain regression | **High** — 12.4x in tinygrad, unknown in pytorch | **Uncertain** — depends on constant folding |
| 2 | T7: Theory-guided autotuning | **High** — reduces compile time for varied shapes | **No** — pytorch autotuner is black-box |
| 3 | T10: OR formulations for kernel optimization | **High** — principled improvement over heuristics | **No** — Inductor uses ad-hoc heuristics |
| 4 | T6: Algebraic fusion cost model | **Medium** — matters for custom ops | **Partially** — softmax is special-cased |
| 5 | T2: Matvec loop ordering | **Medium** — affects LLM decode | **Likely yes** — cuBLAS handles this |
| 6 | T1: CPL scheduling | **Medium** — affects memory-bound kernels | **Likely yes** — LLVM scheduler |
| 7 | T3: Post-TC axis selection | **Low-medium** — autotuner covers this | **Likely yes** — but RDNA4 edge case |
| 8 | T4: Pattern matcher performance | **Low** — pytorch compilation is already C++-heavy | **Mostly yes** |
| 9 | T8: CPython JIT impact | **Low** — pytorch C++ backends bypass CPython | **Mostly yes** |
| 10 | T9: Backend dtype mismatch | **Low** — pytorch has extensive capability checks | **Likely yes** |

---

## Investigation Order

Start with T5 (lazy dequant chains) because:
1. Highest potential impact (12.4x in tinygrad)
2. Most uncertain whether pytorch handles it (no clear constant folding path for quantized weights)
3. Easy to test: load a GPTQ model with `torch.compile`, profile dequant execution count
4. If confirmed: 2-line fix (add constant folding for weight dequant subgraphs)

Then T7 (theory-guided autotuning) because:
1. Addresses a known pytorch pain point (compile latency from autotuning)
2. Novel contribution (no existing work on theory transfer in pytorch's autotuner)
3. Concrete prototype: add shape-similarity cache to `select_algorithm.py`

Then T10 (OR formulations) because:
1. Provides a framework for systematically improving Inductor's heuristics
2. The fusion-as-partitioning formulation is the most actionable
3. The proof manual validation gives a principled way to evaluate improvements

---

## Provenance

Each transplanted hypothesis traces back to a confirmed or strongly-evidenced
finding in the source investigation:

| ID | Source file | Source hypothesis | Confidence | Mode |
|---|---|---|---|---|
| T1 | beam/HYPOTHESIS_GRAPH.md, or/hypothesis-graph.md | H0.5, H1.1a | 95% | Induction |
| T2 | matvec/HYPOTHESIS_GRAPH.md | H1m, H5m | 90% | Deduction + induction |
| T3 | beam/HYPOTHESIS_GRAPH.md | H0.4, H0.5, H0.6a | 95% | Induction |
| T4 | realize/HYPOTHESIS_GRAPH.md | H12, H18, H21, H27 | 95% | Induction |
| T5 | pareto-frontier/HYPOTHESIS_GRAPH.md | H12, H13, H15-H18 | 99% | Induction |
| T6 | or/hypothesis-graph.md | H5 | 80% | Deduction |
| T7 | beam/HYPOTHESIS_GRAPH.md | H0, H2-H5 | 90% | Induction |
| T8 | pareto-frontier/CPYTHON_JIT_HYPOTHESIS_GRAPH.md | H5, H12-H14 | 95% | Induction |
| T9 | beam/HYPOTHESIS_GRAPH_PTX_BF16.md | H0-H6 | 99% | Induction |
| T10 | or/hypothesis-graph.md | H1-H5 | 85% | Deduction |
| T11 | linecount/HYPOTHESIS_GRAPH.md | H0-H4 | 95% | Induction |

---

## Windows CUDA Experiment Queue

Experiments that need a Windows machine with CUDA GPU to convert hypotheses from abduction to induction.

### Experiment 1: Lazy dequant chain (T5)
- **Hypothesis tested:** T5 — does torch.compile constant-fold weight dequant subgraphs?
- **Command:** `python -c "import torch; from transformers import AutoModelForCausalLM; m = AutoModelForCausalLM.from_pretrained('TinyLlama/TinyLlama-1.1B-Chat-v1.0', load_in_4bit=True); m = torch.compile(m); x = torch.randint(0, 1000, (1, 32)).cuda(); import torch.utils.benchmark as b; t = b.Timer('m(x)', globals={'m':m,'x':x}); print(t.blocked_autorange())"`
- **Expected result:** If dequant is NOT folded, each forward pass re-executes the dequant graph. Profile with `torch.profiler` to count dequant kernel launches per forward pass. >1 = T5 confirmed.
- **Requires:** Windows, CUDA GPU (8GB+), bitsandbytes, transformers

### Experiment 2: Matvec loop ordering (T2)
- **Hypothesis tested:** T2 — does Inductor produce pathological matvec kernels for M=1?
- **Command:** `python -c "import torch; torch._inductor.config.trace.enabled = True; @torch.compile def f(x, w): return x @ w.T; x = torch.randn(1, 4096, device='cuda'); w = torch.randn(4096, 4096, device='cuda'); f(x, w); print('Check /tmp/torchinductor_*/triton/*.py for loop ordering')"`
- **Expected result:** Inspect generated Triton kernel. If inner loop strides weight matrix by large offsets instead of unit stride, T2 confirmed. Compare to cuBLAS: `torch.mm(x, w.T)` without compile.
- **Requires:** Windows, CUDA GPU, torch nightly

### Experiment 3: Autotuner shape transfer (T7)
- **Hypothesis tested:** T7 — does Inductor's autotuner retune from scratch for each new shape?
- **Command:** `python -c "import torch, time; torch._inductor.config.max_autotune = True; @torch.compile def f(x, w): return x @ w.T; shapes = [(1,k,k) for k in [512,1024,2048,4096]]; [f(torch.randn(1,k,device='cuda'), torch.randn(k,k,device='cuda')) for _,k,_ in shapes]; t0=time.time(); [f(torch.randn(1,k,device='cuda'), torch.randn(k,k,device='cuda')) for _,k,_ in shapes]; print(f'Second pass: {time.time()-t0:.2f}s')"`
- **Expected result:** If second pass is as slow as first, autotuner doesn't transfer insights across shapes. T7 confirmed.
- **Requires:** Windows, CUDA GPU, torch nightly with max_autotune

### Experiment 4: Fusion score validation (T6)
- **Hypothesis tested:** T6 — does Inductor fuse ops that blow register pressure?
- **Command:** `python -c "import torch; torch._inductor.config.trace.enabled = True; @torch.compile def f(x): return torch.softmax(x.float(), dim=-1).half() * x; x = torch.randn(32, 128, 4096, device='cuda', dtype=torch.float16); f(x); print('Check fusion decisions in trace log')"`
- **Expected result:** If softmax + cast + multiply fuses into one kernel with high register usage, profile occupancy. Low occupancy (<25%) = T6 confirmed (fusion hurt).
- **Requires:** Windows, CUDA GPU, nsight-compute for occupancy

### Experiment 5: CUDA graph batching overhead (T2/analog of H3m)
- **Hypothesis tested:** Does CUDA graph replay overhead matter for small models?
- **Command:** `python -c "import torch; m = torch.nn.Linear(256, 256).cuda(); x = torch.randn(1, 256).cuda(); g = torch.cuda.CUDAGraph(); with torch.cuda.graph(g): y = m(x); import torch.utils.benchmark as b; t1 = b.Timer('g.replay()', globals={'g':g}); t2 = b.Timer('m(x)', globals={'m':m,'x':x}); print('Graph:', t1.blocked_autorange()); print('Eager:', t2.blocked_autorange())"`
- **Expected result:** For tiny model, graph replay overhead may dominate. Compare graph vs eager latency.
- **Requires:** Windows, CUDA GPU

### Experiment 6: bf16 codegen correctness (T9)
- **Hypothesis tested:** T9 — does Inductor handle bf16→fp32 promotion correctly in all ops?
- **Command:** `python -c "import torch; @torch.compile def f(x): return torch.log(x.bfloat16()).float(); x = torch.randn(1024, device='cuda'); y_compiled = f(x); y_eager = torch.log(x.bfloat16()).float(); print('Max diff:', (y_compiled - y_eager).abs().max().item()); assert (y_compiled - y_eager).abs().max() < 1e-5"`
- **Expected result:** If max diff > 1e-5, bf16 promotion is incorrect in compiled path. Test with log, exp, cos, sin, sqrt.
- **Requires:** Windows, CUDA GPU with bf16 support (Ampere+)
