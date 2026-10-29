# Experimental C++ Inductor IR Pipeline

This directory contains an isolated experimental compiler lane with C++ LoopsIR, KernelIR, TiledIR, verifiers, and transformation passes. FX construction, runtime dispatch, and local Triton source emission are Python. Normal Inductor execution is not replaced.

## Representations

- LoopsIR has flat typed buffers, alias Views, materialized Compute results with explicit stores, and symbolic ExternCalls. Physical addresses are structured integer element offsets; no Compute layout or implicit stride application exists.
- KernelIR has generated kernel functions and a program-level dependency DAG, including opaque eager/library/custom-op calls. Functions own logical loop regions. The initial fusion pass is conservative vertical pointwise fusion, not horizontal concat fusion.
- TiledIR is function-local SSA with basic blocks. Tiling makes lane/program indices, coordinate decoding, masks, offsets, tile sizes, grid, and launch metadata explicit before codegen. The initial pointwise function has an entry block and return; reduction and parallel-region control flow are future work.

Codegen translates the already-tiled function's instructions. It does not select tiling, infer hidden layouts, or reuse normal Inductor's scheduler/Python loop IR to implement the new lane.

## Strict Eager Numerics

Supported deterministic operations must preserve eager output bits on the same device/settings. Fusion can remove allocation and memory traffic, but not observable rounding boundaries. There is no floating-point reassociation, FMA contraction, approximate math, subnormal flushing, or algebraic identity folding that changes signed-zero/NaN behavior. Independent operations do not need an artificial dependence chain.

FP32 add and multiply use explicit round-to-nearest PTX operations through Triton. This prevents backend identity folding (for example `x * 1`) from changing NaN payloads; disabling FMA contraction alone is insufficient.

The generated arithmetic subset is static FP32 add, multiply, and ReLU. Precision changes and unsupported operations remain explicit eager boundaries. Future low-precision generation must preserve eager downcast/upcast rounding. Raw-bit tests and generated PTX checks are required; approximate equality and compiler flags alone are insufficient.

## Logical Passes

The initial C++ micro-passes inline view reads, simplify checked integer indices, eliminate unit pointwise dimensions, perform local typed SSA CSE/DCE, and prune dead views/computes while retaining output/alias/extern liveness. General floating-point constant folding is deliberately excluded.

Kernel fusion requires a proven compatible domain/access map and dense/injective result coverage. A temporary disappears only when it does not escape and strict numerical semantics remain intact. Unknown extern effects are barriers. Unsupported fusion patterns remain unfused.

## Eager Extern Boundaries

Executable-mode lowering emits reduction operations as explicit eager ExternCalls. Diagnostic LoopsIR can still represent additive reductions with reduction/post bodies and accumulators. Generated reduction code and producer-to-reduction fusion are deferred.

Extern arguments and keyword arguments are structured values, preserving integer axes, booleans, None, lists/tuples, dtypes, tensor order, and multiple outputs. Opaque boundary tensors may have non-FP32 dtypes, such as int64 argmax indices or boolean reduction results; this does not imply those dtypes have generated arithmetic support. The IR stores symbolic targets, not Python callbacks. Runtime resolves eager/custom-op implementations.

Mutation, randomness, symbolic/data-dependent shapes, and unhandled operations are not silently assumed pure or bitwise deterministic. Eager itself is not guaranteed reproducible for every operator/configuration.

## Separate Executable Backend

From the working source-checkout environment:

```python
import torch
from torch._inductor.loops_ir_backend import loops_ir_backend

function = torch.compile(
    lambda x, w: torch.relu((x * w).sum(dim=1)),
    backend=loops_ir_backend,
    fullgraph=True,
    dynamic=False,
)
x = torch.randn(2, 3, device="cuda", dtype=torch.float32)
w = torch.randn(3, device="cuda", dtype=torch.float32)
with torch.inference_mode():
    result = function(x, w)
```

The multiplication and ReLU use the new generated pointwise path; the reduction is an explicit eager call. No fusion crosses that boundary. The first executable backend is static, single-device CUDA inference. AOT framework adaptation may be reused to lift parameters/functionalize FX, but no Inductor optimization/decomposition or codegen pipeline implements the new backend.

## Existing Diagnostic Hook

```bash
TORCHINDUCTOR_CPP_LOOPS_IR_DUMP=1 python -c 'import torch; f=torch.compile(lambda x,w: torch.relu((x*w).sum(dim=1)),fullgraph=True); print(f(torch.randn(2,3),torch.randn(3)))'
```

This default-off hook still independently constructs and prints the C++ LoopsIR before normal GraphLowering. It does not switch execution to the new backend. Upstream FX transformations may already have changed that graph. An FX cache hit can bypass the hook. Unsupported diagnostic lowering reports a contextual error rather than an incomplete dump.

The C++ extension compiles lazily; importing the lowerer/backend must not trigger a build. A working compiler, Ninja, and PyTorch/pybind headers are required.

## Pointer And Storage Contracts

Input buffers bind to tensor.data_ptr(), not the underlying storage base. Input descriptors have relative offset zero even for sliced inputs. Nested view offsets compose from those pointers; torch.as_strided reconstruction adds the actual tensor.storage_offset to the IR-relative offset. Distinct input names do not establish disjoint storage.

Generated results are allocated as flat spans and exposed through their boundary descriptors. Opaque outputs are acquired from the actual eager call, not copied into a preallocated guessed layout. Padding may remain unwritten; passes and generated execution must distinguish padding from invalid overlapping or incomplete stores.

## Verification And Benchmarking

Core diagnostic/builder tests are `/home/lsakka/pytorch10/pytorch/test/inductor/test_loops_ir.py`. Logical pass tests are `/home/lsakka/pytorch10/pytorch/test/inductor/test_loops_ir_passes.py`. GPU execution tests are `/home/lsakka/pytorch10/pytorch/test/inductor/test_loops_ir_backend.py`.

```bash
CC=/usr/bin/gcc CXX=/usr/bin/g++ MAX_JOBS=2 /home/lsakka/.conda/envs/fresh-test/bin/python /home/lsakka/pytorch10/pytorch/test/inductor/test_loops_ir.py
CC=/usr/bin/gcc CXX=/usr/bin/g++ MAX_JOBS=2 /home/lsakka/.conda/envs/fresh-test/bin/python /home/lsakka/pytorch10/pytorch/test/inductor/test_loops_ir_passes.py
CC=/usr/bin/gcc CXX=/usr/bin/g++ MAX_JOBS=2 /home/lsakka/.conda/envs/fresh-test/bin/python /home/lsakka/pytorch10/pytorch/test/inductor/test_loops_ir_backend.py
```

The model-validation target is the existing TorchBench ResNet18 FP32 inference model in `/home/lsakka/pytorch10/pytorch/benchmark`, with its original weights and inputs shared by eager, the new backend, and Inductor. Eager is the bitwise reference. Report normal Inductor's bitwise status separately, along with generated/extern coverage and measured latency. No toy test or unmeasured performance claim constitutes TorchBench validation.

The benchmark harness is `/home/lsakka/pytorch10/pytorch/benchmarks/dynamo/loops_ir_benchmark.py`. The working design is `/home/lsakka/inductor_ir_design.md`.

Validated ResNet18 FP32 eval (batch 8, H100): the new pipeline is bytewise-equal to eager, with 25 pointwise kernels fused to 17 generated functions and 43 eager/library calls. Median latency is 2.213 ms versus eager 1.960 ms and Inductor 1.947 ms. Performance parity is not yet reached (about 14% slower than Inductor). Inductor differs from eager bits on this run. All 65 tests and Python lint pass. Full evidence: `/home/lsakka/loops_ir_resnet18_fp32.foiegl/report.json` and `/home/lsakka/loops_ir_resnet18_fp32.foiegl/artifacts`.
