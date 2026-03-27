(torch.compiler_inductor_codegen)=

# Code Generation

After the [scheduler](torch.compiler_inductor_scheduler.md) fuses operations
and determines the execution order, TorchInductor generates executable code.
Based on the target hardware and operation types, it produces Triton, C++, or
CUTLASS/CK kernels along with wrapper functions to call them.

**Source**: [torch/_inductor/codegen/](https://github.com/pytorch/pytorch/tree/main/torch/_inductor/codegen)

## Codegen Entry Point

The code generation entry point is the `_codegen` method in the scheduler. It
accepts a `nodes: list[BaseSchedulerNode]`, iterates through each fused node,
and dispatches to the appropriate backend code generator.

The generated output for each compiled graph consists of:

1. **Generated kernels** — the actual compute kernels (Triton, C++, etc.)
2. **Wrapper function** — a Python function that orchestrates calling the
   generated kernels in the correct order, managing tensor allocations and
   passing the right arguments.
3. **`benchmark_compiled_module`** — a convenience function for benchmarking
   the generated code in isolation.

## Backend-Specific Code Generation

### Triton Kernels (GPU)

For NVIDIA, AMD, and Intel GPUs, TorchInductor generates
[Triton](https://triton-lang.org/) kernels. Triton is a Python-based language
for writing GPU kernels that compiles to optimized machine code.

TorchInductor generates Triton code for:

- **Pointwise operations** — element-wise computations fused into single kernels
- **Reductions** — sum, mean, max, and other reduce operations
- **Persistent reductions** — reductions that keep intermediate state in
  registers across iterations
- **Scans** — cumulative operations like `cumsum`
- **Template kernels** — specialized kernels for GEMM and attention patterns
  (enabled by `max_autotune`)

### C++ Kernels (CPU)

For CPU targets, TorchInductor generates vectorized C++ kernels that leverage
SIMD instructions. These kernels are compiled at runtime using the system C++
compiler.

### CUTLASS and Composable Kernel (CK)

When `max_autotune` is enabled, TorchInductor can generate kernels using:

- **CUTLASS** — NVIDIA's CUDA Templates for Linear Algebra Subroutines, used
  for high-performance matrix operations on NVIDIA GPUs.
- **Composable Kernel (CK)** — AMD's library for high-performance operations
  on AMD GPUs.

These template-based kernels are included in
[autotuning](torch.compiler_inductor_autotuning.md) alongside Triton kernels
to select the best-performing implementation.

## Ops Handlers

Ops handlers are the mechanism that translates Inductor IR operations into
backend-specific code. Each backend provides a handler that reinterprets
the `ops.*` calls in the IR's `inner_fn` for its target language.

For example, when generating Triton code, the Triton ops handler translates
`ops.load(...)` into `tl.load(...)`, `ops.store(...)` into `tl.store(...)`,
and arithmetic operations into their Triton equivalents. This design allows
the same IR to target multiple backends by swapping the ops handler.

For details on the base ops interface, see
[ops_handler.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/ops_handler.py).

## Wrapper Code

In addition to the compute kernels, TorchInductor generates a **wrapper
function** that:

- Allocates output tensors
- Computes launch parameters (grid sizes, block sizes)
- Calls the generated kernels in the order determined by the scheduler
- Manages temporary buffers and frees them when no longer needed

The wrapper is a standard Python function, making it easy to call the compiled
graph from regular PyTorch code.

:::{seealso}
For a guide to profiling generated Inductor kernels, including environment
variables for unique kernel naming and individual kernel benchmarking, see
[GPU Profiling](torch.compiler_inductor_profiling.md).
:::
