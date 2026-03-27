(torch.compiler_inductor)=

# TorchInductor

TorchInductor is the default backend compiler for `torch.compile`. It receives
[FX Graphs](https://docs.pytorch.org/docs/stable/fx.html) composed of
[ATen](https://docs.pytorch.org/docs/stable/torch.compiler_ir.html) operators and
generates optimized kernels for various hardware targets. Its most important
optimization is **operator fusion**, which reduces memory bandwidth by combining
multiple operations into single fused kernels.

TorchInductor primarily generates:

- **Triton kernels** for NVIDIA, AMD, and Intel GPUs
- **Vectorized C++ kernels** for CPUs
- **CUTLASS/CK kernels** for specialized matrix operations (when `max_autotune` is enabled)

## Compilation Pipeline

TorchInductor applies a series of passes to transform an FX graph into optimized
executable code:

1. **[Pre-grad passes](torch.compiler_inductor_fx_passes.md)**: Pattern matching and rewrites on high-level Torch IR before autograd.
2. **AOT Autograd**: Traces forward and backward graphs, functionalizes, normalizes, and decomposes to ATen IR.
3. **[Joint graph passes](torch.compiler_inductor_fx_passes.md)**: Optimizations on the combined forward-backward graph.
4. **Partitioner**: Min-cut partitioning of the joint graph into separate forward and backward graphs.
5. **[Post-grad passes](torch.compiler_inductor_fx_passes.md)**: Final high-level optimizations (dead code elimination, pattern matching) before lowering.
6. **[Graph lowering](torch.compiler_inductor_ir.md)**: Converts ATen IR into Inductor IR.
7. **[Scheduling](torch.compiler_inductor_scheduler.md)**: Analyzes dependencies and fuses operations to minimize memory traffic.
8. **[Code generation](torch.compiler_inductor_codegen.md)**: Produces target-specific kernels (Triton, C++, CUTLASS) and wrapper code.

For a walkthrough of these steps with a concrete example, see the
[Architecture & Starter Example](torch.compiler_inductor_overview.md).

```{toctree}
:maxdepth: 1

torch.compiler_inductor_overview.md
torch.compiler_inductor_decomposition.md
torch.compiler_inductor_fx_passes.md
torch.compiler_inductor_ir.md
torch.compiler_inductor_scheduler.md
torch.compiler_inductor_codegen.md
torch.compiler_inductor_autotuning.md
torch.compiler_inductor_caching.md
torch.compiler_inductor_debugging.md
```
