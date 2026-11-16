# CuteDSL Template System

## Quick Start

Writing a CuteDSL template:

```python
from torch._inductor.codegen.cutedsl import CuteDSLTemplate

template_source = """
@cute.kernel
def {{kernel_name}}_kernel(A, B, C):
    # Your CUTLASS kernel logic here
    pass

{{def_kernel("A", "B", "C")}}
    # Call the kernel
    {{kernel_name}}_kernel(A, B, C)
    return C
"""

my_template = CuteDSLTemplate(
    name="my_gemm",
    source=template_source,
)
```

## Architecture

- **[CuteDSLTemplate](cutedsl_template.py#L39)**: Template definition and registration. Generates ChoiceCallers for autotuning.
- **[CuteDSLTemplateKernel](cutedsl_kernel.py#L61)**: Handles code generation, provides template hooks (`def_kernel`), manages args.
- **[CuteDSLScheduling](cutedsl_scheduling.py#L28)**: Integrates with Inductor's scheduler, handles kernel compilation via [`async_compile.cutedsl()`](../../async_compile.py#L756).
- **[CuteDSLTemplateBuffer](../../ir.py)**: IR node representing a CuteDSL template operation in the graph.

### Compilation Process

CuteDSL requires source files for compilation (cannot compile from strings directly). The process:

1. **[CuteDSLScheduling](cutedsl_scheduling.py#L59)** generates the kernel code string and calls [`async_compile.cutedsl()`](../../async_compile.py#L756)
2. **[async_compile.cutedsl()](../../async_compile.py#L756)** uses [`PyCodeCache.write()`](../../codecache.py) to write source to a temporary `.py` file
3. **[PyCodeCache](../../codecache.py)** loads the module from disk, enabling CUTLASS compilation
4. The compiled kernel is wrapped in **[CuteDSLKernelWrapper](cutedsl_kernel.py#L22)** to provide a `.run()` interface
5. The generated Python file is cached via PyCodeCache, but CUTLASS compilation runs every time (no kernel-level caching yet)

**Debug tip**: Use `TORCH_LOGS="kernel_code"` to see the generated kernel source and file path during compilation.

## Writing Templates

Templates use Jinja2 syntax with these available hooks:

- `{{kernel_name}}` - Unique kernel identifier
- `{{def_kernel(args...)}}` - Generates kernel function signature and argument handling
- `{{input_nodes}}` - List of input buffers
- `{{output_node}}` - Output buffer
- `{{gen_defines()}}` - Generates autotunable parameter definitions with proper CuteDSL typing

## Autotunable Parameters

CuteDSL templates support autotunable parameters similar to Triton's `tl.constexpr` system:

```python
template_source = r"""
{{gen_defines()}}

@cute.kernel
def {{kernel_name}}_kernel(gA: cute.Tensor, gB: cute.Tensor, gC: cute.Tensor):
    threads_per_block = THREADS_PER_BLOCK  # Uses autotuned value
    block_size = BLOCK_SIZE
    # ... kernel implementation
"""

# Pass parameters when generating template choices
template.maybe_append_choice(
    choices,
    input_nodes=[a, b],
    layout=layout,
    THREADS_PER_BLOCK=256,    # cutlass.Constexpr = 256
    BLOCK_SIZE=128,           # cutlass.Constexpr = 128
    SCALE_FACTOR=1.5,         # cutlass.Constexpr = 1.5
)
```

Templates must:
1. Define a `@cute.kernel` decorated function
2. Use `{{def_kernel()}}` to create the entry point
3. Return the output tensor
4. Use `{{gen_defines()}}` for autotunable parameters

See [test_cutedsl_template.py](../../../../test/inductor/test_cutedsl_template.py) for complete examples.

## Current Limitations / TODOs

- **No fusion support**: `can_fuse_vertical` and `can_fuse_horizontal` return False
- **Subgraph management**: Bodies and masks not fully implemented
- **File-based compilation**: Requires writing to disk (uses PyCodeCache)
- **Missing epilogue/prologue**: No support for fused operations yet
- **Fixed kernel suffix**: Uses hardcoded "_main" suffix
- **No CUTLASS kernel caching**: Only PyCodeCache works; CUTLASS compilation runs every time (major perf issue)


Note: Requires CUTLASS Python package (`pip install nvidia-cutlass`)