# torch.compiler

This guide provides comprehensive documentation for `torch.compile`, PyTorch's compiler that accelerates model execution by converting PyTorch programs into optimized kernels. The compiler uses graph capture with TorchDynamo and code generation with TorchInductor to deliver significant performance improvements with minimal code changes.

**[Getting Started](getting_started)** - Introduction to torch.compile and basic usage patterns.

**[Core Concepts](core_concepts)** - Understanding the programming model, graph breaks, compilation modes, and tracing behavior.

**[torch.export Concepts](export_concepts)** - Working with torch.export for ahead-of-time graph capture and serialization.

**[Performance](performance)** - Profiling, benchmarking, and optimizing torch.compile performance.

**[Advanced](advanced)** - Deep dives into internals, custom backends, dynamic shapes, and transformations.

**[Troubleshooting](troubleshooting)** - Common issues, debugging techniques, and frequently asked questions.

**[Reference/API](api_reference)** - Complete API documentation and configuration options.

```{toctree}
:maxdepth: 1
:hidden:

getting_started
```
```{toctree}
:maxdepth: 1
:hidden:

core_concepts
```

```{toctree}
:maxdepth: 1
:hidden:


export_concepts
```

```{toctree}
:maxdepth: 1
:hidden:

performance
```

```{toctree}
:maxdepth: 1
:hidden:

advanced
```

```{toctree}
:maxdepth: 1


../../compile/programming_model.observability.md
../../compile/programming_model.reporting_issues.md
../../torch.compiler_troubleshooting.md
../../torch.compiler_faq.md
../../torch.compiler_aot_inductor_debugging_guide.md
../../torch.compiler_aot_inductor_minifier.md
```

```{toctree}
:maxdepth: 1
:hidden:

../../torch.compiler_api.md
../../torch.compiler.config.md
../../torch.compiler_fine_grain_apis.md
../../export/api_reference.md
```
