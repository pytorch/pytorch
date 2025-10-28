# torch.compiler

This guide provides comprehensive documentation for `torch.compile`, PyTorch's compiler that accelerates model execution by converting PyTorch programs into optimized kernels. The compiler uses graph capture with TorchDynamo and code generation with TorchInductor to deliver significant performance improvements with minimal code changes.

**[Getting Started](getting_started.md)** - Introduction to torch.compile and basic usage patterns.

**[Core Concepts](core_concepts.md)** - Understanding the programming model, graph breaks, compilation modes, and tracing behavior.

**[torch.export Concepts](export_concepts.md)** - Working with torch.export for ahead-of-time graph capture and serialization.

**[Performance](performance.md)** - Profiling, benchmarking, and optimizing torch.compile performance.

**[Advanced](advanced.md)** - Deep dives into internals, custom backends, dynamic shapes, and transformations.

**[Debugging Guides](troubleshooting_faqs.md)** - Common issues, debugging techniques, and frequently asked questions.

**[Reference/API](api_reference.md)** - Complete API documentation and configuration options.

```{toctree}
:maxdepth: 1
:hidden:

torch.compiler_get_started.md
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
:hidden:


troubleshooting_faqs
```

```{toctree}
:maxdepth: 1
:hidden:

api_reference
```
