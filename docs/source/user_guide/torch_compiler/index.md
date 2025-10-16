# torch.compiler

This guide provides comprehensive documentation for `torch.compile`, PyTorch's compiler that accelerates model execution by converting PyTorch programs into optimized kernels. The compiler uses graph capture with TorchDynamo and code generation with TorchInductor to deliver significant performance improvements with minimal code changes.

**Getting Started** - Introduction to torch.compile and basic usage patterns.

**Core Concepts** - Understanding the programming model, graph breaks, compilation modes, and tracing behavior.

**torch.export Concepts** - Working with torch.export for ahead-of-time graph capture and serialization.

**Performance** - Profiling, benchmarking, and optimizing torch.compile performance.

**Advanced** - Deep dives into internals, custom backends, dynamic shapes, and transformations.

**Troubleshooting/FAQs** - Common issues, debugging techniques, and frequently asked questions.

**Reference/API** - Complete API documentation and configuration options.

```{toctree}
:maxdepth: 1


getting_started
```
```{toctree}
:maxdepth: 1


core_concepts
```

```{toctree}
:maxdepth: 1



export_concepts
```

```{toctree}
:maxdepth: 1


performance
```

```{toctree}
:maxdepth: 1


advanced
```

```{toctree}
:maxdepth: 1


troubleshooting
```

```{toctree}
:maxdepth: 1


api_reference
```
