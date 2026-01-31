---
name: debugging-torchcompile
description: Debug torch.compile issues including Dynamo, Inductor, AOT Inductor, and Export. Use when investigating compilation errors, graph breaks, recompilations, guard failures, Triton codegen issues, dynamic shape constraints, export failures, or CUDA IMA errors. Also use when the user asks about TORCH_LOGS, TORCH_COMPILE_DEBUG, torch._dynamo.config, torch._inductor.config, aoti_compile_and_package, or why their model is slow after compilation.
---

# Debugging torch.compile

This skill provides guidance for debugging the torch.compile stack, which includes:

- **Dynamo**: The Python bytecode tracer that captures PyTorch operations
- **Inductor**: The default backend that generates optimized Triton/C++ code
- **AOT Inductor**: Ahead-of-time compilation for deployment in non-Python environments
- **Export**: The system for exporting models to portable graph representations

## When to Use Each Guide

| Symptom | Guide |
|---------|-------|
| Graph breaks, guards, recompilations | [dynamo.md](dynamo.md) |
| Slow generated code, fusion issues, codegen errors | [inductor.md](inductor.md) |
| AOT compilation, CUDA IMA, aoti_compile_and_package | [aot_inductor.md](aot_inductor.md) |
| Export failures, dynamic shape issues, constraints | [export.md](export.md) |
| Environment setup, logging, config options | [common.md](common.md) |
| Small tips that don't fit elsewhere | [misc.md](misc.md) |

## Quick Start

### Enable verbose logging
```python
import torch._dynamo
torch._dynamo.config.verbose = True
```

### Get compilation logs
```bash
TORCH_LOGS="+dynamo,+inductor" python script.py
```

### Debug a specific issue
```python
import torch._dynamo
torch._dynamo.config.suppress_errors = False  # Don't silently fall back to eager
```

## Common Debugging Workflow

1. **Identify the component**: Is it a tracing issue (Dynamo), codegen issue (Inductor), or export issue?
2. **Enable relevant logging**: Use TORCH_LOGS or config options
3. **Reproduce minimally**: Use `torch._dynamo.explain()` or create a minimal repro
4. **Check known issues**: Graph breaks, unsupported ops, shape constraints

See the component-specific guides for detailed debugging steps.
