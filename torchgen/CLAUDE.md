# TorchGen/ Directory - Code Generation Engine

TorchGen is PyTorch's code generation system that automatically generates C++ implementations, Python bindings, and dispatch logic from high-level operator definitions.
This is a standalone subset of the tools codegen that is shipped with the PyTorch package so it can be used out of tree more easily.

## 🏗️ Directory Organization

### Core Generation Engine
- **`gen.py`** - Main code generation entry point and orchestration
- **`model.py`** - Data models for operators, arguments, and schemas
- **`context.py`** - Generation context and utilities
- **`utils.py`** - Common utilities for code generation

### API Generation (`api/`)
- **`cpp.py`** - C++ kernel interface generation
- **`python.py`** - Python binding generation
- **`dispatcher.py`** - Dispatch system integration
- **`native.py`** - Native function signature handling
- **`autograd.py`** - Automatic differentiation integration
- **`structured.py`** - Structured kernel generation

### Target Backends (`dest/`)
- **`native_functions.py`** - Native function implementations
- **`register_dispatch_key.py`** - Dispatch key registration
- **`lazy_ir.py`** - Lazy tensor IR generation
- **`ufunc.py`** - Universal function generation

### Packaged Templates (`packaged/`)
- **`ATen/native/native_functions.yaml`** - Operator definitions
- **`ATen/templates/`** - C++ code generation templates
- **`autograd/`** - Autograd-specific templates and derivatives

### Specialized Generators
- **`aoti/`** - Ahead-of-Time Inductor code generation
- **`decompositions/`** - Operation decomposition generation
- **`fuse/`** - Fusion pattern generation
- **`static_runtime/`** - Static runtime optimization
- **`_autoheuristic/`** - Machine learning-based heuristic generation

## 🔧 Key Components

### Operator Definition (`packaged/ATen/native/native_functions.yaml`)
```yaml
- func: add.Tensor(Tensor self, Tensor other, *, Scalar alpha=1) -> Tensor
  device_check: NoCheck
  device_guard: False
  dispatch:
    CPU: add_cpu
    CUDA: add_cuda
  variants: function, method
```

### Code Generation Process
1. **Parse** operator definitions from YAML
2. **Model** operators using data classes in `model.py`
3. **Generate** C++ implementations using templates
4. **Dispatch** routing to appropriate backend implementations
5. **Bind** Python APIs to C++ functions

### Template System
- Uses Jinja2 templates for code generation
- Templates in `packaged/ATen/templates/` and `packaged/autograd/templates/`
- Supports conditional generation based on operator properties

## 🚀 Running Code Generation

Code Generation runs as part of the normal build.

## 🧪 Testing Code Generation

### Integration Tests
```bash
# Test generated code compiles and works
eval $BUILD_CONFIG python setup.py develop
python -c "import torch; torch.add(torch.tensor([1]), torch.tensor([2]))"
```

## 🔧 Advanced Features

### Structured Calling Convention (`api/structured.py`)
- Translates JIT schema to structured functions API (fixes historical native API problems)
- Uses precomputed parameters and argument replacement for optimized implementations

### Auto-Heuristics (`_autoheuristic/`)
- Machine learning-based performance optimization
- Generates heuristics for operator selection and scheduling
- Supports different backends (CUDA A100, H100, etc.)

### Lazy Tensor Generation (`dest/lazy_ir.py`)
- Generates IR for lazy evaluation backends
- Supports compilation and optimization of tensor graphs
- Used by XLA and other graph-based backends

### Functionalization (`gen_functionalization_type.py`)
- Converts in-place operations to out-of-place equivalents
- Supports functional programming paradigms
- Enables optimizations and graph transformations

## 📝 Notes for Claude

- TorchGen is critical infrastructure - changes here affect entire PyTorch
- Heavy use of metaprogramming and template generation
- Operator definitions in YAML are the source of truth for PyTorch operations
- Generated code must be efficient - this is performance-critical path
- Complex interaction between multiple generation phases and backends
- Testing requires both unit tests and integration with full build system