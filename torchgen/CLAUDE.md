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

### Build Integration
```bash
# Code generation runs automatically during build
eval $BUILD_CONFIG python setup.py develop

# Force regeneration
python setup.py clean
eval $BUILD_CONFIG python setup.py develop
```

### Manual Generation
```bash
# Generate all code (typically done during build)
python -m torchgen.gen \
  --source-path aten/src/ATen \
  --install-dir build/aten/src/ATen

# Generate specific components
python -m torchgen.gen_backend_stubs
python -m torchgen.gen_lazy_tensor
```

## 🔄 Development Workflow

### Adding New Operators
1. **Define** operator in `packaged/ATen/native/native_functions.yaml`
2. **Implement** kernels in `aten/src/ATen/native/`
3. **Add derivatives** in `packaged/autograd/derivatives.yaml` (if differentiable)
4. **Regenerate** code using build system
5. **Test** new operator functionality

### Modifying Generation Logic
1. **Update** generation scripts in `api/` or core files
2. **Test** generation with sample operators
3. **Verify** generated code compiles and runs correctly
4. **Run** full test suite to ensure no regressions

### Template Changes
1. **Modify** templates in `packaged/ATen/templates/` or `packaged/autograd/templates/`
2. **Test** with representative operators
3. **Verify** generated code quality and correctness

## 🧪 Testing Code Generation

### Integration Tests
```bash
# Test generated code compiles and works
eval $BUILD_CONFIG python setup.py develop
python -c "import torch; torch.add(torch.tensor([1]), torch.tensor([2]))"
```

## 🔧 Advanced Features

### Structured Kernels (`api/structured.py`)
TODO(Claude): This paragraph is wrong. Also confusing wrt to structured_delegate option in `native_functions.yaml`
- Generates optimized implementations for operations with multiple variants
- Handles broadcasting, type promotion, and memory layout automatically
- Reduces code duplication across similar operations

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

## 🐛 Common Issues

### Generation Errors
- **YAML syntax errors**: Check `native_functions.yaml` for syntax issues
- **Template errors**: Verify Jinja2 template syntax and variable names
- **Missing derivatives**: Add entries to `derivatives.yaml` for differentiable ops

### Build Integration
- **Stale generated code**: Run `python setup.py clean` before rebuilding
- **Template path issues**: Ensure template files are in correct locations
- **Import errors**: Check Python path and module structure

### Performance
- **Slow generation**: Large operator sets can take time to generate
- **Memory usage**: Complex templates may use significant memory
- **Incremental builds**: May not always detect template changes

## 📝 Notes for Claude

- TorchGen is critical infrastructure - changes here affect entire PyTorch
- Heavy use of metaprogramming and template generation
- Operator definitions in YAML are the source of truth for PyTorch operations
- Generated code must be efficient - this is performance-critical path
- Complex interaction between multiple generation phases and backends
- Testing requires both unit tests and integration with full build system