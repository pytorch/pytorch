# ATen/ Directory - C++ Tensor Library

ATen (A Tensor library) is PyTorch's C++ tensor computation library providing the core backend for all tensor operations.

## 🏗️ Directory Organization

### Core Components
- **`src/ATen/`** - Main ATen library with tensor operations and kernels
- **`src/ATen/core/`** - Core tensor types, dispatch system, and utilities
- **`src/ATen/native/`** - Native kernel implementations for CPU/CUDA/other backends
- **`src/ATen/templates/`** - Code generation templates used by torchgen
- **`src/README.md`** - Reference counting and low-level tensor library documentation

### Backend Implementations
- **`src/ATen/cpu/`** - CPU-specific implementations and vectorization
- **`src/ATen/cuda/`** - CUDA GPU implementations and utilities
- **`src/ATen/mps/`** - Apple Metal Performance Shaders backend
- **`src/ATen/hip/`** - AMD ROCm/HIP backend support
- **`src/ATen/vulkan/`** - Vulkan compute backend
- **`src/ATen/xpu/`** - Intel XPU backend support

### Key Subsystems
- **`src/ATen/functorch/`** - Functional transforms (vmap, grad) C++ implementation
- **`src/ATen/quantized/`** - Quantized tensor operations
- **`src/ATen/ops/`** - Operator interface definitions
- **`src/ATen/test/`** - Comprehensive C++ test suite

## 🔧 Key Files & Concepts

### Core Headers
- `src/ATen/ATen.h` - Main ATen header with all tensor operations
- `src/ATen/Tensor.h` - Tensor class definition
- `src/ATen/TensorIterator.h` - Efficient tensor iteration framework
- `src/ATen/core/Tensor.h` - Core tensor interface
- `src/ATen/Dispatch.h` - Type dispatch system for kernels
TODO(Claude): add something about per-op headers as you can see it used at the top of aten/src/ATen/native/TensorShape.cpp where the AT_PER_OPERATOR_HEADERS is used.

### Code Generation
- `src/ATen/native/native_functions.yaml` - Operator definitions for codegen
- `src/ATen/templates/` - Jinja2 templates for generating C++ code
- Generated files appear in build directory TODO(Claude): where should we put a claude.md that describes where to find these generated files in build/?

### Testing
```bash
# Test specific kernel implementations
python test/run_test.py -i test_ops
python test/run_test.py -i test_torch
```

## 🔄 Development Workflow

### After Modifying C++ Code or YAML File
```bash
# Rebuild PyTorch
eval $BUILD_CONFIG python setup.py develop
```

### Adding New Operators
1. Add definition to `src/ATen/native/native_functions.yaml`
2. Implement kernel in `src/ATen/native/` (CPU) and `src/ATen/native/cuda/` (CUDA)
3. Rebuild PyTorch
4. Add entry in OpInfo database in `torch/testing/_internal/common_method_invocation.py` TODO(Claude): check this file and add more details here

## 🐛 Common Issues

### Build Issues
- Just `python setup.py clean` and full rebuild

## 📝 Notes for Claude

- ATen implements the core tensor operations that torch/ Python API calls
- Heavy use of templates, macros, and code generation
- Multi-backend dispatch system routes operations to appropriate implementations
- A lot of documentation is historical and might be inacurate (see src/README.md for example that corresponds to old constructs that don't exists). Always double check facts in the actual code.
- Performance-critical code with extensive vectorization and GPU optimizations