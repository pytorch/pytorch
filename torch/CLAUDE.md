# Torch/ Directory - Main Python API

This is the main PyTorch Python package containing the primary user-facing API, tensor operations, and Python/C++ bindings.

## 🏗️ Directory Organization

### Core Components
- **`__init__.py`** - Main package initialization, tensor creation, and core functions
- **`functional.py`** - Functional interface for operations (F.relu, F.conv2d, etc.)
- **`_tensor.py`** - Core Tensor class implementation
- **`_ops.py`** - Operator definitions and dispatch
- **`serialization.py`** - Model saving/loading (torch.save, torch.load)

### Key Subsystems
- **`nn/`** - Neural network modules, layers, and functional operations
- **`optim/`** - Optimizers (SGD, Adam, AdamW, etc.)
- **`autograd/`** - Automatic differentiation system
- **`cuda/`** - CUDA GPU support and utilities
- **`distributed/`** - Multi-process/multi-GPU training support
- **`jit/`** - TorchScript compilation and optimization
- **`fx/`** - Graph capture and transformation framework

### Advanced Features
- **`_dynamo/`** - PyTorch 2.0 compilation system
- **`_inductor/`** - Compiler backend for optimized kernels
- **`_functorch/`** - Functional transforms (vmap, grad, jacrev, etc.)
- **`export/`** - Model export functionality
- **`_lazy/`** - Lazy tensor evaluation
- **`sparse/`** - Sparse tensor operations

### Utilities & Tools
- **`testing/`** - Testing utilities (torch.testing.assert_close, etc.)
- **`utils/`** - General utilities (checkpoint, hooks, cpp_extension)
- **`profiler/`** - Performance profiling tools
- **`hub.py`** - Model zoo and pre-trained models

## 🔧 Common Development Tasks

### After Modifying Python Files in torch/
```bash
# No rebuild needed - changes available immediately with development install
python -c "import torch; print(torch.__version__)"
```

### After Modifying C++ Extensions (csrc/)
```bash
# Rebuild C++ extensions
eval $BUILD_CONFIG python setup.py develop
```

### Testing Changes
```bash
# Test specific torch functionality
python test/run_test.py -i test_torch
python test/run_test.py -i test_nn
python test/run_test.py -i test_autograd

# Test CUDA functionality
python test/run_test.py -i test_cuda
```

## 📁 Key Implementation Files

### Tensor Operations
- `torch/_tensor.py` - Core Tensor class
- `torch/functional.py` - Functional operations
- `torch/_ops.py` - Operator dispatch system

### Neural Networks
- `torch/nn/__init__.py` - NN module exports
- `torch/nn/functional.py` - Functional NN operations
- `torch/nn/modules/` - Layer implementations

### Autograd System
- `torch/autograd/__init__.py` - Autograd public API
- `torch/autograd/function.py` - Custom function base class
- `torch/csrc/autograd/` - C++ autograd implementation

### CUDA Support
- `torch/cuda/__init__.py` - CUDA device management
- `torch/cuda/memory.py` - GPU memory management
- `torch/csrc/cuda/` - C++ CUDA bindings

## 🐛 Common Issues

### Import Errors
- Check if C++ extensions need rebuilding after updates
- Verify CUDA availability: `torch.cuda.is_available()`

### Memory Issues
- Use `torch.cuda.empty_cache()` to clear GPU memory
- Check memory usage: `torch.cuda.memory_summary()`

### Performance Issues
- Profile with: `torch.profiler.profile()`
- Use TorchScript: `torch.jit.script()` or `torch.jit.trace()`

## 📝 Notes for Claude

- Most user-facing PyTorch functionality lives in this directory
- C++ bindings in `csrc/` require rebuilds, Python files don't
- Testing focuses on core tensor ops, NN modules, and autograd
- Heavy use of dynamic dispatch and operator overloading