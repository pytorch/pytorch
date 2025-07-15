# C10/ Directory - Core C++ Utilities

C10, historically build to support both Caffe2 (C2) and ATen (A10), contains the core C++ utilities, types, and device abstraction layer used throughout PyTorch.

## 🏗️ Directory Organization

### Core Components
- **`core/`** - Fundamental types and interfaces (Tensor, Device, Storage, etc.)
- **`util/`** - General C++ utilities (smart pointers, containers, math, etc.)
- **`macros/`** - Preprocessor macros and build configuration
- **`test/`** - Comprehensive unit tests for C10 components

### Device Backends
- **`cuda/`** - CUDA-specific implementations and utilities
- **`hip/`** - AMD ROCm/HIP backend support
- **`xpu/`** - Intel XPU backend support
- **`mobile/`** - Mobile-optimized allocators and utilities
- **`metal/`** - Apple Metal backend utilities

## 🔧 Key Components

### Core Types (`core/`)
- **`TensorImpl.h`** - Base tensor implementation
- **`Device.h`** - Device abstraction (CPU, CUDA, etc.)
- **`Storage.h`** - Memory storage abstraction
- **`Scalar.h`** - Scalar value types
- **`ScalarType.h`** - Tensor data type definitions
- **`DispatchKey.h`** - Dispatch system for backends
- **`Allocator.h`** - Memory allocation interface

### Utilities (`util/`)
- **`intrusive_ptr.h`** - Reference-counted smart pointers
- **`Optional.h`** - Optional type implementation
- **`ArrayRef.h`** - Non-owning array references
- **`SmallVector.h`** - Stack-optimized vector
- **`Half.h`** - Half-precision floating point
- **`BFloat16.h`** - Brain floating point format
- **`Exception.h`** - Error handling utilities

### Device Abstraction
- **`DeviceGuard.h`** - RAII device context management
- **`Stream.h`** - Asynchronous execution streams
- **`Event.h`** - Synchronization primitives

## 🧪 Testing

### Running C10 Tests
TODO(Claude): There are no real tests that matter here, things need to be tested from python at the top level

## 🔄 Development Workflow

### After Modifying C10 Code
```bash
# C10 is a foundational library, full rebuild recommended
eval $BUILD_CONFIG python setup.py develop
```

## 🐛 Common Issues

### Build Issues
- **Header include order**: C10 headers must be included before system headers

## 📝 Notes for Claude

- C10 is the foundation layer that everything else builds on
- Heavy use of CRTP (Curiously Recurring Template Pattern) and modern C++
- Device-agnostic design with backend-specific implementations
- Critical for performance - changes here affect entire PyTorch
- Extensive use of templates for type safety and performance
