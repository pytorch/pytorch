# CMake Build System Guide

CMake configuration files and modules for PyTorch's complex build system with cross-platform support and dependency management.

## 🏗️ Directory Organization

### Core Configuration
- **`Dependencies.cmake`** - Main dependency detection and configuration
- **`BuildVariables.cmake`** - Build variable definitions and defaults
- **`Summary.cmake`** - Build configuration summary generation
- **`MiscCheck.cmake`** - Miscellaneous build checks and validations

### Platform-Specific
- **`iOS.cmake`** - iOS build configuration and cross-compilation
- **`Metal.cmake`** - Apple Metal GPU backend configuration
- **`VulkanCodegen.cmake`** - Vulkan API code generation
- **`VulkanDependencies.cmake`** - Vulkan dependencies and validation

### Library Integration
- **`BLAS_ABI.cmake`** - BLAS library ABI configuration
- **`FlatBuffers.cmake`** - FlatBuffers serialization library setup
- **`ProtoBuf.cmake`** - Protocol Buffers configuration
- **`Codegen.cmake`** - Code generation utilities

### Find Modules (`Modules/`)
Custom CMake modules for finding libraries:
- **`FindMKL.cmake`** - Intel Math Kernel Library detection
- **`FindCUDAToolkit.cmake`** - CUDA toolkit detection
- **`FindOpenMP.cmake`** - OpenMP parallel programming support
- **`FindMAGMA.cmake`** - MAGMA GPU linear algebra library
- **`FindNCCL.cmake`** - NVIDIA Collective Communications Library

### Public API (`public/`)
Reusable CMake utilities for external projects:
- **`cuda.cmake`** - CUDA configuration utilities
- **`mkl.cmake`** - MKL integration helpers
- **`utils.cmake`** - General CMake utility functions
- **`protobuf.cmake`** - Protocol Buffers integration

### External Dependencies (`External/`)
- **`nnpack.cmake`** - NNPACK neural network acceleration
- **`nccl.cmake`** - NCCL communication library
- **`aotriton.cmake`** - AOTTriton GPU kernel compilation

### Configuration Templates
- **`TorchConfig.cmake.in`** - PyTorch package configuration template
- **`Caffe2Config.cmake.in`** - Caffe2 legacy configuration template
- **`cmake_uninstall.cmake.in`** - Uninstall script template

## 🔧 Key Build Features

### Dependency Detection
```cmake
# Major dependencies handled
- CUDA/ROCm for GPU acceleration
- MKL/OpenBLAS/BLIS for linear algebra
- OpenMP for parallelization
- Protocol Buffers for serialization
- NCCL for distributed training
```

### Cross-Platform Support
- Linux (x86_64, ARM64, PowerPC)
- macOS (Intel, Apple Silicon)
- Windows (MSVC, MinGW)
- iOS cross-compilation
- Android NDK support

### GPU Backend Configuration
```cmake
# CUDA configuration
find_package(CUDAToolkit REQUIRED)
# ROCm/HIP configuration
find_package(HIP REQUIRED)
# Metal configuration (macOS)
find_package(Metal REQUIRED)
```

## 🧪 Build Testing

### Configuration Testing
```bash
# Test CMake configuration
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Debug
cmake --build . --target help
```

### Dependency Verification
```bash
# Check found dependencies
cmake .. -DCMAKE_FIND_DEBUG_MODE=ON
# Dry run to see what would be built
cmake .. --dry-run
```

## 🔧 Development Workflow

### Adding New Dependencies
1. Check with Core Maintainers that we're ok adding a new dependency via github issue
2. Create `FindYourLib.cmake` in `Modules/`
3. Add detection logic in `Dependencies.cmake`
4. Update `BuildVariables.cmake` with new options
5. Add summary in `Summary.cmake`

### Modifying Build Options
```bash
# Common build variables
-DUSE_CUDA=ON/OFF
-DUSE_DISTRIBUTED=ON/OFF
-DUSE_MKL=ON/OFF
-DBUILD_SHARED_LIBS=ON/OFF
-DCMAKE_BUILD_TYPE=Debug/Release
```

### Platform-Specific Builds
```bash
# Custom CUDA architecture
cmake .. -DTORCH_CUDA_ARCH_LIST="7.0;8.0;8.6"
```

## 📁 Key Files

### Build Configuration
- `CMakeLists.txt` (root) - Main build configuration
- `Dependencies.cmake` - Dependency detection logic
- `BuildVariables.cmake` - Build option definitions

### Find Modules
- `Modules/FindMKL.cmake` - Intel MKL detection
- `Modules/FindCUDAToolkit.cmake` - CUDA toolkit discovery
- `Modules/FindOpenMP.cmake` - Modified OpenMP finder with Apple Clang fixes

### Platform Files
- `iOS.cmake` - iOS cross-compilation toolchain
- `Metal.cmake` - Apple Metal GPU backend
- `public/cuda.cmake` - CUDA utilities for external use

## 🐛 Common Issues

### Dependency Detection
- **Missing libraries**: Check `CMAKE_PREFIX_PATH` and library paths
- **Version conflicts**: Ensure compatible versions across dependencies
- **CUDA issues**: Verify CUDA toolkit installation and paths

### Cross-Compilation
- **iOS builds**: Ensure Xcode command line tools are installed
- **Architecture mismatches**: Set correct target architectures
- **Toolchain errors**: Verify cross-compilation toolchain setup

### OpenMP Issues
- **Apple Clang**: Uses custom OpenMP detection with brew compatibility
- **Multiple OpenMP**: Avoid linking multiple OpenMP runtimes (MKL vs system)
- **Link errors**: Check OpenMP library paths and flags

## 📝 Notes for Claude

This CMake build system handles:
- **Complex dependencies**: 50+ external libraries with version constraints
- **Multi-platform**: Linux, macOS, Windows, iOS, Android support
- **GPU backends**: CUDA, ROCm, Metal, Vulkan configuration
- **Optimization**: MKL, OpenBLAS, cuBLAS integration
- **Distributed**: NCCL, RCCL, UCC communication libraries
- **Mobile**: iOS and Android cross-compilation support

Key challenges addressed:
- OpenMP runtime conflicts between MKL and system libraries
- CUDA compute capability detection and compilation
- Apple Silicon vs Intel Mac compatibility
- Static vs shared library linking complexities
- Package config generation for downstream projects

The system uses CMake's modern target-based approach with proper dependency propagation and supports both in-tree and external builds.