# PyTorch Development Guide

This is the main PyTorch repository - a Python package providing tensor computation with GPU acceleration and deep neural networks built on a tape-based autograd system.

## 🏗️ Project Overview

PyTorch is organized into several key components:
- **`torch/`** - Main Python API, frontend and C++/Python binding code
- **`aten/`** - ATen tensor library (C++ backend)
- **`c10/`** - Core C++ utilities, types, and device abstraction
- **`torchgen/`** - Code generation tools for operators used during build time
- **`test/`** - Comprehensive test suite
- **`tools/`** - Build tools and development utilities
- **`functorch/`** - Functional transforms (vmap, grad, etc.), this was moved into torch/
- **`benchmarks/`** - Performance benchmarking tools

## 🚀 Quick Development Setup

### Build PyTorch from Source
```bash
# Default fast CPU build configuration
BUILD_CONFIG="CFLAGS='-DPYBIND11_DETAILED_ERROR_MESSAGES -DHAS_TORCH_SHOW_DISPATCH_TRACE' USE_DISTRIBUTED=0 USE_FLASH_ATTENTION=0 USE_MEM_EFF_ATTENTION=0 USE_MKLDNN=0 USE_CUDA=0 BUILD_TEST=0 USE_FBGEMM=0 USE_NNPACK=0 USE_QNNPACK=0 USE_XNNPACK=0 USE_COLORIZE_OUTPUT=0"

# Install dependencies
pip install -r requirements.txt

# Build and install in development mode (editable install)
eval $BUILD_CONFIG python setup.py develop

# Clean build artifacts
python setup.py clean
```

### Other Environment Variables for Building
```bash
# Debug build with symbols, very expensive, should almost never be used
export DEBUG=1

# Release with debug info
export REL_WITH_DEB_INFO=1

# Enable specific features
export USE_CUDA=1         # Enable CUDA support
export USE_DISTRIBUTED=1  # Enable distributed training
export USE_CUSTOM_DEBINFO='/path/to/file;path/to/other/file' # Build debug symbold only for these files
```

## 🧪 Testing

### Python Tests
You will generally never run all tests, only the relevant files and use github CI for broader testing. Part of development work is figuring out which tests are relevant for your changes and running just those.

```bash
# Run all tests (rarely used)
python test/run_test.py

# Run core test
python test/run_test.py --core

# Run specific tests in test/test_torch.py
python test/run_test.py -i test_torch
```

### C++ Tests
Located in `test/cpp/` - built as part of the main build process.

### Linting and Type Checking
```bash
# Install linter tool
pip install lintrunner

# Initialize linter
lintrunner init

# Run lint for all changes compared to main
lintrunner -m main
```

## 🔧 Common Development Tasks

### Performance Tips
Use `ccache` to speed up C++ compilation - install with your package manager and it will be automatically detected.

### After Modifying Python Files
No rebuild needed with `-e` install - changes are immediately available.

### After Modifying C++/CUDA Files
```bash
# Reinstall to rebuild C++ extensions
eval $BUILD_CONFIG python setup.py develop
```

### Debugging
```bash
# Enable dispatch tracing at runtime (already enabled in build config)
TORCH_SHOW_DISPATCH_TRACE=1 python your_script.py

# Enable C++ stacktraces
TORCH_SHOW_CPP_STACKTRACES=1 python your_script.py
```

## 📁 Key Files & Directories

### Build & Configuration
- `setup.py` - Main build script with extensive configuration options
- `CMakeLists.txt` - CMake build configuration
- `pyproject.toml` - Python packaging configuration
- `requirements.txt` - Python dependencies

### Documentation
- `README.md` - Project overview and installation
- `CONTRIBUTING.md` - Development guidelines and technical details
- `docs/` - Sphinx documentation source

## 🐛 Troubleshooting

### Build Issues
- **Incremental builds not working**: Run `python setup.py clean`
- **CUDA issues**: Set `USE_CUDA=0` for CPU-only builds (already in BUILD_CONFIG)
- **Memory issues during build**: Reduce `MAX_JOBS`

### Runtime Issues
- **Import errors**: Check if you need to reinstall after C++ changes
- **Weird behavior**: Try `python setup.py clean` and rebuild

### Testing Issues
- **Test failures**: Run individual test files to isolate issues

## 🔗 Useful Links

- [Contributing Guide](https://github.com/pytorch/pytorch/wiki/The-Ultimate-Guide-to-PyTorch-Contributions)
- [CI Health Dashboard](https://hud.pytorch.org/ci/pytorch/pytorch/main)
- [Developer Documentation](https://github.com/pytorch/pytorch/wiki)

## 📝 Notes for Claude

This repository uses:
- **Build system**: Python setuptools + CMake + ninja
- **Testing**: custom test runner (with pytest and/or unittest)
- **Code generation**: Custom tools in `torchgen/` and `tools/`
- **CI**: GitHub Actions with extensive matrix testing
- **Documentation**: Sphinx with custom extensions
