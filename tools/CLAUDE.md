# Tools/ Directory - Build Tools and Development Utilities

Collection of build scripts, code generation tools, and development utilities used throughout PyTorch development and CI.

## 🏗️ Directory Organization

### Code Generation
- **`autograd/`** - Autograd code generation and derivative definitions
- **`jit/`** - TorchScript code generation utilities TODO(Claude): can you verify this
- **`pyi/`** - Python stub file generation for type hints

### Build System
- **`setup_helpers/`** - Build configuration and dependency detection to support setup.py
- **`build_pytorch_libs.py`** - Core C++ library build script
- **`build_libtorch.py`** - Standalone C++ library build
- **`cmake.py`** - CMake integration utilities
- **`packaging/`** - Wheel building and distribution

### Development Tools
- **`linter/`** - Code linting and formatting tools for lintrunner
- **`testing/`** - Test selection and CI integration
- **`stats/`** - Performance monitoring and statistics collection
- **`code_analyzer/`** - Static code analysis tools

### Platform Support
- **`amd_build/`** - AMD ROCm/HIP transpilation from CUDA

## 🔧 Key Components

### Code Generation (`autograd/`)
- **`derivatives.yaml`** - Mathematical derivative definitions for operators
- **`gen_autograd.py`** - Main autograd code generator
- **`gen_python_functions.py`** - Python API bindings generation
- **`templates/`** - Jinja2 templates for generated C++ code

### Build Scripts
- **`build_pytorch_libs.py`** - Builds ATen, c10, and other core libraries
- **`build_libtorch.py`** - Builds standalone C++ library
- **`generate_code.py`** - Runs all code generation phases

### Linting Tools (`linter/`)
- **`adapters/`** - Integration with various linters (black, flake8, mypy, etc.)
- **`clang_tidy/`** - C++ static analysis integration
- Supports: Python formatting, C++ formatting, documentation checks

### Testing Infrastructure (`testing/`)
- **`discover_tests.py`** - Test discovery and organization
- **`target_determination/`** - Intelligent test selection based on changes
- **`test_selections.py`** - CI test configuration

## 🚀 Common Development Tasks

### Running Code Generation
```bash
# Regenerate all generated code (done automatically during build)
python tools/setup_helpers/generate_code.py

# Regenerate only autograd code
python tools/autograd/gen_autograd.py

# Regenerate Python stubs
python tools/pyi/gen_pyi.py
```

### Build System Usage
```bash
# Build core libraries only
python tools/build_pytorch_libs.py

# Build libtorch (C++ library)
python tools/build_libtorch.py

# Build with specific configuration
python tools/build_pytorch_libs.py --cmake-only
```

### Linting and Testing
```bash
# Install lintrunner (as mentioned in main CLAUDE.md)
pip install lintrunner
lintrunner init

# Run linters on changes
lintrunner -m main

# Discover available tests
python tools/testing/discover_tests.py
```

## 🔄 Development Workflow

### After Modifying Code Generation
```bash
# Code generation is typically run automatically during build
eval $BUILD_CONFIG python setup.py develop

# For manual regeneration
python tools/setup_helpers/generate_code.py
```

### Adding New Operators
1. Add derivatives to `tools/autograd/derivatives.yaml`
2. Run code generation: `python tools/autograd/gen_autograd.py`
3. Regenerate Python bindings if needed

### Build System Changes
1. Modify relevant files in `setup_helpers/` or build scripts
2. Test with clean build: `python setup.py clean && eval $BUILD_CONFIG python setup.py develop`
3. Test on multiple platforms if possible

## 🧪 Testing and Validation

### Code Generation Testing
```bash
# Test autograd generation
python -m pytest tools/test/test_codegen.py

# Test operator list generation
python -m pytest tools/test/test_gen_oplist_test.py
```

### Build Testing
```bash
# Test build helpers
python -m pytest tools/test/test_cmake.py

# Test selective build functionality
python -m pytest tools/test/test_selective_build.py
```

## 🔧 Advanced Features

### AMD GPU Support (`amd_build/`)
- HIPify transpiler converts CUDA code to AMD HIP
- Automatic conversion of CUDA kernels and API calls
- Build system integration for ROCm support

### Performance Monitoring (`stats/`)
- CI performance tracking and regression detection
- Test execution time monitoring
- Build system performance metrics

### Target Determination (`testing/target_determination/`)
- Intelligent test selection based on code changes
- Historical failure correlation analysis
- Reduces CI time by running only relevant tests

## 🐛 Common Issues

### Code Generation
- **Template errors**: Check Jinja2 template syntax in `templates/`
- **Missing derivatives**: Add to `derivatives.yaml` for new operators
- **Build failures**: Ensure generated code compiles

### Build System
- **Dependency detection**: Check `setup_helpers/` for platform-specific issues
- **CMake errors**: Review CMake configuration and dependencies
- **Linking errors**: May require clean rebuild

### Linting
- **Format conflicts**: Different linters may have conflicting requirements
- **Performance**: Linting large codebases can be slow
- **False positives**: Some checks may need configuration adjustments

## 📝 Notes for Claude

- Tools directory serves dual purpose: build infrastructure and Python module
- Heavy use of code generation to maintain consistency across large codebase
- Build system is complex due to multi-platform and multi-backend support
- Linting infrastructure integrates many external tools
- Test selection system reduces CI load through intelligent targeting
- Code generation templates use Jinja2 for flexibility and maintainability