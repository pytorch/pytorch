# PyTorch Utility Scripts Guide

Collection of build scripts, development tools, and maintenance utilities for PyTorch development and deployment.

## 🏗️ Directory Organization

### Mobile Platform Builds
- **`build_android.sh`** - Android NDK cross-compilation script
- **`build_ios.sh`** - iOS cross-compilation script (macOS only)
- **`build_mobile.sh`** - General mobile platform build utilities
- **`build_pytorch_android.sh`** - PyTorch-specific Android build
- **`build_android_gradle.sh`** - Gradle-based Android build

### Platform-Specific Builds
- **`build_raspbian.sh`** - Raspberry Pi (ARM) build script
- **`build_tegra_x1.sh`** - NVIDIA Tegra X1 build script
- **`build_tizen.sh`** - Samsung Tizen OS build script
- **`build_windows.bat`** - Windows build script
- **`build_local.sh`** - Local development build script

### Development Tools
- **`get_python_cmake_flags.py`** - Generate CMake flags for Python integration
- **`diagnose_protobuf.py`** - Protocol Buffers diagnostic tool
- **`install_triton_wheel.sh`** - Install Triton GPU kernel library
- **`build_host_protoc.sh`** - Build Protocol Buffers compiler

### Code Quality and Maintenance
- **`add_apache_header.sh`** - Add Apache license headers to source files
- **`remove_apache_header.sh`** - Remove Apache license headers
- **`apache_header.txt`** - Apache license header template
- **`lint_urls.sh`** - Check for broken URLs in documentation
- **`lint_xrefs.sh`** - Validate cross-references in documentation

### Testing and Analysis (`analysis/`)
- **`run_test_csv.sh`** - Run test suites and generate CSV reports
- **`format_test_csv.py`** - Format test result CSV files
- **`compile_tests/`** - Compilation test utilities
  - `common.py` - Common test utilities
  - `download_reports.py` - Download CI test reports
  - `failures_histogram.py` - Analyze test failure patterns
  - `passrate.py` - Calculate test pass rates

### Release Management (`release/`)
- **`cut-release-branch.sh`** - Create release branches
- **`apply-release-changes.sh`** - Apply release-specific changes
- **`README.md`** - Release process documentation

### Release Notes (`release_notes/`)
- **`categorize.py`** - Categorize commits for release notes
- **`classifier.py`** - ML-based commit classification
- **`commitlist.py`** - Generate commit lists for releases
- **`apply_categories.py`** - Apply categories to commits
- **`explore.ipynb`** - Interactive release notes exploration

### Specialized Tools
- **`onnx/`** - ONNX integration scripts
  - `install.sh` - Install ONNX dependencies
  - `install-develop.sh` - Install ONNX development dependencies
  - `test.sh` - Run ONNX compatibility tests
- **`jit/`** - TorchScript utilities
  - `log_extract.py` - Extract logs from JIT compilation
- **`export/`** - Export utilities
  - `update_schema.py` - Update export schema definitions
- **`fbcode-dev-setup/`** - Meta/Facebook development setup
  - `ccache_setup.sh` - Configure ccache for faster builds

## 🚀 Common Usage

### Mobile Development
```bash
# Android build
export ANDROID_NDK=/path/to/ndk
bash scripts/build_android.sh

# iOS build (macOS only)
brew install cmake automake libtool
bash scripts/build_ios.sh

# Custom Android build with options
bash scripts/build_android.sh -DBUILD_BINARY=ON
```

### Development Setup
```bash
# Install Triton for GPU kernels
bash scripts/install_triton_wheel.sh

# Get Python CMake flags
python scripts/get_python_cmake_flags.py

# Diagnose Protocol Buffers issues
python scripts/diagnose_protobuf.py
```

### Code Maintenance
```bash
# Add Apache headers to new files
bash scripts/add_apache_header.sh

# Check for broken URLs
bash scripts/lint_urls.sh

# Validate cross-references
bash scripts/lint_xrefs.sh
```

### Testing and Analysis
```bash
# Run test analysis
cd scripts/analysis
bash run_test_csv.sh
python format_test_csv.py

# Compilation test analysis
cd scripts/compile_tests
python download_reports.py
python failures_histogram.py
```

## 🔧 Development Workflow

### Mobile Platform Development
1. Set up platform-specific SDK/NDK
2. Run appropriate build script
3. Copy generated libraries to your mobile project
4. Test on target device

### Release Process
1. Use `cut-release-branch.sh` to create release branch
2. Generate release notes with `release_notes/` tools
3. Apply release changes with `apply-release-changes.sh`
4. Validate builds across platforms

### Adding New Scripts
1. Place in appropriate subdirectory
2. Follow existing naming conventions
3. Add documentation to README.md
4. Test on target platforms

## 📁 Key Files

### Build Scripts
- `build_android.sh` - Primary Android build script
- `build_ios.sh` - Primary iOS build script
- `build_local.sh` - Local development builds

### Development Tools
- `get_python_cmake_flags.py` - CMake configuration helper
- `diagnose_protobuf.py` - Protocol Buffers debugging
- `install_triton_wheel.sh` - GPU kernel library setup

### Maintenance Tools
- `add_apache_header.sh` - License header management
- `lint_urls.sh` - URL validation
- `analysis/run_test_csv.sh` - Test analysis automation

## 🐛 Common Issues

### Mobile Build Issues
- **Android NDK**: Ensure `ANDROID_NDK` environment variable is set
- **iOS builds**: Requires macOS with Xcode and command line tools
- **Cross-compilation**: Check target architecture settings

### Platform Issues
- **Windows builds**: Use `build_windows.bat` instead of shell scripts
- **ARM builds**: May require specific toolchain configuration
- **Missing dependencies**: Install platform-specific requirements

### Development Issues
- **Python CMake flags**: Run `get_python_cmake_flags.py` for configuration
- **Protocol Buffers**: Use `diagnose_protobuf.py` for debugging
- **Triton installation**: Check CUDA compatibility

## 📝 Notes for Claude

This scripts directory provides:
- **Cross-platform builds**: Android, iOS, Windows, ARM platforms
- **Development automation**: CMake flag generation, dependency installation
- **Code quality tools**: License header management, URL validation
- **Release management**: Branch creation, changelog generation
- **Testing infrastructure**: Automated test analysis and reporting
- **Specialized tools**: ONNX integration, JIT utilities, export tools

Key script categories:
- Mobile deployment scripts for Android/iOS development
- Platform-specific builds for embedded systems
- Development workflow automation
- Release process management
- Code quality and maintenance tools
- Testing and analysis utilities

Usage patterns:
- Most scripts should be run from PyTorch root directory
- Mobile builds require platform-specific SDKs
- Release scripts follow semantic versioning
- Analysis tools generate CSV reports for CI integration