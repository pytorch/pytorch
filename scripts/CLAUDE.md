# PyTorch Utility Scripts Guide

Collection of build scripts, development tools, and maintenance utilities for PyTorch development and deployment.

## 🏗️ Directory Organization

### Testing and Analysis (`analysis/`)
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

## 📝 Notes for Claude

This scripts directory provides:
- **Release management**: Branch creation, changelog generation
- **Testing infrastructure**: Automated test analysis and reporting
- **One off scripts**: Should not be used