# PyTorch Copilot Instructions

This is the PyTorch machine learning framework codebase. These instructions help AI agents navigate and contribute effectively.

## Architecture Overview

### Core Components

- **c10/** - Core library (C++-10 compatible) for essential, binary-size-conscious functionality
- **aten/** - ATen tensor library (C++), PyTorch's foundation without autograd
  - `aten/src/ATen/native/` - Modern operator implementations (CPU/CUDA/MPS/sparse)
  - `aten/src/ATen/native/native_functions.yaml` - **Critical**: Declarative operator registry
- **torch/** - Python bindings and public API
  - `torch/csrc/` - C++ Python bindings (hand-written and generated)
  - `torch/csrc/autograd/` - Reverse-mode automatic differentiation
  - `torch/csrc/jit/` - TorchScript JIT compiler
- **torchgen/** - Code generation tooling that reads `native_functions.yaml`
- **tools/** - Build scripts, autograd derivatives, code generation

### The Code Generation Workflow

**Most operator changes require editing `native_functions.yaml`**, not direct C++ files. This YAML file:
1. Declares operator signatures, variants (function/method), and dispatch behavior
2. Gets processed by `torchgen/` to generate C++/Python bindings
3. Produces headers in `build/aten/src/ATen/` during compilation

Example entry structure:
```yaml
- func: my_op(Tensor self, Scalar alpha=1) -> Tensor
  variants: function, method
  dispatch:
    CPU: my_op_cpu
    CUDA: my_op_cuda
```

After editing `native_functions.yaml`, implement kernels in `aten/src/ATen/native/` (see `aten/src/ATen/native/README.md`).

## Development Workflows

### Building from Source

**Never run `setup.py` directly** - use pip with editable install:
```bash
python -m pip install --no-build-isolation -v -e .
```

Speed up builds:
- `DEBUG=1` - Debug symbols with `-g -O0`
- `USE_CUDA=0` - Skip CUDA compilation
- `BUILD_TEST=0` - Skip C++ test binaries
- Install `ninja` (`pip install ninja`) for faster builds
- Use `ccache` for incremental compilation caching

Rebuild specific targets: `(cd build && ninja <target>)`

### Testing

**Critical**: DO NOT run entire test suites. Run specific tests only:
```bash
python test/test_torch.py TestTorch.test_specific_case
```

**Test structure**: All tests use `torch.testing._internal.common_utils`:
```python
from torch.testing._internal.common_utils import run_tests, TestCase

class TestFeature(TestCase):
    def test_something(self):
        # Use self.assertEqual for tensor comparisons
        pass

if __name__ == "__main__":
    run_tests()
```

**For bug fixes**: Create a standalone reproduction script first, verify it fails, then fix and add to appropriate test file.

### Linting

Run linter (not pre-commit): `lintrunner -a` (auto-applies fixes)

## Project-Specific Conventions

### Memory and Storage
- **Storage is never nullptr** (but `StorageImpl.data` may be nullptr for unallocated outputs)
- CUDA device info lives in storage objects

### Python-C++ Integration (`torch/csrc/`)
- Always include `Python.h` **first** to avoid `_XOPEN_SOURCE` redefinition errors
- Use `pybind11::gil_scoped_acquire` before calling Python API or using `THPObjectPtr`
- Wrap entry points with `HANDLE_TH_ERRORS` / `END_HANDLE_TH_ERRORS` for exception conversion

### Dispatch System
- PyTorch uses operator dispatch to route calls to backend-specific kernels
- Prefer `CompositeExplicitAutograd` dispatch when writing device-agnostic compound ops
- See `aten/src/ATen/native/README.md` for dispatch keyword guidance

## Git Workflow (AI Agent Specific)

When preparing PRs from this environment:
```bash
git stash -u
git reset --hard $(cat /tmp/orig_work.txt)  # Reset to LOCAL branch
git stash pop
# Resolve conflicts if necessary
```

## Common Gotchas

1. **Editing generated files** - If it's in `build/`, don't edit it. Edit the source template or `native_functions.yaml`
2. **NVCC template compilation** - NVCC is stricter about C++ than gcc/clang; code working on Linux may fail Windows CI
3. **Windows symbol visibility** - Use `TORCH_API` macros for exported symbols (required on Windows, optional on Linux)
4. **No internet access** - DO NOT attempt to install dependencies during development

## Key Files Reference

- `AGENTS.md` - Instructions specific to AI coding agents
- `CONTRIBUTING.md` - Comprehensive human contributor guide
- `GLOSSARY.md` - Terminology (ATen, kernels, operations, JIT, TorchScript)
- `aten/src/ATen/native/README.md` - Operator implementation guide
- `tools/autograd/derivatives.yaml` - Gradient definitions for autograd

## Performance Debugging

Use `TORCH_SHOW_CPP_STACKTRACES=1` for C++ traces in Python errors. For profiling, prefer `py-spy` over manual instrumentation.
