# PyTorch System Surfaces Map

**Purpose**: Define what is "externally observable" for PyTorch. These are the interfaces, formats, and behaviors that must be protected during refactoring to avoid breaking downstream consumers.

**Audit Date**: 2026-02-08  
**Baseline Commit**: c5f1d40

---

## 1. Public Python API (`torch.*`)

### 1.1 Core Tensor API

**Surface**: Top-level tensor creation, manipulation, and computation functions

**Entry Points**:
- `torch.tensor()`, `torch.Tensor()`
- `torch.zeros()`, `torch.ones()`, `torch.rand()`, `torch.randn()`, etc.
- `torch.add()`, `torch.mul()`, `torch.matmul()`, etc. (200+ ops)
- Tensor methods: `.view()`, `.reshape()`, `.transpose()`, `.detach()`, etc.

**Backward Compatibility Contract**:
- Function signatures must remain stable (kwargs can be added with defaults)
- Tensor behavior (shape, dtype, device) must match documented semantics
- Deprecation requires 2-release warning cycle

**Consumers**:
- ~3 million PyPI downloads/day (estimated from PyPI stats)
- Millions of research scripts, production ML systems
- Downstream libraries: `torchvision`, `torchtext`, `torchaudio`, Hugging Face Transformers, Lightning, etc.

**Verification**:
- `test/test_torch.py` (1,500+ test cases)
- `test/test_ops.py` (comprehensive operator tests)

---

### 1.2 Neural Network API (`torch.nn`)

**Surface**: Module system for building neural networks

**Entry Points**:
- `torch.nn.Module` (base class for all NN components)
- `torch.nn.Linear`, `torch.nn.Conv2d`, `torch.nn.LSTM`, etc. (100+ layers)
- `torch.nn.functional.*` (functional API for stateless ops)
- `torch.nn.Parameter` (trainable parameter wrapper)

**Backward Compatibility Contract**:
- `nn.Module` subclass API (forward, `__init__`, hooks)
- Serialization format (state_dict keys, checkpoint structure)
- Parameter initialization defaults

**Consumers**:
- Every PyTorch model definition
- `torch.hub` pre-trained models
- Model zoos (timm, CLIP, etc.)

**Verification**:
- `test/test_nn.py` (2,000+ test cases)
- `test/test_modules.py`

---

### 1.3 Autograd API (`torch.autograd`)

**Surface**: Automatic differentiation engine

**Entry Points**:
- `torch.autograd.backward()`, `torch.autograd.grad()`
- `torch.autograd.Function` (custom backward pass)
- Context manager: `torch.no_grad()`, `torch.enable_grad()`
- Tensor grad tracking: `.requires_grad`, `.grad`, `.backward()`

**Backward Compatibility Contract**:
- Gradient computation must match mathematical definition
- Hook API (`register_hook()`) must remain stable
- Custom autograd function contract (`forward()`, `backward()`, `setup_context()`)

**Consumers**:
- All training loops
- Research on custom gradients (meta-learning, higher-order gradients)

**Verification**:
- `test/test_autograd.py` (3,000+ test cases)
- `test/autograd/`

---

### 1.4 Distributed Training API (`torch.distributed`)

**Surface**: Multi-node/multi-GPU training primitives

**Entry Points**:
- `torch.distributed.init_process_group()`
- Collectives: `all_reduce()`, `all_gather()`, `broadcast()`, `reduce_scatter()`
- RPC: `torch.distributed.rpc.*`
- Backends: `nccl`, `gloo`, `mpi`
- High-level: `DistributedDataParallel` (DDP), `FullyShardedDataParallel` (FSDP)

**Backward Compatibility Contract**:
- **CRITICAL**: Wire protocol for collectives must remain compatible across versions (old worker, new parameter server scenario)
- RPC serialization format
- DDP/FSDP state dict structure

**Consumers**:
- All large-scale training (multi-node clusters)
- Cloud training services (AWS SageMaker, Azure ML, etc.)

**Verification**:
- `test/distributed/` (500+ test files, multi-process)
- **Gap**: No explicit protocol version test (see Issue I06 in BASELINE_AUDIT.md)

---

### 1.5 TorchScript / JIT API

**Surface**: Ahead-of-time compilation and model export

**Entry Points**:
- `torch.jit.script()`, `torch.jit.trace()`
- `torch.jit.load()`, `torch.jit.save()`
- `torch.jit.ScriptModule`

**Backward Compatibility Contract**:
- Serialization format (`.pt` files) must be backward compatible
- TorchScript IR must remain loadable across versions
- Operator schema changes must be versioned

**Consumers**:
- Mobile deployment (PyTorch Mobile)
- C++ inference (LibTorch)
- ONNX export pipeline (uses TorchScript as intermediate)

**Verification**:
- `test/jit/` (50+ test files)
- `test/test_serialization.py`
- `test/forward_backward_compatibility/` (BC test suite)

---

### 1.6 ONNX Export API

**Surface**: Export PyTorch models to ONNX format for interoperability

**Entry Points**:
- `torch.onnx.export()`
- `torch.onnx.dynamo_export()` (new, uses torch.export)

**Backward Compatibility Contract**:
- ONNX opset version mapping
- Operator export logic must match ONNX spec
- Model must be loadable in ONNX Runtime

**Consumers**:
- ONNX Runtime users
- Inference frameworks (TensorRT, OpenVINO)
- Edge deployment (ONNX-based toolchains)

**Verification**:
- `test/onnx/` (20+ test files)
- External: ONNX Model Zoo compatibility tests

---

### 1.7 Compiler API (`torch.compile`)

**Surface**: Graph-based optimization and compilation

**Entry Points**:
- `torch.compile()` (decorator/wrapper)
- Backends: `"inductor"`, `"aot_eager"`, `"cudagraphs"`
- Mode: `"default"`, `"reduce-overhead"`, `"max-autotune"`

**Backward Compatibility Contract**:
- **BETA STABILITY**: API may change, but documented deprecations required
- Compiled models must produce numerically identical results to eager mode (within tolerance)

**Consumers**:
- Performance-critical training/inference
- Production systems using `torch.compile()` (growing adoption)

**Verification**:
- `test/dynamo/` (100+ test files)
- `test/inductor/` (300+ test files)

---

## 2. C++ API (LibTorch)

### 2.1 ATen Tensor Library

**Surface**: C++ tensor API

**Entry Points**:
- `at::Tensor` class
- Operator functions: `at::add()`, `at::mul()`, `at::matmul()`, etc.
- Factory functions: `at::zeros()`, `at::ones()`, etc.

**Backward Compatibility Contract**:
- ABI stability (within major version)
- Header-only APIs must remain source-compatible

**Consumers**:
- C++ inference applications
- PyTorch C++ extensions
- Mobile (iOS, Android)

**Verification**:
- `c10/test/`, `aten/src/ATen/test/`
- `test/cpp/` (279+ test files)

---

### 2.2 C++ Frontend (torch::nn)

**Surface**: High-level C++ neural network API

**Entry Points**:
- `torch::nn::Module`
- `torch::nn::Linear`, `torch::nn::Conv2d`, etc.
- `torch::optim::SGD`, `torch::optim::Adam`

**Backward Compatibility Contract**:
- API mirrors Python `torch.nn` where possible
- Serialization interop with Python models

**Consumers**:
- C++ training/inference applications
- Embedded systems (limited Python runtime)

**Verification**:
- `test/cpp/api/` (C++ frontend tests)

---

### 2.3 C++ Extensions API

**Surface**: Mechanism for extending PyTorch with custom C++/CUDA ops

**Entry Points**:
- `PYBIND11_MODULE()` macro
- `torch::RegisterOperators()`
- `setuptools` integration (`torch.utils.cpp_extension`)

**Backward Compatibility Contract**:
- `pybind11` binding API must remain stable
- Operator registration ABI must be stable

**Consumers**:
- Third-party extensions (e.g., `flash-attn`, `xformers`)
- Custom kernels in production systems

**Verification**:
- `test/test_cpp_extensions_aot.py`
- `test/test_cpp_extensions_jit.py`
- `test/cpp_extensions/`

---

## 3. Serialization Formats

### 3.1 Checkpoint Format (`.pt`, `.pth`)

**Surface**: PyTorch model and tensor serialization

**Format Details**:
- Based on Python `pickle` (with custom reducers)
- ZIP archive containing:
  - `data.pkl` (pickled state_dict)
  - `version` file (PyTorch version)
  - Tensor data (in various formats: pickle, zipfile, etc.)

**Backward Compatibility Contract**:
- Old checkpoints must load in new PyTorch versions
- Forward compat (new → old) is NOT guaranteed

**Consumers**:
- Every training script saving/loading models
- Model hubs (Hugging Face Hub, PyTorch Hub)

**Verification**:
- `test/test_serialization.py`
- `test/forward_backward_compatibility/`

---

### 3.2 TorchScript Serialized Format

**Surface**: Compiled model serialization

**Format Details**:
- FlatBuffers or Pickle-based (configurable)
- Contains TorchScript IR (operators, constants, control flow)

**Backward Compatibility Contract**:
- **CRITICAL**: Mobile apps cannot update PyTorch easily; models must remain loadable
- Operator versioning required for schema changes

**Consumers**:
- PyTorch Mobile apps
- LibTorch C++ inference

**Verification**:
- `test/jit/`
- `test/test_jit_legacy.py` (BC tests)

---

### 3.3 ONNX Format

**Surface**: Exported ONNX models

**Format Details**:
- ONNX protobuf format
- Opset version determines available operators

**Backward Compatibility Contract**:
- ONNX spec compliance (external dependency)
- Opset version must be documented

**Consumers**:
- ONNX Runtime
- TensorRT, OpenVINO, CoreML converters

**Verification**:
- `test/onnx/`

---

## 4. Command-Line Interfaces (CLI)

### 4.1 Python Module Execution

**Surface**: `python -m torch` utilities

**Entry Points**:
- `python -m torch.utils.collect_env` (environment info)
- `python -m torch.distributed.run` (distributed launcher, replaces `torch.distributed.launch`)

**Backward Compatibility Contract**:
- CLI flags and output format
- Exit codes

**Consumers**:
- Training scripts, CI systems
- Cluster job launchers (SLURM, Kubernetes)

**Verification**:
- `test/distributed/launcher/` (launcher tests)

---

### 4.2 `torch-model-archiver` (TorchServe)

**Surface**: Model packaging for TorchServe

**Entry Points**:
- CLI tool from TorchServe project (external repo)

**Backward Compatibility Contract**:
- Model archive format must load in TorchServe

**Consumers**:
- Production inference deployments

**Verification**:
- External: TorchServe CI

---

## 5. Configuration Files & Environment Variables

### 5.1 Environment Variables

**Surface**: Runtime behavior controlled by env vars

**Key Variables**:
- `TORCH_HOME` (default: `~/.torch/`) - model cache location
- `CUDA_VISIBLE_DEVICES` - GPU selection
- `OMP_NUM_THREADS` - CPU parallelism
- `TORCH_SHOW_CPP_STACKTRACES` - debugging
- `TORCH_COMPILE_DEBUG` - compiler debugging

**Backward Compatibility Contract**:
- Existing env vars must remain functional
- New env vars should have sensible defaults

**Consumers**:
- All users (training scripts, notebooks, production)

**Verification**:
- Documented in `torch.utils.collect_env`
- Ad-hoc tests in various test files

---

### 5.2 Config Files

**Surface**: Per-user or per-project configuration

**Current State**: 
- **None** (PyTorch does not use config files like `.torchrc`)

**Future Consideration**:
- If config files are added, format must be versioned

---

## 6. Model Artifacts & Pretrained Weights

### 6.1 `torch.hub`

**Surface**: Pre-trained model repository

**Entry Points**:
- `torch.hub.load(repo, model, *args, **kwargs)`
- `hubconf.py` in GitHub repos

**Backward Compatibility Contract**:
- `hubconf.py` API must remain stable
- Model download URLs should not break

**Consumers**:
- Researchers using pre-trained models
- Production systems using `torch.hub` for model loading

**Verification**:
- `test/test_hub.py`

---

### 6.2 Model Zoos (External)

**Surface**: Third-party model repositories

**Examples**:
- Hugging Face Transformers
- `timm` (PyTorch Image Models)
- `torchvision.models`

**Backward Compatibility Contract**:
- PyTorch API changes must not break model loading
- State dict keys should remain stable

**Consumers**:
- Entire ecosystem

**Verification**:
- External: downstream library CI

---

## 7. Integration Points

### 7.1 NumPy Interoperability

**Surface**: Tensor ↔ NumPy array conversion

**Entry Points**:
- `torch.from_numpy(ndarray)`
- `tensor.numpy()`
- Zero-copy sharing (when possible)

**Backward Compatibility Contract**:
- Data layout, dtype mapping must remain consistent

**Consumers**:
- Data preprocessing pipelines
- SciPy, scikit-learn integration

**Verification**:
- `test/test_numpy_interop.py`

---

### 7.2 CUDA / ROCm / XPU / MPS

**Surface**: Hardware acceleration backend APIs

**Entry Points**:
- `torch.cuda.*` (NVIDIA CUDA)
- `torch.backends.cuda.*` (CUDA backend config)
- `torch.backends.cudnn.*` (cuDNN config)
- `torch.xpu.*` (Intel XPU)
- `torch.mps.*` (Apple Metal Performance Shaders)

**Backward Compatibility Contract**:
- Device string parsing: `"cuda:0"`, `"cpu"`, `"xpu:0"`, `"mps"`
- Stream, event, memory management APIs

**Consumers**:
- GPU training/inference scripts
- Multi-GPU pipelines

**Verification**:
- `test/test_cuda.py`, `test/test_xpu.py`, `test/test_mps.py`

---

### 7.3 DLPack

**Surface**: Framework-agnostic tensor exchange

**Entry Points**:
- `torch.utils.dlpack.to_dlpack(tensor)`
- `torch.utils.dlpack.from_dlpack(dlpack_tensor)`

**Backward Compatibility Contract**:
- DLPack protocol version compliance

**Consumers**:
- JAX, CuPy, TensorFlow interoperability

**Verification**:
- `test/test_dlpack.py`

---

## 8. External Services & Registries

### 8.1 PyPI Package (`torch`)

**Surface**: Package installation and metadata

**Entry Points**:
- `pip install torch`
- `torch.__version__`

**Backward Compatibility Contract**:
- Semantic versioning (major.minor.patch)
- Deprecation warnings before breaking changes

**Consumers**:
- All PyTorch users

**Verification**:
- PyPI release process (external)

---

### 8.2 Conda Package

**Surface**: Conda distribution

**Entry Points**:
- `conda install pytorch -c pytorch`

**Backward Compatibility Contract**:
- Consistent with PyPI version

**Consumers**:
- Conda users

**Verification**:
- Conda-forge CI (external)

---

## 9. Documentation & Type Stubs

### 9.1 Docstrings

**Surface**: Inline documentation

**Entry Points**:
- `help(torch.tensor)`
- Sphinx-generated docs (pytorch.org)

**Backward Compatibility Contract**:
- Docstrings should remain accurate
- Breaking changes must be documented

**Consumers**:
- All users (via IDE autocomplete, docs site)

**Verification**:
- `test/run_doctests.sh` (partial)

---

### 9.2 Type Stubs (`.pyi`)

**Surface**: Static type information

**Entry Points**:
- `torch/__init__.pyi`
- `torch/_C/*.pyi`

**Backward Compatibility Contract**:
- Type annotations should not break existing typed code

**Consumers**:
- Users with `mypy`, `pyright` type checkers

**Verification**:
- `test/test_type_hints.py`
- `mypy` CI checks

---

## 10. Summary: Critical Surfaces Ranked by Blast Radius

| Rank | Surface | Blast Radius | Refactor Risk | Verification Strength |
|------|---------|--------------|---------------|----------------------|
| 1 | Python API (`torch.*`) | 🔴 **CRITICAL** | Very High | 🟢 Strong |
| 2 | C++ API (ATen, LibTorch) | 🔴 **CRITICAL** | Very High | 🟢 Strong |
| 3 | Checkpoint Format (`.pt`, `.pth`) | 🔴 **CRITICAL** | High | 🟢 Strong |
| 4 | TorchScript Serialization | 🔴 **CRITICAL** | High | 🟡 Medium |
| 5 | Distributed Wire Protocol | 🟠 **HIGH** | High | 🟠 Weak (no explicit version tests) |
| 6 | ONNX Export | 🟠 **HIGH** | Medium | 🟡 Medium |
| 7 | `torch.compile` API | 🟡 **MEDIUM** | Medium (beta) | 🟢 Strong |
| 8 | Environment Variables | 🟡 **MEDIUM** | Low | 🟠 Weak (ad-hoc) |
| 9 | CLI Tools | 🟢 **LOW-MEDIUM** | Low | 🟡 Medium |
| 10 | Type Stubs (`.pyi`) | 🟢 **LOW** | Low | 🟡 Medium |

---

## Appendix: Surface Testing Checklist

For any refactor that touches these surfaces, verify:

- [ ] **Python API**: Run `test/test_torch.py`, `test/test_ops.py`
- [ ] **C++ API**: Run `test/cpp/`, rebuild C++ extensions
- [ ] **Serialization**: Run `test/test_serialization.py`, test old checkpoint loading
- [ ] **Distributed**: Run `test/distributed/` with multi-process launcher
- [ ] **TorchScript**: Run `test/jit/`, test mobile model loading
- [ ] **ONNX**: Run `test/onnx/`, verify ONNX Runtime compatibility
- [ ] **NumPy Interop**: Run `test/test_numpy_interop.py`
- [ ] **Type Stubs**: Run `mypy --strict` on affected modules

---

**End of System Surfaces Map**

