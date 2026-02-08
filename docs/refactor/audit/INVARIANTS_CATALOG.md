# PyTorch Invariants Catalog

**Purpose**: Define the invariants that MUST NOT change during refactoring. These are the promises PyTorch makes to users about behavior, interfaces, and compatibility.

**Audit Date**: 2026-02-08  
**Baseline Commit**: c5f1d40

---

## How to Use This Catalog

For each refactor milestone:
1. Identify which invariants are in scope (could be affected)
2. Add verification tests if coverage is marked "missing" or "partial"
3. Run verification before and after refactor
4. Document any intentional changes with deprecation plan

**Priority Levels**:
- **P0 (Critical)**: Breaking this invariant breaks production systems immediately
- **P1 (High)**: Breaking this causes silent incorrectness or data loss
- **P2 (Medium)**: Breaking this causes user-facing errors but recoverable
- **P3 (Low)**: Breaking this causes inconvenience or requires doc updates

---

## 1. API Stability Invariants

### INV-001: Python Public API Signatures

**Invariant**: Public API functions/methods cannot change signature without deprecation cycle.

**Surface**: `torch.*` (all non-underscore-prefixed names)

**Verification Method**:
- **Static**: Compare function signatures before/after refactor using `inspect.signature()`
- **Dynamic**: Run `test/test_torch.py`, `test/test_ops.py`

**Current Coverage**: 🟢 **Strong** (comprehensive tests exist)

**Priority**: P0 (Critical)

**Notes**:
- Allowed changes: Add kwargs with defaults, add overloads
- Disallowed changes: Remove args, change arg types without coercion, change defaults

**Example**:
```python
# ALLOWED:
def foo(x, y=1):  →  def foo(x, y=1, z=2):

# DISALLOWED:
def foo(x, y=1):  →  def foo(x, z=1):  # removed `y`
```

---

### INV-002: C++ API (ATen) Signatures

**Invariant**: Public C++ API (ATen operators) cannot change without ABI versioning.

**Surface**: `at::*`, `c10::*`, `torch::*` (C++ namespace)

**Verification Method**:
- **Static**: ABI compatibility checker (e.g., `abi-compliance-checker`)
- **Dynamic**: Run `test/cpp/` suite

**Current Coverage**: 🟡 **Partial** (tests exist, but no automated ABI checker in CI)

**Priority**: P0 (Critical)

**Notes**:
- C++ lacks Python's dynamic typing; signature changes = ABI break
- Requires major version bump or ABI versioning scheme

**Recommendation**: Add ABI compliance check to CI (deferred to milestone M15)

---

### INV-003: `torch.nn.Module` Subclass Contract

**Invariant**: `nn.Module` subclassing requirements must remain stable:
- `__init__()` → register parameters/buffers
- `forward()` → compute outputs
- Hooks: `register_forward_hook()`, `register_backward_hook()`
- State dict: `state_dict()`, `load_state_dict()`

**Surface**: `torch.nn.Module`, all `nn.*` layers

**Verification Method**:
- **Dynamic**: `test/test_nn.py`, `test/test_modules.py`
- **Integration**: Test custom user-defined modules

**Current Coverage**: 🟢 **Strong**

**Priority**: P0 (Critical)

**Notes**:
- Breaking this breaks every PyTorch model definition

---

### INV-004: Autograd Custom Function Contract

**Invariant**: `torch.autograd.Function` subclass API must remain stable:
- `forward(ctx, *args, **kwargs)` → outputs
- `backward(ctx, *grad_outputs)` → grad_inputs
- `setup_context(ctx, inputs, output)` (new-style)

**Surface**: `torch.autograd.Function`

**Verification Method**:
- **Dynamic**: `test/test_autograd.py` (custom function tests)

**Current Coverage**: 🟢 **Strong**

**Priority**: P0 (Critical)

**Notes**:
- Many third-party extensions use custom autograd functions (flash-attn, xformers, etc.)

---

## 2. Numerical Correctness Invariants

### INV-010: Operator Output Correctness

**Invariant**: Operators must produce mathematically correct results (within numerical tolerance).

**Surface**: All `torch.*` operators (`add`, `mul`, `matmul`, `conv2d`, etc.)

**Verification Method**:
- **Dynamic**: `test/test_ops.py` (OpInfo-based tests with reference implementations)
- **Numerical**: Compare against NumPy, SciPy, or analytical solutions

**Current Coverage**: 🟢 **Strong** (OpInfo framework covers 500+ operators)

**Priority**: P1 (High)

**Notes**:
- Tolerance: typically `atol=1e-5, rtol=1e-4` for float32
- Some ops (e.g., `sqrt`, `log`) have documented edge case behavior (NaN, inf)

**Verification Command**:
```bash
pytest test/test_ops.py -k test_operator_name
```

---

### INV-011: Gradient Correctness

**Invariant**: Computed gradients must match analytical gradients (within tolerance).

**Surface**: All differentiable operators

**Verification Method**:
- **Dynamic**: `test/test_ops_gradients.py` (gradcheck, gradgradcheck)
- **Numerical**: Finite difference approximation

**Current Coverage**: 🟢 **Strong** (gradcheck in OpInfo tests)

**Priority**: P1 (High)

**Notes**:
- Breaking this causes silent training failures (wrong updates)
- Double-backward (gradgradcheck) is tested for most ops

---

### INV-012: Determinism (When Requested)

**Invariant**: If deterministic mode is enabled, operations must produce identical results across runs.

**Surface**: `torch.use_deterministic_algorithms(True)`

**Verification Method**:
- **Dynamic**: `test/test_* --deterministic` flag (partial coverage)

**Current Coverage**: 🟠 **Weak** (not all tests run in deterministic mode)

**Priority**: P2 (Medium)

**Notes**:
- Some ops (e.g., `scatter_add_`) are inherently non-deterministic on GPU
- Determinism is best-effort, not guaranteed

**Recommendation**: Add determinism test harness (deferred to milestone M16)

---

## 3. Serialization Invariants

### INV-020: Checkpoint Backward Compatibility

**Invariant**: Checkpoints saved in version N must load in version N+k (forward compatibility).

**Surface**: `torch.save()`, `torch.load()`

**Verification Method**:
- **Dynamic**: `test/test_serialization.py`
- **Integration**: `test/forward_backward_compatibility/` (loads old checkpoints)

**Current Coverage**: 🟢 **Strong**

**Priority**: P0 (Critical)

**Notes**:
- Pickle format evolution is managed via `torch.serialization._rebuild_*` functions
- State dict key changes break compatibility

**Test Matrix**:
- Load PyTorch 1.x checkpoint in 2.x: ✅ Required
- Load PyTorch 2.x checkpoint in 1.x: ❌ Not guaranteed

---

### INV-021: TorchScript Model Compatibility

**Invariant**: TorchScript models (.pt) saved in version N must load in version N+k.

**Surface**: `torch.jit.save()`, `torch.jit.load()`

**Verification Method**:
- **Dynamic**: `test/jit/`, `test/test_jit_legacy.py`

**Current Coverage**: 🟡 **Partial** (BC tests exist, but not comprehensive)

**Priority**: P0 (Critical)

**Notes**:
- Mobile deployment cannot easily update PyTorch; models must remain loadable
- Operator schema versioning required

**Recommendation**: Improve BC test coverage (deferred to milestone M17)

---

### INV-022: State Dict Key Stability

**Invariant**: `state_dict()` keys for `nn.Module` subclasses must remain stable.

**Surface**: All `torch.nn.*` modules

**Verification Method**:
- **Snapshot**: Save state dict, compare keys after refactor

**Current Coverage**: 🟠 **Weak** (no systematic key stability tests)

**Priority**: P1 (High)

**Notes**:
- Breaking this breaks checkpoint loading for all downstream users
- Example: Renaming `linear.weight` → `linear.W` would break all saved models

**Recommendation**: Add state dict key regression test (milestone M18)

---

## 4. Distributed System Invariants

### INV-030: Collective Wire Protocol Compatibility

**Invariant**: Distributed collectives (all_reduce, all_gather, etc.) must work across heterogeneous PyTorch versions.

**Surface**: `torch.distributed.*` collectives

**Verification Method**:
- **Integration**: Multi-process test with mixed versions

**Current Coverage**: 🔴 **Missing** (no explicit cross-version tests)

**Priority**: P0 (Critical)

**Notes**:
- **RISK**: No protocol version checks; breaking changes are silent failures
- Production clusters often have mixed PyTorch versions during rolling updates

**Recommendation**: **HIGH PRIORITY** - Add protocol version + cross-version tests (milestone M11)

---

### INV-031: RPC Serialization Compatibility

**Invariant**: RPC messages must deserialize across PyTorch versions.

**Surface**: `torch.distributed.rpc.*`

**Verification Method**:
- **Integration**: Multi-process test with mixed versions

**Current Coverage**: 🔴 **Missing**

**Priority**: P1 (High)

**Notes**:
- RPC uses pickle-based serialization; version skew can cause deserialization failures

**Recommendation**: Add RPC cross-version test (milestone M11)

---

### INV-032: DistributedDataParallel (DDP) Hook API

**Invariant**: DDP communication hooks must remain compatible.

**Surface**: `torch.nn.parallel.DistributedDataParallel`, `register_comm_hook()`

**Verification Method**:
- **Dynamic**: `test/distributed/test_c10d_common.py`

**Current Coverage**: 🟡 **Partial**

**Priority**: P2 (Medium)

**Notes**:
- Third-party hooks (e.g., gradient compression) depend on this API

---

## 5. Device & Hardware Invariants

### INV-040: Device String Parsing

**Invariant**: Device strings (`"cpu"`, `"cuda:0"`, `"xpu:1"`, `"mps"`) must parse correctly.

**Surface**: `torch.device()`, `tensor.to(device)`

**Verification Method**:
- **Dynamic**: `test/test_torch.py` (device tests)

**Current Coverage**: 🟢 **Strong**

**Priority**: P0 (Critical)

**Notes**:
- Breaking this breaks all multi-GPU code

---

### INV-041: CUDA Stream Semantics

**Invariant**: CUDA stream creation, synchronization must behave as documented.

**Surface**: `torch.cuda.Stream`, `torch.cuda.synchronize()`, `with torch.cuda.stream(s):`

**Verification Method**:
- **Dynamic**: `test/test_cuda.py`

**Current Coverage**: 🟢 **Strong**

**Priority**: P1 (High)

**Notes**:
- Async execution correctness depends on this

---

### INV-042: Memory Format Preservation

**Invariant**: Tensor memory format (contiguous, channels_last, etc.) must be preserved across operations.

**Surface**: `tensor.contiguous()`, `tensor.is_contiguous()`, `memory_format` arg

**Verification Method**:
- **Dynamic**: `test/test_torch.py` (memory format tests)

**Current Coverage**: 🟡 **Partial**

**Priority**: P2 (Medium)

**Notes**:
- Performance optimization (channels_last for conv nets) depends on this

---

## 6. Build & Packaging Invariants

### INV-050: Import Path Stability

**Invariant**: Public imports must remain accessible from documented paths.

**Surface**: `from torch import *`, `from torch.nn import *`, etc.

**Verification Method**:
- **Static**: Import smoke test (can be done without C++ build)

**Current Coverage**: 🟡 **Partial** (no systematic import smoke test)

**Priority**: P0 (Critical)

**Notes**:
- Example: `from torch.nn import Linear` must work forever

**Recommendation**: Add import smoke test (milestone M01)

---

### INV-051: `torch.__version__` Format

**Invariant**: Version string must follow semantic versioning (major.minor.patch).

**Surface**: `torch.__version__`

**Verification Method**:
- **Static**: Regex check `\d+\.\d+\.\d+`

**Current Coverage**: 🟢 **Strong** (hardcoded in release process)

**Priority**: P2 (Medium)

---

## 7. Backward Compatibility Invariants

### INV-060: Deprecation Cycle Enforcement

**Invariant**: Breaking changes require 2-release deprecation warnings.

**Surface**: All public APIs

**Verification Method**:
- **Policy**: Manual review in PR process

**Current Coverage**: 🟡 **Partial** (enforced by maintainers, not automated)

**Priority**: P0 (Critical)

**Notes**:
- PyTorch 2.0 → 2.1: Deprecation warnings added
- PyTorch 2.1 → 2.2: Deprecated features removed

**Recommendation**: Add automated deprecation policy checker (milestone M19)

---

### INV-061: `torch.nn.functional` API Stability

**Invariant**: Functional API must remain stable (stateless op interface).

**Surface**: `torch.nn.functional.*`

**Verification Method**:
- **Dynamic**: `test/test_nn.py`

**Current Coverage**: 🟢 **Strong**

**Priority**: P0 (Critical)

**Notes**:
- Used heavily in functional programming style (e.g., `F.relu(x)`)

---

## 8. Documentation & Type Invariants

### INV-070: Type Stub Correctness

**Invariant**: `.pyi` stubs must match runtime behavior.

**Surface**: `torch/*.pyi`, `torch/_C/*.pyi`

**Verification Method**:
- **Static**: `mypy --strict`, `pyright`

**Current Coverage**: 🟡 **Partial** (type checks in CI, but not comprehensive)

**Priority**: P2 (Medium)

**Notes**:
- Incorrect stubs cause false positives/negatives in type checkers

---

### INV-071: Docstring Accuracy

**Invariant**: Docstrings must match actual function behavior.

**Surface**: All public APIs

**Verification Method**:
- **Dynamic**: `pytest --doctest-modules` (partial)

**Current Coverage**: 🟠 **Weak** (doctests not comprehensive)

**Priority**: P3 (Low)

**Notes**:
- Outdated docstrings mislead users, but don't break code

**Recommendation**: Improve doctest coverage (low priority, milestone M20)

---

## 9. Performance Invariants

### INV-080: No Performance Regression (Critical Ops)

**Invariant**: Core operations (matmul, conv2d, etc.) must not regress >5% without justification.

**Surface**: High-frequency operators

**Verification Method**:
- **Benchmark**: `benchmarks/operator_benchmark/`, `benchmarks/dynamo/`

**Current Coverage**: 🟡 **Partial** (benchmarks exist, but not run on every PR)

**Priority**: P2 (Medium)

**Notes**:
- Refactors that simplify code may cause small regressions; 5% threshold is policy decision

**Recommendation**: Add perf regression gate (milestone M21)

---

### INV-081: Memory Overhead Bound

**Invariant**: Peak memory usage should not increase unbounded for fixed-size inputs.

**Surface**: All operators (especially gradient computation)

**Verification Method**:
- **Benchmark**: Memory profiling (`torch.cuda.max_memory_allocated()`)

**Current Coverage**: 🟠 **Weak** (no systematic memory regression tests)

**Priority**: P2 (Medium)

**Notes**:
- OOM errors are harder to debug than performance regressions

**Recommendation**: Add memory regression test (milestone M22)

---

## 10. Summary: Critical Invariants by Priority

### P0 (Critical) - Must Never Break

| ID | Invariant | Verification Strength | Action Required |
|----|-----------|----------------------|-----------------|
| INV-001 | Python API Signatures | 🟢 Strong | None |
| INV-002 | C++ API Signatures | 🟡 Partial | Add ABI checker (M15) |
| INV-003 | `nn.Module` Contract | 🟢 Strong | None |
| INV-004 | Autograd Function Contract | 🟢 Strong | None |
| INV-020 | Checkpoint Backward Compat | 🟢 Strong | None |
| INV-021 | TorchScript Compat | 🟡 Partial | Improve BC tests (M17) |
| INV-030 | Distributed Wire Protocol | 🔴 **MISSING** | **ADD TESTS (M11)** |
| INV-040 | Device String Parsing | 🟢 Strong | None |
| INV-050 | Import Path Stability | 🟡 Partial | Add smoke test (M01) |
| INV-060 | Deprecation Cycle | 🟡 Partial | Automate check (M19) |
| INV-061 | `torch.nn.functional` API | 🟢 Strong | None |

### P1 (High) - Silent Failures

| ID | Invariant | Verification Strength | Action Required |
|----|-----------|----------------------|-----------------|
| INV-010 | Operator Correctness | 🟢 Strong | None |
| INV-011 | Gradient Correctness | 🟢 Strong | None |
| INV-022 | State Dict Key Stability | 🟠 Weak | Add regression test (M18) |
| INV-031 | RPC Serialization | 🔴 **MISSING** | **ADD TESTS (M11)** |
| INV-041 | CUDA Stream Semantics | 🟢 Strong | None |

### P2 (Medium) - Recoverable Errors

| ID | Invariant | Verification Strength | Action Required |
|----|-----------|----------------------|-----------------|
| INV-012 | Determinism | 🟠 Weak | Add test harness (M16) |
| INV-032 | DDP Hook API | 🟡 Partial | Improve coverage |
| INV-042 | Memory Format | 🟡 Partial | Improve coverage |
| INV-051 | Version String Format | 🟢 Strong | None |
| INV-070 | Type Stub Correctness | 🟡 Partial | Improve type checks |
| INV-080 | Performance (No Regression) | 🟡 Partial | Add perf gate (M21) |
| INV-081 | Memory Overhead | 🟠 Weak | Add mem test (M22) |

### P3 (Low) - Inconvenience

| ID | Invariant | Verification Strength | Action Required |
|----|-----------|----------------------|-----------------|
| INV-071 | Docstring Accuracy | 🟠 Weak | Improve doctests (M20) |

---

## Appendix A: Verification Quick Reference

| Verification Method | Command | When to Use |
|---------------------|---------|-------------|
| **Python API Tests** | `pytest test/test_torch.py` | After any Python API change |
| **C++ API Tests** | `python test/run_test.py --cpp` | After C++ API change |
| **Operator Tests** | `pytest test/test_ops.py` | After operator impl change |
| **Gradient Check** | `pytest test/test_ops_gradients.py` | After autograd change |
| **Serialization** | `pytest test/test_serialization.py` | After model/tensor save/load change |
| **Distributed** | `pytest test/distributed/` | After distributed change |
| **Type Stubs** | `mypy --strict torch/` | After adding/changing APIs |
| **Import Smoke Test** | `python -c "import torch; print(torch.__version__)"` | After restructuring |

---

## Appendix B: Invariant Violation Protocol

If a refactor **must** break an invariant:

1. **Document the break**: Why is this necessary?
2. **Assess blast radius**: Which downstream consumers are affected?
3. **Deprecation plan**: Add warnings in N, break in N+2 releases
4. **Migration guide**: Provide clear upgrade path
5. **Announce**: Blog post, release notes, GitHub issue
6. **Rollback plan**: How to revert if adoption is poor

**Example**: PyTorch 2.0 breaking TorchScript compatibility
- Rationale: Enable new compiler (`torch.compile`)
- Deprecation: Warnings in 1.13, break in 2.0
- Migration: Use `torch.export` instead of `torch.jit.trace`
- Announcement: Blog post, migration guide
- Rollback: Not feasible (architectural change)

---

**End of Invariants Catalog**

