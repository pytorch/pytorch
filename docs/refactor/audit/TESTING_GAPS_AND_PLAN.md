# PyTorch Testing Gaps & Plan

**Purpose**: Identify testing gaps and propose minimal additions to support safe refactoring.

**Audit Date**: 2026-02-08  
**Baseline Commit**: c5f1d40

---

## 1. Test Inventory

| Test Type | Count | Location | Execution Time | Coverage Est. |
|-----------|-------|----------|----------------|---------------|
| **Python Unit** | 1,353+ files | `test/*.py` | ~2-4 hours (sharded) | 70-80% |
| **C++ Unit** | 279+ files | `test/cpp/`, `c10/test/`, `aten/test/` | ~30 min | 60-70% |
| **Distributed Integration** | 100+ files | `test/distributed/` | ~1 hour (multi-process) | 65-75% |
| **Compiler Tests** | 300+ files | `test/dynamo/`, `test/inductor/` | ~1.5 hours | 60-70% |
| **JIT Tests** | 50+ files | `test/jit/` | ~20 min | 70% |
| **ONNX Tests** | 20+ files | `test/onnx/` | ~15 min | 60% |
| **Doctests** | Inline | Various | Not run systematically | <10% |

**Total Test Execution**: ~5-7 hours (full suite, unsharded)  
**CI Sharding**: 5 shards (default), 3 shards (distributed) → ~1 hour wall time

---

## 2. Coverage Snapshot (Baseline)

**Unable to run locally** (no build env per AGENTS.md), but CI workflows include coverage reporting.

**Estimated Coverage by Module**:
- `torch/`: 75-85% (high test density)
- `torch/nn/`: 80-90% (comprehensive module tests)
- `torch/autograd/`: 85-90% (gradcheck framework)
- `torch/distributed/`: 65-75% (complex multi-process)
- `torch/_dynamo/`: 60-70% (rapidly evolving)
- `torch/_inductor/`: 60-70% (rapidly evolving)
- `c10/`: 70-80% (unit tests exist)
- `aten/`: 75-85% (OpInfo tests cover 500+ ops)
- `caffe2/`: 30-40% (deprecated, low priority)

**Overall Estimated Coverage**: **70-75%**

---

## 3. Testing Gaps by Priority

### Gap 1: No Import Smoke Test (P0)

**Description**: No test verifies Python imports work without C++ build.

**Impact**: Refactors can break import paths, only discovered during full CI build.

**Recommended Fix**: **M01** - Add `test/test_import_smoke.py`

**Test Plan**:
```python
# test/test_import_smoke.py
def test_torch_import():
    import torch
    assert torch.__version__

def test_nn_import():
    from torch import nn
    assert nn.Module

def test_optim_import():
    from torch import optim
    assert optim.SGD
```

**Verification**: Runs in <1 second, no C++ build required

---

### Gap 2: No Distributed Protocol Version Test (P0)

**Description**: No test verifies distributed collectives work across PyTorch versions.

**Impact**: Silent failures in production clusters during rolling updates.

**Recommended Fix**: **M11, M12** - Add protocol version + cross-version tests

**Test Plan**:
- Start two processes with different PyTorch versions
- Attempt `all_reduce()`
- Expect: Graceful failure with version mismatch error

---

### Gap 3: No State Dict Key Regression Test (P1)

**Description**: No test prevents accidental state dict key renames (e.g., `linear.weight` → `linear.W`).

**Impact**: Breaks all saved model checkpoints for downstream users.

**Recommended Fix**: **M18** - Add `test/test_state_dict_keys.py`

**Test Plan**:
- Snapshot state dict keys for all `nn.*` modules
- On future runs, compare keys; fail if changed

---

### Gap 4: Weak Determinism Testing (P2)

**Description**: Tests exist, but not run systematically in deterministic mode.

**Impact**: Non-deterministic ops may be used in contexts requiring determinism.

**Recommended Fix**: **M16** - Add `test/test_determinism.py`

**Test Plan**:
- Enable `torch.use_deterministic_algorithms(True)`
- Run ops twice with same seed
- Assert outputs are identical

---

### Gap 5: No ABI Compatibility Checker (P1)

**Description**: C++ ABI breaks are not detected before release.

**Impact**: Users upgrading PyTorch cannot use pre-compiled C++ extensions.

**Recommended Fix**: **M15** - Add ABI checker to CI

**Tool**: `abi-compliance-checker` or `abi-dumper`

---

### Gap 6: No Performance Regression Gate (P2)

**Description**: Performance regressions are not caught automatically.

**Impact**: Refactors may slow down critical ops without detection.

**Recommended Fix**: **M21** (deferred) - Add perf regression CI job

**Challenge**: Benchmarks are noisy; requires stable hardware + statistical analysis

---

### Gap 7: No Memory Regression Test (P2)

**Description**: Memory usage increases are not tracked.

**Impact**: Refactors may cause OOM errors in production.

**Recommended Fix**: **M22** (deferred) - Add memory profiling tests

---

### Gap 8: Weak Doctest Coverage (P3)

**Description**: Docstring examples are not tested.

**Impact**: Documentation examples may be outdated or incorrect.

**Recommended Fix**: **M20** (deferred) - Add doctests to core functions

---

## 4. Flake & Nondeterminism Risks

### Known Flaky Tests (Examples)

**Note**: Flaky tests should be tracked in CI artifacts. This is a preliminary list.

| Test | Flake Rate | Cause | Mitigation |
|------|-----------|-------|------------|
| `test_distributed_*` (multi-process) | ~2-5% | Timing, port conflicts | Retry logic, unique ports |
| `test_cuda_*` (GPU tests) | ~1-3% | GPU OOM, race conditions | Retry, explicit sync |
| `test_dynamo_*` (compiler) | ~1-2% | Bytecode version sensitivity | Pin Python version |

**Recommendation**: Add flake tracking dashboard (e.g., GitHub issue labels, or tool like `pytest-flakefinder`)

---

## 5. Test Gap Summary Table

| Gap ID | Description | Priority | Refactor Risk | Milestone | Effort |
|--------|-------------|---------|---------------|-----------|--------|
| **G01** | No import smoke test | P0 | High | M01 | 4h |
| **G02** | No distributed protocol version test | P0 | Very High | M11, M12 | 28h |
| **G03** | No state dict key regression test | P1 | High | M18 | 8h |
| **G04** | Weak determinism testing | P2 | Medium | M16 | 8h |
| **G05** | No ABI compatibility checker | P1 | High | M15 | 16h |
| **G06** | No perf regression gate | P2 | Medium | M21 | 20h (deferred) |
| **G07** | No memory regression test | P2 | Medium | M22 | 16h (deferred) |
| **G08** | Weak doctest coverage | P3 | Low | M20 | 20h (deferred) |

**Total Effort (P0-P1)**: 56 hours  
**Total Effort (All)**: 120 hours

---

## 6. Minimal Test Additions (Quick Wins)

These can be added immediately to improve refactor safety:

### 6.1 Import Smoke Test (M01)
- **File**: `test/test_import_smoke.py` (~50 lines)
- **Time**: 4 hours
- **Benefit**: Catches import breakage before full CI build

### 6.2 State Dict Key Snapshot (M18)
- **File**: `test/test_state_dict_keys.py` (~100 lines) + `test/expect/state_dict_keys.json` (snapshot)
- **Time**: 8 hours
- **Benefit**: Prevents checkpoint incompatibility

### 6.3 Distributed Version Check (M11)
- **File**: `torch/distributed/_protocol_version.py` (~20 lines) + version check in `init_process_group()`
- **Time**: 16 hours
- **Benefit**: Prevents silent cross-version failures

---

## 7. Test Execution Strategy for Refactors

### 7.1 Smoke Test (Fast, <1 min)
- Import smoke test (M01)
- Syntax check (`python -m py_compile`)

### 7.2 Unit Tests (Medium, ~10-30 min)
- Relevant module tests (e.g., `pytest test/test_nn.py` for `torch.nn` refactor)

### 7.3 Integration Tests (Slow, ~1-2 hours)
- Full test suite (if touching core like `c10/`, `aten/`)

### 7.4 CI Verification (Slowest, ~1 hour wall time)
- All tests, sharded across 5+ workers
- Required for merge

---

## 8. Coverage Goals (Post-Refactor Program)

| Module | Current Coverage | Target Coverage | Strategy |
|--------|-----------------|----------------|----------|
| `c10/` | 70% | 80% | Add edge case tests (device, dtype) |
| `aten/` | 80% | 85% | Expand OpInfo tests |
| `torch/nn/` | 85% | 90% | Add module hook tests |
| `torch/distributed/` | 70% | 80% | Add cross-version tests |
| `torch/_dynamo/` | 65% | 75% | Add bytecode edge cases |
| `torch/_inductor/` | 65% | 75% | Add codegen edge cases |

**Overall Target**: **75-80%** (from baseline 70-75%)

---

## 9. Test Infrastructure Needs

### 9.1 Faster Feedback Loop
- **Current**: Full CI = ~1 hour (sharded)
- **Goal**: Critical tests in <10 min
- **Solution**: Tier tests (smoke → unit → integration)

### 9.2 Better Flake Tracking
- **Current**: Manual issue reports
- **Goal**: Automated flake dashboard
- **Solution**: CI artifact analysis (e.g., `pytest --reruns=3`)

### 9.3 Performance Baseline
- **Current**: Benchmarks exist, not run on PRs
- **Goal**: Perf regression gate (M21)
- **Challenge**: Noisy, requires stable hardware

---

## 10. Verification Command Quick Reference

| What | Command | When |
|------|---------|------|
| **Import Smoke** | `python test/test_import_smoke.py` | After restructuring |
| **Unit Tests** | `pytest test/test_torch.py -v` | After API changes |
| **Operator Tests** | `pytest test/test_ops.py -k test_add` | After op impl change |
| **Gradient Tests** | `pytest test/test_ops_gradients.py` | After autograd change |
| **Distributed** | `pytest test/distributed/ -v` | After distributed change |
| **JIT** | `pytest test/jit/ -v` | After TorchScript change |
| **Full Suite** | `python test/run_test.py` | Before merge (locally) |
| **CI** | Push to PR branch | Always (required for merge) |

---

**End of Testing Gaps & Plan**

