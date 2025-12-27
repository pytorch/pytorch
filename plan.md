# torch.compile Cold Compile Time Optimization Plan

Inspired by Abseil's fast tips (abseil.io/fast/hints.html), this plan focuses on:
- Lazy initialization to defer work until needed
- Avoiding expensive global constructors
- Reducing import time overhead
- Caching to avoid repeated computation
- Profiling-driven optimization

---

## Phase 1: Measurement & Profiling

### 1.1 Establish baseline cold compile metrics
- [x] Create benchmark script measuring end-to-end cold compile time
- [x] Break down time by phase: dynamo tracing, AOT autograd, inductor lowering, codegen, autotuning

**Test:**
```python
import subprocess, time
script = '''
import torch
@torch.compile
def f(x): return x.sin() + x.cos()
f(torch.randn(100))
'''
start = time.time()
subprocess.run(['python', '-c', script], env={**os.environ, 'TORCH_COMPILE_DISABLE_CACHE': '1'})
print(f"Cold compile: {time.time() - start:.2f}s")
```

### 1.2 Profile import times
- [x] Measure import time for torch._dynamo, torch._inductor, triton
- [x] Identify heavy imports that could be deferred

**Test:**
```bash
python -X importtime -c "import torch._dynamo" 2>&1 | head -50
python -X importtime -c "import torch._inductor" 2>&1 | head -50
```

---

## Phase 2: Lazy Initialization (Abseil Pattern: Defer Work)

### 2.1 Lazy import for sympy in symbolic shapes
- [x] Defer sympy import until first symbolic shape is created
- [ ] Use `torch._dynamo.utils.lazy_import` pattern

**Test:**
```python
import torch
# Should NOT import sympy yet
assert 'sympy' not in sys.modules
x = torch.randn(3, 3)
# Only import when needed
torch._dynamo.symbolic_convert.something_needing_sympy()
assert 'sympy' in sys.modules
```

### 2.2 Lazy import for Triton compiler
- [ ] Defer triton import in async_compile.py until actually compiling GPU kernels
- [ ] Move `pre_fork_setup()` triton key computation to first kernel compile

**Test:**
```python
import torch._inductor.async_compile
# Triton should not be imported until GPU kernel compilation
assert 'triton' not in sys.modules
```

### 2.3 Lazy initialization for device properties
- [ ] Cache device properties lazily in `caching_device_properties()`
- [ ] Only query GPU properties when first GPU kernel is compiled

**Test:**
```python
import torch._inductor
# No CUDA calls should happen until we compile a CUDA kernel
# Verify with CUDA_VISIBLE_DEVICES="" python -c "import torch._inductor"
```

---

## Phase 3: Avoid Expensive Global Constructors (Abseil Pattern)

### 3.1 Convert module-level initialization to function-level
- [ ] Audit `torch/_dynamo/__init__.py` for eager initialization
- [ ] Move config parsing to first use
- [ ] Defer logger/handler setup

**Test:**
```python
import time
start = time.time()
import torch._dynamo
print(f"Import time: {time.time() - start:.3f}s")
# Target: < 0.5s
```

### 3.2 Use POD types for TLS dispatch keys
- [ ] Ensure LocalDispatchKeySet uses zero-initialized POD (already done in c10)
- [ ] Audit Python-side TLS for similar patterns

**Test:**
```python
# Verify no expensive constructors run on import
import torch
# Check that dispatch key initialization is lazy
```

### 3.3 Defer FX graph pass registration
- [ ] Register inductor passes lazily instead of at module import
- [ ] Use decorator pattern that registers on first compile

**Test:**
```python
import torch._inductor
# Passes should not be instantiated until first compile
@torch.compile
def f(x): return x + 1
f(torch.randn(10))  # Now passes are registered
```

---

## Phase 4: Caching Improvements (Abseil Pattern: Avoid Repeated Work)

### 4.1 FakeTensor metadata caching
- [ ] Extend FakeTensor cache to cover more graph patterns
- [ ] Cache intermediate AOT autograd results

**Test:**
```python
import torch
@torch.compile
def f(x): return x.sin().cos().tan()

# First compile
f(torch.randn(100))

# Second compile with same shape should be faster
import time
start = time.time()
f(torch.randn(100))
print(f"Warm compile: {time.time() - start:.3f}s")  # Target: < 0.1s
```

### 4.2 Guard expression caching
- [ ] Cache compiled guard check functions
- [ ] Avoid re-compiling same guard expressions

**Test:**
```python
# Run same model twice, second should have cached guards
model = torch.nn.Linear(10, 10)
compiled = torch.compile(model)
compiled(torch.randn(5, 10))
# Guards should be cached for second call with same shape
compiled(torch.randn(5, 10))
```

### 4.3 Subprocess pool pre-warming
- [ ] Call `maybe_warm_pool()` earlier in compilation pipeline
- [ ] Make pool warming async/background

**Test:**
```python
import torch._inductor.async_compile as ac
# Pool should be warmed without blocking
ac.maybe_warm_pool()
# Verify pool is ready
```

---

## Phase 5: Reduce AOT Autograd Overhead

### 5.1 Merge AOT dispatcher passes
- [ ] Combine forward and backward tracing into single pass where possible
- [ ] Reduce FakeTensor propagation overhead

**Test:**
```python
import torch
from torch._dynamo.utils import CompileProfiler

with CompileProfiler() as prof:
    @torch.compile
    def f(x):
        return x.sin().sum()
    f(torch.randn(100, requires_grad=True))

# Check AOT time is reduced
print(prof.report())
```

### 5.2 Optimize sympy expression handling
- [ ] Cache simplified symbolic expressions
- [ ] Use faster sympy operations for common patterns

**Test:**
```python
import torch
# Dynamic shapes should not cause sympy explosion
@torch.compile(dynamic=True)
def f(x): return x.sum()
f(torch.randn(100))
f(torch.randn(200))  # Different shape, should reuse cached sympy work
```

---

## Phase 6: Autotuning Optimization

### 6.1 Defer autotuning to warm runs
- [ ] Use heuristic-based kernel selection for first run
- [ ] Run autotuning in background after first execution

**Test:**
```python
import torch
torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_gemm_backends = "ATen"  # Faster default

@torch.compile
def f(x, y): return x @ y
# First run should use heuristics, not autotune
start = time.time()
f(torch.randn(1000, 1000), torch.randn(1000, 1000))
print(f"First run: {time.time() - start:.2f}s")  # Should be fast
```

### 6.2 Integrate torch-diode for ML-based tuning
- [ ] Replace coordinate descent with ML model predictions
- [ ] Target ~500ms instead of seconds for autotuning

**Test:**
```python
# With torch-diode, autotuning should be <1s even for complex models
import torch
torch._inductor.config.use_diode_autotuner = True  # hypothetical config
```

---

## Phase 7: Import Graph Optimization

### 7.1 Reduce transitive imports
- [ ] Audit import graph for torch._dynamo
- [ ] Break circular imports that force eager loading

**Test:**
```bash
# Count number of modules loaded
python -c "import sys; import torch._dynamo; print(len(sys.modules))"
# Target: reduce by 20%
```

### 7.2 Lazy load optional backends
- [ ] Only import ONNX backend when explicitly requested
- [ ] Defer cudagraphs import until cuda compilation

**Test:**
```python
import torch._dynamo
# Optional backends should not be imported
assert 'torch._dynamo.backends.onnxrt' not in sys.modules
```

---

## Phase 8: Hierarchical Compilation

### 8.1 Enable regional compilation for repeated structures
- [ ] Use InvokeSubgraph HOP for transformer layers
- [ ] Cache compiled subgraphs for reuse

**Test:**
```python
import torch
import torch.nn as nn

class TransformerBlock(nn.Module):
    # ... standard transformer block
    pass

model = nn.Sequential(*[TransformerBlock() for _ in range(12)])
compiled = torch.compile(model)

# Should compile one block and reuse for others
start = time.time()
compiled(torch.randn(1, 100, 512))
print(f"12-layer compile: {time.time() - start:.2f}s")
# Target: ~1.5x single layer time, not 12x
```

---

## Summary Metrics

| Metric | Current | Target | Test Command |
|--------|---------|--------|--------------|
| Cold import time | ~2s | <1s | `python -X importtime -c "import torch._dynamo"` |
| Simple function compile | ~3s | <1s | `time python -c "import torch; torch.compile(lambda x: x+1)(torch.randn(10))"` |
| ResNet50 first compile | ~60s | <20s | Benchmark script |
| Transformer layer compile | ~10s | <5s | Benchmark script |
| Warm compile (cached) | ~0.5s | <0.1s | Second run of same model |

---

## Priority Order

1. **Phase 1**: Measurement (required for all other phases)
2. **Phase 2.1-2.2**: Lazy sympy/triton imports (high impact, low risk)
3. **Phase 4.1**: FakeTensor caching improvements (already in progress)
4. **Phase 5.1**: AOT pass merging (high impact, medium complexity)
5. **Phase 6.1**: Deferred autotuning (high impact for max-autotune mode)
6. **Phase 3**: Global constructor cleanup (medium impact, low risk)
7. **Phase 7**: Import graph optimization (medium impact, high complexity)
8. **Phase 8**: Hierarchical compilation (long-term, highest impact)
