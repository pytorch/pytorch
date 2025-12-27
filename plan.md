# torch.compile Cold Compile Time Optimization Plan

Inspired by Abseil's performance tips (lazy initialization, avoiding global state, deferring work until needed), this plan identifies opportunities to reduce cold compile times in torch.compile.

## Priority 1: Lazy Import Optimization

### 1.1 Defer polyfill loading until first use
**File:** `torch/_dynamo/polyfills/loader.py` (lines 34-47)

Currently, all 12+ polyfill modules are imported and registered at `torch._dynamo` import time. This adds overhead even when torch.compile is never called.

**TODO:**
- [x] Wrap polyfill loading in a `@functools.cache` decorated function
- [x] Call the loader lazily on first `torch.compile()` invocation instead of at import time
- [x] Move the `trace_rules._builtin_function_ids.remove()` calls to lazy initialization

**Test:**
```python
import time
start = time.perf_counter()
import torch._dynamo
end = time.perf_counter()
print(f"torch._dynamo import time: {end - start:.3f}s")
# Before: ~X.XXs, After: should be measurably faster
```

---

### 1.2 Lazy load convert_frame heavy imports
**File:** `torch/_dynamo/convert_frame.py` (lines 24-173)

This file has 173+ symbols across 80+ import statements that execute on first compile.

**TODO:**
- [x] Identify imports only used in rare code paths (error handling, debugging)
- [x] Move rarely-used imports inside functions that use them
- [x] Group related imports and use lazy loading pattern from `compile_fx.py`

**Test:**
```python
import torch._dynamo.utils as utils
utils.counters.clear()

@torch.compile
def f(x): return x + 1

# Measure first compile time
import time
start = time.perf_counter()
f(torch.randn(10))
first_compile = time.perf_counter() - start

torch._dynamo.reset()
start = time.perf_counter()
f(torch.randn(10))
second_compile = time.perf_counter() - start

print(f"First compile: {first_compile:.3f}s, Second: {second_compile:.3f}s")
```

---

### 1.3 Extend lazy import pattern in compile_fx.py
**File:** `torch/_inductor/compile_fx.py` (lines 238-294)

Already has `_lazy_import_*` functions. Extend this pattern to more modules.

**TODO:**
- [x] Audit remaining top-level imports for candidates to make lazy
- [x] Convert `from torch._subclasses import FakeTensorMode` and similar to lazy
- [x] Add `@functools.cache` wrappers for expensive module imports

**Test:**
```python
import sys
before_modules = set(sys.modules.keys())

import torch
@torch.compile(backend="inductor")
def f(x): return x * 2

after_import = set(sys.modules.keys())
print(f"Modules loaded on import: {len(after_import - before_modules)}")

f(torch.randn(10))
after_compile = set(sys.modules.keys())
print(f"Modules loaded on first compile: {len(after_compile - after_import)}")
```

---

## Priority 2: Global State Initialization Deferral

### 2.1 Defer trace_rules dictionary construction
**File:** `torch/_dynamo/trace_rules.py` (lines 151-599)

The `manual_torch_name_rule_map` (200+ entries) and `torch_c_binding_in_graph_functions` (200+ entries) are built at import time.

**TODO:**
- [x] Wrap dictionary construction in `@functools.cache` decorated functions
- [x] Only build dictionaries on first access
- [x] Consider using `__getattr__` module-level lazy loading

**Test:**
```python
import time
start = time.perf_counter()
import torch._dynamo.trace_rules
end = time.perf_counter()
print(f"trace_rules import time: {end - start:.4f}s")
# Should be faster after deferral
```

---

### 2.2 Lazy initialize utils.py global counters
**File:** `torch/_dynamo/utils.py` (lines 164-187)

Global counters and metrics dicts are created at import time.

**TODO:**
- [x] Use lazy initialization for `counters`, `optimus_scuba_log`, `compilation_time_metrics`
- [x] Initialize these on first access rather than import

**Test:**
```python
import torch._dynamo.utils as utils
# Verify counters work after lazy init
utils.counters["test"]["count"] += 1
assert utils.counters["test"]["count"] == 1
```

---

## Priority 3: Cache Warm-up Optimization

### 3.1 Parallelize async compile pool warm-up
**File:** `torch/_inductor/async_compile.py` (lines 290-298, 724-745)

The `warm_pool()` and `maybe_warm_pool()` functions warm up compilation pools.

**TODO:**
- [x] Make pool warm-up truly async (non-blocking)
- [x] Start warm-up earlier in the compile pipeline
- [x] Consider background thread for pool initialization

**Test:**
```python
import torch
import time

# Measure with pool warming
torch._inductor.async_compile.warm_async_compile_pool()
start = time.perf_counter()
@torch.compile
def f(x): return x + 1
f(torch.randn(10).cuda())
warm_time = time.perf_counter() - start

# Reset and measure cold
torch._dynamo.reset()
import os
os.environ["TORCH_WARM_POOL"] = "0"
start = time.perf_counter()
@torch.compile
def g(x): return x + 1
g(torch.randn(10).cuda())
cold_time = time.perf_counter() - start

print(f"Warm pool: {warm_time:.3f}s, Cold pool: {cold_time:.3f}s")
```

---

### 3.2 Preload autotune cache more aggressively
**File:** `torch/_inductor/runtime/autotune_cache.py` (lines 41-72)

The `_preload_autotune_cache()` function loads cache files into memory.

**TODO:**
- [x] Call preload earlier (on first torch.compile, not first kernel compile)
- [x] Use background thread for cache preloading
- [ ] Add progress indicator for large cache directories

**Test:**
```python
import torch._inductor.runtime.autotune_cache as ac
import time

start = time.perf_counter()
ac._preload_autotune_cache()
load_time = time.perf_counter() - start
print(f"Autotune cache preload time: {load_time:.3f}s")
print(f"Cache entries loaded: {len(ac._autotune_cache_memory)}")
```

---

## Priority 4: Scheduler/Fusion Pass Optimization

### 4.1 Reduce fusion pass iterations
**File:** `torch/_inductor/scheduler.py` (lines 3597-3631)

The fusion loop runs up to 10 iterations by default.

**TODO:**
- [ ] Profile to determine if fewer iterations suffice for most models
- [ ] Add early termination when no fusions occur
- [ ] Consider making max iterations configurable

**Test:**
```python
import torch
import torch._dynamo.utils as utils

utils.counters.clear()
@torch.compile
def f(x):
    return x.sin().cos().exp().log()

f(torch.randn(1000, 1000).cuda())
print(f"Fusion iterations: {utils.counters}")
```

---

### 4.2 Optimize get_possible_fusions() pairwise comparisons
**File:** `torch/_inductor/scheduler.py` (lines 4327-4381)

This function has O(n�) complexity for node comparisons.

**TODO:**
- [ ] Add early-exit heuristics for nodes that can't fuse
- [ ] Use indexing structures to reduce pairwise comparisons
- [ ] Cache fusion compatibility results

**Test:**
```python
import torch
import torch._dynamo.utils as utils

# Large model with many nodes
model = torch.nn.Sequential(*[torch.nn.Linear(100, 100) for _ in range(20)])
model = model.cuda()

utils.counters.clear()
compiled = torch.compile(model)
compiled(torch.randn(32, 100).cuda())

print(f"Scheduler metrics: {dict(utils.counters.get('inductor', {}))}")
```

---

## Priority 5: Codegen Initialization

### 5.1 Lazy load Triton helpers
**File:** `torch/_inductor/runtime/triton_helpers.py` (lines 23-65)

Triton driver initialization happens on first access.

**TODO:**
- [ ] Defer driver initialization until actually needed
- [ ] Cache device properties computation
- [ ] Parallelize GPU property queries for multi-GPU systems

**Test:**
```python
import time
start = time.perf_counter()
from torch._inductor.runtime import triton_helpers
import_time = time.perf_counter() - start
print(f"triton_helpers import time: {import_time:.4f}s")
```

---

### 5.2 Reduce CUTLASS operation fetching overhead
**File:** `torch/_inductor/codegen/cuda/cutlass_cache.py` (lines 64-100)

CUTLASS ops are fetched on first GEMM compilation.

**TODO:**
- [ ] Cache CUTLASS operations to local file after first fetch
- [ ] Make fetch async/non-blocking
- [ ] Add fallback timeout for network issues

**Test:**
```python
import torch
import torch._inductor.codegen.cuda.cutlass_cache as cc

# Clear cache and measure fetch time
cc.maybe_fetch_ops.cache_clear()
import time
start = time.perf_counter()
ops = cc.maybe_fetch_ops()
fetch_time = time.perf_counter() - start
print(f"CUTLASS ops fetch time: {fetch_time:.3f}s, ops count: {len(ops) if ops else 0}")
```

---

## Priority 6: Import-Time Code Execution

### 6.1 Defer dynamo __init__.py initialization
**File:** `torch/_dynamo/__init__.py` (lines 118-127)

Code runs at import time to modify torch.manual_seed.

**TODO:**
- [ ] Move manual_seed patching to first compile
- [ ] Use lazy initialization for serialization.add_safe_globals

**Test:**
```python
import time
start = time.perf_counter()
import torch._dynamo
import_time = time.perf_counter() - start
print(f"torch._dynamo import time: {import_time:.3f}s")
```

---

### 6.2 Profile and reduce TYPE_CHECKING import overhead
**Files:** Various files in `torch/_dynamo/` and `torch/_inductor/`

Some files use `if TYPE_CHECKING:` blocks inconsistently.

**TODO:**
- [ ] Audit files for heavy imports that should be in TYPE_CHECKING blocks
- [ ] Ensure type-only imports don't execute at runtime
- [ ] Add linting rule to enforce TYPE_CHECKING usage

**Test:**
```python
import sys
import typing

# Count imports that are only for typing
typing_only_imports = sum(1 for m in sys.modules if 'typing' in m.lower())
print(f"Typing-related modules loaded: {typing_only_imports}")
```

---

## Verification: Overall Cold Compile Time Benchmark

```python
import subprocess
import time
import statistics

def measure_cold_compile():
    """Measure cold compile time in fresh process."""
    code = '''
import time
import torch

start = time.perf_counter()

@torch.compile
def f(x):
    return x.sin().cos().exp()

x = torch.randn(1000, 1000).cuda()
f(x)
torch.cuda.synchronize()

end = time.perf_counter()
print(f"{end - start:.3f}")
'''
    result = subprocess.run(
        ['python', '-c', code],
        capture_output=True,
        text=True,
        env={**os.environ, 'TORCHINDUCTOR_FORCE_DISABLE_CACHES': '1'}
    )
    return float(result.stdout.strip())

# Run multiple times and report statistics
times = [measure_cold_compile() for _ in range(5)]
print(f"Cold compile times: {times}")
print(f"Mean: {statistics.mean(times):.3f}s, Stdev: {statistics.stdev(times):.3f}s")
```

---

## Profiling Commands

```bash
# Profile with cProfile
TORCH_COMPILE_CPROFILE=1 python your_script.py

# Profile with strobelight
TORCH_COMPILE_STROBELIGHT=TRUE python your_script.py

# Get compile time breakdown
TORCH_COMPILE_PROFILE=1 python your_script.py

# Disable all caches for true cold start measurement
TORCHINDUCTOR_FORCE_DISABLE_CACHES=1 python your_script.py
```

---

## Success Metrics

| Metric | Current Baseline | Target |
|--------|-----------------|--------|
| torch._dynamo import time | TBD | -20% |
| First compile time (simple function) | TBD | -15% |
| First compile time (ResNet-50) | TBD | -10% |
| Modules loaded on first compile | TBD | -10% |
