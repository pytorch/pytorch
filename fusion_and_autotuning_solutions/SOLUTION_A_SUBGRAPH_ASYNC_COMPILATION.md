# 方案 A: 修复 SubgraphChoiceCaller 异步编译

**Date**: 2025-11-07
**Status**: Implementation Ready
**Complexity**: High
**Timeline**: 4-6 weeks

---

## Executive Summary

**目标**: 给 `SubgraphChoiceCaller` 添加 `precompile()` 方法，使其能够参与并行预编译流程，消除 async compilation bottleneck。

**核心问题**:
- SubgraphChoiceCaller 缺少 `precompile()` 方法
- 无法参与 ThreadPoolExecutor 并行预编译
- 导致 benchmark 阶段串行编译，成为性能瓶颈

**预期收益**:
- ✅ 编译时间减少 30-40%
- ✅ 保持完整的异步编译架构
- ✅ 无需 scheduler 改动
- ✅ 向后兼容

---

## 目录

1. [问题分析](#1-问题分析)
2. [解决方案设计](#2-解决方案设计)
3. [实现细节](#3-实现细节)
4. [测试方案](#4-测试方案)
5. [风险评估](#5-风险评估)
6. [实施路线图](#6-实施路线图)
7. [验收标准](#7-验收标准)

---

## 1. 问题分析

### 1.1 当前状态

**文件**: `/torch/_inductor/codegen/subgraph.py` (Lines 43-167)

```python
class SubgraphChoiceCaller(ir.ChoiceCaller):
    """Represents a subgraph autotuning choice."""

    def __init__(self, gm, input_nodes, layout, ...):
        self.gm = gm  # FX GraphModule
        self.original_inputs = input_nodes
        # ...

    def benchmark(self, *args, out):
        """Compile and benchmark subgraph."""
        # ❌ PROBLEM: Compilation happens HERE (synchronous)
        bm_graph_lowering = GraphLowering(
            self.gm,
            example_inputs=args,
        )
        mod = bm_graph_lowering.compile_to_module()  # SYNC!

        return benchmarker.benchmark(...)

    # ❌ MISSING: def precompile(self): ...
```

### 1.2 性能影响

**Benchmark 场景**: tuned_mm with decompose_k + 3 Triton configs

```
Without precompile():
────────────────────────────────────────────
Thread 1: [Triton 1: 500ms] → [Benchmark: 100ms]
Thread 2: [Triton 2: 500ms] → [Benchmark: 100ms]
Thread 3: [Triton 3: 500ms] → [Benchmark: 100ms]
Main:     [Wait: 500ms]      → [Subgraph compile: 500ms + Benchmark: 100ms]
────────────────────────────────────────────
Total: 1100ms (500ms parallel + 600ms serial)

With precompile():
────────────────────────────────────────────
Thread 1: [Triton 1: 500ms] → [Benchmark: 100ms]
Thread 2: [Triton 2: 500ms] → [Benchmark: 100ms]
Thread 3: [Triton 3: 500ms] → [Benchmark: 100ms]
Thread 4: [Subgraph: 500ms] → [Benchmark: 100ms]
────────────────────────────────────────────
Total: 600ms (all parallel)

Speedup: 1.83x (45% faster)
```

### 1.3 Root Cause

**文件**: `/torch/_inductor/select_algorithm.py` (Lines 3112-3128)

```python
for c in choices:
    if hasattr(c, "precompile"):  # ← Gate
        # Submit to thread/process pool
        future = executor.submit(precompile_with_captured_stdout, c)
        futures[future] = c
    else:
        # ❌ SubgraphChoiceCaller skipped!
        pass
```

**结论**: 没有 `precompile()` → 不进入并行编译流程 → 串行 bottleneck

---

## 2. 解决方案设计

### 2.1 架构设计

```
┌─────────────────────────────────────────────────────────┐
│  SubgraphChoiceCaller (Enhanced)                        │
│  ────────────────────────────────────────────────────   │
│                                                          │
│  + _compiled_module: Optional[Module] = None           │
│  + _precompile_lock: threading.Lock                    │
│  + _precompile_done: bool = False                      │
│                                                          │
│  + precompile() → None                                 │
│    ├─ Generate fake inputs                             │
│    ├─ Create GraphLowering                             │
│    ├─ Compile to module (cache result)                 │
│    └─ Thread-safe                                      │
│                                                          │
│  + benchmark(*args, out) → float                       │
│    ├─ Check _compiled_module                           │
│    ├─ Use cached if available                          │
│    └─ Fallback to on-demand compile                    │
│                                                          │
│  + _generate_fake_inputs() → List[FakeTensor]          │
│    └─ Create fake tensors from original_inputs         │
└─────────────────────────────────────────────────────────┘
```

### 2.2 关键设计决策

| 决策 | 理由 |
|------|------|
| **Cache compiled module** | 避免重复编译，benchmark 直接使用 |
| **Thread-safe lock** | ThreadPoolExecutor 可能并发调用 precompile() |
| **Fake tensor mode** | Precompile 不需要真实数据，使用 fake tensors |
| **Graceful fallback** | Precompile 失败时 benchmark 仍可工作 |
| **Preserve hash_key()** | 保持 cache 兼容性，不破坏现有逻辑 |

### 2.3 数据流

```
make_precompile_fn()
    │
    ├─ For SubgraphChoiceCaller:
    │  └─ executor.submit(choice.precompile)  ← NEW!
    │     └─ Parallel thread execution
    │        ├─ Generate fake inputs
    │        ├─ GraphLowering(gm, fake_inputs)
    │        ├─ compile_to_module()
    │        └─ Cache in _compiled_module
    │
    └─ wait_on_futures()
       └─ All precompiles complete (including subgraph)

benchmark_fn()
    │
    └─ SubgraphChoiceCaller.benchmark()
       ├─ Check _compiled_module
       ├─ Use cached module (fast path) ← NEW!
       └─ Or compile on-demand (fallback)
```

---

## 3. 实现细节

### 3.1 核心实现

**文件**: `/torch/_inductor/codegen/subgraph.py`

```python
import threading
from typing import Optional
from torch._subclasses import FakeTensorMode
from torch._inductor.graph import GraphLowering

class SubgraphChoiceCaller(ir.ChoiceCaller):
    def __init__(
        self,
        gm: torch.fx.GraphModule,
        input_nodes: Sequence[ir.IRNode],
        layout: ir.Layout,
        **kwargs,
    ):
        super().__init__(...)
        self.gm = gm
        self.original_inputs = input_nodes

        # ═══════════════════════════════════════════════════════
        # NEW: Precompilation support
        # ═══════════════════════════════════════════════════════
        self._compiled_module: Optional[Any] = None
        self._precompile_lock = threading.Lock()
        self._precompile_done = False
        self._precompile_error: Optional[Exception] = None

    def _generate_fake_inputs(self) -> List[torch.Tensor]:
        """
        Generate fake tensor inputs for precompilation.

        Returns:
            List of FakeTensor matching original_inputs shapes/dtypes
        """
        fake_inputs = []

        for inp in self.original_inputs:
            # Extract metadata from IRNode
            size = inp.get_size()
            stride = inp.get_stride()
            dtype = inp.get_dtype()
            device = inp.get_device()

            # Create fake tensor
            fake_tensor = torch.empty_strided(
                size, stride,
                dtype=dtype,
                device=device
            )
            fake_inputs.append(fake_tensor)

        return fake_inputs

    def precompile(self) -> None:
        """
        Precompile subgraph for async compilation.

        This method is called by ThreadPoolExecutor during the
        precompilation phase. It compiles the FX graph to a module
        and caches the result for later benchmarking.

        Thread-safe: Multiple threads may call this concurrently.
        """
        # Fast path: Already precompiled
        if self._precompile_done:
            return

        # Thread-safe precompilation
        with self._precompile_lock:
            # Double-check after acquiring lock
            if self._precompile_done:
                return

            try:
                log.debug(
                    "Precompiling subgraph: %s (hash=%s)",
                    self.name,
                    self.kernel_hash_key(),
                )

                # Generate fake inputs for compilation
                fake_inputs = self._generate_fake_inputs()

                # Create GraphLowering with fake mode
                fake_mode = FakeTensorMode()
                with V.set_fake_mode(fake_mode):
                    bm_graph_lowering = GraphLowering(
                        self.gm,
                        example_inputs=fake_inputs,
                        shape_env=None,  # Use default shape env
                        num_static_inputs=len(fake_inputs),
                        cpp_wrapper=False,
                    )

                    # Compile to module (expensive operation)
                    self._compiled_module = bm_graph_lowering.compile_to_module()

                self._precompile_done = True
                log.debug(
                    "Precompilation complete: %s",
                    self.name,
                )

            except Exception as e:
                # Record error but don't fail
                self._precompile_error = e
                self._precompile_done = False
                log.warning(
                    "Precompile failed for %s: %s (will fallback to on-demand compile)",
                    self.name,
                    e,
                    exc_info=True,
                )

    def benchmark(self, *args, out) -> float:
        """
        Benchmark subgraph with actual inputs.

        Uses pre-compiled module if available, otherwise compiles on-demand.

        Args:
            *args: Actual tensor inputs
            out: Output tensor

        Returns:
            Benchmark time in milliseconds
        """
        # Fast path: Use pre-compiled module
        if self._compiled_module is not None:
            log.debug(
                "Using pre-compiled module for benchmark: %s",
                self.name,
            )
            mod = self._compiled_module
        else:
            # Fallback: Compile on-demand
            if self._precompile_error is not None:
                log.debug(
                    "Precompile failed, compiling on-demand: %s",
                    self.name,
                )
            else:
                log.debug(
                    "No pre-compiled module, compiling on-demand: %s",
                    self.name,
                )

            # Standard compilation path
            bm_graph_lowering = GraphLowering(
                self.gm,
                example_inputs=list(args),
            )
            mod = bm_graph_lowering.compile_to_module()

        # Benchmark execution
        benchmarker = get_benchmarker()

        # Call the compiled function
        def call_subgraph():
            return mod.run(*args)

        # Measure time
        ms = benchmarker.benchmark(call_subgraph, (), {})

        return ms

    def hash_key(self) -> str:
        """
        Compute hash key for caching.

        Preserves existing behavior for cache compatibility.
        """
        # Existing implementation (unchanged)
        return f"subgraph_{self.name}_{hash(self.gm.code)}"
```

### 3.2 关键实现点

#### 3.2.1 Fake Tensor Generation

```python
def _generate_fake_inputs(self) -> List[torch.Tensor]:
    """Why fake tensors?"""
    # Precompilation doesn't need real data
    # Only needs shapes, dtypes, strides for compilation
    # Much faster than creating real tensors

    fake_inputs = []
    for inp in self.original_inputs:
        fake_tensor = torch.empty_strided(
            inp.get_size(),      # Shape: [M, K]
            inp.get_stride(),    # Stride: [K, 1] or [1, M]
            dtype=inp.get_dtype(),  # torch.float16, etc.
            device=inp.get_device(), # 'cuda:0', etc.
        )
        fake_inputs.append(fake_tensor)

    return fake_inputs
```

#### 3.2.2 Thread Safety

```python
def precompile(self) -> None:
    # Thread-safe double-check locking pattern
    if self._precompile_done:  # Fast check (no lock)
        return

    with self._precompile_lock:  # Acquire lock
        if self._precompile_done:  # Double-check
            return

        # Critical section: compile and cache
        self._compiled_module = ...
        self._precompile_done = True
```

**为什么需要线程安全？**
- ThreadPoolExecutor 可能并发调用同一个 choice
- 防止重复编译浪费资源
- 确保 cache 一致性

#### 3.2.3 Error Handling

```python
try:
    self._compiled_module = compile(...)
    self._precompile_done = True
except Exception as e:
    # Record error but don't crash
    self._precompile_error = e
    self._precompile_done = False
    log.warning("Precompile failed, will fallback")

# Later in benchmark():
if self._compiled_module is not None:
    mod = self._compiled_module  # Use cache
else:
    mod = compile_on_demand()  # Graceful fallback
```

**设计理念**: Precompile 是优化，失败不应影响 correctness

---

## 4. 测试方案

### 4.1 单元测试

**文件**: `test/inductor/test_subgraph_async_compile.py`

```python
import torch
import unittest
from torch._inductor.codegen.subgraph import SubgraphChoiceCaller
from torch._inductor.select_algorithm import autotune_select_algorithm
from concurrent.futures import ThreadPoolExecutor

class TestSubgraphAsyncCompilation(unittest.TestCase):
    """Test SubgraphChoiceCaller.precompile() functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    def test_precompile_basic(self):
        """Test basic precompile() functionality."""
        # Create a simple subgraph
        def simple_graph(a, b):
            return a + b

        gm = torch.fx.symbolic_trace(simple_graph)

        # Create SubgraphChoiceCaller
        input_nodes = [...]  # Mock IRNode objects
        layout = ...  # Mock Layout

        caller = SubgraphChoiceCaller(gm, input_nodes, layout)

        # Test precompile()
        caller.precompile()

        # Verify compiled module cached
        self.assertIsNotNone(caller._compiled_module)
        self.assertTrue(caller._precompile_done)

    def test_precompile_thread_safety(self):
        """Test thread-safe precompile()."""
        caller = SubgraphChoiceCaller(...)

        # Call precompile() from multiple threads
        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = [executor.submit(caller.precompile) for _ in range(10)]
            for f in futures:
                f.result()

        # Should only compile once
        self.assertTrue(caller._precompile_done)
        self.assertIsNotNone(caller._compiled_module)

    def test_benchmark_uses_cached_module(self):
        """Test benchmark() uses pre-compiled module."""
        caller = SubgraphChoiceCaller(...)

        # Precompile
        caller.precompile()
        cached_module = caller._compiled_module

        # Benchmark should use cached module
        a = torch.randn(10, 10, device=self.device)
        b = torch.randn(10, 10, device=self.device)
        out = torch.empty_like(a)

        ms = caller.benchmark(a, b, out=out)

        # Verify same module used
        self.assertIs(caller._compiled_module, cached_module)
        self.assertGreater(ms, 0)

    def test_benchmark_fallback_on_precompile_failure(self):
        """Test graceful fallback when precompile fails."""
        caller = SubgraphChoiceCaller(...)

        # Simulate precompile failure
        caller._precompile_error = RuntimeError("Mock error")
        caller._precompile_done = False
        caller._compiled_module = None

        # Benchmark should still work (compile on-demand)
        a = torch.randn(10, 10, device=self.device)
        b = torch.randn(10, 10, device=self.device)
        out = torch.empty_like(a)

        ms = caller.benchmark(a, b, out=out)

        # Should succeed despite precompile failure
        self.assertGreater(ms, 0)

    def test_async_compilation_integration(self):
        """Test integration with make_precompile_fn()."""
        from torch._inductor.select_algorithm import AlgorithmSelectorCache

        # Create choices including SubgraphChoiceCaller
        choices = [
            TritonTemplateCaller(...),
            SubgraphChoiceCaller(...),  # Should participate in async compile
            ExternKernelCaller(...),
        ]

        # Create precompile function
        selector = AlgorithmSelectorCache()
        precompile_fn = selector.make_precompile_fn(
            choices,
            name="test",
            inputs_key="test_key",
        )

        # Execute precompilation
        precompile_fn()

        # Verify SubgraphChoiceCaller was precompiled
        subgraph_caller = choices[1]
        self.assertTrue(subgraph_caller._precompile_done)
        self.assertIsNotNone(subgraph_caller._compiled_module)

if __name__ == '__main__':
    unittest.main()
```

### 4.2 集成测试

**文件**: `test/inductor/test_decompose_k_async.py`

```python
import torch
from torch._inductor.kernel.mm import decompose_k_subgraph_template
from torch.testing._internal.inductor_utils import HAS_CUDA

@unittest.skipIf(not HAS_CUDA, "CUDA required")
class TestDecomposeKAsyncCompilation(unittest.TestCase):
    """Test decompose_k with async compilation."""

    def test_decompose_k_parallel_precompile(self):
        """Test decompose_k choices are precompiled in parallel."""
        # Enable max_autotune and parallel compilation
        with torch._inductor.config.patch(
            max_autotune=True,
            max_autotune_gemm_threads=4,
        ):
            @torch.compile
            def matmul(a, b):
                return a @ b

            a = torch.randn(1024, 8192, device='cuda', dtype=torch.float16)
            b = torch.randn(8192, 2048, device='cuda', dtype=torch.float16)

            # First run: precompile + benchmark
            import time
            start = time.time()
            result = matmul(a, b)
            compile_time = time.time() - start

            # Verify decompose_k was used
            # (check generated code or logs)

            # Verify parallel speedup
            # With async: ~2-3x faster than serial
            self.assertLess(compile_time, 10.0)  # Should be fast

    def test_decompose_k_with_fusion(self):
        """Test decompose_k + relu fusion with async compile."""
        with torch._inductor.config.patch(
            max_autotune=True,
            max_autotune_gemm_threads=4,
        ):
            @torch.compile
            def matmul_relu(a, b):
                return (a @ b).relu()

            a = torch.randn(1024, 8192, device='cuda', dtype=torch.float16)
            b = torch.randn(8192, 2048, device='cuda', dtype=torch.float16)

            result = matmul_relu(a, b)

            # Verify fusion happened
            # (check for fused kernel in generated code)
```

### 4.3 性能测试

**文件**: `benchmarks/inductor/bench_subgraph_async.py`

```python
import torch
import time
from torch._inductor.kernel.mm import decompose_k_subgraph_template

def benchmark_compilation_time(enable_async: bool):
    """Benchmark compilation time with/without async."""
    config = {
        'max_autotune': True,
        'max_autotune_gemm_threads': 8 if enable_async else 1,
    }

    with torch._inductor.config.patch(**config):
        @torch.compile
        def matmul(a, b):
            return a @ b

        a = torch.randn(1024, 8192, device='cuda', dtype=torch.float16)
        b = torch.randn(8192, 2048, device='cuda', dtype=torch.float16)

        start = time.time()
        result = matmul(a, b)
        compile_time = time.time() - start

    return compile_time

# Benchmark
serial_time = benchmark_compilation_time(enable_async=False)
async_time = benchmark_compilation_time(enable_async=True)

print(f"Serial compilation: {serial_time:.2f}s")
print(f"Async compilation: {async_time:.2f}s")
print(f"Speedup: {serial_time / async_time:.2f}x")
```

**Expected Results**:
- Serial: ~10-12s
- Async: ~6-7s
- Speedup: ~1.5-2x

---

## 5. 风险评估

### 5.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **Fake tensor 不兼容** | 中 | 高 | Fallback to real tensors if fake mode fails |
| **Cache coherency 问题** | 低 | 高 | Thread-safe locking, extensive testing |
| **编译失败率增加** | 低 | 中 | Graceful fallback, error logging |
| **内存泄漏** | 低 | 中 | Clear cache after benchmark, memory profiling |

### 5.2 性能风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **Precompile 开销过大** | 低 | 中 | Profile and optimize fake tensor generation |
| **Cache miss 导致重复编译** | 中 | 低 | Improve hash_key() uniqueness |
| **并行度不够** | 低 | 低 | Tune max_workers based on CPU cores |

### 5.3 兼容性风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **破坏现有 cache** | 低 | 中 | Preserve hash_key() behavior |
| **与其他 features 冲突** | 低 | 中 | Integration testing with AOT, dynamic shapes |
| **Backward compatibility** | 低 | 低 | Precompile is optional optimization |

---

## 6. 实施路线图

### Phase 1: Core Implementation (Week 1-2)

**Week 1**: Basic precompile() implementation
```
Day 1-2: Implement _generate_fake_inputs()
Day 3-4: Implement precompile() with caching
Day 5: Implement thread-safe locking
```

**Week 2**: Error handling and fallback
```
Day 1-2: Add graceful fallback in benchmark()
Day 3-4: Implement error logging and diagnostics
Day 5: Code review and refactoring
```

### Phase 2: Testing (Week 3-4)

**Week 3**: Unit and integration tests
```
Day 1-2: Write unit tests for precompile()
Day 3-4: Write integration tests with tuned_mm
Day 5: Write decompose_k specific tests
```

**Week 4**: Performance testing
```
Day 1-2: Compilation time benchmarks
Day 3-4: Memory profiling
Day 5: Thread pool scalability tests
```

### Phase 3: Validation (Week 5)

```
Day 1-2: Run on vLLM workloads
Day 3-4: Run PyTorch CI test suite
Day 5: Address any failures
```

### Phase 4: Deployment (Week 6)

```
Day 1-2: Code review and approval
Day 3: Documentation updates
Day 4: Merge to master
Day 5: Monitor for regressions
```

---

## 7. 验收标准

### 7.1 功能验收

- ✅ SubgraphChoiceCaller has `precompile()` method
- ✅ Precompile uses fake tensors successfully
- ✅ Compiled module cached correctly
- ✅ Benchmark uses cached module
- ✅ Graceful fallback on precompile failure
- ✅ Thread-safe concurrent precompile

### 7.2 性能验收

- ✅ Compilation time reduced by ≥30%
- ✅ Parallel speedup scales with num_workers
- ✅ No memory leaks in precompile/benchmark cycle
- ✅ Cache hit rate ≥90% for repeated compilations

### 7.3 质量验收

- ✅ All unit tests pass
- ✅ All integration tests pass
- ✅ PyTorch CI test suite passes
- ✅ No regressions in existing benchmarks
- ✅ Code coverage ≥85% for new code

### 7.4 文档验收

- ✅ Docstrings for all new methods
- ✅ Implementation guide updated
- ✅ Performance benchmarks documented
- ✅ Known limitations documented

---

## 附录

### A. 相关文件清单

| 文件 | 改动类型 | 行数估计 |
|-----|---------|---------|
| `subgraph.py` | 修改 | +150 |
| `test_subgraph_async_compile.py` | 新增 | +200 |
| `test_decompose_k_async.py` | 新增 | +100 |
| `bench_subgraph_async.py` | 新增 | +80 |

**Total**: ~530 lines of code

### B. 依赖关系

```
SubgraphChoiceCaller.precompile()
    ├─ Depends on: FakeTensorMode
    ├─ Depends on: GraphLowering
    ├─ Depends on: ThreadPoolExecutor (select_algorithm.py)
    └─ Used by: make_precompile_fn()
```

### C. 参考实现

**TritonTemplateCaller.precompile()** (select_algorithm.py:2257-2259):
```python
def precompile(self):
    assert self.bmreq is not None
    self.bmreq.precompile()
```

**ExternKernelCaller.precompile()** (select_algorithm.py:2394):
```python
def precompile(self):
    pass  # No-op for extern kernels
```

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Author**: Collective Op Autotuning Team
**Reviewers**: TBD
**Status**: Ready for Implementation
