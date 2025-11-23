# 为什么 precompile() 是异步编译的关键

**Date**: 2025-11-07
**Purpose**: 理解 precompile() 方法在 async compilation 中的核心作用

---

## TL;DR - 直接回答

### Q: precompile() 是唯一激活异步 compile 的 function 吗？

**A: 是的！** precompile() 是**进入异步编译流程的唯一入口**。

**关键机制**：
```python
# select_algorithm.py:3112-3128
if hasattr(c, "precompile"):  # ← KEY GATE
    # HAS precompile() → 并行编译路径
    if triton_cuda_choice and async_compile.use_process_pool():
        future = async_compile.triton(...)  # Process pool
    else:
        future = executor.submit(precompile_with_captured_stdout, c)  # Thread pool
    futures[c] = future  # 收集 Future 对象
else:
    # NO precompile() → 跳过，后续同步编译
    pass  # 这个 choice 不会被预编译！
```

**结果**：
- ✅ **有 precompile()**: 异步并行编译（ThreadPoolExecutor / ProcessPoolExecutor）
- ❌ **无 precompile()**: 跳过预编译，benchmark 时同步编译（串行 bottleneck）

---

## 详细解释

### 1. Async Compilation 完整流程

```
┌─────────────────────────────────────────────────────────┐
│  autotune_select_algorithm()                            │
│  ├─ choices = [TritonCaller, ExternCaller, SubgraphCaller]
│  └─ do_autotuning(choices, ...)                        │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│  do_autotuning()                                        │
│  ├─ precompile_fn = make_precompile_fn(choices)        │
│  ├─ precompile_fn()  ← 并行预编译所有 choices          │
│  └─ benchmark(choices)  ← 所有 choices 已预编译        │
└──────────────┬──────────────────────────────────────────┘
               │
               ▼
┌─────────────────────────────────────────────────────────┐
│  make_precompile_fn(choices) ← KEY FUNCTION            │
│                                                          │
│  1. Create ThreadPoolExecutor(max_workers=N)           │
│  2. Create AsyncCompile()                              │
│  3. For each choice:                                    │
│     if hasattr(choice, "precompile"):  ← GATE          │
│        └─ Submit to executor/process pool              │
│     else:                                               │
│        └─ SKIP (no precompile, will compile later)     │
│                                                          │
│  4. Return wait_on_futures() function                  │
└─────────────────────────────────────────────────────────┘
```

### 2. 关键代码分析

**File**: `/torch/_inductor/select_algorithm.py` (Lines 3112-3128)

```python
# Iterate through all choices
for c in choices:
    # Skip duplicate choices
    if c.kernel_hash_key() in seen_choices:
        continue
    seen_choices.add(c.kernel_hash_key())

    # ═══════════════════════════════════════════════════════
    # KEY GATE: Check if choice has precompile() method
    # ═══════════════════════════════════════════════════════
    if hasattr(c, "precompile"):

        # Determine compilation path
        triton_cuda_choice = (
            isinstance(c, TritonTemplateCaller) and
            isinstance(c.bmreq, TritonGPUBenchmarkRequest)
        )

        # PATH 1: Triton CUDA → Process Pool (fastest)
        if triton_cuda_choice and async_compile.use_process_pool():
            with open(c.bmreq.module_path) as file:
                source_code = file.read()
            future = async_compile.triton(
                kernel_name=c.bmreq.kernel_name,
                source_code=source_code
            ).future

        # PATH 2: Other choices → Thread Pool
        else:
            future = executor.submit(precompile_with_captured_stdout, c)

        # Register future for tracking
        future.add_done_callback(on_complete)
        futures[future] = c

    # ═══════════════════════════════════════════════════════
    # NO precompile() → SKIP! Will compile synchronously later
    # ═══════════════════════════════════════════════════════
```

### 3. precompile() 的作用

**对于 TritonTemplateCaller**:

```python
# select_algorithm.py:2257-2259
class TritonTemplateCaller(ir.TritonTemplateCallerBase):
    def precompile(self):
        assert self.bmreq is not None
        self.bmreq.precompile()  # Compile Triton kernel to LLVM IR
```

**TritonBenchmarkRequest.precompile()** (在 `autotune_process.py`):
```python
def precompile(self):
    """
    Pre-compile the Triton kernel without benchmarking.

    This compiles the kernel source code to:
    1. LLVM IR
    2. PTX (NVIDIA) or AMDGCN (AMD)
    3. Cached binary

    Result: Kernel is ready to run, no compilation needed during benchmark
    """
    if not self._precompiled:
        # Compile kernel
        self.kernel = triton.compile(self.kernel_code, ...)
        self._precompiled = True
```

**对于 ExternKernelCaller**:

```python
# select_algorithm.py:2394
class ExternKernelCaller(ChoiceCaller):
    def precompile(self):
        """No-op for extern kernels (already compiled)."""
        pass  # cuBLAS/MKL kernels are pre-compiled
```

**对于 SubgraphChoiceCaller** (MISSING!):

```python
# subgraph.py:43-167
class SubgraphChoiceCaller(ir.ChoiceCaller):
    # ❌ NO precompile() method
    #    Cannot participate in async compilation!

    def benchmark(self, *args, out):
        # Must compile synchronously during benchmark
        bm_graph_lowering = GraphLowering(...)
        mod = bm_graph_lowering.compile_to_module()  # SYNC compilation
        return benchmarker.benchmark(...)
```

### 4. 没有 precompile() 会怎样？

#### Scenario A: 有 precompile()

```
Time:    0ms          500ms         1000ms        1500ms
         │             │             │             │
Thread 1 │[Triton 1 precompile]    │[Triton 1 benchmark]
Thread 2 │[Triton 2 precompile]    │[Triton 2 benchmark]
Thread 3 │[Extern 1 precompile]    │[Extern 1 benchmark]
Thread 4 │[Subgraph precompile]    │[Subgraph benchmark]
         │                          │
         └─ All parallel            └─ All parallel
         └─ Total: 500ms            └─ Total: 1500ms
```

**Total compilation time**: ~500ms (parallel)
**Total benchmark time**: ~1000ms (parallel)
**Total end-to-end**: ~1500ms

#### Scenario B: 无 precompile() (SubgraphChoiceCaller)

```
Time:    0ms          500ms         1000ms        1500ms        2000ms
         │             │             │             │             │
Thread 1 │[Triton 1 precompile]    │[Triton 1 benchmark]       │
Thread 2 │[Triton 2 precompile]    │[Triton 2 benchmark]       │
Thread 3 │[Extern 1 precompile]    │[Extern 1 benchmark]       │
Main     │[Wait...................]│[Subgraph compile + benchmark]
         │                          │                           │
         └─ 3 parallel              └─ SERIAL BOTTLENECK       │
         └─ Total: 500ms            └─ Total: 2000ms
```

**Total compilation time**: ~500ms (parallel) + **500ms (serial)** = 1000ms
**Total benchmark time**: ~1000ms
**Total end-to-end**: ~2000ms (**33% slower**)

### 5. 为什么只能线性 compile？

**原因 1: 没有 Future 对象**

```python
# make_precompile_fn()
futures = {}  # Future 对象收集器

for choice in choices:
    if hasattr(choice, "precompile"):
        future = executor.submit(...)
        futures[future] = choice  # ← 收集 Future
    else:
        # NO future created!
        pass  # 这个 choice 没有 Future，无法追踪状态
```

**结果**: SubgraphChoiceCaller 没有 Future 对象，`wait_on_futures()` 无法等待它的编译完成。

**原因 2: benchmark() 时才编译**

```python
# do_autotuning() → benchmark()
def benchmark_fn(choices):
    for choice in choices:
        if choice has Future:
            # Use pre-compiled kernel
            ms = choice.benchmark(...)  # Fast (already compiled)
        else:
            # Must compile NOW (synchronous)
            ms = choice.benchmark(...)  # Slow (compile + benchmark)
            #        ↑
            #        └─ SubgraphChoiceCaller.benchmark() 里调用 compile_to_module()
            #           这是 SYNCHRONOUS 的，阻塞主线程！
```

**原因 3: 无法并行化 subgraph 编译**

SubgraphChoiceCaller 的编译发生在 `benchmark()` 调用中：

```python
def benchmark(self, *args, out):
    # Create GraphLowering
    bm_graph_lowering = GraphLowering(self.gm, ...)

    # ═══════════════════════════════════════════════════════
    # SYNCHRONOUS COMPILATION (blocks thread)
    # ═══════════════════════════════════════════════════════
    mod = bm_graph_lowering.compile_to_module()

    # Benchmark
    return benchmarker.benchmark(...)
```

这个编译是**同步的**，发生在 benchmark 阶段，无法提前并行化。

---

## 关键设计原理

### 为什么要 precompile()?

**设计目标**: **分离编译和测量**

1. **Precompilation Phase** (可并行):
   - 编译所有 kernel choices
   - CPU 密集，可以并行
   - 使用 ThreadPoolExecutor/ProcessPoolExecutor
   - 不运行 kernel，只编译

2. **Benchmarking Phase** (需串行):
   - 运行已编译的 kernels
   - GPU 密集，需要 GPU 独占
   - 测量精确时间
   - 编译开销已消除

**Benefits**:
- ✅ **编译并行化**: N 个 worker 同时编译
- ✅ **Benchmark 精确**: 无编译开销影响
- ✅ **Cache 友好**: 编译结果可缓存
- ✅ **Scalability**: 随 worker 数量线性加速

### 为什么 SubgraphChoiceCaller 缺少 precompile()?

**历史原因**:

1. **实现复杂性**: Subgraph 编译需要 GraphLowering，比 Triton kernel 编译复杂
2. **Fake Tensor Mode**: 需要处理 fake tensors 生成 example inputs
3. **Cache Coherency**: Subgraph 编译结果缓存更复杂
4. **优先级**: Triton/cuBLAS choices 更常用，subgraph 是 edge case

**当前状态**: Subgraph 编译只能在 benchmark 时同步进行，无法并行预编译。

---

## 总结表

| Aspect | 有 precompile() | 无 precompile() |
|--------|----------------|----------------|
| **编译时机** | Precompilation phase (并行) | Benchmark phase (串行) |
| **并行化** | ✅ YES (ThreadPoolExecutor) | ❌ NO (主线程阻塞) |
| **Future 对象** | ✅ YES (可追踪状态) | ❌ NO (无法追踪) |
| **编译开销** | ✅ 分摊到并行阶段 | ❌ 全部在 benchmark 阶段 |
| **Benchmark 精度** | ✅ 高（无编译开销） | ⚠️ 低（包含编译时间） |
| **Scalability** | ✅ 随 worker 数线性加速 | ❌ 串行瓶颈 |
| **Examples** | TritonTemplateCaller, ExternKernelCaller | SubgraphChoiceCaller |

---

## 修复方案预览

### 给 SubgraphChoiceCaller 添加 precompile()

```python
# subgraph.py
class SubgraphChoiceCaller(ir.ChoiceCaller):
    def __init__(self, gm, input_nodes, ...):
        self.gm = gm
        self.original_inputs = input_nodes
        self._compiled_module = None  # Cache
        self._precompile_done = False

    def precompile(self):
        """Precompile subgraph for async compilation."""
        if self._precompile_done:
            return

        # Create fake inputs
        fake_inputs = self._generate_fake_inputs()

        # Compile subgraph
        bm_graph_lowering = GraphLowering(
            self.gm,
            example_inputs=fake_inputs,
        )

        # Cache compiled module
        self._compiled_module = bm_graph_lowering.compile_to_module()
        self._precompile_done = True

    def benchmark(self, *args, out):
        """Use cached compiled module."""
        if self._compiled_module is not None:
            mod = self._compiled_module  # Use cache
        else:
            # Fallback: compile on-demand
            mod = GraphLowering(...).compile_to_module()

        return benchmarker.benchmark(...)
```

**效果**:
- ✅ SubgraphChoiceCaller 可以参与并行预编译
- ✅ 消除 benchmark 阶段的编译瓶颈
- ✅ 总编译时间减少 ~33%

---

## 参考文献

### 关键文件

| 文件 | 行号 | 功能 |
|-----|------|------|
| `select_algorithm.py` | 3009-3180 | `make_precompile_fn()` - 并行预编译 orchestrator |
| `select_algorithm.py` | 3112-3128 | `hasattr(c, "precompile")` gate - 异步编译入口 |
| `select_algorithm.py` | 2257-2259 | `TritonTemplateCaller.precompile()` |
| `subgraph.py` | 43-167 | `SubgraphChoiceCaller` (缺少 precompile) |

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Author**: Collective Op Autotuning Team
