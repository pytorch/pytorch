# Inline Fusion vs Async Compilation 深度分析

**Date**: 2025-11-07
**Purpose**: 理解 inline fusion 与 async compilation 的冲突，以及如何解决

---

## TL;DR - 核心问题回答

### Q1: Custom Op 要走 defer 路线（MultiTemplateBuffer）还需要做什么？

**A**: 需要 5 个关键改动：

1. ✅ **Config 标志**: 添加 `config.benchmark_collective_epilogue_fusion = True`
2. ✅ **Custom Op 调用**: 传递 `return_multi_template=True` 到 `autotune_select_algorithm()`
3. ⚠️ **Scheduler 集成**: 扩展 `finalize_multi_template_buffers()` 处理 collective ops
4. ⚠️ **Precompile 方法**: SubgraphChoiceCaller 需要实现 `precompile()` 方法（**关键**）
5. ⚠️ **Async 兼容性**: 确保 subgraph 可以并行 benchmark（避免 serialization bottleneck）

### Q2: Inline Fusion 为什么没用在 internal？

**A**: 因为 **breaks async compilation**，具体原因：

1. **SubgraphChoiceCaller 缺少 `precompile()` 方法**，无法参与并行预编译
2. **Async compilation pipeline 无法处理 fused subgraphs**，导致 serialization bottleneck
3. **ThreadPoolExecutor 的并行性被破坏**，fusion 必须同步编译
4. **Default config 是 `benchmark_epilogue_fusion=False`**，所以 inline fusion 不启用

### Q3: tuned_mm 的 decompose_k 能做 epilogue fusion 吗？

**A**: **YES！**✅

- `decompose_k_subgraph_template` 通过 **SubgraphBuffer** 支持 epilogue fusion
- Test 证据: `(a @ b).relu()` 生成 `triton_.*_fused_mm_0.run`（融合 kernel）
- 机制: SubgraphBuffer → inline_subgraph_to_ir_nodes() → 生成可融合的 IR nodes

### Q4: 如何把 decompose_k inline fusion 集成到 internal mm？

**A**: 有 **3 种方案**，推荐程度从高到低：

1. 🥇 **修复 async compilation** (推荐，但复杂)
   - 给 SubgraphChoiceCaller 添加 `precompile()` 方法
   - 支持并行预编译 subgraphs

2. 🥈 **使用 MultiTemplateBuffer** (中等复杂)
   - 延迟 benchmarking 到 scheduler 阶段
   - 避免 async precompilation 问题

3. 🥉 **禁用并行预编译** (简单但慢)
   - 设置 `config.max_autotune_gemm_threads = 1`
   - 所有 choices 串行编译

---

## 目录

1. [Current Custom Op Inline Fusion 实现](#1-current-custom-op-inline-fusion-实现)
2. [Decompose K Subgraph 机制](#2-decompose-k-subgraph-机制)
3. [Async Compilation 架构](#3-async-compilation-架构)
4. [为什么 Inline Fusion 破坏 Async Compilation](#4-为什么-inline-fusion-破坏-async-compilation)
5. [解决方案对比](#5-解决方案对比)
6. [实现路线图](#6-实现路线图)

---

## 1. Current Custom Op Inline Fusion 实现

### 位置与代码

**文件**: `/torch/_inductor/kernel/custom_op.py` (Lines 373-389)

```python
# Apply inlining for fusion if winning_choice has graph;
# otherwise return result as-is (default fallback impl)
if winning_choice.gm is not None:
    log.debug(
        "Inlining winning choice: %s (name=%s)",
        getattr(winning_choice, "name", type(winning_choice).__name__),
        name,
    )
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes

    return inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)

log.debug(
    "Winning choice does not support inlining: %s (name=%s)",
    getattr(winning_choice, "name", type(winning_choice).__name__),
    name,
)
return selected_result
```

### Inline Fusion 机制

```
Winning Choice (has .gm attribute)
        ↓
inline_subgraph_to_ir_nodes(gm, inputs, name)
        ↓ (subgraph.py:27-40)
process_subgraph_nodes(gm, inputs)
        ↓ (lowering.py:7310-7336)
For each FX node:
├─ placeholder → map to input args
├─ compute nodes → V.graph.run_node()
└─ output → extract result
        ↓
Returns: TensorBox with individual IR nodes (fusable!)
```

**关键特性**:
1. ✅ **FX graph 分解**: 每个操作变成独立的 ComputedBuffer
2. ✅ **Fusable IR**: 可以与后续 epilogue 操作融合
3. ✅ **已实现**: 代码已在 custom_op.py 中

**为什么没用在 internal？**
- ❌ **Breaks async compilation** (下面会详细解释)
- ❌ **Default config 不启用**: `benchmark_epilogue_fusion=False`
- ❌ **SubgraphChoiceCaller 缺少 precompile()**

---

## 2. Decompose K Subgraph 机制

### Definition 与 Algorithm

**文件**: `/torch/_inductor/kernel/mm.py` (Lines 998-1047)

```python
def decomposeK(a, b, k_splits):
    """
    Decompose large K dimension into batched matmuls.

    Strategy:
    1. Reshape K into B (batch) dimension
    2. Use torch.bmm (batched matmul)
    3. Reduce results across batch dimension

    Example: (m, k) @ (k, n) with k_splits=32
    → Reshape: (m, 32, k//32) @ (32, k//32, n)
    → BMM: (32, m, k//32) @ (32, k//32, n) → (32, m, n)
    → Sum: (32, m, n) → (m, n)
    """
    m = a.shape[0]
    n = b.shape[1]
    k = a.shape[1]

    k_parts = k // k_splits
    B = k_splits
    a_reshaped = torch.permute(a.reshape(m, B, k_parts), (1, 0, 2))  # [B, m, k_parts]
    b_reshaped = b.reshape(B, k_parts, n)                            # [B, k_parts, n]
    result = torch.bmm(a_reshaped, b_reshaped, out_dtype=torch.float32)  # [B, m, n]
    reduced_buf = torch.sum(result, 0)  # Sum across B dimension
    return reduced_buf.to(a.dtype)
```

### Epilogue Fusion 支持

**✅ YES - 通过 SubgraphBuffer**

**Test 证据** (`test_max_autotune.py`, Lines 1527-1541):
```python
# Test adding epilogue also equivalent to eager
compiled_func = torch.compile(lambda a, b: (a @ b).relu(), dynamic=dynamic)
out, code = run_and_get_code(compiled_func, a, b)

FileCheck().check("extern_kernels.bmm_dtype").check_regex(
    "triton_.*_fused_mm_0.run"  # <-- "fused_mm_0" = decompose_k + relu fusion!
).check("decompose_k").run(code[0])
```

### Fusion 实现路径

```
tuned_mm() 选择 decompose_k_subgraph_template
        ↓
generate() 创建 SubgraphChoiceCaller
        ↓
Autotuning 选择最佳 choice
        ↓
custom_op.py: if winning_choice.gm is not None
        ↓
inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)
        ↓
FX graph nodes → Individual IR nodes (ComputedBuffer)
        ↓
Epilogue (e.g., relu) 可以融合到最后一个 IR node
```

**关键点**:
1. ✅ **SubgraphBuffer** 生成可融合的 IR nodes
2. ✅ **Test 验证**: decompose_k + relu 融合成功
3. ✅ **Production ready**: 机制已存在

---

## 3. Async Compilation 架构

### ThreadPoolExecutor Pipeline

**文件**: `/torch/_inductor/select_algorithm.py` (Lines 3009-3181)

```python
def make_precompile_fn(self, choices, name, inputs_key, precompilation_timeout_seconds):
    """
    Parallel precompilation of all choices using ThreadPoolExecutor.
    """
    num_workers = inductor_config.compile.max_workers()
    executor = ThreadPoolExecutor(max_workers=num_workers)  # Line 3095
    async_compile = torch._inductor.async_compile.AsyncCompile()  # Line 3096

    futures = {}

    for c in choices:
        if hasattr(c, "precompile"):  # Line 3112 - KEY CHECK
            triton_cuda_choice = isinstance(c, TritonTemplateCaller) and isinstance(
                c.bmreq, TritonGPUBenchmarkRequest)

            if triton_cuda_choice and async_compile.use_process_pool():
                # TRITON PATH: Async process pool compilation
                future = async_compile.triton(
                    kernel_name=c.bmreq.kernel_name,
                    source_code=source_code
                ).future  # Lines 3119-3121
            else:
                # FALLBACK PATH: Thread pool compilation
                future = executor.submit(precompile_with_captured_stdout, c)
                # Line 3124-3125

            futures[c] = future
        # else: NO precompile() → skipped!

    return precompile_fn
```

### Async Compilation Decision Tree

```
Choice 需要 precompile?
    │
    ├─ NO → 跳过预编译，后续同步编译（SLOW）
    │
    └─ YES → 并行预编译
           │
           ├─ TritonTemplateCaller + CUDA?
           │  └─ YES → AsyncCompile.triton() (Process Pool)
           │          ├─ 异步进程池编译
           │          └─ 返回 Future
           │
           └─ Other choices?
              └─ YES → ThreadPoolExecutor.submit()
                       ├─ 线程池并行编译
                       └─ 返回 Future
```

### Key Methods

| Class | Method | Purpose |
|-------|--------|---------|
| `ThreadPoolExecutor` | `submit(fn, *args)` | 提交任务到线程池 |
| `AsyncCompile` | `triton()` | Triton kernel 异步编译 |
| `TritonTemplateCaller` | `precompile()` | 预编译 Triton kernel |
| `SubgraphChoiceCaller` | ❌ **MISSING** `precompile()` | **问题所在** |

---

## 4. 为什么 Inline Fusion 破坏 Async Compilation

### Root Cause: SubgraphChoiceCaller 缺少 precompile()

**文件**: `/torch/_inductor/codegen/subgraph.py` (Lines 43-167)

```python
class SubgraphChoiceCaller(ir.ChoiceCaller):
    def __init__(self, gm, input_nodes, ...):
        self.gm = gm  # FX GraphModule
        self.original_inputs = input_nodes
        # ...

    def benchmark(self, *args, out):
        """Benchmark by compiling subgraph on-the-fly."""
        # Create GraphLowering
        bm_graph_lowering = GraphLowering(...)

        # Compile to module (SYNCHRONOUS!)
        mod = bm_graph_lowering.compile_to_module()

        # Benchmark
        return benchmarker.benchmark(...)

    # ❌ MISSING: def precompile(self): ...
    #            Cannot participate in async compilation!
```

**对比 TritonTemplateCaller**:

```python
class TritonTemplateCaller(ir.TritonTemplateCallerBase):
    def __init__(self, ..., bmreq):
        self.bmreq = bmreq  # TritonBenchmarkRequest

    # ✅ HAS precompile() method
    def precompile(self):
        assert self.bmreq is not None
        self.bmreq.precompile()  # Can be called from thread/process pool
```

### Failure Chain

```
1. tuned_mm() 收集 choices
   ├─ TritonTemplateCaller (有 precompile())
   ├─ ExternKernelCaller (有 precompile())
   └─ decompose_k_subgraph_template → SubgraphChoiceCaller (❌ 无 precompile())

2. make_precompile_fn() 开始并行预编译
   ├─ for choice in choices:
   │  └─ if hasattr(choice, "precompile"):  # Line 3112
   │     ├─ TritonTemplateCaller → submit to pool ✅
   │     ├─ ExternKernelCaller → submit to pool ✅
   │     └─ SubgraphChoiceCaller → SKIP (no precompile) ❌
   │
   └─ SubgraphChoiceCaller 不会被预编译

3. 后续 benchmark 阶段
   ├─ TritonTemplateCaller → 已预编译，快速 benchmark ✅
   ├─ ExternKernelCaller → 已预编译，快速 benchmark ✅
   └─ SubgraphChoiceCaller → 必须同步编译 (SLOW) ❌
                            └─ Serialization bottleneck!

4. 结果
   ├─ Async 并行性被破坏
   ├─ SubgraphChoiceCaller 成为瓶颈
   └─ 总编译时间显著增加
```

### Benchmark Serialization Problem

```
Without precompile():
────────────────────────────────────────────────────────
Thread 1: [Triton 1 precompile] [Triton 1 benchmark]
Thread 2: [Triton 2 precompile] [Triton 2 benchmark]
Thread 3: [Extern 1 precompile] [Extern 1 benchmark]
Main:     [Wait.....................] [Subgraph compile + benchmark]
                                      ↑ BOTTLENECK (synchronous)
────────────────────────────────────────────────────────

With precompile():
────────────────────────────────────────────────────────
Thread 1: [Triton 1 precompile] [Triton 1 benchmark]
Thread 2: [Triton 2 precompile] [Triton 2 benchmark]
Thread 3: [Extern 1 precompile] [Extern 1 benchmark]
Thread 4: [Subgraph precompile] [Subgraph benchmark]
                                ↑ PARALLEL (no bottleneck)
────────────────────────────────────────────────────────
```

---

## 5. 解决方案对比

### 方案 1: 修复 Async Compilation (推荐 🥇)

**目标**: 给 SubgraphChoiceCaller 添加 `precompile()` 方法

#### 实现步骤

**Step 1: 添加 precompile() 方法**

```python
# subgraph.py
class SubgraphChoiceCaller(ir.ChoiceCaller):
    def __init__(self, gm, input_nodes, ...):
        self.gm = gm
        self.original_inputs = input_nodes
        self._compiled_module = None  # Cache compiled module
        self._precompile_done = False

    def precompile(self):
        """Precompile subgraph for async compilation."""
        if self._precompile_done:
            return

        try:
            # Create GraphLowering with example inputs
            fake_mode = torch._subclasses.FakeTensorMode()
            with V.set_fake_mode(fake_mode):
                # Generate fake inputs
                fake_inputs = [
                    torch.empty_strided(
                        inp.get_size(), inp.get_stride(),
                        dtype=inp.get_dtype(), device=inp.get_device()
                    ) for inp in self.original_inputs
                ]

                # Compile subgraph
                bm_graph_lowering = GraphLowering(
                    self.gm,
                    example_inputs=fake_inputs,
                    ...
                )

                # Cache compiled module
                self._compiled_module = bm_graph_lowering.compile_to_module()
                self._precompile_done = True
        except Exception as e:
            log.warning(f"Precompile failed for {self.name}: {e}")
            self._precompile_done = False

    def benchmark(self, *args, out):
        """Use cached compiled module if available."""
        if self._compiled_module is not None:
            # Use pre-compiled module
            mod = self._compiled_module
        else:
            # Fallback: compile on-demand
            bm_graph_lowering = GraphLowering(...)
            mod = bm_graph_lowering.compile_to_module()

        return benchmarker.benchmark(...)
```

**Step 2: 测试并行预编译**

```python
# test_subgraph_parallel_precompile.py
import torch
from torch._inductor.kernel.mm import decompose_k_subgraph_template
from torch._inductor.select_algorithm import autotune_select_algorithm

@torch.compile
def test_decompose_k_parallel(a, b):
    return (a @ b).relu()

# Enable parallel precompilation
torch._inductor.config.max_autotune_gemm_threads = 8

a = torch.randn(1024, 8192, device='cuda', dtype=torch.float16)
b = torch.randn(8192, 2048, device='cuda', dtype=torch.float16)

# Should use parallel precompilation
result = test_decompose_k_parallel(a, b)
```

#### 优点 & 缺点

| 优点 | 缺点 |
|------|------|
| ✅ 完全并行化 | ⚠️ 实现复杂 |
| ✅ 无性能损失 | ⚠️ 需要处理 fake tensor mode |
| ✅ 保持 async 架构 | ⚠️ 可能有 cache coherency 问题 |
| ✅ 长期最优解 | ⚠️ 需要大量测试 |

---

### 方案 2: 使用 MultiTemplateBuffer (中等复杂 🥈)

**目标**: 延迟 benchmarking 到 scheduler 阶段，避免 async precompilation

#### 架构

```
custom_op.py
    ↓
autotune_select_algorithm(
    choices=[...],
    return_multi_template=True  # 启用延迟
)
    ↓
AlgorithmSelectorCache.__call__()
    ├─ 不立即 benchmark
    ├─ 创建 MultiTemplateBuffer
    └─ 返回包含所有 choices 的 buffer
    ↓
Scheduler: finalize_multi_template_buffers()
    ├─ 检测 epilogue fusion 机会
    ├─ 对每个 choice (包括 subgraph):
    │  └─ Benchmark with epilogue fused
    └─ 选择最佳 choice
```

#### 实现步骤

**Step 1: 启用 MultiTemplateBuffer**

```python
# custom_op.py
def call_function(self, target, args, kwargs):
    # ... (existing detection code)

    if is_collective:
        return autotune_select_algorithm(
            f"custom_op_{op_overload}",
            choices=choices,
            is_collective=True,
            process_group=process_group,
            return_multi_template=True,  # NEW!
        )
```

**Step 2: Scheduler 集成**

```python
# scheduler.py
def finalize_multi_template_buffers(self, nodes):
    for node in nodes:
        multi_node = node.node

        # Check if has subgraph choices
        has_subgraph = any(
            isinstance(c, SubgraphChoiceCaller)
            for c in multi_node.unfiltered_choices
        )

        if has_subgraph:
            # Sequential benchmarking for subgraphs (no async)
            self._finalize_with_subgraph_choices(node, multi_node)
        else:
            # Standard parallel benchmarking
            self._finalize_compute_multi_template(node, multi_node)

def _finalize_with_subgraph_choices(self, node, multi_node):
    """
    Benchmark subgraph choices sequentially to avoid async issues.
    """
    timings = {}

    for choice in multi_node.unfiltered_choices:
        if isinstance(choice, SubgraphChoiceCaller):
            # Sequential benchmark (no precompile)
            with multi_node.swap_as_triton_caller(choice):
                ms = self._benchmark_single_choice(choice)
                timings[choice] = ms
        else:
            # Use cached timings from async precompilation
            cached_timings = multi_node.choice_timings()
            timings[choice] = cached_timings[choice]

    # Select best
    best_choice = min(timings, key=timings.__getitem__)
    multi_node.finalize_as_triton_caller(best_choice)
```

#### 优点 & 缺点

| 优点 | 缺点 |
|------|------|
| ✅ 避免 async precompile 问题 | ⚠️ Subgraph 仍然串行 benchmark |
| ✅ 支持 epilogue fusion | ⚠️ 增加 scheduler 复杂度 |
| ✅ 渐进式实现 | ⚠️ 编译时间可能更长 |
| ✅ 可以与方案 1 结合 | ⚠️ 需要 scheduler 改动 |

---

### 方案 3: 禁用并行预编译 (简单但慢 🥉)

**目标**: 强制所有 choices 串行编译

#### 实现

```python
# config.py 或 runtime
torch._inductor.config.max_autotune_gemm_threads = 1  # Disable parallelism

# 或者在 tuned_mm() 中
if any(isinstance(c, SubgraphChoiceCaller) for c in choices):
    # Disable async precompilation for this tuned_mm call
    with torch._inductor.config.patch(max_autotune_gemm_threads=1):
        return autotune_select_algorithm(...)
```

#### 优点 & 缺点

| 优点 | 缺点 |
|------|------|
| ✅ 最简单实现 | ❌ 编译时间显著增加 |
| ✅ 无需代码改动 | ❌ 浪费 CPU 核心 |
| ✅ 测试/调试友好 | ❌ 不可扩展 |
| ✅ 快速 workaround | ❌ 长期不可行 |

---

## 6. 实现路线图

### Phase 1: Quick Fix (方案 3) - Week 1
```
Goal: 先让 inline fusion 跑起来
├─ 1.1: 设置 config.max_autotune_gemm_threads = 1
├─ 1.2: 测试 decompose_k + relu fusion
├─ 1.3: 验证 functional correctness
└─ 1.4: Benchmark performance (baseline)
```

### Phase 2: MultiTemplateBuffer (方案 2) - Week 2-3
```
Goal: 支持 epilogue fusion，但保持串行 subgraph benchmark
├─ 2.1: 启用 return_multi_template=True
├─ 2.2: Scheduler 集成
│  ├─ 扩展 finalize_multi_template_buffers()
│  └─ 实现 _finalize_with_subgraph_choices()
├─ 2.3: 测试 fusion benchmarking
└─ 2.4: Performance 对比 (vs Phase 1)
```

### Phase 3: Async Fix (方案 1) - Week 4-6
```
Goal: 完全并行化，无 bottleneck
├─ 3.1: SubgraphChoiceCaller.precompile() 实现
│  ├─ Fake tensor mode 支持
│  ├─ Compiled module caching
│  └─ Error handling
├─ 3.2: Async compilation 集成测试
│  ├─ Thread pool utilization
│  ├─ Cache coherency
│  └─ Race condition 检查
├─ 3.3: Performance benchmarking
│  ├─ Compilation time 对比
│  ├─ Parallel scalability (1/2/4/8 threads)
│  └─ Memory usage
└─ 3.4: Production rollout
   ├─ Default config 调整
   └─ Documentation
```

---

## 关键文件改动总结

### 方案 1 (Async Fix)

| 文件 | 改动 | 行数估计 |
|-----|------|---------|
| `subgraph.py` | 添加 `SubgraphChoiceCaller.precompile()` | +50 |
| `select_algorithm.py` | 无需改动（自动支持） | 0 |
| `test_subgraph_choice.py` | 测试并行预编译 | +100 |

### 方案 2 (MultiTemplateBuffer)

| 文件 | 改动 | 行数估计 |
|-----|------|---------|
| `custom_op.py` | 传递 `return_multi_template=True` | +5 |
| `scheduler.py` | 扩展 `finalize_multi_template_buffers()` | +80 |
| `config.py` | 添加 `benchmark_collective_epilogue_fusion` | +5 |
| `test_collective_autotuning.py` | 测试 fusion benchmarking | +150 |

### 方案 3 (Quick Fix)

| 文件 | 改动 | 行数估计 |
|-----|------|---------|
| `mm.py` 或 runtime | 设置 `max_autotune_gemm_threads=1` | +3 |

---

## 推荐策略

### 短期 (1-2 周)
使用 **方案 3** 快速验证 inline fusion 的 correctness 和 performance gains。

```python
# Quick test in mm.py
if any(isinstance(c, SubgraphChoiceCaller) for c in choices):
    with torch._inductor.config.patch(max_autotune_gemm_threads=1):
        return autotune_select_algorithm(...)
```

### 中期 (3-4 周)
实现 **方案 2** (MultiTemplateBuffer) 以支持 epilogue fusion benchmarking。

优先级:
1. Custom op autotuning 走 defer 路线
2. Scheduler 集成
3. Fusion benchmarking 测试

### 长期 (5-8 周)
实现 **方案 1** (Async Fix) 作为最终优化。

重点:
1. SubgraphChoiceCaller.precompile() 实现
2. 充分测试并行性和 cache coherency
3. Performance profiling 和优化

---

## 总结

| 问题 | 回答 | 关键文件 |
|-----|------|---------|
| Custom op 要走 defer 路线还需要做什么？ | 5 个改动（见 TL;DR） | custom_op.py, scheduler.py, subgraph.py |
| Inline fusion 为什么没用在 internal？ | Breaks async compilation | select_algorithm.py:3112 |
| decompose_k 能做 epilogue fusion 吗？ | YES ✅ (via SubgraphBuffer) | mm.py, test_max_autotune.py |
| 如何集成 decompose_k inline fusion？ | 3 种方案（推荐方案 1 长期） | 见实现路线图 |

**Next Step**: 选择一个方案开始实现！推荐从方案 3 开始快速验证。

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Author**: Collective Op Autotuning Team
