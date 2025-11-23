# Fusion and Autotuning Solutions - 完整指南

**创建日期**: 2025-11-07
**目标**: 为 Custom Op / Collective Op 实现 Fusion 和高效 Autotuning

---

## 📚 文档目录

本文件夹包含了关于 **Inline Fusion** 和 **Async Compilation** 的完整技术方案和背景知识文档。

### 背景知识文档

| 文档 | 内容 | 阅读时间 | 优先级 |
|------|------|---------|--------|
| [MULTITEMPLATEBUFFER_SUMMARY.md](./MULTITEMPLATEBUFFER_SUMMARY.md) | MultiTemplateBuffer 快速总结（中文） | 10 分钟 | ⭐⭐⭐ 必读 |
| [MULTITEMPLATEBUFFER_ANALYSIS.md](./MULTITEMPLATEBUFFER_ANALYSIS.md) | MultiTemplateBuffer 深度分析（完整版） | 30 分钟 | ⭐⭐⭐ 必读 |
| [WHY_PRECOMPILE_IS_KEY.md](./WHY_PRECOMPILE_IS_KEY.md) | 为什么 precompile() 是异步编译的关键 | 15 分钟 | ⭐⭐⭐ 必读 |
| [INLINE_FUSION_AND_ASYNC_COMPILATION.md](./INLINE_FUSION_AND_ASYNC_COMPILATION.md) | Inline Fusion vs Async Compilation 深度分析 | 25 分钟 | ⭐⭐ 推荐 |

### 实施方案文档

| 文档 | 内容 | 复杂度 | 时间线 | 优先级 |
|------|------|--------|--------|--------|
| [SOLUTION_A_SUBGRAPH_ASYNC_COMPILATION.md](./SOLUTION_A_SUBGRAPH_ASYNC_COMPILATION.md) | 方案 A: 修复 SubgraphChoiceCaller 异步编译 | 高 | 4-6 周 | ⭐⭐⭐ 推荐先做 |
| [SOLUTION_B_MULTITEMPLATE_RECURSIVE_FUSION.md](./SOLUTION_B_MULTITEMPLATE_RECURSIVE_FUSION.md) | 方案 B: Custom Op MultiTemplateBuffer + 递归 Fusion | 极高 | 8-12 周 | ⭐⭐ 在方案 A 之后 |

---

## 🎯 核心问题与解决方案

### 问题 1: tuned_mm 的 decompose_k 能做 epilogue fusion 吗？

**✅ 是的！** decompose_k_subgraph_template 通过 SubgraphBuffer 支持 epilogue fusion。

**证据**:
```python
# Test 证明 (test_max_autotune.py)
compiled_func = torch.compile(lambda a, b: (a @ b).relu(), dynamic=dynamic)
# 生成: triton_.*_fused_mm_0.run (融合 kernel)
```

### 问题 2: Inline fusion 为什么没用在 internal？

**❌ 因为 breaks async compilation**

**Root Cause**: SubgraphChoiceCaller 缺少 `precompile()` 方法
- 无法参与 ThreadPoolExecutor 并行预编译
- Benchmark 阶段必须同步编译（serialization bottleneck）
- 总编译时间增加 30-40%

### 问题 3: Custom op 走 defer 路线（MultiTemplateBuffer）还需要做什么？

**需要 5 个关键改动**:
1. ✅ Config 标志: `config.benchmark_collective_epilogue_fusion = True`
2. ✅ 传递 `return_multi_template=True` 到 `autotune_select_algorithm()`
3. ⚠️ Scheduler 集成: 扩展 `finalize_multi_template_buffers()`
4. ⚠️ Precompile 方法: SubgraphChoiceCaller 实现 `precompile()`
5. ⚠️ Async 兼容性: 确保 subgraph 并行 benchmark

---

## 💡 两种解决方案对比

### 方案 A: 修复 Subgraph Async Compilation

**目标**: 给 SubgraphChoiceCaller 添加 `precompile()` 方法，消除编译瓶颈

**核心改动**:
```python
# subgraph.py
class SubgraphChoiceCaller(ir.ChoiceCaller):
    def __init__(self, ...):
        self._compiled_module = None  # 缓存编译结果
        self._precompile_lock = threading.Lock()

    def precompile(self):
        """预编译 subgraph（可并行）"""
        with self._precompile_lock:
            fake_inputs = self._generate_fake_inputs()
            self._compiled_module = GraphLowering(...).compile_to_module()

    def benchmark(self, *args, out):
        """使用缓存的模块"""
        if self._compiled_module is not None:
            mod = self._compiled_module  # 快速路径
        else:
            mod = compile_on_demand()  # Fallback
        return benchmarker.benchmark(...)
```

**优势**:
- ✅ 编译时间减少 30-40%
- ✅ 完全并行化，无性能损失
- ✅ 保持 async 架构
- ✅ 向后兼容

**劣势**:
- ⚠️ 实现复杂（fake tensor mode）
- ⚠️ 需要处理 cache coherency

**复杂度**: 高
**时间线**: 4-6 周
**推荐**: ⭐⭐⭐ **先做方案 A**

---

### 方案 B: Custom Op MultiTemplateBuffer + 递归 Fusion

**目标**: 实现完整的 fusion 框架，支持 epilogue/prologue/cross-subgraph fusion

**核心创新**:
1. **延迟选择**: 使用 MultiTemplateBuffer 延迟到 scheduler 阶段
2. **打开 subgraph boundary**: 暴露首尾 nodes 给 scheduler
3. **递归 fusion**: 探索所有 fusion 组合
4. **动态 choice 生成**: 根据 fusion 机会生成 fused choices

**核心架构**:
```
custom_op.py (return_multi_template=True)
    ↓
MultiTemplateBuffer (延迟选择)
    ↓
scheduler.py:finalize_multi_template_buffers()
    ↓
FusionOptimizer (NEW)
    ├─ SubgraphBoundaryInfo (打开 boundary)
    ├─ FusedChoiceCaller (fused choices)
    ├─ recursive_fusion_optimization()
    └─ fusion heuristics
    ↓
Benchmark all choices (original + fused)
    ↓
Select best choice
```

**优势**:
- ✅ 完整的 epilogue/prologue fusion
- ✅ 跨 subgraph fusion（首创）
- ✅ 递归 fusion 探索
- ✅ 与方案 A 完美互补

**劣势**:
- ⚠️ 极高复杂度
- ⚠️ 编译时间可能增加
- ⚠️ FX graph 合并复杂
- ⚠️ 需要大量测试

**复杂度**: 极高
**时间线**: 8-12 周
**推荐**: ⭐⭐ **在方案 A 之后再做**

---

## 🚀 推荐实施路线

### 阶段 1: 快速验证 (Week 1)

使用 **Quick Fix** 验证 correctness:
```python
# mm.py 或 custom_op.py
if any(isinstance(c, SubgraphChoiceCaller) for c in choices):
    with torch._inductor.config.patch(max_autotune_gemm_threads=1):
        return autotune_select_algorithm(...)
```

**目标**: 验证 inline fusion 的功能正确性和性能收益

---

### 阶段 2: 方案 A 实施 (Week 2-7)

**Phase 1**: Core Implementation (Week 2-3)
- 实现 `_generate_fake_inputs()`
- 实现 `precompile()` with caching
- 实现 thread-safe locking

**Phase 2**: Testing (Week 4-5)
- Unit tests for precompile()
- Integration tests with tuned_mm
- Performance benchmarks

**Phase 3**: Validation (Week 6-7)
- Run on vLLM workloads
- PyTorch CI test suite
- Address any failures

**验收标准**:
- ✅ 编译时间减少 ≥30%
- ✅ 所有测试通过
- ✅ 无内存泄漏

---

### 阶段 3: 方案 B 实施 (Week 8-19) - Optional

**Phase 1**: Foundation (Week 8-9)
- MultiTemplateBuffer support in custom_op.py
- SubgraphBoundaryInfo implementation

**Phase 2**: Epilogue Fusion (Week 10-12)
- FusionOptimizer 基础
- FusedChoiceCaller implementation
- Scheduler integration

**Phase 3**: Prologue Fusion (Week 13-14)
- Prologue detection
- Integration & testing

**Phase 4**: Cross-Subgraph Fusion (Week 15-17)
- Adjacent subgraph detection
- Subgraph fusion implementation
- Comprehensive testing

**Phase 5**: Recursive Fusion (Week 18)
- Recursive fusion algorithm
- Fusion heuristics

**Phase 6**: Validation (Week 19)
- Performance benchmarks
- Production rollout

**验收标准**:
- ✅ 运行时间减少 ≥25%（fusion）
- ✅ Epilogue fusion 正常工作
- ✅ 所有测试通过

---

## 📊 预期性能提升

### 方案 A 单独

| 指标 | Baseline | 方案 A | 提升 |
|------|---------|--------|------|
| 编译时间 | 10.0s | 6.0-7.0s | 30-40% ⬇️ |
| 运行时间 | 5.0ms | 5.0ms | 无变化 |
| 总端到端 | 10.5s | 6.5s | 38% ⬇️ |

### 方案 A + 方案 B 组合

| 指标 | Baseline | 方案 A+B | 提升 |
|------|---------|----------|------|
| 编译时间 | 10.0s | 6.0-7.0s | 30-40% ⬇️ |
| 运行时间 | 5.0ms | 3.5-3.8ms | 25-30% ⬇️ |
| 总端到端 | 10.5s | 4.2-4.8s | 55-60% ⬇️ |

---

## 🔍 关键技术点

### 1. MultiTemplateBuffer 机制

**作用**: 延迟 kernel 选择到 scheduler 阶段

```python
# 创建时不 benchmark，只传入 lazy 函数
MultiTemplateBuffer(
    layout=layout,
    inputs=input_nodes,
    choice_timings_fn=get_timings,  # LAZY callable
    unfiltered_choices=choices,
)

# Scheduler 阶段才真正 benchmark
def finalize_multi_template_buffers(self, nodes):
    for node in nodes:
        # 此时才调用 choice_timings_fn()
        timings = node.choice_timings()
        best = min(timings, key=timings.__getitem__)
```

### 2. precompile() 的核心作用

**关键**: `precompile()` 是进入异步编译流程的**唯一入口**

```python
# select_algorithm.py
for choice in choices:
    if hasattr(choice, "precompile"):  # ← GATE
        # 有 precompile() → 并行编译
        future = executor.submit(choice.precompile)
        futures[choice] = future
    else:
        # 无 precompile() → 跳过，后续串行编译
        pass
```

**结果**:
- 有 precompile(): ThreadPoolExecutor 并行编译
- 无 precompile(): benchmark 时同步编译（bottleneck）

### 3. Subgraph Boundary 打开

**问题**: Subgraph 是黑盒，scheduler 看不到内部结构

**解决**: 提取 first_nodes 和 last_nodes

```python
class SubgraphBoundaryInfo:
    first_nodes: List[torch.fx.Node]  # Entry points
    last_nodes: List[torch.fx.Node]   # Exit points

    def can_fuse_epilogue(self, epilogue_op):
        for last_node in self.last_nodes:
            if is_pointwise(epilogue_op) and is_compatible(last_node, epilogue_op):
                return True
        return False
```

### 4. Recursive Fusion 策略

```python
# Iteration 0: Original choices
[all_reduce_nccl, all_reduce_triton]

# Iteration 1: + Epilogue fusion
[all_reduce_nccl, all_reduce_triton,
 all_reduce_nccl+relu, all_reduce_triton+relu]

# Iteration 2: + Double epilogue
[..., all_reduce_nccl+relu+scale, all_reduce_triton+relu+scale]

# Benchmark all, select fastest
```

---

## 📖 阅读指南

### 如果你是第一次接触这个项目

**推荐阅读顺序**:
1. 📄 [MULTITEMPLATEBUFFER_SUMMARY.md](./MULTITEMPLATEBUFFER_SUMMARY.md) - 快速了解 MultiTemplateBuffer
2. 📄 [WHY_PRECOMPILE_IS_KEY.md](./WHY_PRECOMPILE_IS_KEY.md) - 理解 async compilation 的关键
3. 📄 [SOLUTION_A_SUBGRAPH_ASYNC_COMPILATION.md](./SOLUTION_A_SUBGRAPH_ASYNC_COMPILATION.md) - 查看方案 A 实施细节

### 如果你要实施方案 A

**必读文档**:
1. 📄 [WHY_PRECOMPILE_IS_KEY.md](./WHY_PRECOMPILE_IS_KEY.md) - 理解问题根源
2. 📄 [SOLUTION_A_SUBGRAPH_ASYNC_COMPILATION.md](./SOLUTION_A_SUBGRAPH_ASYNC_COMPILATION.md) - 完整实施方案

**参考实现**:
- `torch/_inductor/select_algorithm.py` - TritonTemplateCaller.precompile()
- `torch/_inductor/codegen/subgraph.py` - SubgraphChoiceCaller

### 如果你要实施方案 B

**必读文档**:
1. 📄 [MULTITEMPLATEBUFFER_ANALYSIS.md](./MULTITEMPLATEBUFFER_ANALYSIS.md) - 理解 MultiTemplateBuffer 机制
2. 📄 [INLINE_FUSION_AND_ASYNC_COMPILATION.md](./INLINE_FUSION_AND_ASYNC_COMPILATION.md) - 理解 inline fusion
3. 📄 [SOLUTION_B_MULTITEMPLATE_RECURSIVE_FUSION.md](./SOLUTION_B_MULTITEMPLATE_RECURSIVE_FUSION.md) - 完整实施方案

**前置条件**:
- ⚠️ **必须先完成方案 A**，否则编译性能会很差

---

## 🛠️ 配置示例

### 方案 A: Async Compilation Fix

```python
# 启用并行预编译（默认）
torch._inductor.config.max_autotune_gemm_threads = 8

# 测试时可以禁用（验证 correctness）
torch._inductor.config.max_autotune_gemm_threads = 1
```

### 方案 B: Custom Op Fusion

```python
# 启用 custom op fusion
torch._inductor.config.enable_custom_op_fusion = True

# Fusion types
torch._inductor.config.custom_op_fusion_types = [
    'epilogue',        # all_reduce + relu
    'prologue',        # relu + all_reduce
    'cross_subgraph',  # subgraph_A + subgraph_B
]

# 递归 fusion
torch._inductor.config.enable_recursive_fusion = True
torch._inductor.config.max_fusion_depth = 3

# Fusion threshold
torch._inductor.config.fusion_speedup_threshold = 1.1  # 10% faster
```

### 完整示例

```python
import torch
import torch.distributed as dist

# 初始化分布式
dist.init_process_group(backend='nccl')

# 配置
with torch._inductor.config.patch(
    max_autotune=True,
    max_autotune_gemm_threads=8,          # 方案 A
    enable_custom_op_fusion=True,         # 方案 B
    enable_recursive_fusion=True,         # 方案 B
):
    @torch.compile
    def distributed_compute(x, w):
        y = x @ w
        y = torch.ops._c10d_functional.all_reduce_(y, "sum", "default")
        y = y.relu()
        y = y * 2.0
        return y

    x = torch.randn(1024, 2048, device='cuda', dtype=torch.float16)
    w = torch.randn(2048, 1024, device='cuda', dtype=torch.float16)

    result = distributed_compute(x, w)
    # 生成单个融合 kernel: all_reduce + relu + scale
```

---

## 🧪 测试策略

### 方案 A 测试

```python
# test/inductor/test_subgraph_async_compile.py
class TestSubgraphAsyncCompilation(unittest.TestCase):
    def test_precompile_basic(self):
        """Test SubgraphChoiceCaller.precompile()"""
        caller = SubgraphChoiceCaller(...)
        caller.precompile()
        self.assertIsNotNone(caller._compiled_module)

    def test_benchmark_uses_cached_module(self):
        """Test benchmark uses pre-compiled module"""
        caller.precompile()
        ms = caller.benchmark(...)
        self.assertGreater(ms, 0)
```

### 方案 B 测试

```python
# test/inductor/test_custom_op_fusion.py
class TestCustomOpFusion(unittest.TestCase):
    def test_epilogue_fusion_basic(self):
        """Test all_reduce + relu fusion"""
        with config.patch(enable_custom_op_fusion=True):
            @torch.compile
            def test_func(x):
                y = torch.ops._c10d_functional.all_reduce_(x, "sum", "default")
                return y.relu()

            result = test_func(torch.randn(1024, device='cuda'))
            # Verify fusion happened
```

---

## 📝 总结

### 方案选择建议

| 场景 | 推荐方案 | 理由 |
|------|---------|------|
| **短期目标 (1-2 个月)** | 方案 A | 实现复杂度合理，收益明显 |
| **长期目标 (3-4 个月)** | 方案 A + B | 最大化性能提升 |
| **只想验证 correctness** | Quick Fix | 禁用并行编译，最简单 |
| **Production 部署** | 方案 A | 稳定可靠，风险可控 |

### 关键收益

**方案 A**:
- ✅ 编译时间减少 30-40%
- ✅ 实现复杂度可控
- ✅ 向后兼容
- ✅ 为方案 B 打基础

**方案 A + B**:
- ✅ 编译时间减少 30-40%
- ✅ 运行时间减少 25-35%
- ✅ 端到端性能提升 55-60%
- ✅ 完整的 fusion 支持

### 下一步行动

1. **阅读背景知识文档** (1-2 小时)
2. **选择实施方案** (方案 A 或 A+B)
3. **开始 Phase 1 实施**
4. **逐步测试和验证**
5. **Production 部署**

---

## 📞 联系方式

**Document Author**: Collective Op Autotuning Team
**Last Updated**: 2025-11-07
**Version**: 1.0

如有问题，请参考各个文档中的详细说明，或查看相关源代码：
- `/torch/_inductor/select_algorithm.py`
- `/torch/_inductor/codegen/subgraph.py`
- `/torch/_inductor/scheduler.py`
- `/torch/_inductor/kernel/mm.py`

---

**Happy Coding! 🚀**
