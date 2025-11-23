# MultiTemplateBuffer 快速总结 (Quick Summary)

**目标**: 理解 MultiTemplateBuffer 的运作机制，为 Collective Op Autotuning V2 做准备

---

## 核心问题回答

### Q: `tuned_mm` 是否使用了 MultiTemplateBuffer？

**A: 是的，但仅在特定配置下**

- **默认行为** (`config.benchmark_epilogue_fusion=False`): **不使用** MultiTemplateBuffer，直接立即选择最佳 kernel
- **启用融合优化** (`config.benchmark_epilogue_fusion=True`): **使用** MultiTemplateBuffer，延迟选择以便在 scheduling 阶段与 epilogue 操作一起 benchmark

---

## MultiTemplateBuffer 是什么？

### 定义
一种**延迟内核选择机制**，将 benchmark 和选择推迟到 scheduling 阶段，以便：
1. 与 epilogue 操作融合后 benchmark（如 matmul + relu）
2. 根据不同的 tensor 大小选择不同的 kernel
3. 在完整的上下文（fusion 机会、tensor shapes）已知后再做决策

### 核心设计：懒加载 (Lazy Evaluation)

```python
class MultiTemplateBuffer(TritonTemplateBuffer):
    def __init__(self, ..., choice_timings_fn, ...):
        # 关键：choice_timings_fn 是一个"懒"函数
        # 创建时不执行，需要时才调用
        self._choice_timings_fn = choice_timings_fn

    def choice_timings(self, hint_override=None):
        # 第一次调用时才真正 benchmark
        if hint_override not in self._choice_timings:
            self._choice_timings[hint_override] = (
                self._choice_timings_fn(hint_override)  # 现在才执行！
            )
        return self._choice_timings[hint_override]
```

---

## 两种 Autotuning 路径

### Path A: MultiTemplateBuffer（延迟选择）

**触发条件**: `return_multi_template=True`

```
tuned_mm()
  └─► autotune_select_algorithm(
        choices=[...],
        return_multi_template=True  # 启用延迟
      )
        └─► 创建 MultiTemplateBuffer
              ├─ 不立即 benchmark
              ├─ 返回包含所有 choices 的 buffer
              └─ 等待 scheduler 调用 choice_timings()
```

**优点**:
- ✅ 可以与 epilogue 融合后 benchmark
- ✅ 支持 multi-kernel dispatch（不同大小用不同 kernel）
- ✅ 充分利用运行时上下文信息

**缺点**:
- ⚠️ 复杂度高，需要 scheduler 集成

### Path B: Direct Selection（立即选择）

**触发条件**: `return_multi_template=False`（大多数操作的默认）

```
tuned_mm()
  └─► autotune_select_algorithm(
        choices=[...],
        return_multi_template=False  # 默认
      )
        ├─ 立即 benchmark 所有 choices
        ├─ 选择最快的 choice
        └─ 直接返回 output_node
```

**优点**:
- ✅ 简单直接
- ✅ 不需要 scheduler 集成

**缺点**:
- ❌ 无法考虑 fusion 机会
- ❌ 不支持 multi-kernel dispatch

---

## 当前 Collective Op 实现状态

### V1 实现（已完成）

**使用**: Path B（立即选择）

```python
# custom_op.py
return autotune_select_algorithm(
    f"custom_op_{op_overload}",
    choices=choices,
    is_collective=True,
    # 未传 return_multi_template → 默认 False → Path B
)
```

**限制**:
1. ❌ 不支持 epilogue fusion（无法 benchmark all_reduce + relu 融合）
2. ❌ 不支持 multi-kernel dispatch（所有消息大小用同一个 kernel）
3. ❌ 无法利用延迟选择的优势

---

## V2 改进方向

### 目标

让 collective ops 也支持 MultiTemplateBuffer，实现：
1. **Epilogue fusion**: Benchmark all_reduce + elementwise ops 融合版本
2. **Multi-kernel dispatch**: 小消息用一个 kernel，大消息用另一个
3. **Deferred selection**: 在分布式上下文完全已知后再选择

### 架构改动

#### 1. 启用 MultiTemplateBuffer

```python
# custom_op.py
return autotune_select_algorithm(
    f"custom_op_{op_overload}",
    choices=choices,
    is_collective=True,
    return_multi_template=config.benchmark_collective_epilogue_fusion,  # 新增！
)
```

#### 2. 添加配置

```python
# config.py
benchmark_collective_epilogue_fusion: bool = False  # 启用延迟选择
collective_multi_kernel_hints: list[int] = []  # 消息大小 hints
```

#### 3. Scheduler 集成

```python
# scheduler.py
def finalize_multi_template_buffers(self, nodes):
    for node in nodes:
        multi_node = node.node

        if getattr(multi_node, 'is_collective', False):
            # 新增：collective-specific 处理
            self._finalize_collective_multi_template(node, multi_node)
        else:
            # 现有：matmul/conv 处理
            self._finalize_compute_multi_template(node, multi_node)
```

#### 4. Fusion Benchmarking

```python
def _finalize_collective_multi_template(self, node, multi_node):
    # 1. 检测 epilogue（如 all_reduce 后面的 relu）
    epilogue_ops = self._detect_collective_epilogue(node)

    # 2. Benchmark 每个 choice 与 epilogue 融合后的性能
    if epilogue_ops:
        for choice in multi_node.unfiltered_choices:
            with multi_node.swap_as_triton_caller(choice):
                ms_fused = self._benchmark_collective_fusion(...)
                timings[choice] = ms_fused

    # 3. 根据消息大小 hints 选择不同 kernels
    if config.collective_multi_kernel_hints:
        for hint in config.collective_multi_kernel_hints:
            best_for_hint = select_best_for_size(hint)
            callers[hint] = best_for_hint
        multi_node.finalize_as_triton_callers(callers)  # multi-kernel!
```

---

## 实现路线图

```
Phase 1: Foundation (Week 1) ✅ DONE
├─ CollectiveBenchmarker
├─ Integration with select_algorithm.py
└─ Tests (test_collective_autotuning.py)

Phase 2: MultiTemplateBuffer Integration (Week 2)
├─ Add config.benchmark_collective_epilogue_fusion
├─ Modify custom_op.py to pass return_multi_template=True
└─ Test basic MultiTemplateBuffer creation

Phase 3: Scheduler Integration (Week 3)
├─ Extend finalize_multi_template_buffers()
├─ Implement _finalize_collective_multi_template()
├─ Add epilogue detection
└─ Test fusion benchmarking

Phase 4: Multi-Kernel Dispatch (Week 4)
├─ Add config.collective_multi_kernel_hints
├─ Implement size-based hint override
├─ Extend multi_kernel.py for collective dispatch
└─ Test runtime dispatch

Phase 5: Testing & Validation (Week 5)
├─ Comprehensive test suite
├─ Performance benchmarks
├─ Integration with vLLM
└─ Documentation
```

---

## 关键文件位置

| 文件 | 行号 | 功能 |
|-----|------|------|
| `/torch/_inductor/ir.py` | 5269-5357 | **MultiTemplateBuffer 定义** |
| `/torch/_inductor/select_algorithm.py` | 2927-2971 | **Path A: 延迟选择逻辑** |
| `/torch/_inductor/select_algorithm.py` | 2973-3007 | **Path B: 立即选择逻辑** |
| `/torch/_inductor/kernel/mm.py` | 1100-1329 | **tuned_mm 实现** |
| `/torch/_inductor/scheduler.py` | 3412-3489 | **finalize_multi_template_buffers()** |
| `/torch/_inductor/runtime/collective_benchmarking.py` | 全文 | **我们的 V1 实现** |
| `/torch/_inductor/kernel/custom_op.py` | 324-370 | **Collective op 入口** |

---

## 总结

1. ✅ **tuned_mm 确实使用 MultiTemplateBuffer**，但仅在 `config.benchmark_epilogue_fusion=True` 时
2. ✅ **我们的 V1 实现**使用了 Path B（立即选择），功能正常但无 fusion 支持
3. ✅ **V2 应该采用 MultiTemplateBuffer**，实现 epilogue fusion 和 multi-kernel dispatch
4. ✅ **基础设施已存在**，只需扩展到 collective ops
5. ✅ **逐步实施**，先 fusion 后 multi-kernel，保持向后兼容

---

## 下一步

1. 阅读详细分析文档：`MULTITEMPLATEBUFFER_ANALYSIS.md`
2. 开始 Phase 2 实现：添加 config flags
3. 测试 MultiTemplateBuffer creation for collective ops
4. 逐步推进到 scheduler 集成

---

**文档版本**: 1.0
**最后更新**: 2025-11-07
**作者**: Collective Op Autotuning Team
