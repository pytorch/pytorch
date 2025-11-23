# Collective Op Autotuning - 方案澄清和对比

## 🔍 问题澄清

### 你的核心问题

1. **V1的"单独sync"是什么意思？**
2. **V2的"统一sync"为什么更高效？**
3. **V2需要custom op支持MultiTemplateBuffer吗？**
4. **目前SubgraphTemplate只支持不带MultiTemplateBuffer的路径？**
5. **如果走MultiTemplateBuffer，fusion就能做到了吗？**
6. **Subgraph的inline fusion和MultiTemplateBuffer fusion的区别？**
7. **各个方案的取舍是什么？**

让我逐一回答。

---

## 1️⃣ V1 "单独sync"的含义

### 场景
假设你有3个custom collective ops在同一个模型里:
```python
def model(x):
    y1 = my_allreduce_1(x)      # Custom collective op 1
    y2 = my_allreduce_2(y1)     # Custom collective op 2
    y3 = my_allgather(y2)       # Custom collective op 3
    return y3
```

### V1的行为 (当前方案)

**编译时** (第一次运行):
```python
# Lowering阶段 - 遇到第一个collective op
my_allreduce_1 lowering触发:
├─ 生成choices: [impl1, impl2, impl3]
├─ 调用 autotune_select_algorithm()
├─ 【同步点1】所有ranks barrier同步
├─ Benchmark choice 1
│   └─ barrier → cuda.sync → time → barrier
├─ Benchmark choice 2
│   └─ barrier → cuda.sync → time → barrier
├─ Benchmark choice 3
│   └─ barrier → cuda.sync → time → barrier
└─ 选择最优 → inline_subgraph_to_ir_nodes() → 返回IR nodes

# Lowering阶段 - 遇到第二个collective op
my_allreduce_2 lowering触发:
├─ 生成choices: [impl1, impl2, impl3]
├─ 调用 autotune_select_algorithm()
├─ 【同步点2】所有ranks barrier同步
├─ Benchmark choice 1
│   └─ barrier → cuda.sync → time → barrier
├─ ...
└─ 选择最优 → inline_subgraph_to_ir_nodes()

# Lowering阶段 - 遇到第三个collective op
my_allgather lowering触发:
├─ 生成choices: [impl1, impl2]
├─ 调用 autotune_select_algorithm()
├─ 【同步点3】所有ranks barrier同步
├─ Benchmark choice 1
│   └─ barrier → cuda.sync → time → barrier
├─ ...
└─ 选择最优 → inline_subgraph_to_ir_nodes()
```

**关键点**:
- ❌ 每个collective op **单独**进入autotuning
- ❌ 每个op都要重新sync所有ranks
- ❌ 如果有N个collective ops，需要N次"大同步"

**时间开销**:
```
3个collective ops × 5ms sync开销 = 15ms
3个collective ops × 3 choices × 10ms benchmark = 90ms
总计: ~105ms
```

---

## 2️⃣ V2 "统一sync"的含义

### V2的行为 (MultiTemplateBuffer方案)

**编译时** (第一次运行):
```python
# Phase 1: Lowering阶段 - 创建MultiTemplateBuffer (不benchmark)
my_allreduce_1 lowering:
├─ 生成choices: [impl1, impl2, impl3]
├─ 调用 autotune_select_algorithm(return_multi_template=True)
└─ 返回 CollectiveMultiTemplateBuffer (延迟选择)
    └─ 包含3个choices，还没benchmark

my_allreduce_2 lowering:
├─ 生成choices: [impl1, impl2, impl3]
└─ 返回 CollectiveMultiTemplateBuffer (延迟选择)
    └─ 包含3个choices，还没benchmark

my_allgather lowering:
├─ 生成choices: [impl1, impl2]
└─ 返回 CollectiveMultiTemplateBuffer (延迟选择)
    └─ 包含2个choices，还没benchmark

# Phase 2: Scheduler阶段 - 统一处理所有MultiTemplateBuffers
scheduler.finalize_multi_template_buffers():
├─ 【第1步】collect_collective_nodes()
│   └─ 发现3个CollectiveMultiTemplateBuffer nodes
│
├─ 【第2步】try_sync_collective_nodes() 
│   └─ 【唯一的大同步】5ms timeout检测所有ranks是否ready
│       └─ 成功: 所有ranks准备好了
│
└─ 【第3步】遍历每个MultiTemplateBuffer并finalize
    │
    ├─ For my_allreduce_1:
    │   ├─ Benchmark choice 1 (内部有小barrier)
    │   ├─ Benchmark choice 2 (内部有小barrier)
    │   ├─ Benchmark choice 3 (内部有小barrier)
    │   └─ 选择最优 → finalize
    │
    ├─ For my_allreduce_2:
    │   ├─ Benchmark choice 1 (内部有小barrier)
    │   ├─ Benchmark choice 2 (内部有小barrier)
    │   ├─ Benchmark choice 3 (内部有小barrier)
    │   └─ 选择最优 → finalize
    │
    └─ For my_allgather:
        ├─ Benchmark choice 1 (内部有小barrier)
        ├─ Benchmark choice 2 (内部有小barrier)
        └─ 选择最优 → finalize
```

**关键点**:
- ✅ 所有collective ops在lowering时只创建MultiTemplateBuffer，不benchmark
- ✅ 在scheduler阶段**统一收集**所有collective nodes
- ✅ **只需一次大同步** (5ms pre-sync) 检测ranks就绪
- ✅ 之后所有benchmark在已经同步的ranks上进行

**时间开销**:
```
1次大同步: 5ms
3个collective ops × 3 choices × 10ms benchmark = 90ms
总计: ~95ms (比V1节省10ms)
```

**为什么更高效？**
因为那个5ms的"pre-sync"只是快速检测"所有ranks是否准备好开始benchmark"，成功后就不需要每个op都重新协调所有ranks了。

---

## 3️⃣ MultiTemplateBuffer vs 直接benchmark

### 现状分析

**当前Custom Op的路径** (autotune_custom_op, Line 325-350):
```python
# Line 325: 调用autotune_select_algorithm
selected_result, winning_choice = autotune_select_algorithm(
    name=name,
    choices=choices,
    input_nodes=list(inputs),
    layout=choices[0].layout,
    input_gen_fns=input_gen_fns,
    return_choice=True,  # ← 关键: 没有return_multi_template=True
)

# Line 335-343: 获胜后立即inline
if winning_choice.gm is not None:
    return inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)
```

**关键观察**:
- ❌ `return_multi_template` **没有设置为True**
- ❌ 所以不会创建MultiTemplateBuffer
- ✅ 直接benchmark选出winner，然后inline返回IR nodes

**这意味着**:
```python
# V1路径 (当前)
custom_op → autotune_select_algorithm(return_multi_template=False)
         → 立即benchmark
         → 选出winner
         → inline_subgraph_to_ir_nodes() 
         → 返回IR nodes (fusable)
         → 后续可以epilogue fusion

# V2路径 (如果改成return_multi_template=True)
custom_op → autotune_select_algorithm(return_multi_template=True)
         → 创建CollectiveMultiTemplateBuffer (延迟选择)
         → 返回MultiTemplateBuffer
         → 到scheduler阶段才finalize choice
         → 可以benchmark with/without epilogue fusion
```

---

## 4️⃣ 关键代码位置和修改

### 目前的实现路径

```python
# torch/_inductor/select_algorithm.py, Line ~2945
def autotune_select_algorithm(..., return_multi_template=False):
    cache = get_algorithm_selector_cache()
    
    if return_multi_template:
        # 创建MultiTemplateBuffer (延迟benchmark)
        return MultiTemplateBuffer(...)
    else:
        # 立即benchmark并返回winning choice
        return benchmark_and_select_winner(...)
```

**当前custom op调用**:
```python
# custom_op.py, Line 325
autotune_select_algorithm(
    ...,
    return_choice=True,
    # ❌ 没有 return_multi_template=True
)
```

### V2需要的修改

**修改1: custom_op.py, Line 325**
```python
# 添加参数
selected_result, winning_choice = autotune_select_algorithm(
    name=name,
    choices=choices,
    input_nodes=list(inputs),
    layout=choices[0].layout,
    input_gen_fns=input_gen_fns,
    return_choice=True,
    return_multi_template=True,  # ← NEW: 请求MultiTemplateBuffer
    is_collective=is_collective,  # ← NEW: 标记为collective
    process_group=process_group,  # ← NEW: 传递process group
)
```

**修改2: select_algorithm.py**
```python
# Line ~2945 - AlgorithmSelectorCache.__call__
if return_multi_template:
    if is_collective and dist.is_initialized():
        # 创建CollectiveMultiTemplateBuffer
        return CollectiveMultiTemplateBuffer(...)
    else:
        # 创建普通MultiTemplateBuffer
        return MultiTemplateBuffer(...)
```

**修改3: scheduler.py**
```python
# 在finalize_multi_template_buffers()中添加collective处理
def finalize_multi_template_buffers(self):
    # Step 1: 收集collective nodes
    collective_nodes = self.collect_collective_nodes()
    
    # Step 2: 统一pre-sync (5ms timeout)
    if collective_nodes:
        sync_ok = self.try_sync_collective_nodes()
    
    # Step 3: Finalize每个MultiTemplateBuffer
    for node in self.nodes:
        if isinstance(node, CollectiveMultiTemplateBuffer):
            # 使用distributed benchmarking
            ...
```

---

## 5️⃣ Fusion的两种形式

### Inline Fusion (当前V1使用)

**发生时机**: Lowering阶段，benchmark完后立即发生

**代码位置**: custom_op.py, Line 335-343
```python
# 选出winning choice后
if winning_choice.gm is not None:
    # 立即inline这个subgraph到IR nodes
    return inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)
```

**效果**:
```python
# 假设winning choice是一个subgraph: all_reduce + relu
winning_choice.gm = {
    input → all_reduce → relu → output
}

# Inline后变成IR nodes:
return TensorBox(
    ComputedBuffer(all_reduce_ir),
    ComputedBuffer(relu_ir),
)
```

**这样的IR nodes可以被scheduler fusion**:
```python
# 如果后续有epilogue
x = my_allreduce(x)  # inline成IR nodes
y = x + 1            # 后续epilogue

# Scheduler可以fuse: all_reduce + relu + add
```

**限制**:
- ✅ 只能fuse **winning choice内部**已经包含的ops
- ❌ 不能benchmark "all_reduce vs all_reduce+epilogue"
- ❌ 只是让winning choice的ops变成可fuse的IR nodes

### MultiTemplateBuffer Fusion (V2可以做的)

**发生时机**: Scheduler阶段，finalize时

**代码位置**: scheduler.py, finalize_multi_template_buffers()
```python
# Scheduler阶段识别fusion机会
if can_fuse(collective_node, epilogue_node):
    # Benchmark WITH epilogue
    time_fused = benchmark(collective_choice_fused_with_epilogue)
    
    # Benchmark WITHOUT epilogue
    time_unfused = benchmark(collective_choice_alone)
    
    if time_fused < time_unfused:
        # Fuse!
        finalize_as_fused(collective_node, epilogue_node)
```

**效果**:
```python
# 可以benchmark多种配置
Config 1: all_reduce alone           → 10ms
Config 2: all_reduce + add           → 9ms  ← Better!
Config 3: all_reduce (with add later) → 11ms

# 选择Config 2 (fused)
```

**优势**:
- ✅ 可以benchmark **有无epilogue**的性能差异
- ✅ 自动选择是否fusion更快
- ✅ 支持更复杂的fusion pattern

---

## 6️⃣ 方案对比表

| 维度 | V1 (现有Inline) | V2 (MultiTemplateBuffer) |
|-----|----------------|-------------------------|
| **实现位置** | custom_op.py | custom_op.py + scheduler.py |
| **Benchmark时机** | Lowering阶段(立即) | Scheduler阶段(延迟) |
| **Sync策略** | 每个op单独sync | 统一pre-sync一次 |
| **Fusion类型** | Inline fusion only | Epilogue fusion benchmark |
| **Fusion能力** | Winning choice内部 + 后续ops | 可benchmark with/without epilogue |
| **代码修改** | 小 (custom_op.py) | 中 (custom_op.py + select_algorithm.py + scheduler.py) |
| **实现复杂度** | ⭐⭐ 简单 | ⭐⭐⭐⭐ 中等 |
| **开发时间** | 1-2天 | 3-4天 |
| **N个collective ops开销** | N × 5ms sync | 1 × 5ms sync |
| **适用场景** | 简单custom op | 复杂场景，多collective ops |

---

## 7️⃣ 具体方案选择建议

### 方案A: V1 - 快速验证 (推荐先做)

**什么不改**:
- ❌ 不需要`return_multi_template=True`
- ❌ 不需要修改scheduler.py
- ❌ 保持现有的inline fusion机制

**只需要改**:
- ✅ 在`autotune_select_algorithm`调用前检测是否collective
- ✅ 如果是collective，使用`CollectiveBenchmarker`
- ✅ 添加timeout保护

**修改点**:
```python
# custom_op.py, Line 324
# 检测是否collective
is_collective = False
process_group = None
if op_overload:
    from torch._inductor.runtime.collective_benchmarking import is_collective_op
    op_name = str(op_overload)
    is_collective = is_collective_op(op_name)
    if is_collective:
        # 从non_tensor_args提取process_group
        for kwargs in non_tensor_args:
            if 'group' in kwargs:
                process_group = kwargs['group']
                break

# Line 325: 传递collective信息
selected_result, winning_choice = autotune_select_algorithm(
    name=name,
    choices=choices,
    input_nodes=list(inputs),
    layout=choices[0].layout,
    input_gen_fns=input_gen_fns,
    return_choice=True,
    is_collective=is_collective,  # NEW
    process_group=process_group,  # NEW
)
```

```python
# select_algorithm.py - AlgorithmSelectorCache.__call__
# 在benchmark阶段检测is_collective
if is_collective and dist.is_initialized():
    from torch._inductor.runtime.collective_benchmarking import (
        CollectiveBenchmarker
    )
    benchmarker = CollectiveBenchmarker(
        process_group=process_group,
        nruns=config.benchmark_kernel_nruns,
    )
    # 使用specialized benchmarking
    # ... benchmark with sync ...
```

**优势**:
- ✅ 最小改动
- ✅ 快速验证collective autotuning可行性
- ✅ 保留inline fusion能力
- ✅ 1-2天完成

**劣势**:
- ❌ 多个collective ops时sync overhead较大
- ❌ 不能benchmark with/without epilogue

---

### 方案B: V2 - 完整方案 (后续升级)

**需要改**:
- ✅ custom_op.py添加`return_multi_template=True`
- ✅ select_algorithm.py支持创建`CollectiveMultiTemplateBuffer`
- ✅ scheduler.py添加unified sync和finalize逻辑
- ✅ ir.py添加`CollectiveMultiTemplateBuffer`类

**修改点**:
```python
# custom_op.py, Line 325
selected_result, winning_choice = autotune_select_algorithm(
    ...,
    return_multi_template=True,  # NEW
    is_collective=is_collective,
    process_group=process_group,
)

# 注意: 如果return_multi_template=True，不能立即inline
# 因为返回的是MultiTemplateBuffer，要等scheduler finalize
if return_multi_template:
    # 直接返回MultiTemplateBuffer
    return selected_result
else:
    # 原有的inline逻辑
    if winning_choice.gm is not None:
        return inline_subgraph_to_ir_nodes(...)
```

**优势**:
- ✅ 统一sync，多collective ops更高效
- ✅ 支持epilogue fusion benchmark
- ✅ 更通用，适用所有MultiTemplateBuffer场景

**劣势**:
- ❌ 实现复杂度高
- ❌ 需要修改scheduler核心逻辑
- ❌ 3-4天开发时间

---

## 8️⃣ 推荐实施路径

### Phase 1: V1 基础 (Week 1-2)
**目标**: 让collective op autotuning基础功能work

**任务**:
1. ✅ 已完成: `collective_benchmarking.py`
2. 🔲 修改`custom_op.py`添加detection
3. 🔲 修改`select_algorithm.py`使用CollectiveBenchmarker
4. 🔲 测试vLLM场景

**交付**:
- 能autotune custom collective ops
- 有timeout保护
- 保留inline fusion能力

---

### Phase 2: 评估和决策 (Week 3)
**目标**: 决定是否需要V2

**评估标准**:
1. **性能需求**: 是否真的有多个collective ops导致sync overhead明显？
2. **Fusion需求**: 是否需要benchmark with/without epilogue？
3. **开发资源**: 是否有时间实现V2？

**决策**:
- 如果只有1-2个collective ops → V1足够
- 如果有3+个collective ops → V2有明显收益
- 如果需要fusion优化 → V2必要

---

### Phase 3: V2 实施 (Week 4-5, 可选)
**前提**: Phase 1稳定，且评估显示V2有必要

**任务**:
1. 🔲 创建`CollectiveMultiTemplateBuffer`类
2. 🔲 修改scheduler添加unified sync
3. 🔲 实现epilogue fusion benchmark
4. 🔲 完整测试和优化

---

## 9️⃣ FAQ

### Q: V1的inline fusion和V2的epilogue fusion有什么区别？

**A**: 
```python
# V1 Inline Fusion (发生在lowering)
my_allreduce(x)  # winning choice已经是fused subgraph
↓ inline
all_reduce_ir + internal_ops  # 变成IR nodes
↓ scheduler可以继续fuse
all_reduce_ir + internal_ops + epilogue_ops

# V2 Epilogue Fusion (发生在scheduler)
MultiTemplateBuffer(all_reduce)  # 还没选择实现
↓ scheduler识别有epilogue
benchmark(all_reduce alone)          # 10ms
benchmark(all_reduce + epilogue)     # 9ms  ← 直接测试fused版本
↓ 选择fused版本
finalize_as_fused()
```

V2可以**直接benchmark fused版本的性能**，而V1只是让ops变成fusable然后依赖scheduler去fuse。

### Q: 如果custom op本身没有epilogue需求，V2还有意义吗？

**A**: 如果：
- 只有1个collective op → V2收益不大
- 有3+个collective ops → V2的统一sync仍然有价值（节省sync overhead）

### Q: Subgraph是否支持MultiTemplateBuffer？

**A**: 
- **当前**: SubgraphTemplate生成的choices可以放入MultiTemplateBuffer，但custom_op.py默认不使用（`return_multi_template=False`）
- **V2需要**: 设置`return_multi_template=True`，让SubgraphChoiceCaller生成的choices延迟到scheduler选择

---

## 🎯 总结

### 核心区别

**V1 (Inline Fusion)**:
```
Lowering: collective_op → benchmark → inline → IR nodes → [scheduler fusion]
         └─ 每个op单独sync
```

**V2 (MultiTemplateBuffer)**:
```
Lowering: collective_op → MultiTemplateBuffer (延迟)
Scheduler: [unified sync] → benchmark all → finalize → [可测试epilogue fusion]
          └─ 统一sync一次
```

### 建议

1. **立即**: 实施V1，验证功能
2. **1周后**: 评估是否需要V2
3. **按需**: 如果有明显收益，再实施V2

V1足够处理大多数场景，V2是性能和通用性的进一步提升。

---

**希望这个文档彻底解答了你的疑问！** 🎉
