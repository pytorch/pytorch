# Collective Op Autotuning - 完整实施指南

**版本**: V1 实用方案
**日期**: 2024-11
**状态**: Ready for Implementation

---

## 📋 目录

1. [快速开始](#快速开始)
2. [核心设计](#核心设计)
3. [实施步骤](#实施步骤)
4. [V2预留设计](#v2预留设计)
5. [测试计划](#测试计划)
6. [常见问题](#常见问题)

---

## 🚀 快速开始

### 目标
为PyTorch Inductor添加collective operations (all_reduce, all_gather等) 的autotuning支持，特别针对vLLM等分布式场景。

### V1方案概述
- **兼容性优先**: 与现有subgraph/custom op完全兼容
- **最小侵入**: 只修改custom_op.py和select_algorithm.py
- **实用导向**: 先让功能work，为V2留出复用空间
- **开发时间**: 1-2天

### 核心创新
1. ✅ **Timeout保护**: 防止某rank hang导致系统卡死
2. ✅ **跨rank同步**: Barrier + all_reduce确保准确benchmark
3. ✅ **保留fusion**: Inline fusion机制不变，scheduler可继续fuse

---

## 🏗️ 核心设计

### 架构图

```
┌─────────────────────────────────────────────────────────┐
│  Custom Op Registration                                 │
│  register_custom_op_autotuning(my_allreduce, configs)  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Lowering Phase (autotune_custom_op)                   │
│  ├─ 检测是否collective op                               │
│  ├─ 提取process_group                                   │
│  └─ 生成choices (SubgraphChoiceCaller)                  │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Autotuning Phase (autotune_select_algorithm)          │
│  ├─ 如果is_collective: 使用CollectiveBenchmarker        │
│  │   ├─ Pre-sync with timeout (~5ms)                   │
│  │   ├─ Benchmark each choice with barriers            │
│  │   └─ All-reduce timing (max across ranks)           │
│  └─ 否则: 使用regular benchmarker                       │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│  Choice Selection & Inlining                            │
│  ├─ 选择最优choice                                       │
│  ├─ inline_subgraph_to_ir_nodes() (如果有gm)            │
│  └─ 返回fusable IR nodes                                │
└─────────────────────────────────────────────────────────┘
```

### 关键决策

**Q: 为什么不用MultiTemplateBuffer (V2)?**

A: V1目标是**快速验证**和**兼容性**：
- ✅ 保持与现有custom op流程一致
- ✅ 不修改scheduler.py (风险低)
- ✅ 保留inline fusion (scheduler可继续fuse epilogue)
- ✅ 1-2天完成，快速迭代

**Q: V1能做fusion吗?**

A: ✅ 可以！通过inline fusion:
```python
# Winning choice的subgraph会被inline成IR nodes
my_allreduce(x) → inline → IR nodes (all_reduce_ir + ...)

# 后续epilogue仍然可以被scheduler fuse
y = my_allreduce(x)  # IR nodes
z = y + 1            # Scheduler可以fuse: all_reduce + add
```

**限制**: 不能benchmark "有无epilogue"的性能差异 (这是V2的优势)

---

## 🔧 实施步骤

### Step 1: 已完成 ✅
- `collective_benchmarking.py` (包含timeout机制)
  - `is_collective_op()` - 检测collective ops
  - `benchmark_collective_op()` - 跨rank benchmarking
  - `sync_with_timeout()` - 防止hang
  - `CollectiveBenchmarker` - 封装类

### Step 2: 修改 custom_op.py

**文件**: `/data/users/tianren/pytorch/torch/_inductor/kernel/custom_op.py`

**修改位置**: Line 324-332 (autotune_custom_op函数)

```python
def autotune_custom_op(
    name: str,
    decompositions: list[Callable[..., Any]],
    inputs: list[Any],
    non_tensor_args: list[dict[str, Any]],
    op_overload: torch._ops.OpOverload,
    user_input_gen_fns: Optional[...] = None,
) -> Union[TensorBox, Any]:
    # ... existing code ...

    # ============ NEW CODE START ============
    # 检测是否为collective operation
    is_collective = False
    process_group = None

    if op_overload:
        from torch._inductor.runtime.collective_benchmarking import is_collective_op

        op_name = str(op_overload)
        is_collective = is_collective_op(op_name)

        if is_collective:
            # 尝试从non_tensor_args中提取process_group
            for kwargs_dict in non_tensor_args:
                if 'group' in kwargs_dict:
                    process_group = kwargs_dict['group']
                    break
                elif 'process_group' in kwargs_dict:
                    process_group = kwargs_dict['process_group']
                    break

            # Log collective op detection
            import torch.distributed as dist
            if dist.is_initialized():
                rank = dist.get_rank()
                log.info(
                    f"[Rank {rank}] Detected collective op: {op_name} "
                    f"(process_group={'default' if process_group is None else 'custom'})"
                )
    # ============ NEW CODE END ============

    # ... existing choice generation code ...

    # Line 325: 传递collective信息给autotune_select_algorithm
    selected_result, winning_choice = autotune_select_algorithm(
        name=name,
        choices=choices,
        input_nodes=list(inputs),
        layout=choices[0].layout,
        input_gen_fns=input_gen_fns,
        return_choice=True,
        is_collective=is_collective,      # NEW
        process_group=process_group,      # NEW
    )

    # ... existing inline code (不变) ...
```

**关键点**:
1. ✅ 检测collective op使用`is_collective_op()`
2. ✅ 从kwargs提取process_group (优先'group'，其次'process_group')
3. ✅ 传递`is_collective`和`process_group`给autotuning
4. ✅ **保持inline逻辑不变** (兼容现有机制)

---

### Step 3: 修改 select_algorithm.py

**文件**: `/data/users/tianren/pytorch/torch/_inductor/select_algorithm.py`

#### 修改3.1: autotune_select_algorithm函数签名

**位置**: Line ~3908

```python
def autotune_select_algorithm(
    name,
    choices,
    input_nodes,
    layout,
    *,
    input_gen_fns=None,
    return_choice=False,
    is_collective=False,      # NEW
    process_group=None,        # NEW
    **kwargs,
):
    """
    Autotune a group of choices and select the best one.

    NEW: Supports collective operations with distributed synchronization.
    """
    cache = get_algorithm_selector_cache()

    if "return_multi_template" not in kwargs:
        kwargs["return_multi_template"] = (
            torch._inductor.config.benchmark_epilogue_fusion
        )

    if "precompilation_timeout_seconds" not in kwargs:
        kwargs["precompilation_timeout_seconds"] = config.precompilation_timeout_seconds

    # 传递新参数给cache
    return cache(
        name,
        choices,
        input_nodes,
        layout,
        input_gen_fns=input_gen_fns,
        return_choice=return_choice,
        is_collective=is_collective,      # NEW
        process_group=process_group,      # NEW
        **kwargs,
    )
```

#### 修改3.2: AlgorithmSelectorCache.__call__方法

**位置**: 找到`class AlgorithmSelectorCache`的`__call__`方法

```python
class AlgorithmSelectorCache:
    def __call__(
        self,
        name,
        choices,
        input_nodes,
        layout,
        *,
        input_gen_fns=None,
        return_choice=False,
        return_multi_template=False,
        is_collective=False,      # NEW
        process_group=None,        # NEW
        **kwargs,
    ):
        # ... existing cache key generation and lookup ...

        # ============ 找到benchmark代码的位置 ============
        # 通常在cached result miss之后，需要实际benchmark

        # ============ NEW CODE - 添加collective benchmarking分支 ============
        if is_collective:
            import torch.distributed as dist

            if not dist.is_initialized():
                log.warning(
                    f"Collective op '{name}' detected but distributed not initialized. "
                    f"Falling back to regular benchmarking."
                )
                is_collective = False
            else:
                # 使用CollectiveBenchmarker
                from torch._inductor.runtime.collective_benchmarking import (
                    CollectiveBenchmarker,
                )

                rank = dist.get_rank(process_group)
                log.info(
                    f"[Rank {rank}] Using CollectiveBenchmarker for '{name}' "
                    f"with {len(choices)} choices"
                )

                # 创建specialized benchmarker
                collective_benchmarker = CollectiveBenchmarker(
                    process_group=process_group,
                    nruns=config.benchmark_kernel_nruns,
                    estimate=False,
                )

                # 使用collective benchmarking逻辑
                # 注意: 这里需要集成到现有的benchmark流程中
                # 可以通过替换benchmarker实例或者添加条件分支

                # TODO: 具体实现需要查看benchmark代码的结构
                # 关键是在调用choice.benchmark()时使用我们的CollectiveBenchmarker

        # ... 继续现有的benchmark和selection逻辑 ...
```

**注意**: 这部分需要根据实际的benchmark代码结构调整。关键是在benchmark choices时，如果`is_collective=True`，使用`CollectiveBenchmarker`而不是默认的benchmarker。

---

### Step 4: 集成Benchmarking (关键)

在`select_algorithm.py`中找到实际benchmark choices的代码，通常是：

```python
# 找到类似这样的代码
for choice in choices:
    timing = choice.benchmark(*args, out=out)
    timings[choice] = timing
```

**修改为**:

```python
# 如果是collective op，需要特殊处理
if is_collective:
    from torch._inductor.runtime.collective_benchmarking import (
        try_collective_benchmark_with_timeout,
    )

    for choice in choices:
        # 使用specialized collective benchmarking
        # 注意: 这里需要适配choice.benchmark的接口

        # 尝试benchmark with timeout
        timing = try_collective_benchmark_with_timeout(
            comm_func=choice.kernel if hasattr(choice, 'kernel') else choice,
            comm_func_name=choice.name,
            input_tensors=prepared_inputs,
            output_tensor=prepared_output,
            process_group=process_group,
            nruns=config.benchmark_kernel_nruns,
            timeout_seconds=30.0,
        )

        if timing is not None:
            timings[choice] = timing
        else:
            # Timeout or failure
            log.warning(
                f"[Collective] Choice {choice.name} timed out, using inf"
            )
            timings[choice] = float('inf')
else:
    # 现有的regular benchmarking
    for choice in choices:
        timing = choice.benchmark(*args, out=out)
        timings[choice] = timing
```

---

## 🔄 V2预留设计 (可复用部分)

### 完全可复用的组件

1. ✅ **collective_benchmarking.py** - 100%复用
   - `is_collective_op()` - V2也需要检测
   - `benchmark_collective_op()` - V2的核心benchmark函数
   - `sync_with_timeout()` - V2的pre-sync会用
   - `CollectiveBenchmarker` - V2也用这个类

2. ✅ **custom_op.py中的detection逻辑** - 部分复用
   - V2仍然需要检测是否collective
   - V2仍然需要提取process_group
   - **区别**: V2会设置`return_multi_template=True`

3. ✅ **Timeout机制** - 100%复用
   - V2的pre-sync和benchmark都需要timeout保护

### V2需要新增的部分

1. 🆕 **CollectiveMultiTemplateBuffer类** (ir.py)
   - 继承自MultiTemplateBuffer
   - 包含process_group和collective_op_type

2. 🆕 **Scheduler的unified sync** (scheduler.py)
   - `collect_collective_nodes()` - 收集所有collective nodes
   - `try_sync_collective_nodes()` - 统一pre-sync
   - `_finalize_collective_choice()` - Specialized finalize

3. 🆕 **select_algorithm.py的MultiTemplateBuffer创建**
   - 当`return_multi_template=True`且`is_collective=True`时
   - 创建CollectiveMultiTemplateBuffer而不是普通MultiTemplateBuffer

### V1到V2的升级路径

```python
# V1 (当前实施)
custom_op.py:
  is_collective = detect_collective()
  autotune_select_algorithm(..., is_collective=is_collective)

select_algorithm.py:
  if is_collective:
    use CollectiveBenchmarker  # ← V2可以复用
  benchmark and select winner
  return winning_result

custom_op.py:
  inline_subgraph_to_ir_nodes()
  return IR nodes

# V2 (未来升级)
custom_op.py:
  is_collective = detect_collective()  # ← 复用V1的检测逻辑
  autotune_select_algorithm(...,
                           is_collective=is_collective,
                           return_multi_template=True)  # ← 新增

select_algorithm.py:
  if return_multi_template and is_collective:
    return CollectiveMultiTemplateBuffer(...)  # ← 延迟benchmark
  # 不立即benchmark

scheduler.py:
  unified_sync()  # ← 新增: 统一同步
  for each CollectiveMultiTemplateBuffer:
    use CollectiveBenchmarker  # ← 复用V1的benchmarker
    finalize_choice()
```

**关键**: `CollectiveBenchmarker`在V1和V2中都用，是100%可复用的核心组件。

---

## 🧪 测试计划

### Phase 1: 单个Collective Op, 2 Ranks

**目标**: 验证基础功能

**测试代码**:
```python
# test/inductor/test_collective_autotuning.py

import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)

class TestCollectiveAutotuning(MultiProcessTestCase):

    @skip_if_lt_x_gpu(2)
    def test_single_allreduce_2ranks(self):
        """Test single all_reduce with 2 ranks"""

        # Initialize distributed
        dist.init_process_group(backend='nccl')
        rank = dist.get_rank()

        # Define custom collective op
        @torch.library.custom_op("test::my_allreduce", mutates_args=())
        def my_allreduce(x: torch.Tensor) -> torch.Tensor:
            return torch.ops._c10d_functional.all_reduce_(x, "sum")

        # Implementation 1: Direct NCCL
        def allreduce_nccl(x):
            return torch.ops._c10d_functional.all_reduce_(x, "sum")

        # Implementation 2: Simulate chunked (for testing)
        def allreduce_chunked(x, chunk_size=1024):
            return torch.ops._c10d_functional.all_reduce_(x, "sum")

        # Register autotuning
        from torch._inductor.kernel.custom_op import (
            register_custom_op_autotuning,
            CustomOpConfig,
        )

        register_custom_op_autotuning(
            my_allreduce,
            configs=[
                CustomOpConfig(allreduce_nccl),
                CustomOpConfig(allreduce_chunked, chunk_size=1024),
            ],
        )

        # Test model
        class SimpleModel(torch.nn.Module):
            def forward(self, x):
                return my_allreduce(x)

        model = torch.compile(SimpleModel())

        # Run
        x = torch.randn(128, 128, device=f'cuda:{rank}')
        y = model(x)

        # Verify
        expected = x * 2  # sum across 2 ranks
        torch.testing.assert_close(y, expected)

        if rank == 0:
            print("✅ Single allreduce test passed!")

        dist.destroy_process_group()
```

**验证点**:
- ✅ 能检测到collective op
- ✅ Timeout机制不触发 (正常完成)
- ✅ 2 ranks同步正常
- ✅ Benchmark结果合理
- ✅ 选择的实现能正确运行

---

### Phase 2: 多个Collective Ops, 2 Ranks

**目标**: 验证多个ops的sync overhead

**测试代码**:
```python
@skip_if_lt_x_gpu(2)
def test_multiple_collectives_2ranks(self):
    """Test 3 collective ops with 2 ranks"""

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()

    # Define 3 different collective ops
    # ... (注册my_allreduce, my_allgather, my_reduce_scatter)

    # Test model with multiple collective ops
    class MultiCollectiveModel(torch.nn.Module):
        def forward(self, x):
            y1 = my_allreduce(x)          # Collective 1
            y2 = my_allgather(y1)         # Collective 2
            y3 = my_reduce_scatter(y2)    # Collective 3
            return y3

    model = torch.compile(MultiCollectiveModel())

    # Measure compilation time
    import time
    start = time.time()
    x = torch.randn(128, 128, device=f'cuda:{rank}')
    y = model(x)
    compile_time = time.time() - start

    if rank == 0:
        print(f"✅ Multiple collectives test passed!")
        print(f"   Compilation time: {compile_time:.2f}s")
        print(f"   Expected: 3 ops × ~50ms sync = ~150ms overhead")

    dist.destroy_process_group()
```

**验证点**:
- ✅ 3个collective ops都能正确autotune
- ✅ 编译时间合理 (~150ms sync overhead for V1)
- ✅ 结果正确性
- ⚠️ 注意观察sync overhead (为V2提供数据支持)

---

### Phase 3: 更多Ops, 更多Ranks

**目标**: 压力测试和scalability验证

**测试代码**:
```python
@skip_if_lt_x_gpu(4)
def test_scalability_4ranks(self):
    """Test 5 collective ops with 4 ranks"""

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    # 5个collective ops
    class LargeCollectiveModel(torch.nn.Module):
        def forward(self, x):
            y1 = my_allreduce(x)
            y2 = my_allreduce(y1)
            y3 = my_allgather(y2)
            y4 = my_reduce_scatter(y3)
            y5 = my_allreduce(y4)
            return y5

    model = torch.compile(LargeCollectiveModel())

    # Measure
    import time
    start = time.time()
    x = torch.randn(256, 256, device=f'cuda:{rank}')
    y = model(x)
    compile_time = time.time() - start

    if rank == 0:
        print(f"✅ Scalability test (4 ranks, 5 ops) passed!")
        print(f"   Compilation time: {compile_time:.2f}s")
        print(f"   Expected V1: 5 ops × ~50ms = ~250ms overhead")
        print(f"   Expected V2: 1 × 5ms = ~5ms overhead (potential savings)")

    dist.destroy_process_group()
```

**验证点**:
- ✅ 4 ranks能正常同步
- ✅ 5个ops都能autotune
- ✅ 结果正确性
- 📊 **关键数据**: 如果sync overhead > 200ms，V2有明显价值

---

### Phase 4: Timeout测试

**目标**: 验证timeout保护机制

**测试代码**:
```python
@skip_if_lt_x_gpu(2)
def test_timeout_protection(self):
    """Test that timeout mechanism prevents hang"""

    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()

    # 模拟: rank 1 sleep，rank 0 正常
    if rank == 1:
        import time
        time.sleep(10)  # Simulate hang

    # 注册一个会timeout的实现
    # ... register with short timeout ...

    model = torch.compile(SimpleModel())

    # Should NOT hang, should timeout gracefully
    try:
        x = torch.randn(128, 128, device=f'cuda:{rank}')
        y = model(x)

        if rank == 0:
            print("✅ Timeout protection worked! Did not hang.")
    except Exception as e:
        if rank == 0:
            print(f"⚠️ Expected timeout exception: {e}")

    dist.destroy_process_group()
```

**验证点**:
- ✅ 不会indefinitely hang
- ✅ Timeout后能fallback
- ✅ 有清晰的warning/error message

---

### 测试运行命令

```bash
# 2 ranks
torchrun --nproc_per_node=2 -m pytest test/inductor/test_collective_autotuning.py::TestCollectiveAutotuning::test_single_allreduce_2ranks -v

# 4 ranks
torchrun --nproc_per_node=4 -m pytest test/inductor/test_collective_autotuning.py::TestCollectiveAutotuning::test_scalability_4ranks -v
```

---

## ❓ 常见问题

### Q1: V1和现有custom op有什么区别？

**A**: V1完全兼容现有机制，只是在collective op场景下：
- 使用`CollectiveBenchmarker`替代regular benchmarker
- 添加了timeout保护
- 跨rank同步benchmark
- 其他流程(inline fusion等)完全不变

### Q2: V1会影响non-collective ops吗？

**A**: 不会。检测逻辑只在`is_collective=True`时触发，其他ops走原有路径。

### Q3: 如果distributed没有initialized怎么办？

**A**: 自动fallback到regular benchmarking，并打印warning。

### Q4: V1的性能overhead是多少？

**A**:
- 单个collective op: ~50ms (和regular autotuning类似)
- N个collective ops: N × 50ms (每个op单独sync)
- V2可以优化到: 5ms + N × benchmark_time

### Q5: 什么时候应该升级到V2？

**A**: 当满足以下任一条件：
- 有3+个collective ops (sync overhead > 150ms)
- 需要benchmark epilogue fusion的性能
- V1稳定运行后，有开发时间

---

## 📁 文件清单

### 已实现
- ✅ `torch/_inductor/runtime/collective_benchmarking.py` (完整实现)

### 待修改
- 🔲 `torch/_inductor/kernel/custom_op.py` (添加detection)
- 🔲 `torch/_inductor/select_algorithm.py` (集成CollectiveBenchmarker)

### 待创建
- 🔲 `test/inductor/test_collective_autotuning.py` (测试)

### 文档
- ✅ 本文档 (MASTER_GUIDE.md)
- ✅ collective_benchmarking.py的docstrings

---

## 🎯 实施Checklist

### Week 1: 核心实现
- [ ] 修改custom_op.py添加detection逻辑
- [ ] 修改select_algorithm.py集成CollectiveBenchmarker
- [ ] 编写Phase 1测试 (单个op, 2 ranks)
- [ ] 验证基础功能

### Week 2: 完善和测试
- [ ] 编写Phase 2测试 (多个ops, 2 ranks)
- [ ] 编写Phase 3测试 (更多ops, 更多ranks)
- [ ] 编写Phase 4测试 (timeout)
- [ ] 性能数据收集
- [ ] 决定是否需要V2

### Week 3+: V2实施 (可选)
- [ ] 创建CollectiveMultiTemplateBuffer类
- [ ] 修改scheduler.py添加unified sync
- [ ] 完整测试和优化
- [ ] 文档更新

---

## 📊 成功指标

### V1成功标准
1. ✅ 能正确autotune custom collective ops
2. ✅ Timeout机制有效，不会hang
3. ✅ 2-4 ranks测试通过
4. ✅ 结果正确性验证通过
5. ✅ 编译时间在预期范围内

### 性能目标
- 单个collective op: < 100ms autotuning overhead
- 多个collective ops: 可以接受线性增长 (V1限制)
- 无hang或crash

### 为V2做准备
- 收集sync overhead数据
- 确认可复用组件
- 验证设计方向

---

## 🚀 开始实施

**推荐流程**:
1. 阅读本文档
2. 查看`collective_benchmarking.py`了解API
3. 修改`custom_op.py` (Step 2)
4. 修改`select_algorithm.py` (Step 3-4)
5. 编写Phase 1测试
6. 迭代优化

**联系方式**:
- Owner: PyTorch Inductor Team
- Module: `torch._inductor`

---

**Let's build it!** 🎉
