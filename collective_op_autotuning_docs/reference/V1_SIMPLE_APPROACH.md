# Collective Op Autotuning 设计文档

## 1. 概述 (Overview)

本设计实现了对分布式collective operations (如`all_reduce`, `all_gather`, `reduce_scatter`等)的custom op autotuning支持。与常规ops不同,collective ops需要在autotuning时进行跨rank同步,确保所有ranks同时开始benchmark。

### 1.1 目标

1. ✅ 复用现有的custom op autotuning基础设施
2. ✅ 针对collective ops添加specialized benchmarking机制
3. ✅ 保证跨rank的同步和准确计时
4. ✅ 最小化对现有代码的侵入性修改

### 1.2 关键挑战

- **跨rank同步**: 所有ranks必须同时开始benchmark才能获得准确的时间测量
- **时间聚合**: 需要收集所有ranks的时间,选择最差情况(max)作为保守估计
- **兼容性**: 必须与现有的autotuning流程无缝集成

---

## 2. 架构设计 (Architecture)

### 2.1 核心组件

```
┌─────────────────────────────────────────────────────────────────┐
│                    1. Detection Layer                           │
│         检测是否为collective op (custom_op.py)                  │
└────────────────────┬────────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────────┐
│                    2. Routing Layer                             │
│    根据op类型路由到不同的benchmarker (select_algorithm.py)       │
└────────────────────┬────────────────────────────────────────────┘
                     │
         ┌───────────┴───────────┐
         │                       │
         ▼                       ▼
┌──────────────────┐    ┌──────────────────────────┐
│  Regular         │    │  Collective              │
│  Benchmarker     │    │  Benchmarker (NEW)       │
│  (existing)      │    │  collective_benchmarking │
└──────────────────┘    └──────────────────────────┘
```

### 2.2 文件结构

#### 新增文件
- **`torch/_inductor/runtime/collective_benchmarking.py`** ✅ 已创建
  - `is_collective_op()`: 检测是否为collective op
  - `benchmark_collective_op()`: 核心benchmarking函数
  - `CollectiveBenchmarker`: 封装的benchmarker类

#### 需要修改的文件
1. **`torch/_inductor/kernel/custom_op.py`**
   - 添加collective op检测
   - 传递process_group metadata

2. **`torch/_inductor/select_algorithm.py`**
   - 在`autotune_select_algorithm`中检测collective ops
   - 路由到`CollectiveBenchmarker`

---

## 3. 详细实现方案

### 3.1 Phase 1: Detection (检测阶段)

**位置**: `torch/_inductor/kernel/custom_op.py` → `autotune_custom_op()`

**修改点**:
```python
def autotune_custom_op(...):
    # ... existing code ...
    
    # NEW: 检测是否为collective op
    is_collective = False
    process_group = None
    
    # 从op_overload或者decompositions中提取信息
    if op_overload:
        op_name = str(op_overload)
        from torch._inductor.runtime.collective_benchmarking import is_collective_op
        is_collective = is_collective_op(op_name)
        
        # 如果是collective op,尝试提取process_group
        if is_collective:
            # 从non_tensor_args中提取process_group
            for kwargs_dict in non_tensor_args:
                if 'group' in kwargs_dict:
                    process_group = kwargs_dict['group']
                    break
    
    # 传递给autotune_select_algorithm
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

### 3.2 Phase 2: Routing (路由阶段)

**位置**: `torch/_inductor/select_algorithm.py` → `autotune_select_algorithm()`

**修改点1**: 修改函数签名
```python
def autotune_select_algorithm(
    name: str,
    choices: list,
    input_nodes: list,
    layout,
    *,
    input_gen_fns=None,
    return_choice=False,
    is_collective=False,  # NEW
    process_group=None,   # NEW
    **kwargs,
):
    cache = get_algorithm_selector_cache()
    
    # 传递给cache
    return cache(
        name,
        choices,
        input_nodes,
        layout,
        input_gen_fns=input_gen_fns,
        return_choice=return_choice,
        is_collective=is_collective,  # NEW
        process_group=process_group,  # NEW
        **kwargs,
    )
```

**修改点2**: 修改`AlgorithmSelectorCache.__call__`
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
        is_collective=False,  # NEW
        process_group=None,   # NEW
        **kwargs,
    ):
        # ... existing cache lookup code ...
        
        if cached_result is not None:
            return cached_result
        
        # NEW: 根据是否为collective op选择benchmarker
        if is_collective:
            from torch._inductor.runtime.collective_benchmarking import (
                CollectiveBenchmarker,
            )
            benchmarker = CollectiveBenchmarker(
                process_group=process_group,
                nruns=config.benchmark_kernel_nruns,
            )
            # 使用specialized benchmarking路径
            result = self._autotune_collective(
                name,
                choices,
                input_nodes,
                layout,
                benchmarker,
                input_gen_fns,
                **kwargs,
            )
        else:
            # 使用现有的regular autotuning路径
            result = self._autotune_regular(
                name, choices, input_nodes, layout, input_gen_fns, **kwargs
            )
        
        # Cache and return
        return result
```

### 3.3 Phase 3: Benchmarking (benchmark阶段)

**位置**: `torch/_inductor/select_algorithm.py` → `AlgorithmSelectorCache._autotune_collective()`

**新增方法**:
```python
class AlgorithmSelectorCache:
    def _autotune_collective(
        self,
        name,
        choices,
        input_nodes,
        layout,
        benchmarker,
        input_gen_fns,
        **kwargs,
    ):
        """Autotune collective operations with cross-rank synchronization."""
        
        # 1. 生成输入数据 (与regular autotuning相同)
        input_tensors = self._generate_inputs(input_nodes, input_gen_fns)
        
        # 2. 遍历所有choices进行benchmarking
        timings = []
        for choice in choices:
            try:
                # 关键: 所有ranks必须同时进入这个benchmark
                # CollectiveBenchmarker内部会做barrier同步
                
                # 准备输出tensor (如果需要)
                output_tensor = self._prepare_output_tensor(choice, layout)
                
                # Benchmark这个choice
                time_us = benchmarker.benchmark(
                    comm_func=choice.kernel,
                    comm_func_name=choice.name,
                    input_tensors=input_tensors,
                    output_tensor=output_tensor,
                )
                
                timings.append((choice, time_us))
                
            except Exception as e:
                log.warning(f"Choice {choice.name} failed: {e}")
                timings.append((choice, float('inf')))
        
        # 3. 选择最佳choice
        # 注意: timings已经是所有ranks的max值(在benchmark_collective_op中处理)
        best_choice, best_time = min(timings, key=lambda x: x[1])
        
        log.info(
            f"[Collective Autotune] {name}: "
            f"Selected {best_choice.name} with time {best_time:.2f} us"
        )
        
        # 4. 返回结果
        if kwargs.get('return_choice', False):
            return self._call_choice(best_choice, input_nodes), best_choice
        else:
            return self._call_choice(best_choice, input_nodes)
```

### 3.4 Phase 4: Synchronization (同步机制)

**位置**: `torch/_inductor/runtime/collective_benchmarking.py` → `benchmark_collective_op()`

**关键代码** (已在文件中实现):

```python
def benchmark_collective_op(...):
    # ... 准备输入参数 ...
    
    # Warmup
    torch.cuda.synchronize()
    comm_func(**input_args, group=process_group)
    torch.cuda.synchronize()
    
    comm_time = 0.0
    for _ in range(nruns):
        # 🔑 关键1: Barrier确保所有ranks同时开始
        dist.barrier(group=process_group)
        torch.cuda.synchronize()
        
        # 🔑 关键2: 使用CUDA events精确计时
        start_evt = torch.cuda.Event(enable_timing=True)
        end_evt = torch.cuda.Event(enable_timing=True)
        
        start_evt.record()
        comm_func(**input_args, group=process_group)
        end_evt.record()
        end_evt.synchronize()
        
        comm_time += start_evt.elapsed_time(end_evt)
    
    comm_time = (comm_time / nruns) * 1000.0  # ms -> us
    
    # 🔑 关键3: All-reduce获取所有ranks的最大时间
    if process_group is not None:
        comm_time_tensor = torch.tensor([comm_time], device=device)
        dist.all_reduce(comm_time_tensor, op=dist.ReduceOp.MAX, group=process_group)
        comm_time = comm_time_tensor.item()
    
    return comm_time
```

**同步流程**:
```
Rank 0                  Rank 1                  Rank N
  │                       │                       │
  ├─ barrier() ──────────┼─ barrier() ──────────┼─ barrier()
  │  (wait for all)       │  (wait for all)       │  (wait for all)
  │                       │                       │
  ├─ cuda.sync()          ├─ cuda.sync()          ├─ cuda.sync()
  │                       │                       │
  ├─ start_event.record() ├─ start_event.record() ├─ start_event.record()
  ├─ collective_op()      ├─ collective_op()      ├─ collective_op()
  ├─ end_event.record()   ├─ end_event.record()   ├─ end_event.record()
  │                       │                       │
  ├─ measure time: t0     ├─ measure time: t1     ├─ measure time: tN
  │                       │                       │
  ├─ all_reduce(MAX) ─────┼─ all_reduce(MAX) ─────┼─ all_reduce(MAX)
  │                       │                       │
  └─ final_time = max(t0, t1, ..., tN) on all ranks
```

---

## 4. 使用示例

### 4.1 注册collective op autotuning

```python
import torch
from torch._inductor.kernel.custom_op import (
    register_custom_op_autotuning,
    CustomOpConfig,
)

# 定义custom collective op
@torch.library.custom_op("mylib::my_allreduce", mutates_args=())
def my_allreduce(tensor: torch.Tensor, group_name: str = "default"):
    return torch.ops._c10d_functional.all_reduce_(
        tensor, "sum", group_name=group_name
    )

# 实现1: NCCL版本
def allreduce_nccl(tensor, group_name="default"):
    return torch.ops._c10d_functional.all_reduce_(
        tensor, "sum", group_name=group_name
    )

# 实现2: 自定义分段allreduce
def allreduce_chunked(tensor, group_name="default", chunk_size=1024):
    # Custom implementation with chunking
    ...

# 注册autotuning
register_custom_op_autotuning(
    my_allreduce,
    configs=[
        CustomOpConfig(allreduce_nccl),
        CustomOpConfig(allreduce_chunked, chunk_size=1024),
        CustomOpConfig(allreduce_chunked, chunk_size=2048),
    ],
    input_gen_fns={
        "tensor": lambda fake: torch.randn_like(fake, device='cuda'),
    },
)
```

### 4.2 在分布式训练中使用

```python
import torch.distributed as dist

# 初始化分布式环境
dist.init_process_group(backend='nccl')

# 编译模型 (inductor会自动autotune collective ops)
model = torch.compile(model)

# 训练时,collective ops会使用autotuned实现
output = model(input)  # 内部的my_allreduce会被autotuned
```

---

## 5. 实现注意事项

### 5.1 必须注意的问题

1. **Barrier同步开销**
   - 每次benchmark都需要barrier,会有额外开销
   - 解决方案: 减少nruns数量,或者使用cached结果

2. **Process Group传递**
   - 必须确保process_group正确传递到benchmarking层
   - 如果missing,默认使用`dist.group.WORLD`

3. **错误处理**
   - 如果某个rank的benchmark失败,整个autotuning会hang
   - 解决方案: 添加timeout机制,或者让所有ranks同时抛出异常

4. **Cache Key生成**
   - Collective ops的cache key需要包含world_size和rank信息
   - 不同的world_size可能有不同的最佳实现

### 5.2 可选优化

1. **Time Estimator**
   - 使用`dist._time_estimator`可以更快地估计时间
   - 但准确性较低,适合快速原型

2. **分层Autotuning**
   - 先用estimator快速筛选,再用实际benchmark精确测量
   - 可以显著减少总autotuning时间

3. **Cached Results共享**
   - Rank 0做autotuning,然后broadcast结果给其他ranks
   - 需要确保所有ranks的硬件配置相同

---

## 6. 与现有Autotuning的对比

| 维度 | Regular Autotuning | Collective Autotuning |
|------|-------------------|----------------------|
| **同步** | 不需要 | 必须barrier同步 |
| **计时** | 单rank | 所有ranks的max |
| **缓存** | 基于shape/dtype | 额外包含world_size |
| **失败处理** | 单rank重试 | 所有ranks同时处理 |
| **开销** | 低 | 中等(barrier开销) |

---

## 7. 测试计划

### 7.1 单元测试

```python
# test/inductor/test_collective_autotuning.py
class TestCollectiveAutotuning(unittest.TestCase):
    def test_allreduce_autotuning(self):
        # 测试all_reduce的autotuning
        ...
    
    def test_allgather_autotuning(self):
        # 测试all_gather的autotuning
        ...
    
    def test_sync_correctness(self):
        # 验证跨rank同步是否正确
        ...
```

### 7.2 集成测试

```python
def test_end_to_end_collective_autotuning():
    # 模拟真实的分布式训练场景
    # 验证autotuned collective op的正确性和性能
    ...
```

---

## 8. 下一步工作

### 8.1 必须完成
- [ ] 修改`custom_op.py`添加detection逻辑
- [ ] 修改`select_algorithm.py`添加routing逻辑
- [ ] 实现`_autotune_collective`方法
- [ ] 编写单元测试和集成测试

### 8.2 可选增强
- [ ] 实现time estimator快速模式
- [ ] 添加更多collective ops支持(broadcast, scatter等)
- [ ] 实现分层autotuning优化
- [ ] 添加性能监控和日志

---

## 9. 参考资料

1. **Autoparallel Reference**
   - https://github.com/meta-pytorch/autoparallel/blob/main/autoparallel/autobucketing_util/estimation_utils.py
   - `benchmark_comm_func`函数的实现

2. **Inductor Autotuning**
   - `/data/users/tianren/pytorch/torch/_inductor/select_algorithm.py`
   - 现有的autotuning基础设施

3. **Custom Op Framework**
   - `/data/users/tianren/pytorch/torch/_inductor/kernel/custom_op.py`
   - Custom op的lowering和autotuning机制

---

## 10. 附录: 关键代码片段索引

### A. Collective Op检测
- 文件: `collective_benchmarking.py`
- 函数: `is_collective_op()`
- 行: 37-47

### B. Benchmarking核心逻辑
- 文件: `collective_benchmarking.py`
- 函数: `benchmark_collective_op()`
- 行: 70-197

### C. 同步机制
- 文件: `collective_benchmarking.py`
- 函数: `benchmark_collective_op()` 中的barrier和all_reduce部分
- 行: 166-193

---

## 联系方式

如有问题或建议,请联系:
- Owner: PyTorch Inductor Team
- Module: `torch._inductor`
