# Collective Op Autotuning - 实现路线图 & 关键注意事项

## 📋 总结 (Executive Summary)

本方案设计了一套完整的collective ops autotuning机制,在现有custom op autotuning基础上添加了分布式同步支持。

**核心创新点**:
1. ✅ 复用现有autotuning基础设施,最小化代码修改
2. ✅ 专门的benchmark同步机制确保跨rank准确计时
3. ✅ 模块化设计,易于扩展到更多collective ops

---

## 🎯 实现优先级

### P0 - 必须完成 (核心功能)

1. **✅ 已完成: 创建`collective_benchmarking.py`**
   - 文件路径: `/data/users/tianren/pytorch/torch/_inductor/runtime/collective_benchmarking.py`
   - 包含核心benchmarking逻辑和同步机制

2. **🔲 待完成: 修改`custom_op.py`**
   - 文件: `/data/users/tianren/pytorch/torch/_inductor/kernel/custom_op.py`
   - 函数: `autotune_custom_op()`
   - 修改内容:
     ```python
     # 在autotune_custom_op()中添加:
     from torch._inductor.runtime.collective_benchmarking import is_collective_op

     is_collective = False
     process_group = None

     if op_overload:
         is_collective = is_collective_op(str(op_overload))
         if is_collective:
             for kwargs in non_tensor_args:
                 if 'group' in kwargs:
                     process_group = kwargs['group']
                     break

     # 传递给autotune_select_algorithm
     selected_result, winning_choice = autotune_select_algorithm(
         ...,
         is_collective=is_collective,
         process_group=process_group,
     )
     ```

3. **🔲 待完成: 修改`select_algorithm.py`**
   - 文件: `/data/users/tianren/pytorch/torch/_inductor/select_algorithm.py`
   - 修改点:
     - `autotune_select_algorithm()`函数签名
     - `AlgorithmSelectorCache.__call__()`方法
     - 新增`AlgorithmSelectorCache._autotune_collective()`方法

### P1 - 重要优化 (性能提升)

1. **Cache优化**
   - 修改cache key生成,包含world_size
   - 避免重复autotuning

2. **Time Estimator支持**
   - 添加快速估计模式
   - 在`collective_benchmarking.py`中已有框架,可以启用

3. **错误处理增强**
   - 添加timeout机制
   - 处理部分rank失败的情况

### P2 - 可选功能 (扩展性)

1. **支持更多collective ops**
   - broadcast, scatter, gather等
   - 参考`COLLECTIVE_OPS`集合扩展

2. **分层autotuning**
   - 先用estimator筛选,再精确benchmark
   - 减少总autotuning时间

3. **性能监控**
   - 添加logging和metrics
   - 记录autotuning结果供分析

---

## ⚠️ 关键注意事项

### 1. 同步相关

**问题**: 所有ranks必须同时进入benchmark,否则会hang

**解决方案**:
- ✅ 在`benchmark_collective_op()`中使用`dist.barrier()`
- ✅ 每次benchmark前都同步
- ⚠️ 确保所有ranks的choices顺序一致

**代码位置**:
```python
# collective_benchmarking.py, line ~166
for _ in range(nruns):
    dist.barrier(group=process_group)  # 关键同步点
    torch.cuda.synchronize()
    # ... benchmark ...
```

### 2. Process Group传递

**问题**: process_group可能在kwargs中,也可能是默认的

**解决方案**:
```python
# 优先级:
# 1. 从kwargs中提取 'group' 或 'process_group'
# 2. 如果没有,使用 dist.group.WORLD (默认)
# 3. 如果distributed未初始化,抛出清晰的错误

process_group = kwargs.get('group') or kwargs.get('process_group') or None
```

**代码位置**: `custom_op.py`的detection阶段

### 3. 时间聚合策略

**问题**: 不同ranks可能有不同的timing

**解决方案**:
- ✅ 使用`all_reduce(MAX)`获取最慢的rank时间
- 原因: 保守估计,确保所有ranks都能完成

**代码位置**:
```python
# collective_benchmarking.py, line ~188
comm_time_tensor = torch.tensor([comm_time], device=device)
dist.all_reduce(comm_time_tensor, op=dist.ReduceOp.MAX)
comm_time = comm_time_tensor.item()  # 所有ranks都会得到相同的max值
```

### 4. Cache Key生成

**问题**: 不同world_size可能有不同的最佳实现

**建议修改** (在`select_algorithm.py`):
```python
def _make_cache_key(..., is_collective=False, process_group=None):
    key = [name, str(layout), ...]

    if is_collective and dist.is_initialized():
        world_size = dist.get_world_size(process_group)
        key.append(f"ws_{world_size}")

    return tuple(key)
```

### 5. 错误处理

**场景1: 某个rank的benchmark失败**
```python
# 建议: 让所有ranks都抛出相同的错误,避免hang
try:
    time_us = benchmarker.benchmark(...)
except Exception as e:
    # Broadcast error to all ranks
    error_flag = torch.tensor([1], device='cuda')
    dist.all_reduce(error_flag, op=dist.ReduceOp.MAX)
    raise RuntimeError(f"Benchmark failed on at least one rank: {e}")
```

**场景2: Distributed未初始化**
```python
# 在CollectiveBenchmarker.__init__中已有检查
if not dist.is_initialized():
    log.warning("Distributed not initialized")
    # 运行时会抛出清晰的错误
```

### 6. 输入输出Tensor准备

**问题**: 某些collective ops需要额外的output tensor

**解决方案** (在`collective_benchmarking.py`中已实现):
```python
# all_gather: 需要world_size倍大小的output
if "all_gather" in comm_func_name:
    output_tensor = torch.empty(
        world_size * input_tensor.numel(),
        dtype=input_tensor.dtype,
        device=input_tensor.device
    )

# reduce_scatter: 需要1/world_size大小的output
elif "reduce_scatter" in comm_func_name:
    output_tensor = torch.empty(
        input_tensor.numel() // world_size,
        ...
    )
```

---

## 🔧 实现步骤 (Step-by-Step)

### Step 1: 修改`custom_op.py` (15分钟)

```python
# 在 autotune_custom_op() 函数中,line ~324附近

# 1. 添加import
from torch._inductor.runtime.collective_benchmarking import is_collective_op

# 2. 在调用autotune_select_algorithm之前添加检测逻辑
is_collective = False
process_group = None

if op_overload:
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

# 3. 传递给autotune_select_algorithm
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
```

### Step 2: 修改`select_algorithm.py` - Part A (20分钟)

**位置**: `autotune_select_algorithm()`函数,line ~3908

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
    cache = get_algorithm_selector_cache()

    if "return_multi_template" not in kwargs:
        kwargs["return_multi_template"] = (
            torch._inductor.config.benchmark_epilogue_fusion
        )

    if "precompilation_timeout_seconds" not in kwargs:
        kwargs["precompilation_timeout_seconds"] = config.precompilation_timeout_seconds

    # 传递新参数
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

### Step 3: 修改`select_algorithm.py` - Part B (30分钟)

**位置**: `AlgorithmSelectorCache.__call__()`,需要找到具体行数

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
        is_collective=False,      # NEW
        process_group=None,        # NEW
        **kwargs,
    ):
        # ... existing preprocessing and cache lookup code ...

        # NEW: 添加collective ops的路由逻辑
        if is_collective:
            import torch.distributed as dist
            from torch._inductor.runtime.collective_benchmarking import (
                CollectiveBenchmarker,
            )

            if not dist.is_initialized():
                log.warning(
                    f"Collective op '{name}' requires distributed initialization. "
                    f"Falling back to regular autotuning."
                )
                is_collective = False
            else:
                benchmarker = CollectiveBenchmarker(
                    process_group=process_group,
                    nruns=config.benchmark_kernel_nruns,
                )

                # 使用specialized collective benchmarking
                result = self._autotune_collective(
                    name,
                    choices,
                    input_nodes,
                    layout,
                    benchmarker,
                    input_gen_fns,
                    return_choice=return_choice,
                    **kwargs,
                )

                # Cache the result
                # TODO: 需要修改cache key包含world_size

                return result

        # Regular autotuning path (existing code)
        # ... continue with existing logic ...
```

### Step 4: 实现`_autotune_collective()`方法 (45分钟)

**位置**: `AlgorithmSelectorCache`类中新增方法

```python
class AlgorithmSelectorCache:
    # ... existing methods ...

    def _autotune_collective(
        self,
        name,
        choices,
        input_nodes,
        layout,
        benchmarker,
        input_gen_fns,
        return_choice=False,
        **kwargs,
    ):
        """Autotune collective operations with cross-rank synchronization.

        This method benchmarks collective operations ensuring all ranks
        synchronize before and during benchmarking for accurate timing.
        """
        import torch.distributed as dist

        log.info(
            f"[Collective Autotune] Starting autotuning for {name} "
            f"with {len(choices)} choices"
        )

        # 1. 准备输入数据
        # 复用现有的input generation逻辑
        # TODO: 需要找到现有代码中生成输入的部分

        # 2. 遍历所有choices进行benchmarking
        timings = []

        for idx, choice in enumerate(choices):
            try:
                log.debug(
                    f"[Collective Autotune] Benchmarking choice {idx}: {choice.name}"
                )

                # TODO: 根据choice类型准备output tensor
                # 对于all_gather: output_size = input_size * world_size
                # 对于reduce_scatter: output_size = input_size / world_size

                # 注意: 这里需要实际的input tensors,不是IR nodes
                # 可能需要参考现有的benchmark代码如何生成real tensors

                # Benchmark this choice
                # time_us = benchmarker.benchmark(
                #     comm_func=choice.kernel,
                #     comm_func_name=choice.name,
                #     input_tensors=[...],  # TODO: convert IR nodes to tensors
                #     output_tensor=output_tensor,
                # )

                # timings.append((choice, time_us))

                # PLACEHOLDER: 暂时使用inf作为占位
                timings.append((choice, float('inf')))

            except Exception as e:
                log.warning(
                    f"[Collective Autotune] Choice {choice.name} failed: {e}"
                )
                timings.append((choice, float('inf')))

        # 3. 选择最佳choice
        if not timings:
            raise RuntimeError(f"No valid choices for collective op {name}")

        best_choice, best_time = min(timings, key=lambda x: x[1])

        rank = dist.get_rank()
        if rank == 0:
            log.info(
                f"[Collective Autotune] {name}: "
                f"Selected {best_choice.name} with time {best_time:.2f} us"
            )

        # 4. 调用winning choice生成结果
        # TODO: 需要参考现有代码如何调用choice并获取结果
        # result = self._call_choice(best_choice, input_nodes)

        # 暂时返回None
        if return_choice:
            return None, best_choice  # TODO: fix
        else:
            return None  # TODO: fix
```

---

## 🧪 测试策略

### 单元测试

创建 `test/inductor/test_collective_autotuning.py`:

```python
import unittest
import torch
import torch.distributed as dist
from torch.testing._internal.common_distributed import (
    MultiProcessTestCase,
    skip_if_lt_x_gpu,
)

class TestCollectiveAutotuning(MultiProcessTestCase):
    @skip_if_lt_x_gpu(2)
    def test_allreduce_benchmark(self):
        """Test benchmarking of all_reduce operation."""
        from torch._inductor.runtime.collective_benchmarking import (
            benchmark_collective_op,
            is_collective_op,
        )

        # Initialize distributed
        dist.init_process_group(backend='nccl')

        # Test is_collective_op detection
        self.assertTrue(
            is_collective_op("torch.ops._c10d_functional.all_reduce_.default")
        )

        # Test benchmarking
        tensor = torch.randn(1024, device='cuda')
        comm_func = torch.ops._c10d_functional.all_reduce_.default

        time_us = benchmark_collective_op(
            comm_func=comm_func,
            comm_func_name="all_reduce",
            input_tensors=[tensor],
            output_tensor=None,
            nruns=2,
        )

        self.assertGreater(time_us, 0)

        dist.destroy_process_group()
```

### 集成测试

```python
def test_custom_collective_op_autotuning():
    """End-to-end test of collective op autotuning."""

    # Define custom collective op
    @torch.library.custom_op("test::my_allreduce", mutates_args=())
    def my_allreduce(x: torch.Tensor):
        return torch.ops._c10d_functional.all_reduce_(x, "sum")

    # Register autotuning
    from torch._inductor.kernel.custom_op import (
        register_custom_op_autotuning,
        CustomOpConfig,
    )

    register_custom_op_autotuning(
        my_allreduce,
        configs=[CustomOpConfig()],
    )

    # Test in distributed setting
    # ...
```

---

## 📊 性能预期

### Benchmark开销分析

**Regular Op Autotuning**:
- 单个choice benchmark: ~1-10ms
- 10个choices总时间: ~10-100ms

**Collective Op Autotuning** (with barriers):
- 单个choice benchmark: ~5-20ms (包含barrier开销)
- 10个choices总时间: ~50-200ms
- 额外开销主要来自: barrier同步 + all_reduce聚合

**优化建议**:
1. 减少choices数量 (只选择最有希望的实现)
2. 使用time estimator进行初筛
3. Cache结果避免重复autotuning

---

## 🔗 相关文件索引

### 已创建
- ✅ `/data/users/tianren/pytorch/torch/_inductor/runtime/collective_benchmarking.py`
- ✅ `/data/users/tianren/pytorch/COLLECTIVE_OP_AUTOTUNING_DESIGN.md`
- ✅ `/data/users/tianren/pytorch/COLLECTIVE_OP_ROADMAP.md` (本文件)

### 待修改
- 🔲 `/data/users/tianren/pytorch/torch/_inductor/kernel/custom_op.py`
  - 函数: `autotune_custom_op()` (line ~324)

- 🔲 `/data/users/tianren/pytorch/torch/_inductor/select_algorithm.py`
  - 函数: `autotune_select_algorithm()` (line ~3908)
  - 类方法: `AlgorithmSelectorCache.__call__()` (需要定位)
  - 新增方法: `AlgorithmSelectorCache._autotune_collective()` (新建)

### 待创建
- 🔲 `test/inductor/test_collective_autotuning.py` (单元测试)

---

## ❓ FAQ

### Q1: 为什么不直接修改现有的benchmarking逻辑?

**A**: 为了保持代码清晰和可维护性。Collective ops有独特的同步需求,混在一起会让代码变得复杂且难以调试。

### Q2: 如果只有部分ranks需要autotuning怎么办?

**A**: 目前设计要求所有ranks都参与。如果某些ranks不需要,可以考虑:
- 让它们也进入benchmark但忽略结果
- 或者只在需要的ranks上做autotuning,然后broadcast结果

### Q3: 不同hardware配置的ranks怎么处理?

**A**: 使用`all_reduce(MAX)`确保选择所有ranks都能接受的配置。如果性能差异很大,建议:
- 分别对不同类型的ranks做autotuning
- 或者为heterogeneous设置添加专门的逻辑

### Q4: 如何debug collective autotuning?

**A**:
1. 设置`TORCH_LOGS="+inductor"` 查看详细日志
2. 每个rank单独输出日志到不同文件
3. 使用smaller world_size (2 ranks) 简化调试
4. 添加额外的logging在关键同步点

---

## 总结

这个设计方案提供了一个完整的、模块化的collective ops autotuning实现。关键优势:

✅ **最小侵入**: 只需修改3个文件,新增1个模块
✅ **复用现有**: 充分利用现有autotuning基础设施
✅ **易于扩展**: 模块化设计,容易添加新的collective ops
✅ **性能优化**: 准确的跨rank同步保证benchmark质量

下一步: 按照roadmap依次实现P0功能,然后逐步添加优化和测试。
