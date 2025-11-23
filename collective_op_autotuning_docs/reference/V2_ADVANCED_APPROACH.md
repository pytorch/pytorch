# Collective Op Autotuning - V2升级方案 (基于Scheduler MultiTemplateBuffer)

## 背景

基于与TL的讨论和对MultiTemplateBuffer的理解,这是一个更完整、更general的collective op autotuning方案。

### V1 vs V2 对比

| 特性 | V1 (Custom Op Layer) | V2 (Scheduler Layer) |
|-----|---------------------|---------------------|
| **集成点** | custom_op.py (lowering) | scheduler.py (scheduler phase) |
| **触发时机** | 立即在lowering时 | 在scheduler finalize MultiTemplateBuffer时 |
| **Fusion支持** | ❌ 不支持 | ✅ 支持epilogue fusion |
| **通用性** | 仅custom ops | 所有产生MultiTemplateBuffer的ops |
| **同步点** | 每个op单独sync | 统一在scheduler阶段sync (~5ms window) |
| **实现复杂度** | 简单 | 中等 |

---

## V2 设计方案

### 核心理念

TL建议的流程:
```
Lowering → Scheduler → MultiTemplateBuffer gets realized → 
短暂sync (~5ms) 收集需要collective benchmark的nodes → 
如果同步失败 fallback → 
所有ranks同时benchmark多个choices
```

### 架构

```
┌─────────────────────────────────────────────────────────┐
│          Phase 1: Lowering                              │
│    Custom op降低为MultiTemplateBuffer                   │
│    (包含multiple choices: Triton/ExternKernel)          │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Phase 2: Scheduler                             │
│    识别所有包含collective ops的MultiTemplateBuffers     │
└────────────────────┬────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────┐
│          Phase 3: Pre-Benchmark Sync (NEW)              │
│    ⏱ ~5ms timeout window                                │
│    尝试同步所有ranks,收集需要benchmark的nodes            │
│    如果失败 → fallback to default                        │
└────────────────────┬────────────────────────────────────┘
                     │
                ┌────┴────┐
                │ Success │
                └────┬────┘
                     ▼
┌─────────────────────────────────────────────────────────┐
│    Phase 4: finalize_multi_template_buffers() (MODIFIED)│
│    对于collective ops:                                   │
│    - 跨rank同时benchmark所有choices                      │
│    - 支持fusion: benchmark with/without epilogue         │
│    - 选择最优choice并finalize                            │
└─────────────────────────────────────────────────────────┘
```

---

## 关键组件

### 1. 新增: CollectiveMultiTemplateBuffer (ir.py)

```python
# torch/_inductor/ir.py

class CollectiveMultiTemplateBuffer(MultiTemplateBuffer):
    """
    A MultiTemplateBuffer specifically for collective operations.
    
    Extends MultiTemplateBuffer to handle distributed synchronization
    and benchmarking requirements for collective ops like all_reduce, all_gather.
    """
    
    def __init__(
        self,
        layout: Layout,
        inputs: Sequence[IRNode],
        choice_timings_fn: Callable,
        unfiltered_choices: list[ChoiceCaller],
        allowed_prologue_inps: OrderedSet[str],
        process_group: Optional[dist.ProcessGroup] = None,  # NEW
        collective_op_type: str = "unknown",  # NEW: e.g., "all_reduce"
    ):
        super().__init__(
            layout, inputs, choice_timings_fn, 
            unfiltered_choices, allowed_prologue_inps
        )
        self.process_group = process_group
        self.collective_op_type = collective_op_type
        self._sync_succeeded = False  # Track if pre-sync succeeded
    
    def is_collective(self) -> bool:
        """Mark this as a collective operation buffer."""
        return True
    
    def benchmark_choices_distributed(
        self,
        hint_override: Optional[int] = None,
        timeout_seconds: float = 30.0,
    ) -> dict[ChoiceCaller, float]:
        """
        Benchmark choices with distributed synchronization.
        
        Uses CollectiveBenchmarker to ensure all ranks benchmark simultaneously.
        """
        from torch._inductor.runtime.collective_benchmarking import (
            CollectiveBenchmarker,
            try_collective_benchmark_with_timeout,
        )
        
        benchmarker = CollectiveBenchmarker(
            process_group=self.process_group,
            nruns=config.benchmark_kernel_nruns,
        )
        
        timings = {}
        for choice in self.unfiltered_choices:
            # Try benchmarking with timeout
            time_us = try_collective_benchmark_with_timeout(
                comm_func=choice.kernel,
                comm_func_name=choice.name,
                input_tensors=...,  # prepared inputs
                output_tensor=...,  # prepared output
                process_group=self.process_group,
                timeout_seconds=timeout_seconds,
            )
            
            if time_us is not None:
                timings[choice] = time_us
            else:
                # Timeout or failure, use inf to deprioritize
                timings[choice] = float('inf')
        
        return timings
```

### 2. 修改: scheduler.py - 添加Pre-Benchmark Sync

```python
# torch/_inductor/scheduler.py

class Scheduler:
    def __init__(self, nodes: list[Any]):
        # ... existing init code ...
        self.collective_nodes: list[MultiTemplateBuffer] = []
        self.collective_sync_window = 5.0  # 5ms timeout for initial sync
    
    def collect_collective_nodes(self) -> None:
        """
        Identify all MultiTemplateBuffer nodes that contain collective ops.
        
        This should be called before finalize_multi_template_buffers().
        """
        for node in self.nodes:
            if isinstance(node, SchedulerNode) and isinstance(
                node.node, (MultiTemplateBuffer, CollectiveMultiTemplateBuffer)
            ):
                # Check if this is a collective op
                multi_node = node.node
                if isinstance(multi_node, CollectiveMultiTemplateBuffer):
                    self.collective_nodes.append(multi_node)
                elif self._is_collective_multitemplate(multi_node):
                    # Convert regular MultiTemplateBuffer to Collective version
                    # if it contains collective ops
                    self.collective_nodes.append(multi_node)
    
    def _is_collective_multitemplate(self, node: MultiTemplateBuffer) -> bool:
        """Check if a MultiTemplateBuffer contains collective operations."""
        from torch._inductor.runtime.collective_benchmarking import is_collective_op
        
        # Check choices for collective ops
        for choice in node.unfiltered_choices:
            if hasattr(choice, 'kernel'):
                kernel_name = str(choice.kernel)
                if is_collective_op(kernel_name):
                    return True
        return False
    
    def try_sync_collective_nodes(self) -> bool:
        """
        Attempt to synchronize all ranks before collective benchmarking.
        
        This is the ~5ms sync window mentioned by TL. If sync fails,
        we fallback to regular autotuning without collective sync.
        
        Returns:
            True if sync succeeded, False if timeout/failure
        """
        if not self.collective_nodes:
            return True  # No collective nodes, no need to sync
        
        import torch.distributed as dist
        from torch._inductor.runtime.collective_benchmarking import sync_with_timeout
        
        if not dist.is_initialized():
            log.warning(
                "Distributed not initialized but found collective nodes. "
                "Falling back to regular autotuning."
            )
            return False
        
        rank = dist.get_rank()
        log.info(
            f"[Rank {rank}] Found {len(self.collective_nodes)} collective nodes. "
            f"Attempting sync with {self.collective_sync_window}s timeout..."
        )
        
        # Try to sync all ranks
        sync_ok = sync_with_timeout(
            process_group=None,  # Use default world group
            timeout_seconds=self.collective_sync_window,
        )
        
        if sync_ok:
            log.info(f"[Rank {rank}] Collective sync succeeded!")
            for node in self.collective_nodes:
                if isinstance(node, CollectiveMultiTemplateBuffer):
                    node._sync_succeeded = True
        else:
            log.warning(
                f"[Rank {rank}] Collective sync timeout. "
                f"Falling back to regular autotuning."
            )
        
        return sync_ok
    
    def finalize_multi_template_buffers(self) -> None:
        """
        Finalize backing choices for MultiTemplateBuffers.
        
        MODIFIED to handle collective operations specially.
        """
        # NEW: Step 1 - Collect collective nodes
        self.collect_collective_nodes()
        
        # NEW: Step 2 - Try to sync before benchmarking
        collective_sync_ok = self.try_sync_collective_nodes()
        
        # Existing code continues...
        for i, node in enumerate(self.nodes):
            if isinstance(node, SchedulerNode) and isinstance(
                node.node, MultiTemplateBuffer
            ):
                multi_node = node.node
                
                # NEW: Check if this is a collective node
                is_collective = isinstance(
                    multi_node, CollectiveMultiTemplateBuffer
                ) or multi_node in self.collective_nodes
                
                if is_collective and collective_sync_ok:
                    # Use distributed benchmarking
                    min_node_unfused, min_time = (
                        self._finalize_collective_choice(multi_node)
                    )
                else:
                    # Regular autotuning (existing code)
                    if not config.test_configs.force_extern_kernel_in_multi_template:
                        min_node_unfused, _ = multi_node.get_min_choice()
                    else:
                        # ... existing extern kernel logic ...
                        pass
                
                # ... rest of existing finalization code ...
    
    def _finalize_collective_choice(
        self, multi_node: MultiTemplateBuffer
    ) -> tuple[ChoiceCaller, float]:
        """
        Finalize choice for a collective operation MultiTemplateBuffer.
        
        Uses distributed benchmarking with synchronization.
        """
        if isinstance(multi_node, CollectiveMultiTemplateBuffer):
            # Use the specialized benchmarking method
            timings = multi_node.benchmark_choices_distributed(
                timeout_seconds=30.0  # Full benchmark timeout
            )
        else:
            # Fallback to regular timing
            timings = multi_node.choice_timings()
        
        if not timings:
            raise RuntimeError(
                f"No valid choices for collective MultiTemplateBuffer"
            )
        
        min_choice = min(timings, key=timings.get)
        min_time = timings[min_choice]
        
        return min_choice, min_time
```

### 3. 修改: select_algorithm.py - 创建CollectiveMultiTemplateBuffer

```python
# torch/_inductor/select_algorithm.py

def autotune_select_algorithm(
    name,
    choices,
    input_nodes,
    layout,
    *,
    input_gen_fns=None,
    return_choice=False,
    is_collective=False,  # NEW
    process_group=None,   # NEW
    **kwargs,
):
    cache = get_algorithm_selector_cache()
    
    # ... existing parameter processing ...
    
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
        process_group=None,       # NEW
        **kwargs,
    ):
        # ... existing preprocessing ...
        
        if return_multi_template:
            # NEW: Check if this should be a CollectiveMultiTemplateBuffer
            if is_collective and dist.is_initialized():
                from torch._inductor.ir import CollectiveMultiTemplateBuffer
                
                # Determine collective op type
                collective_op_type = self._infer_collective_type(choices)
                
                return torch._inductor.ir.TensorBox.create(
                    CollectiveMultiTemplateBuffer(
                        layout,
                        input_nodes,
                        get_timings,
                        choices,
                        allowed_prologue_inps,
                        process_group=process_group,          # NEW
                        collective_op_type=collective_op_type, # NEW
                    )
                )
            else:
                # Regular MultiTemplateBuffer
                return torch._inductor.ir.TensorBox.create(
                    torch._inductor.ir.MultiTemplateBuffer(
                        layout,
                        input_nodes,
                        get_timings,
                        choices,
                        allowed_prologue_inps,
                    )
                )
        
        # ... existing non-multi-template code ...
    
    def _infer_collective_type(self, choices: list[ChoiceCaller]) -> str:
        """Infer the type of collective operation from choices."""
        from torch._inductor.runtime.collective_benchmarking import is_collective_op
        
        for choice in choices:
            if hasattr(choice, 'kernel'):
                kernel_name = str(choice.kernel)
                if 'all_reduce' in kernel_name:
                    return 'all_reduce'
                elif 'all_gather' in kernel_name:
                    return 'all_gather'
                elif 'reduce_scatter' in kernel_name:
                    return 'reduce_scatter'
                elif 'all_to_all' in kernel_name:
                    return 'all_to_all'
        
        return 'unknown'
```

---

## 关键优势

### 1. **支持Fusion** ✅
V2可以在scheduler阶段benchmark with/without epilogue fusion:
```python
# 在scheduler.py的fusion逻辑中
if can_fuse(node1, node2):
    # 对于collective MultiTemplateBuffer, 也可以benchmark fused version
    fused_time = benchmark_fused(node1, node2, is_collective=True)
    unfused_time = benchmark_unfused(node1, is_collective=True)
    
    if fused_time < unfused_time:
        fuse_nodes(node1, node2)
```

### 2. **统一同步点** ✅
只需要一次~5ms的pre-sync,而不是每个op都sync:
```python
# V1: 每个op都sync (N次sync)
for collective_op in collective_ops:
    sync_and_benchmark(op)  # barrier每次

# V2: 统一sync一次 (1次sync)
if try_sync_collective_nodes():  # 一次barrier
    for collective_op in collective_ops:
        benchmark(op)  # 内部barrier
```

### 3. **更好的Fallback机制** ✅
如果sync失败,自动fallback到regular autotuning:
```python
if not sync_succeeded:
    # Fallback: 使用第一个choice或者extern kernel
    use_fallback_choice()
    log.warning("Using fallback due to sync timeout")
```

### 4. **通用性** ✅
不仅限于custom ops,任何产生MultiTemplateBuffer的op都能使用:
- Custom ops
- Matmul with collective
- Any fused collective + other ops

---

## 实现时间线

### Phase 1: V1 - 简单版本 (1-2 days)
- ✅ 已完成: collective_benchmarking.py
- 🔲 实现custom_op.py的detection和routing
- 🔲 实现select_algorithm.py的基础集成
- **目标**: 能够为简单的custom collective ops做autotuning

### Phase 2: V1.5 - 添加Timeout (0.5 day)
- ✅ 已完成: sync_with_timeout()
- ✅ 已完成: try_collective_benchmark_with_timeout()
- **目标**: 不会因为某个rank卡住而hang

### Phase 3: V2 - MultiTemplateBuffer集成 (3-4 days)
- 🔲 创建CollectiveMultiTemplateBuffer类
- 🔲 修改scheduler.py添加pre-sync
- 🔲 修改select_algorithm.py创建Collective版本
- 🔲 实现distributed benchmarking in finalize_multi_template_buffers
- **目标**: 支持fusion和更通用的场景

### Phase 4: V2.5 - 优化和测试 (2-3 days)
- 🔲 优化sync window时间
- 🔲 添加详细logging和metrics
- 🔲 编写comprehensive tests
- 🔲 性能优化和cache key改进

---

## 使用示例

### V2使用 - vLLM场景

```python
# vLLM的tensor parallel allreduce
import torch
import torch.distributed as dist
from torch._inductor.kernel.custom_op import (
    register_custom_op_autotuning,
    CustomOpConfig,
)

@torch.library.custom_op("vllm::allreduce_tp", mutates_args=())
def allreduce_tp(
    tensor: torch.Tensor,
    tp_group: str = "default",
) -> torch.Tensor:
    return torch.ops._c10d_functional.all_reduce_(
        tensor, "sum", group_name=tp_group
    )

# 实现1: Standard NCCL
def allreduce_nccl(tensor, tp_group="default"):
    return torch.ops._c10d_functional.all_reduce_(
        tensor, "sum", group_name=tp_group
    )

# 实现2: Ring allreduce (for large tensors)
def allreduce_ring(tensor, tp_group="default", chunk_size=1024**2):
    # Custom ring allreduce implementation
    ...

# 实现3: Tree allreduce (for small tensors)
def allreduce_tree(tensor, tp_group="default"):
    # Custom tree allreduce implementation
    ...

# 注册autotuning - 会自动创建CollectiveMultiTemplateBuffer
register_custom_op_autotuning(
    allreduce_tp,
    configs=[
        CustomOpConfig(allreduce_nccl),
        CustomOpConfig(allreduce_ring, chunk_size=1024**2),
        CustomOpConfig(allreduce_tree),
    ],
    input_gen_fns={
        "tensor": lambda fake: torch.randn_like(fake, device='cuda'),
    },
)

# 在模型中使用
class TPLinear(torch.nn.Module):
    def forward(self, x):
        # Local matmul
        y = x @ self.weight
        
        # Collective op - 会被autotuned, 并且可能和其他ops fusion
        y = allreduce_tp(y, tp_group=self.tp_group)
        
        # Epilogue
        y = y + self.bias
        return y

# Compile模型
model = torch.compile(TPLinear())

# 第一次运行时:
# 1. Lowering阶段: allreduce_tp -> CollectiveMultiTemplateBuffer
# 2. Scheduler阶段: 识别collective node
# 3. Pre-sync (~5ms): 尝试同步所有ranks
# 4. Finalize: benchmark 3个choices (nccl, ring, tree)
#    - 可能还会benchmark with bias fusion
# 5. 选择最优的实现

output = model(input)
```

---

## 与V1的关系

### 共存策略

V1和V2可以共存,按需使用:

```python
# V1: 用于简单场景,custom ops without fusion
# - 在custom_op.py层面直接处理
# - 适合: 单个collective op, 不需要fusion

# V2: 用于复杂场景,需要fusion
# - 在scheduler.py层面处理
# - 适合: collective op + epilogue fusion, 多个collective ops

# 选择策略:
if config.enable_collective_multitemplate:
    # Use V2 - create CollectiveMultiTemplateBuffer
    return_multi_template = True
else:
    # Use V1 - direct benchmarking
    return_multi_template = False
```

### 迁移路径

1. **现在**: 实现V1,验证基础功能
2. **后续**: 逐步迁移到V2,获得fusion支持
3. **最终**: V2成为主要方案,V1作为fallback

---

## 配置选项

```python
# torch/_inductor/config.py

# V1相关
collective_autotune_timeout = 30.0  # Benchmark timeout (seconds)
collective_benchmark_nruns = 3      # Number of runs for benchmarking

# V2相关
enable_collective_multitemplate = True  # 使用V2方案
collective_pre_sync_timeout = 5.0       # Pre-sync window (seconds)
collective_fusion_enabled = True         # 允许collective ops fusion
```

---

## 性能预期

### V1 vs V2 - Overhead对比

| 场景 | V1 Overhead | V2 Overhead | 说明 |
|-----|------------|------------|-----|
| **单个allreduce** | ~50-100ms | ~55-105ms | V2多一次5ms pre-sync |
| **3个allreduce** | ~150-300ms | ~55-105ms | V2只需一次pre-sync |
| **allreduce + fusion** | N/A (不支持) | ~60-120ms | V2支持fusion benchmark |

### V2的收益

对于vLLM这种有多个collective ops的场景:
- **3个allreduce ops**: V1需要3次sync (~150ms), V2只需1次 (~55ms)
- **节省**: ~95ms per compilation
- **Fusion额外收益**: 如果能fusion,运行时性能提升5-15%

---

## 总结

### V1 (Current)
- ✅ 简单,快速实现
- ✅ 足够处理基础场景
- ❌ 不支持fusion
- ❌ 多个collective ops时overhead较大

### V2 (Upgrade)
- ✅ 支持fusion (关键!)
- ✅ 更高效的同步机制
- ✅ 更通用,适用所有MultiTemplateBuffer场景
- ✅ 更好的fallback机制
- ❌ 实现复杂度较高
- ❌ 需要修改scheduler核心逻辑

### 建议

**分阶段实施**:
1. **立即**: 完成V1 + timeout (P0) - 让基础功能work
2. **近期**: 优化和测试V1 (P1) - 稳定后上线
3. **中期**: 实现V2 (P2) - 获得fusion支持
4. **长期**: V2成为默认,V1保留作为简单场景的fast path

这样既能快速deliver功能,又为未来的优化留下空间。
