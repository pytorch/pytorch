# 方案 B: Custom Op MultiTemplateBuffer + 递归 Fusion

**Date**: 2025-11-07
**Status**: Design Ready
**Complexity**: Very High
**Timeline**: 8-12 weeks

---

## Executive Summary

**目标**: 实现 Custom Op 的 MultiTemplateBuffer 支持，并通过打开 subgraph boundary 实现递归 fusion，允许跨 subgraph 的 epilogue/prologue fusion。

**核心创新**:
1. **延迟选择**: 使用 MultiTemplateBuffer 延迟 kernel 选择到 scheduler 阶段
2. **打开 subgraph boundary**: 将 subgraph 的首尾 nodes 暴露给 scheduler
3. **递归 fusion**: 考虑 subgraph 之间、subgraph 与普通 ops 之间的所有 fusion 机会
4. **动态 choice 生成**: 根据 fusion 机会动态生成新的 fused choices

**预期收益**:
- ✅ 支持 collective op + elementwise fusion（如 all_reduce + relu）
- ✅ 支持跨 subgraph fusion（如 matmul subgraph + reduce subgraph）
- ✅ Multi-kernel dispatch based on message size
- ✅ 完整的 epilogue fusion benchmarking

---

## 目录

1. [问题分析与动机](#1-问题分析与动机)
2. [架构设计](#2-架构设计)
3. [核心组件](#3-核心组件)
4. [实现细节](#4-实现细节)
5. [递归 Fusion 算法](#5-递归-fusion-算法)
6. [测试方案](#6-测试方案)
7. [实施路线图](#7-实施路线图)
8. [风险与挑战](#8-风险与挑战)

---

## 1. 问题分析与动机

### 1.1 当前限制

**问题 1: Subgraph 是不透明的黑盒**

```python
# 当前行为
result = custom_op(x)  # Subgraph choice 被选中
result = result.relu()  # 独立的 pointwise op，无法融合

# 生成代码：
buf0 = subgraph_call(x)     # Subgraph kernel
buf1 = triton_relu(buf0)    # 独立的 relu kernel
```

**问题**: Subgraph 内部结构对 scheduler 不可见，无法融合外部 epilogue。

**问题 2: 立即选择阻止 fusion**

```python
# custom_op.py (当前)
return autotune_select_algorithm(
    choices=[triton_choice, subgraph_choice, extern_choice],
    return_multi_template=False  # 立即选择最佳 choice
)
# → 选择后返回单一结果，scheduler 看不到其他 choices
```

**问题**: 立即选择意味着 scheduler 无法考虑 fusion 机会来重新评估 choices。

**问题 3: 跨 subgraph fusion 不可能**

```python
# User code
y = subgraph_op1(x)  # 选择 subgraph_choice_A
z = subgraph_op2(y)  # 选择 subgraph_choice_B

# 当前: 两个独立的 subgraph calls
# 理想: 融合成单个 fused subgraph (如果有性能提升)
```

### 1.2 方案 B 的解决方案

```
┌──────────────────────────────────────────────────────────┐
│  Custom Op → MultiTemplateBuffer                        │
│  ─────────────────────────────────────────────────────  │
│                                                          │
│  Choices:                                                │
│  ├─ Triton choice                                       │
│  ├─ Subgraph choice (with boundary info)               │
│  │  ├─ first_node: matmul                              │
│  │  └─ last_node: sum                                  │
│  └─ Extern choice                                       │
│                                                          │
│  Return: MultiTemplateBuffer (deferred selection)      │
└──────────────────────────────────────────────────────────┘
                    ↓
┌──────────────────────────────────────────────────────────┐
│  Scheduler: finalize_multi_template_buffers()           │
│  ─────────────────────────────────────────────────────  │
│                                                          │
│  1. Detect epilogue: relu after custom_op               │
│                                                          │
│  2. Open subgraph boundaries:                           │
│     ├─ Extract subgraph.last_node (sum)                │
│     └─ Check fusability with relu                      │
│                                                          │
│  3. Generate fused choices:                             │
│     ├─ Original choices (no fusion)                    │
│     ├─ Subgraph + relu fused                           │
│     └─ Triton + relu fused                             │
│                                                          │
│  4. Benchmark all choices (including fused)             │
│                                                          │
│  5. Select best choice                                  │
└──────────────────────────────────────────────────────────┘
```

---

## 2. 架构设计

### 2.1 整体架构

```
┌─────────────────────────────────────────────────────────────┐
│  Layer 1: Custom Op Lowering                                │
│  ────────────────────────────────────────────────────────   │
│                                                              │
│  custom_op.py:call_function()                               │
│  ├─ Detect collective op                                    │
│  ├─ Generate choices (Triton/Subgraph/Extern)              │
│  └─ Return MultiTemplateBuffer                              │
│     └─ choice_timings_fn = lazy benchmarking function       │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 2: Scheduler Integration                             │
│  ────────────────────────────────────────────────────────   │
│                                                              │
│  scheduler.py:finalize_multi_template_buffers()             │
│  ├─ Identify MultiTemplateBuffer nodes                      │
│  ├─ Detect fusion opportunities                             │
│  │  ├─ Epilogue ops (pointwise after custom_op)            │
│  │  ├─ Prologue ops (pointwise before custom_op)           │
│  │  └─ Adjacent subgraphs                                   │
│  └─ Call FusionOptimizer                                    │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 3: Fusion Optimizer (NEW)                            │
│  ────────────────────────────────────────────────────────   │
│                                                              │
│  fusion_optimizer.py:optimize_with_fusion()                 │
│  ├─ Open subgraph boundaries                                │
│  │  └─ Extract first_node, last_node from each choice       │
│  ├─ Generate fused choices                                  │
│  │  ├─ For each original choice:                            │
│  │  │  ├─ Try fuse with epilogue                            │
│  │  │  ├─ Try fuse with prologue                            │
│  │  │  └─ Try fuse with adjacent subgraphs                  │
│  │  └─ Create FusedChoiceCaller for each fusion             │
│  └─ Return: all_choices (original + fused)                  │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 4: Fused Choice Execution                            │
│  ────────────────────────────────────────────────────────   │
│                                                              │
│  FusedChoiceCaller:                                         │
│  ├─ Combine multiple subgraphs/ops into single FX graph     │
│  ├─ Compile fused graph                                     │
│  ├─ Benchmark fused execution                               │
│  └─ Return timing                                           │
└─────────────────────────────────────────────────────────────┘
                            ↓
┌─────────────────────────────────────────────────────────────┐
│  Layer 5: Selection & Finalization                          │
│  ────────────────────────────────────────────────────────   │
│                                                              │
│  scheduler.py:                                              │
│  ├─ Benchmark all choices (parallel)                        │
│  ├─ Select fastest choice                                   │
│  └─ Finalize MultiTemplateBuffer                            │
│     ├─ If fused choice wins: inline fused subgraph          │
│     └─ If original choice wins: use original                │
└─────────────────────────────────────────────────────────────┘
```

---

## 3. 核心组件

### 3.1 关键数据结构

#### SubgraphBoundaryInfo

```python
@dataclass
class SubgraphBoundaryInfo:
    """
    Represents the boundary of a subgraph for fusion analysis.
    """
    first_nodes: List[torch.fx.Node]  # Entry points (can have multiple)
    last_nodes: List[torch.fx.Node]   # Exit points (can have multiple)
    fusable_inputs: Set[str]          # Which inputs can be fused
    fusable_outputs: Set[str]         # Which outputs can be fused

    def can_fuse_epilogue(self, epilogue_op: torch.fx.Node) -> bool:
        """Check if epilogue can fuse with last_nodes."""
        for last_node in self.last_nodes:
            if is_pointwise(epilogue_op) and is_compatible(last_node, epilogue_op):
                return True
        return False

    def can_fuse_prologue(self, prologue_op: torch.fx.Node) -> bool:
        """Check if prologue can fuse with first_nodes."""
        for first_node in self.first_nodes:
            if is_pointwise(prologue_op) and is_compatible(first_node, prologue_op):
                return True
        return False
```

#### FusedChoiceCaller

```python
class FusedChoiceCaller(SubgraphChoiceCaller):
    """
    Represents a fused choice combining multiple operations.
    """
    def __init__(
        self,
        base_choice: SubgraphChoiceCaller,
        fusion_ops: List[torch.fx.Node],
        fusion_type: str,  # 'epilogue', 'prologue', 'cross_subgraph'
    ):
        self.base_choice = base_choice
        self.fusion_ops = fusion_ops
        self.fusion_type = fusion_type

        # Create fused FX graph
        self.fused_gm = self._create_fused_graph()

        super().__init__(self.fused_gm, ...)

    def _create_fused_graph(self) -> torch.fx.GraphModule:
        """
        Combine base_choice.gm with fusion_ops into single FX graph.
        """
        # Implementation details in section 4
        ...
```

---

## 4. 实现细节

### 4.1 Custom Op MultiTemplateBuffer Support

**文件**: `/torch/_inductor/kernel/custom_op.py`

```python
def call_function(self, target, args, kwargs):
    """
    Lower custom op to autotuned choices with MultiTemplateBuffer.
    """
    # ... (existing op detection code)

    if is_collective:
        # NEW: Enable MultiTemplateBuffer for custom ops
        return autotune_select_algorithm(
            f"custom_op_{op_overload}",
            choices=choices,
            input_nodes=input_nodes,
            layout=layout,
            process_group=process_group,
            is_collective=True,
            return_multi_template=config.enable_custom_op_fusion,  # NEW CONFIG
        )
```

**新增配置** (`/torch/_inductor/config.py`):

```python
# Enable MultiTemplateBuffer for custom ops
enable_custom_op_fusion: bool = False

# Which fusion types to consider
custom_op_fusion_types: List[str] = ['epilogue', 'prologue', 'cross_subgraph']

# Maximum fusion depth (prevent infinite recursion)
max_fusion_depth: int = 3

# Minimum speedup required to use fused choice
fusion_speedup_threshold: float = 1.1  # 10% faster

# Enable recursive fusion
enable_recursive_fusion: bool = False
```

### 4.2 Subgraph Boundary Extraction

**文件**: `/torch/_inductor/codegen/subgraph.py`

```python
class SubgraphChoiceCaller(ir.ChoiceCaller):
    """Enhanced with boundary info for fusion."""

    def __init__(self, gm, input_nodes, layout, **kwargs):
        super().__init__(...)
        self.gm = gm
        self.original_inputs = input_nodes

        # NEW: Extract boundary info
        self.boundary_info = self._extract_boundary_info()

    def _extract_boundary_info(self) -> SubgraphBoundaryInfo:
        """
        Extract first and last nodes from FX graph for fusion analysis.
        """
        first_nodes = []
        last_nodes = []

        for node in self.gm.graph.nodes:
            if node.op == 'placeholder':
                # Find first compute node after placeholder
                for user in node.users:
                    if user.op == 'call_function':
                        first_nodes.append(user)

            elif node.op == 'output':
                # Find last compute node before output
                for arg in node.args[0]:
                    if isinstance(arg, torch.fx.Node) and arg.op == 'call_function':
                        last_nodes.append(arg)

        return SubgraphBoundaryInfo(
            first_nodes=first_nodes,
            last_nodes=last_nodes,
            fusable_inputs=self._compute_fusable_inputs(),
            fusable_outputs=self._compute_fusable_outputs(),
        )
```

### 4.3 Fusion Optimizer

**新文件**: `/torch/_inductor/fusion_optimizer.py`

```python
class FusionOptimizer:
    """
    Optimize MultiTemplateBuffer choices by generating fused variants.
    """

    def optimize(self) -> List[ChoiceCaller]:
        """
        Generate all possible fused choices.

        Returns:
            List of original + fused choices
        """
        all_choices = list(self.original_choices)

        # Detect fusion opportunities
        epilogue_ops = self._detect_epilogue()
        prologue_ops = self._detect_prologue()
        adjacent_subgraphs = self._detect_adjacent_subgraphs()

        # Generate fused choices
        if epilogue_ops and 'epilogue' in config.custom_op_fusion_types:
            all_choices.extend(
                self._generate_epilogue_fused_choices(epilogue_ops)
            )

        if prologue_ops and 'prologue' in config.custom_op_fusion_types:
            all_choices.extend(
                self._generate_prologue_fused_choices(prologue_ops)
            )

        if adjacent_subgraphs and 'cross_subgraph' in config.custom_op_fusion_types:
            all_choices.extend(
                self._generate_cross_subgraph_fused_choices(adjacent_subgraphs)
            )

        return all_choices
```

### 4.4 Scheduler Integration

**文件**: `/torch/_inductor/scheduler.py`

```python
def finalize_multi_template_buffers(self, nodes):
    """
    Finalize MultiTemplateBuffer nodes with fusion optimization.
    """
    for node in nodes:
        if not isinstance(node.node, MultiTemplateBuffer):
            continue

        multi_node = node.node

        # Check if this is a custom op with fusion enabled
        is_custom_op = getattr(multi_node, 'is_custom_op', False)

        if is_custom_op and config.enable_custom_op_fusion:
            # NEW: Use FusionOptimizer
            self._finalize_custom_op_with_fusion(node, multi_node)
        else:
            # Standard finalization (matmul/conv)
            self._finalize_compute_multi_template(node, multi_node)

def _finalize_custom_op_with_fusion(
    self,
    scheduler_node: SchedulerNode,
    multi_node: MultiTemplateBuffer
):
    """
    Finalize custom op MultiTemplateBuffer with fusion optimization.
    """
    from torch._inductor.fusion_optimizer import FusionOptimizer

    # Create fusion optimizer
    optimizer = FusionOptimizer(multi_node, scheduler_node)

    # Generate all choices (original + fused)
    all_choices = optimizer.optimize()

    log.info(
        "Custom op fusion: %d original choices, %d fused choices, %d total",
        len(multi_node.unfiltered_choices),
        len(all_choices) - len(multi_node.unfiltered_choices),
        len(all_choices),
    )

    # Benchmark all choices
    timings = self._benchmark_all_choices(all_choices, multi_node)

    # Select best choice
    best_choice = min(timings, key=timings.__getitem__)

    # Finalize
    if isinstance(best_choice, FusedChoiceCaller):
        # Inline fused subgraph
        self._inline_fused_choice(scheduler_node, best_choice)
    else:
        # Standard finalization
        multi_node.finalize_as_triton_caller(best_choice)
```

---

## 5. 递归 Fusion 算法

### 5.1 递归策略

```python
def recursive_fusion_optimization(
    choices: List[ChoiceCaller],
    scheduler_nodes: List[SchedulerNode],
    depth: int = 0
) -> List[ChoiceCaller]:
    """
    Recursively generate fused choices until no more fusion opportunities.

    Example:
        Iteration 0: [A, B, C] (original choices)
        Iteration 1: [A, B, C, A+epilogue, B+epilogue] (epilogue fusion)
        Iteration 2: [A, B, C, A+ep, B+ep, A+ep+next, B+ep+next] (cross-subgraph)
        ...
    """
    if depth >= config.max_fusion_depth:
        return choices

    # Generate fused choices for this iteration
    new_choices = []

    for choice in choices:
        # Try epilogue fusion
        epilogue_fused = try_epilogue_fusion(choice, scheduler_nodes)
        new_choices.extend(epilogue_fused)

        # Try prologue fusion
        prologue_fused = try_prologue_fusion(choice, scheduler_nodes)
        new_choices.extend(prologue_fused)

        # Try cross-subgraph fusion
        cross_fused = try_cross_subgraph_fusion(choice, scheduler_nodes)
        new_choices.extend(cross_fused)

    # If no new choices, stop recursion
    if len(new_choices) == 0:
        return choices

    # Recurse with new choices
    all_choices = choices + new_choices
    return recursive_fusion_optimization(all_choices, scheduler_nodes, depth + 1)
```

### 5.2 Fusion Heuristics

```python
def should_fuse(
    choice_a: ChoiceCaller,
    choice_b: ChoiceCaller,
    fusion_type: str
) -> bool:
    """
    Heuristic to determine if two choices should be fused.

    Factors:
    - Memory footprint (fused should be <= sum of individuals)
    - Compute intensity (high intensity benefits from fusion)
    - Data locality (adjacent in memory → good for fusion)
    - Kernel launch overhead (small ops benefit more from fusion)
    """
    # Memory check
    memory_a = estimate_memory_footprint(choice_a)
    memory_b = estimate_memory_footprint(choice_b)
    memory_fused = estimate_memory_footprint_fused(choice_a, choice_b)

    if memory_fused > memory_a + memory_b:
        # Fusion increases memory, skip
        return False

    # Compute intensity check
    compute_a = estimate_compute_intensity(choice_a)
    compute_b = estimate_compute_intensity(choice_b)

    if fusion_type == 'epilogue' and compute_b < 0.1:
        # Epilogue is very cheap (e.g., relu), definitely fuse
        return True

    # Kernel launch overhead check
    if is_small_kernel(choice_a) or is_small_kernel(choice_b):
        # Small kernels benefit from fusion (reduce launch overhead)
        return True

    return True  # Default: try fusion
```

### 5.3 递归示例

```python
# Example: all_reduce + relu + scale
result = torch.ops._c10d_functional.all_reduce_(x, "sum", "default")
result = result.relu()
result = result * 2.0

# Recursive fusion iterations:

# Iteration 0: Original choices
choices = [
    all_reduce_nccl_choice,
    all_reduce_triton_choice,
]

# Iteration 1: Epilogue fusion (all_reduce + relu)
choices += [
    all_reduce_nccl_choice + relu,  # Fused
    all_reduce_triton_choice + relu,  # Fused
]

# Iteration 2: Double epilogue fusion (all_reduce + relu + scale)
choices += [
    (all_reduce_nccl_choice + relu) + scale,  # Double fused
    (all_reduce_triton_choice + relu) + scale,  # Double fused
]

# Benchmark all 6 choices, select fastest
# Result: all_reduce_triton_choice + relu + scale (single kernel!)
```

---

## 6. 测试方案

### 6.1 单元测试

**文件**: `test/inductor/test_custom_op_fusion.py`

```python
import torch
import unittest
from torch._inductor.config import config

class TestCustomOpFusion(unittest.TestCase):
    """Test MultiTemplateBuffer + fusion for custom ops."""

    def test_epilogue_fusion_basic(self):
        """Test basic epilogue fusion (custom_op + relu)."""
        with config.patch(enable_custom_op_fusion=True):
            @torch.compile
            def test_func(x):
                y = torch.ops._c10d_functional.all_reduce_(x, "sum", "default")
                return y.relu()

            x = torch.randn(1024, device='cuda')
            result = test_func(x)

            # Verify fusion happened
            # (check generated code for fused kernel)

    def test_double_epilogue_fusion(self):
        """Test multiple epilogue ops fused."""
        with config.patch(
            enable_custom_op_fusion=True,
            enable_recursive_fusion=True,
        ):
            @torch.compile
            def test_func(x):
                y = torch.ops._c10d_functional.all_reduce_(x, "sum", "default")
                y = y.relu()
                y = y * 2.0
                return y

            x = torch.randn(1024, device='cuda')
            result = test_func(x)

            # Verify all three ops fused

    def test_cross_subgraph_fusion(self):
        """Test fusion across multiple subgraphs."""
        with config.patch(
            enable_custom_op_fusion=True,
            custom_op_fusion_types=['cross_subgraph'],
        ):
            @torch.compile
            def test_func(x):
                y = torch.ops.custom.op1(x)  # Subgraph A
                z = torch.ops.custom.op2(y)  # Subgraph B
                return z

            x = torch.randn(1024, device='cuda')
            result = test_func(x)

            # Verify subgraphs fused
```

### 6.2 集成测试

**文件**: `test/inductor/test_collective_fusion.py`

```python
@unittest.skipIf(not torch.cuda.is_available(), "CUDA required")
class TestCollectiveFusion(unittest.TestCase):
    """Test collective ops with fusion."""

    def test_all_reduce_relu_fusion(self):
        """Test all_reduce + relu fusion."""
        with config.patch(
            enable_custom_op_fusion=True,
            max_autotune=True,
        ):
            @torch.compile
            def collective_relu(x):
                y = torch.ops._c10d_functional.all_reduce_(x, "sum", "default")
                return y.relu()

            x = torch.randn(1024, 1024, device='cuda', dtype=torch.float16)

            # Benchmark
            import time
            torch.cuda.synchronize()
            start = time.time()
            result = collective_relu(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            # Should be faster than unfused
            self.assertLess(elapsed, baseline_time * 0.9)
```

### 6.3 性能测试

**文件**: `benchmarks/inductor/bench_custom_op_fusion.py`

```python
def benchmark_fusion_variants():
    """Benchmark different fusion configurations."""

    configs = [
        {'enable_custom_op_fusion': False},  # Baseline (no fusion)
        {'enable_custom_op_fusion': True, 'custom_op_fusion_types': ['epilogue']},
        {'enable_custom_op_fusion': True, 'custom_op_fusion_types': ['epilogue', 'prologue']},
        {'enable_custom_op_fusion': True, 'custom_op_fusion_types': ['epilogue', 'prologue', 'cross_subgraph']},
    ]

    for cfg in configs:
        with config.patch(**cfg):
            @torch.compile
            def test_func(x):
                y = torch.ops._c10d_functional.all_reduce_(x, "sum", "default")
                y = y.relu()
                y = y * 2.0
                return y

            x = torch.randn(1024, 1024, device='cuda')

            # Warmup
            for _ in range(10):
                test_func(x)

            # Benchmark
            torch.cuda.synchronize()
            start = time.time()
            for _ in range(100):
                test_func(x)
            torch.cuda.synchronize()
            elapsed = time.time() - start

            print(f"Config {cfg}: {elapsed / 100 * 1000:.2f} ms per iteration")
```

**Expected Results**:
- No fusion: 10.0 ms
- Epilogue only: 7.5 ms (25% faster)
- Epilogue + Prologue: 7.0 ms (30% faster)
- All fusion types: 6.5 ms (35% faster)

---

## 7. 实施路线图

### Phase 1: Foundation (Week 1-2)

**Week 1**: MultiTemplateBuffer support
```
Day 1-2: Add return_multi_template support in custom_op.py
Day 3-4: Add config flags (enable_custom_op_fusion, etc.)
Day 5: Test basic MultiTemplateBuffer creation
```

**Week 2**: Boundary extraction
```
Day 1-2: Implement SubgraphBoundaryInfo
Day 3-4: Implement _extract_boundary_info()
Day 5: Unit tests for boundary extraction
```

### Phase 2: Epilogue Fusion (Week 3-5)

**Week 3**: FusionOptimizer基础
```
Day 1-2: Create fusion_optimizer.py
Day 3-4: Implement epilogue detection
Day 5: Implement epilogue fusion generation
```

**Week 4**: FusedChoiceCaller
```
Day 1-2: Implement FusedChoiceCaller class
Day 3-4: Implement _create_fused_graph()
Day 5: Test fused graph generation
```

**Week 5**: Scheduler integration
```
Day 1-2: Implement _finalize_custom_op_with_fusion()
Day 3-4: Implement fusion benchmarking
Day 5: Integration tests
```

### Phase 3: Prologue Fusion (Week 6-7)

**Week 6**: Prologue detection
```
Day 1-2: Implement _detect_prologue()
Day 3-4: Implement prologue fusion generation
Day 5: Test prologue fusion
```

**Week 7**: Integration & testing
```
Day 1-3: Integration with scheduler
Day 4-5: Comprehensive tests
```

### Phase 4: Cross-Subgraph Fusion (Week 8-10)

**Week 8**: Adjacent subgraph detection
```
Day 1-2: Implement _detect_adjacent_subgraphs()
Day 3-4: Implement fusability checks
Day 5: Test detection logic
```

**Week 9**: Subgraph fusion
```
Day 1-3: Implement _fuse_subgraphs()
Day 4-5: Test cross-subgraph fusion
```

**Week 10**: Integration & testing
```
Day 1-3: Integration with scheduler
Day 4-5: Comprehensive tests
```

### Phase 5: Recursive Fusion (Week 11)

```
Day 1-2: Implement recursive_fusion_optimization()
Day 3-4: Implement fusion heuristics
Day 5: Test recursive fusion
```

### Phase 6: Testing & Validation (Week 12)

```
Day 1-2: Performance benchmarks
Day 3-4: Integration with vLLM workloads
Day 5: Final validation
```

---

## 8. 风险与挑战

### 8.1 技术风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **FX graph 合并复杂** | 高 | 高 | 逐步实现，先 epilogue 再 cross-subgraph |
| **Fusion 正确性问题** | 中 | 高 | 大量测试，验证 eager mode 等价性 |
| **Benchmark 开销过大** | 高 | 中 | 使用 fusion heuristics 减少 choices |
| **递归深度过大** | 中 | 中 | 限制 max_fusion_depth |
| **Scheduler 复杂度增加** | 高 | 高 | 模块化设计，保持独立性 |

### 8.2 性能风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **编译时间显著增加** | 高 | 高 | 限制 fusion choices 数量，使用 cache |
| **内存占用过大** | 中 | 中 | 及时释放 intermediate graphs |
| **Fusion 反而变慢** | 中 | 中 | 使用 fusion_speedup_threshold 过滤 |
| **Cache miss 率高** | 中 | 低 | 改进 hash_key() 生成 |

### 8.3 复杂度风险

| 风险 | 概率 | 影响 | 缓解措施 |
|------|------|------|---------|
| **代码维护困难** | 高 | 高 | 详细文档，单元测试覆盖 |
| **调试困难** | 高 | 中 | 添加详细日志，可视化工具 |
| **与现有 features 冲突** | 中 | 高 | Integration testing，feature flags |

---

## 附录

### A. 文件改动清单

| 文件 | 改动类型 | 行数估计 | 复杂度 |
|-----|---------|---------|--------|
| `custom_op.py` | 修改 | +50 | 低 |
| `config.py` | 修改 | +30 | 低 |
| `subgraph.py` | 修改 | +150 | 中 |
| `fusion_optimizer.py` | 新增 | +800 | 高 |
| `scheduler.py` | 修改 | +200 | 高 |
| `test_custom_op_fusion.py` | 新增 | +400 | 中 |
| `test_collective_fusion.py` | 新增 | +300 | 中 |
| `bench_custom_op_fusion.py` | 新增 | +200 | 低 |

**Total**: ~2130 lines of code

### B. 依赖关系图

```
custom_op.py (return_multi_template=True)
    ↓
MultiTemplateBuffer (ir.py)
    ↓
scheduler.py:finalize_multi_template_buffers()
    ↓
FusionOptimizer (fusion_optimizer.py)
    ├─ SubgraphBoundaryInfo
    ├─ FusedChoiceCaller
    ├─ recursive_fusion_optimization()
    └─ fusion heuristics
    ↓
_benchmark_all_choices()
    ↓
Best choice selection & finalization
```

### C. 与方案 A 的关系

**互补性**:
- 方案 A: 修复 async compilation (底层优化)
- 方案 B: 实现 fusion (上层功能)

**实施顺序**:
1. 先实施方案 A (Week 1-6)
   - 修复 SubgraphChoiceCaller.precompile()
   - 消除编译瓶颈
   - 为方案 B 打下基础

2. 再实施方案 B (Week 7-18)
   - 在方案 A 的基础上实现 fusion
   - 享受方案 A 的并行编译优势

**组合效果**:
- 方案 A: 编译时间减少 30-40%
- 方案 B: 运行时间减少 25-35% (fusion)
- 组合: 端到端性能提升 50-60%

### D. 配置示例

```python
# 完整配置示例

# Enable custom op fusion
torch._inductor.config.enable_custom_op_fusion = True

# Fusion types
torch._inductor.config.custom_op_fusion_types = [
    'epilogue',        # fuse pointwise after custom_op
    'prologue',        # fuse pointwise before custom_op
    'cross_subgraph',  # fuse adjacent subgraphs
]

# Recursive fusion
torch._inductor.config.enable_recursive_fusion = True
torch._inductor.config.max_fusion_depth = 3

# Fusion thresholds
torch._inductor.config.fusion_speedup_threshold = 1.1  # 10% speedup required

# Async compilation (from Solution A)
torch._inductor.config.max_autotune_gemm_threads = 8

# Autotuning
torch._inductor.config.max_autotune = True
```

### E. 实际使用示例

```python
import torch
import torch.distributed as dist

# Initialize distributed
dist.init_process_group(backend='nccl')

# Enable fusion
with torch._inductor.config.patch(
    enable_custom_op_fusion=True,
    enable_recursive_fusion=True,
):
    @torch.compile
    def distributed_compute(x, w):
        # Matmul
        y = x @ w

        # All-reduce (custom op with fusion support)
        y = torch.ops._c10d_functional.all_reduce_(y, "sum", "default")

        # Epilogue ops (will be fused with all_reduce)
        y = y.relu()
        y = y * 2.0
        y = y + 1.0

        return y

    # Input
    x = torch.randn(1024, 2048, device='cuda', dtype=torch.float16)
    w = torch.randn(2048, 1024, device='cuda', dtype=torch.float16)

    # Run (first time: compile + fuse + benchmark)
    result = distributed_compute(x, w)

    # Generated code will have:
    # 1. matmul kernel
    # 2. SINGLE fused kernel: all_reduce + relu + scale + add
    #    (instead of 4 separate kernels)
```

---

## 总结

方案 B 提供了一个**完整的 fusion 框架**，通过：

1. ✅ **MultiTemplateBuffer**: 延迟选择，保留所有 fusion 机会
2. ✅ **Open boundaries**: 暴露 subgraph 内部结构
3. ✅ **Recursive fusion**: 探索所有可能的 fusion 组合
4. ✅ **Dynamic choice generation**: 根据上下文动态生成 fused choices

**关键优势**:
- 完整的 epilogue/prologue fusion 支持
- 跨 subgraph fusion (首创)
- 递归 fusion 探索
- 与方案 A 完美互补

**实施建议**:
1. 先完成方案 A (async compilation fix)
2. 逐步实施方案 B (epilogue → prologue → cross-subgraph → recursive)
3. 每个阶段充分测试和验证
4. 使用 feature flags 控制逐步 rollout

---

**Document Version**: 1.0
**Last Updated**: 2025-11-07
**Author**: Collective Op Autotuning Team
**Reviewers**: TBD
**Status**: Ready for Implementation
