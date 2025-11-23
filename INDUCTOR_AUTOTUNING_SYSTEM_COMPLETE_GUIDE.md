# PyTorch Inductor Autotuning System - 完整技术文档

> **作者**: Research Analysis
> **日期**: 2025-01-10
> **版本**: 1.0
> **基于代码版本**: PyTorch main branch

---

## 📖 目录

1. [系统架构概览](#系统架构概览)
2. [核心系统对比](#核心系统对比)
3. [Kernel类型映射表](#kernel类型映射表)
4. [Heuristic系统详解](#heuristic系统详解)
5. [Max Autotune与Exhaustive模式](#max-autotune与exhaustive模式)
6. [Template Heuristics机制](#template-heuristics机制)
7. [配置数量完整表](#配置数量完整表)
8. [代码位置索引](#代码位置索引)
9. [实践建议](#实践建议)

---

## 系统架构概览

### 整体设计

PyTorch Inductor的autotuning系统采用**分层架构**，而非多套独立系统：

```
┌─────────────────────────────────────────────────────────────────┐
│              PyTorch Inductor Autotuning 系统                    │
│                                                                  │
│  Layer 1: Algorithm/Backend Selection (高层决策)                │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  autotune_select_algorithm                               │  │
│  │  - 决策: 使用哪个backend实现？                            │  │
│  │  - 输入: [Triton, CUTLASS, ATen, CK, CPP, ...]          │  │
│  │  - 输出: 最优ChoiceCaller 或 MultiTemplateBuffer         │  │
│  │  - 场景: Template kernels (GEMM, Conv, Attention)       │  │
│  └──────────────────┬───────────────────────────────────────┘  │
│                     │                                           │
│                     ▼ (选中Triton template)                     │
│                                                                  │
│  Layer 2: Config Parameter Tuning (低层优化)                    │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  CachingAutotuner                                        │  │
│  │  - 决策: 哪个config参数最优？                             │  │
│  │  - 输入: 单个Triton kernel + configs                     │  │
│  │  - 输出: 最优launcher                                     │  │
│  │  - 场景: Triton kernel内部 + Codegen fusion kernels     │  │
│  └──────────────────────────────────────────────────────────┘  │
│                                                                  │
│  Heuristics: Config生成器 (支撑层)                              │
│  ┌──────────────────────────────────────────────────────────┐  │
│  │  Runtime Heuristics (triton_heuristics.py)              │  │
│  │  - pointwise(), reduction(), persistent_reduction()     │  │
│  │  - 生成: CachingAutotuner的configs                       │  │
│  │                                                          │  │
│  │  Template Heuristics (template_heuristics/)             │  │
│  │  - CUDAMMTemplateConfigHeuristic, ROCmMM, ...           │  │
│  │  - 生成: autotune_select_algorithm的configs             │  │
│  └──────────────────────────────────────────────────────────┘  │
└─────────────────────────────────────────────────────────────────┘
```

### 关键洞察

**这不是两套系统，而是一个系统的两个层级！**

- **Layer 1 (autotune_select_algorithm)**: 选择开哪辆车（backend）
- **Layer 2 (CachingAutotuner)**: 调整车的参数（config）
- **Heuristics**: 提供候选选项（生成器）

---

## 核心系统对比

### 1. CachingAutotuner vs autotune_select_algorithm

#### 代码位置

| 组件 | 文件 | 行号 |
|------|------|------|
| **CachingAutotuner** | `/torch/_inductor/runtime/triton_heuristics.py` | 260-1456 |
| **autotune_select_algorithm** | `/torch/_inductor/select_algorithm.py` | 4076-4087 |
| **AlgorithmSelectorCache** | `/torch/_inductor/select_algorithm.py` | 2619-3050 |
| **MultiTemplateBuffer** | `/torch/_inductor/ir.py` | 5269-5357 |

#### 功能对比表

| 维度 | CachingAutotuner | autotune_select_algorithm |
|------|-----------------|---------------------------|
| **层级** | 低层 - Config tuning | 高层 - Backend selection |
| **输入** | 单个Triton kernel + configs列表 | 多个ChoiceCaller (跨backend) |
| **输出** | 最优launcher | ChoiceCaller或MultiTemplateBuffer |
| **决策** | BLOCK_M=64还是128? | 用Triton还是CUTLASS? |
| **缓存** | 磁盘 (size_hints key) | 内存 + 磁盘 (operation key) |
| **时机** | 运行时 | 编译时 |
| **使用场景** | Triton内部优化 + Codegen kernels | Template kernels (MM/Conv/Attn) |

#### 决策树

```
Operation需要编译
    │
    ├─ 是Template kernel? (MM/Conv/Attention)
    │  │
    │  └─ YES → autotune_select_algorithm()
    │      ├─ 收集choices: [Triton, CUTLASS, ATen, CK, ...]
    │      ├─ 预编译所有choices
    │      ├─ Benchmark每个choice
    │      ├─ 选择最优 → 假设选中Triton
    │      │
    │      └─ Triton内部 → CachingAutotuner
    │          ├─ configs: [Config(BLOCK_M=64), Config(BLOCK_M=128), ...]
    │          ├─ Benchmark所有configs
    │          └─ 选择最优launcher
    │
    └─ 是Codegen fusion? (pointwise/reduction)
       │
       └─ YES → 直接 CachingAutotuner
           ├─ @cached_autotune装饰器
           ├─ configs由heuristic生成
           └─ Benchmark选最优
```

### 2. MultiTemplateBuffer

**定义**: 延迟backend选择的容器，支持epilogue fusion优化

**创建位置**: **仅在** `select_algorithm.py:2963`

```python
# 创建条件
if return_multi_template and (config.max_autotune or config.max_autotune_gemm):
    return ir.TensorBox.create(
        ir.MultiTemplateBuffer(
            layout,
            input_nodes,
            get_timings,              # 延迟benchmark函数
            choices,                  # 所有backend choices
            allowed_prologue_inps,
        )
    )
```

**使用场景**:
1. ✅ autotune_select_algorithm创建
2. ✅ Scheduler中进行epilogue fusion
3. ✅ 联合benchmark (kernel + fusion)
4. ❌ CachingAutotuner **不使用** MultiTemplateBuffer

**为什么CachingAutotuner不用?**
- 处理单个kernel，无需跨backend
- 立即benchmark，不延迟
- 已在Layer 2，无需更高层抽象

---

## Kernel类型映射表

### Template Kernels → autotune_select_algorithm

| Operation | ATen操作符 | 文件位置 | Backend选项 | MultiTemplate支持 |
|-----------|-----------|---------|------------|------------------|
| **Matrix Multiply** | `aten.mm` | `mm.py:1323` | Triton, CUTLASS, ATen, CK, CPP | ✅ |
| **Batch MM** | `aten.bmm` | `bmm.py:135` | Triton, CUTLASS, ATen, CK | ✅ |
| **Add MM** | `aten.addmm` | `mm.py:1378` | Triton, CUTLASS, ATen, CK | ✅ |
| **Scaled MM** | `aten._scaled_mm` | `mm.py:1772` | Triton, CUTLASS, ATen | ✅ |
| **Grouped MM** | `aten._grouped_mm` | `mm_grouped.py:791` | Triton, CUTLASS, ATen | ✅ |
| **Convolution** | `aten.convolution` | `conv.py:650` | Triton, CK, ATen | ✅ |
| **Flex Attention** | custom | `flex_attention.py:429` | Triton (多variants) | ✅ |
| **Flex Decoding** | custom | `flex_decoding.py:388` | Triton (多variants) | ✅ |
| **Custom Op** | user-defined | `custom_op.py:320` | 用户定义 | ✅ |

### Codegen Fusion Kernels → CachingAutotuner

| Kernel类型 | Heuristic函数 | Config范围 | 文件位置 |
|-----------|--------------|-----------|---------|
| **Pointwise 1D** | `pointwise()` | XBLOCK, num_warps | `triton_heuristics.py:2599` |
| **Pointwise 2D** | `pointwise()` | XBLOCK, YBLOCK, num_warps | `triton_heuristics.py:2673` |
| **Pointwise 3D** | `pointwise()` | XBLOCK, YBLOCK, ZBLOCK | `triton_heuristics.py:2712` |
| **Reduction** | `reduction()` | XBLOCK, R0_BLOCK, num_warps | `triton_heuristics.py:2798` |
| **Persistent Reduction** | `persistent_reduction()` | XBLOCK, R0_BLOCK | `triton_heuristics.py:3396` |
| **Foreach** | `foreach()` | num_warps | `triton_heuristics.py:3613` |
| **Split Scan** | `split_scan()` | XBLOCK, R0_BLOCK | `triton_heuristics.py:3463` |

### 判断规则

```python
# 伪代码
if kernel有@register_lowering且是template:
    system = "autotune_select_algorithm"
    configs_source = "template_heuristics/"
elif kernel是codegen生成:
    system = "CachingAutotuner"
    configs_source = "triton_heuristics.py"
else:
    system = "Direct execution"
    configs_source = None
```

---

## Heuristic系统详解

### 1. Heuristic的本质

**Heuristic ≠ Autotuning**

```
Heuristic = Config生成器
Autotuning = Performance测试器

┌──────────────┐
│ Input Shapes │
│ size_hints   │
└──────┬───────┘
       │
       ▼
┌────────────────────────┐
│  Heuristic函数         │
│  根据shape生成configs  │
└──────┬─────────────────┘
       │
       ├─ max_autotune=OFF → [1个config]
       └─ max_autotune=ON  → [4-13个configs]
       │
       ▼
┌────────────────────────┐
│  Autotuner             │
│  Benchmark这些configs  │
│  选择最优               │
└────────────────────────┘
```

### 2. Runtime Heuristics (triton_heuristics.py)

#### Pointwise Heuristics

**文件位置**: `triton_heuristics.py:2599-2740`

**决策逻辑**:
```python
def pointwise(size_hints, inductor_meta, ...):
    max_autotune = inductor_meta.get("max_autotune") or \
                   inductor_meta.get("max_autotune_pointwise")

    if len(size_hints) == 1:  # 1D
        if not autotune_pointwise and not max_autotune:
            return [1个config]  # 快速路径
        else:
            return [2-3个基础 + ROCm额外5个]  # 完整autotuning

    elif len(size_hints) == 2:  # 2D
        if not autotune_pointwise and not max_autotune:
            return [1个config(32,32)]
        else:
            return [6个基础 + ROCm额外4个]

    elif len(size_hints) == 3:  # 3D
        if not autotune_pointwise:
            return [1个config]
        else:
            return [7个基础configs]
```

#### Reduction Heuristics

**文件位置**: `triton_heuristics.py:2798-3000`

**决策逻辑**:
```python
def _reduction_configs(size_hints, inductor_meta, ...):
    max_autotune = inductor_meta.get("max_autotune") or \
                   inductor_meta.get("max_autotune_pointwise")

    # 生成基础configs
    contiguous_config = make_config(x=1, r=min(rnumel, 2048))
    outer_config = make_config(x=64, r=8)
    tiny_config = make_config(...)

    # 检查快速路径
    if not max_autotune:
        if reduction_hint == ReductionHint.INNER:
            return [contiguous_config]  # 1个
        elif reduction_hint == ReductionHint.OUTER:
            return [outer_config]  # 1个
        elif reduction_hint == ReductionHint.OUTER_TINY:
            return [tiny_config]  # 1个

    # 完整autotuning路径
    return [
        contiguous_config,
        outer_config,
        tiny_config,
        make_config(64, 64),
        make_config(8, 512),
        make_config(64, 4, num_warps=8),
        # + ROCm额外2个
    ]  # 7-9个
```

#### Persistent Reduction Heuristics

**文件位置**: `triton_heuristics.py:3396-3460`

**xblock值范围**:
```python
if torch.version.hip:  # ROCm
    xblock_vals = [1, 4, 8, 16, 32, 64, 128, 256]  # 8个
else:  # CUDA
    xblock_vals = [1, 8, 32, 128]  # 4个
```

**决策逻辑**:
```python
def _persistent_reduction_configs(...):
    configs = [生成xblock_vals的configs]

    if not max_autotune:
        if reduction_hint == INNER and rnumel >= 256:
            return configs[:1]  # 仅第一个
        elif reduction_hint == OUTER:
            return configs[-1:]  # 仅最后一个
        # ...
    else:
        # 返回所有configs
        return configs  # 4-9个
```

### 3. Template Heuristics (template_heuristics/)

#### 目录结构

```
template_heuristics/
├── __init__.py              # 包初始化
├── registry.py              # 注册系统
├── base.py                  # TemplateConfigHeuristics基类
├── params.py                # 参数配置类
├── triton.py               # 主要实现 (~2600行)
├── gemm.py                 # GEMM基类
├── cutedsl.py              # CuTe DSL支持
├── decompose_k.py          # K分解策略
├── contiguous_mm.py        # 连续性优化
└── aten.py                 # ATen后端
```

#### 注册机制

**文件位置**: `template_heuristics/registry.py`

```python
# 注册装饰器
@register_template_heuristic(
    template_name="mm",      # 模板名
    device_type="cuda",      # 设备类型
    op_name="addmm",        # 操作名（可选）
    register=True           # 条件注册
)
class CUDAAddMMTemplateConfigHeuristic(BaseHeuristic):
    pass

# 查询优先级
def get_template_heuristic(template_name, device_type, op_name):
    """
    优先级（从高到低）：
    1. (template_name, device_type, op_name)  # 最具体
    2. (template_name, None, op_name)          # 跨设备
    3. (template_name, device_type, None)      # 跨操作
    4. (template_name, None, None)             # 通用
    """
```

#### 与select_algorithm集成

**调用链**:
```
kernel/mm.py (get_mm_configs)
    ↓
choices.py (get_ktc方法)
    ↓
registry.get_template_heuristic(template_name, device_type, op_name)
    ↓
heuristic.get_template_configs(kernel_inputs, op_name)
    ↓
make_ktc_generator() → KernelTemplateChoice
    ↓
select_algorithm.py (ChoiceCaller生成与autotuning)
```

---

## Max Autotune与Exhaustive模式

### 1. 配置标志定义

**文件位置**: `config.py:459-543`

```python
# 总开关
max_autotune = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE") == "1"

# 独立开关
max_autotune_pointwise = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_POINTWISE") == "1"
max_autotune_gemm = os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM") == "1"

# GEMM搜索空间
max_autotune_gemm_search_space: Literal["DEFAULT", "EXHAUSTIVE"] = \
    os.environ.get("TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE", "DEFAULT").upper()
```

### 2. 三种模式对比

| 配置 | max_autotune | max_autotune_gemm | search_space | 行为 |
|------|-------------|-------------------|--------------|------|
| **默认** | False | False | DEFAULT | 无autotuning或轻量级 |
| **仅GEMM** | False | True | DEFAULT | 仅GEMM使用20个configs |
| **全局** | True | True (推导) | DEFAULT | 所有ops使用完整configs |
| **GEMM穷举** | True | True | EXHAUSTIVE | GEMM使用1875个configs |

### 3. max_autotune的作用

#### Layer 1: Backend选择

```python
# kernel/mm.py:1200-1250
if not (max_autotune or max_autotune_gemm):
    # 快速路径：仅ATen
    choices = [aten_mm]
else:
    # 完整路径：多backend
    choices = [
        aten_mm,
        triton_mm_template,
        cutlass_template,
        ck_template,
    ]
```

#### Layer 2: Config数量

```python
# triton_heuristics.py:2629-2633 (pointwise例子)
if not max_autotune and not max_autotune_pointwise:
    configs = [1个config]
else:
    configs = [2-10个configs]
```

### 4. EXHAUSTIVE搜索空间

#### GEMM配置生成

**文件位置**: `template_heuristics/triton.py:253-261`

```python
exhaustive_configs = [
    GemmConfig(BLOCK_M, BLOCK_N, BLOCK_K, num_stages, num_warps, group_m)
    for BLOCK_M in [16, 32, 64, 128, 256]      # 5个
    for BLOCK_N in [16, 32, 64, 128, 256]      # 5个
    for BLOCK_K in [16, 32, 64, 128, 256]      # 5个
    for num_stages in [1, 2, 3, 4, 5]          # 5个
    for num_warps in [2, 4, 8]                 # 3个
    for group_m in [8]                          # 1个
]
# 总计: 5 * 5 * 5 * 5 * 3 * 1 = 1875个configs
```

#### 穷举剪枝策略

**文件位置**: `template_heuristics/triton.py:679-704`

```python
def _prune_exhaustive_configs(configs, dtype_size):
    """
    剪枝条件:
    1. Shared memory超限
    2. Register数量 > 255 (必然spill)
    """
    pruned = []
    for config in configs:
        # 检查register压力
        acc_regs = math.ceil(
            config.block_m * config.block_n / (config.num_warps * 32)
        )
        if acc_regs > 255:
            continue  # 跳过

        pruned.append(config)
    return pruned
```

#### 不同设备的EXHAUSTIVE

| 设备 | 基础configs | 额外参数 | 总数 |
|------|-----------|---------|------|
| **CUDA** | 1875 | - | **1875** |
| **ROCm** | 1875 | matrix_instr×2, waves_per_eu×2, kpack×1 | **7500** |
| **Flex Attention** | - | BLOCK_M×4, BLOCK_N×3, stages×4, warps×3 | **144** |

---

## 配置数量完整表

### 1. Triton Codegen Kernels (CachingAutotuner)

#### Pointwise

| 维数 | max_autotune=OFF | max_autotune=ON | +ROCm | 总计(ON) |
|------|-----------------|-----------------|-------|---------|
| **1D** | 1 | 2-3 | +5 | **7-10** |
| **2D** | 1 | 6 | +4 | **10-13** |
| **3D** | 1 | 7 | 0 | **7-10** |

**关键因素**:
- `autotune_pointwise` flag
- `TileHint.SQUARE` (2D情况)
- `AutotuneHint` 额外configs

**代码位置**: `triton_heuristics.py:2599-2740`

#### Reduction

| 场景 | max_autotune=OFF | max_autotune=ON | +ROCm | deterministic |
|------|-----------------|-----------------|-------|---------------|
| **with hint (INNER/OUTER/TINY)** | 1 | 7 | +2 | 1 |
| **without hint** | 6 | 7 | +2 | 1 |
| **3D tiling** | 可变 | 可变 | - | 1 |

**关键因素**:
- `reduction_hint` (INNER/OUTER/OUTER_TINY/DEFAULT)
- `deterministic` mode强制过滤到1个
- `force_filter_reduction_configs`

**代码位置**: `triton_heuristics.py:2798-3000`

#### Persistent Reduction

| 场景 | max_autotune=OFF | max_autotune=ON | xblock_vals | 总计(ON) |
|------|-----------------|-----------------|------------|---------|
| **CUDA** | 1 | 4 | [1,8,32,128] | **4** |
| **ROCm** | 1 | 8-9 | [1,4,8,16,32,64,128,256] | **8-9** |

**关键因素**:
- `reduction_hint`
- `rnumel` (reduction元素数量)
- 平台 (CUDA vs ROCm)

**代码位置**: `triton_heuristics.py:3396-3460`

#### Foreach

| 模式 | max_autotune=OFF | max_autotune=ON |
|------|-----------------|-----------------|
| **num_warps** | [8] | [1, 2, 4, 8] |
| **总计** | **1** | **4** |

**代码位置**: `triton_heuristics.py:3613-3635`

### 2. Template Kernels (autotune_select_algorithm)

#### GEMM Templates

| Template | Device | DEFAULT | EXHAUSTIVE | 文件位置 |
|----------|--------|---------|------------|---------|
| **MM** | CUDA | 20 | 1875 | `triton.py:253` |
| **MM** | ROCm | 20 | 7500 | `triton.py:1161` |
| **Persistent TMA MM** | CUDA | 15 | - | - |
| **Scaled MM** | CUDA | 18 | - | - |
| **Blackwell MM** | CUDA | 12 | - | - |

**DEFAULT configs示例** (triton.py:60-81):
```python
mm_configs = [
    GemmConfig(64, 64, 32, 2, 4, 8),
    GemmConfig(64, 128, 32, 3, 4, 8),
    GemmConfig(128, 64, 32, 3, 4, 8),
    GemmConfig(128, 128, 32, 3, 4, 8),
    GemmConfig(256, 64, 32, 4, 4, 8),
    # ... 共20个
]
```

#### Attention Templates

| Template | DEFAULT | EXHAUSTIVE | 说明 |
|----------|---------|------------|------|
| **Flex Attention Forward** | 18 | 144 | BLOCK_M×4, BLOCK_N×3, stages×4, warps×3 |
| **Flex Attention Backward** | 10 | 120 | - |
| **Flex Decoding** | 12 | 96 | - |

**代码位置**: `template_heuristics/triton.py:495-519`

#### Grouped GEMM

| Template | DEFAULT | EXHAUSTIVE |
|----------|---------|------------|
| **Grouped MM (Triton)** | 15 | 600 |
| **Grouped MM (CuTe)** | 8 | 128 |

### 3. 完整决策矩阵

```
┌─────────────────────┬──────────────────┬──────────────────┬──────────────────┐
│ Kernel类型          │ max_autotune=OFF │ max_autotune=ON  │ EXHAUSTIVE模式   │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ POINTWISE_1D        │ 1                │ 7-10             │ 同ON             │
│ POINTWISE_2D        │ 1                │ 10-13            │ 同ON             │
│ POINTWISE_3D        │ 1                │ 7-10             │ 同ON             │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ REDUCTION (hint)    │ 1                │ 7-9              │ 同ON             │
│ REDUCTION (no hint) │ 6                │ 7-9              │ 同ON             │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ PERSISTENT (CUDA)   │ 1                │ 4                │ 同ON             │
│ PERSISTENT (ROCm)   │ 1                │ 8-9              │ 同ON             │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ FOREACH             │ 1                │ 4                │ 同ON             │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ MM (CUDA)           │ 1 (ATen only)    │ 20               │ 1875             │
│ MM (ROCm)           │ 1 (ATen only)    │ 20               │ 7500             │
├─────────────────────┼──────────────────┼──────────────────┼──────────────────┤
│ FLEX_ATTN_FWD       │ N/A              │ 18               │ 144              │
│ FLEX_ATTN_BWD       │ N/A              │ 10               │ 120              │
│ FLEX_DECODING       │ N/A              │ 12               │ 96               │
└─────────────────────┴──────────────────┴──────────────────┴──────────────────┘
```

**说明**:
- Codegen kernels的EXHAUSTIVE与ON相同（无template_heuristics）
- Template kernels的EXHAUSTIVE显著增加configs数量

---

## Template Heuristics机制

### 1. 工作流程

```
1. Kernel Lowering
   ├─ tuned_mm(mat1, mat2) 被调用
   └─ 需要生成backend choices
      ↓
2. Template Heuristic查询
   ├─ get_template_heuristic("mm", "cuda", "mm")
   └─ 返回 CUDAMMTemplateConfigHeuristic 实例
      ↓
3. Config生成
   ├─ heuristic.get_template_configs(kernel_inputs, "mm")
   ├─ 内部调用 _get_config_generator()
   │  ├─ search_space="DEFAULT" → get_mm_configs()
   │  └─ search_space="EXHAUSTIVE" → get_exhaustive_mm_configs()
   └─ 返回 configs iterator
      ↓
4. Config预处理
   ├─ preprocess_mm_configs(m, n, k, configs, ...)
   ├─ _filter_configs() - 设备特定过滤
   ├─ _scale_mm_configs() - 根据shape缩放
   ├─ _prune_exceeding_max_shared_mem_configs()
   └─ _prune_exhaustive_configs() (if EXHAUSTIVE)
      ↓
5. 生成KernelTemplateChoice
   ├─ make_ktc_generator(template, configs, ...)
   └─ 每个config生成一个choice
      ↓
6. Autotuning
   ├─ autotune_select_algorithm(choices)
   └─ 选择最优choice
```

### 2. Config缩放逻辑

**文件位置**: `template_heuristics/triton.py:762-882`

```python
def _scale_mm_configs(self, m, n, k, configs, scale, ...):
    """
    根据M/N/K大小动态缩放config参数
    """
    # 小shape检测
    if m <= 32 and n <= 32:
        # 使用tiny configs
        configs = [c for c in configs if c.block_m <= 32 and c.block_n <= 32]

    # 大shape检测
    if m >= 2048 and n >= 2048:
        # 使用大block configs
        configs = [c for c in configs if c.block_m >= 128]

    # K维度调整
    if k <= 64:
        configs = [c for c in configs if c.block_k <= 64]

    # 缩放因子应用
    for config in configs:
        config.block_m = min(config.block_m * scale, m)
        config.block_n = min(config.block_n * scale, n)

    return configs
```

### 3. 设备特定优化

#### CUDA优化 (triton.py:697-742)

```python
class CUDAConfigHeuristic(BaseConfigHeuristic):
    def _filter_configs(self, configs):
        # CUDA特定过滤逻辑
        # 1. 移除num_stages=0的configs
        # 2. 调整num_warps基于SM数量
        return filtered_configs
```

#### ROCm优化 (triton.py:1068-1208)

```python
class ROCmConfigHeuristic(BaseConfigHeuristic):
    def _filter_configs(self, configs):
        # ROCm特定优化
        # 1. matrix_instr_nonkdim参数
        # 2. waves_per_eu调整
        # 3. kpack设置
        # 4. num_stages限制（通常≤2）
        return filtered_configs
```

---

## 代码位置索引

### 核心系统文件

| 文件 | 路径 | 关键内容 |
|------|------|---------|
| **CachingAutotuner** | `/torch/_inductor/runtime/triton_heuristics.py` | 行260-1456 |
| **autotune_select_algorithm** | `/torch/_inductor/select_algorithm.py` | 行4076-4087 |
| **AlgorithmSelectorCache** | `/torch/_inductor/select_algorithm.py` | 行2619-3050 |
| **MultiTemplateBuffer** | `/torch/_inductor/ir.py` | 行5269-5357 |
| **CoordescTuner** | `/torch/_inductor/runtime/coordinate_descent_tuner.py` | 全文件 |

### Runtime Heuristics

| 函数 | 文件 | 行号 |
|------|------|------|
| `pointwise()` | `triton_heuristics.py` | 2599-2740 |
| `reduction()` | `triton_heuristics.py` | 3187-3224 |
| `_reduction_configs()` | `triton_heuristics.py` | 2798-3000 |
| `persistent_reduction()` | `triton_heuristics.py` | 3396-3460 |
| `foreach()` | `triton_heuristics.py` | 3613-3635 |
| `split_scan()` | `triton_heuristics.py` | 3463-3498 |
| `template()` | `triton_heuristics.py` | 3503-3536 |
| `user_autotune()` | `triton_heuristics.py` | 3590-3610 |

### Template Heuristics

| 文件 | 路径 | 内容 |
|------|------|------|
| **registry.py** | `/torch/_inductor/template_heuristics/` | 注册系统 |
| **triton.py** | `/torch/_inductor/template_heuristics/` | CUDA/ROCm GEMM configs |
| **gemm.py** | `/torch/_inductor/template_heuristics/` | GEMM基类 |
| **cutedsl.py** | `/torch/_inductor/template_heuristics/` | CuTe DSL |
| **decompose_k.py** | `/torch/_inductor/template_heuristics/` | K分解策略 |

### Kernel Implementations

| 操作 | 文件 | 行号 |
|------|------|------|
| **tuned_mm** | `/torch/_inductor/kernel/mm.py` | 1100-1329 |
| **tuned_addmm** | `/torch/_inductor/kernel/mm.py` | 1370-1433 |
| **tuned_bmm** | `/torch/_inductor/kernel/bmm.py` | 135+ |
| **convolution** | `/torch/_inductor/kernel/conv.py` | 650+ |
| **flex_attention** | `/torch/_inductor/kernel/flex/flex_attention.py` | 429+ |
| **grouped_mm** | `/torch/_inductor/kernel/mm_grouped.py` | 791+ |

### Configuration

| 配置 | 文件 | 行号 |
|------|------|------|
| **max_autotune** | `/torch/_inductor/config.py` | 459 |
| **max_autotune_gemm** | `/torch/_inductor/config.py` | 465 |
| **max_autotune_pointwise** | `/torch/_inductor/config.py` | 462 |
| **max_autotune_gemm_search_space** | `/torch/_inductor/config.py` | 541-543 |
| **coordinate_descent_tuning** | `/torch/_inductor/config.py` | 583-591 |

---

## 实践建议

### 1. 不同场景的配置建议

#### 场景1: 模型开发/调试

```python
# 优先编译速度
torch._inductor.config.max_autotune = False
torch._inductor.config.max_autotune_gemm = False
torch._inductor.config.coordinate_descent_tuning = False

# 预期：
# - 编译时间: ~1-5秒
# - 性能: 基线 (70-80% of optimal)
```

#### 场景2: 训练 (动态shape)

```python
# 平衡编译和性能
torch._inductor.config.max_autotune = False
torch._inductor.config.max_autotune_gemm = True  # 仅GEMM
torch._inductor.config.max_autotune_gemm_search_space = "DEFAULT"

# 预期：
# - 编译时间: ~10-30秒
# - 性能: 85-90% of optimal
```

#### 场景3: 推理 (固定shape)

```python
# 极致性能
torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_gemm = True
torch._inductor.config.max_autotune_gemm_search_space = "DEFAULT"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.coordinate_descent_check_all_directions = True

# 预期：
# - 编译时间: ~1-5分钟
# - 性能: 95-99% of optimal
```

#### 场景4: 生产部署 (极限优化)

```python
# 穷举搜索
torch._inductor.config.max_autotune = True
torch._inductor.config.max_autotune_gemm = True
torch._inductor.config.max_autotune_gemm_search_space = "EXHAUSTIVE"
torch._inductor.config.coordinate_descent_tuning = True
torch._inductor.config.coordinate_descent_check_all_directions = True
torch._inductor.config.coordinate_descent_search_radius = 2

# 预期：
# - 编译时间: ~10-30分钟 (一次性)
# - 性能: 接近理论最优
# - 适用: 编译一次，运行百万次
```

### 2. 编译时间 vs 性能权衡

| 配置 | 编译时间 | 运行性能 | 适用场景 |
|------|---------|---------|---------|
| **默认** | 1-5s | 70-80% | 开发/调试 |
| **仅GEMM** | 10-30s | 85-90% | 训练（GEMM密集型） |
| **全局ON** | 1-5min | 95-98% | 固定shape推理 |
| **EXHAUSTIVE** | 10-30min | 98-100% | 生产部署 |

### 3. 常见问题排查

#### Q1: 编译太慢怎么办？

```python
# 方案1: 减少autotuning范围
config.max_autotune = False
config.max_autotune_gemm = True  # 仅优化GEMM

# 方案2: 使用子进程autotuning
config.autotune_in_subproc = True

# 方案3: 调整超时
config.precompilation_timeout_seconds = 300  # 5分钟
```

#### Q2: 性能不如预期？

```python
# 检查1: 确认autotuning已启用
print(f"max_autotune: {config.max_autotune}")
print(f"max_autotune_gemm: {config.max_autotune_gemm}")

# 检查2: 查看缓存命中
import torch._inductor.select_algorithm as sa
print(sa.get_algorithm_selector_cache().cache_info())

# 检查3: 启用更激进的优化
config.max_autotune_gemm_search_space = "EXHAUSTIVE"
config.coordinate_descent_tuning = True
```

#### Q3: 内存不足？

```python
# 方案1: 减少并行编译
config.compile_threads = 1

# 方案2: 禁用某些backend
config.max_autotune_gemm_backends = "TRITON,ATEN"  # 移除CUTLASS

# 方案3: 增加shared memory剪枝
config.max_autotune_prune_choices_based_on_shared_mem = True
```

### 4. 性能分析工具

```python
# 启用详细日志
import logging
logging.getLogger("torch._inductor").setLevel(logging.DEBUG)

# 启用kernel性能分析
config.triton.cudagraphs = True
config.benchmark_kernel = True

# 导出autotuning结果
config.trace.enabled = True
config.trace.log_autotuning_results = True

# 运行后分析
# 查看 .torch_inductor/autotune_cache/ 目录
# 查看编译日志找到最优configs
```

### 5. 缓存管理

```python
# 清除缓存（强制重新autotuning）
import shutil
import os
cache_dir = os.path.expanduser("~/.triton/cache")
shutil.rmtree(cache_dir, ignore_errors=True)

# 启用远程缓存（团队共享）
config.autotune_remote_cache = "s3://my-bucket/inductor-cache"

# 启用FX graph缓存（跨编译复用）
config.fx_graph_cache = True
```

---

## 附录

### A. 环境变量速查

```bash
# 启用max_autotune
export TORCHINDUCTOR_MAX_AUTOTUNE=1

# 仅GEMM
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM=1

# 穷举搜索
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_SEARCH_SPACE=EXHAUSTIVE

# Coordinate descent
export TORCHINDUCTOR_COORDINATE_DESCENT_TUNING=1
export TORCHINDUCTOR_COORDINATE_DESCENT_CHECK_ALL_DIRECTIONS=1
export TORCHINDUCTOR_COORDINATE_DESCENT_RADIUS=2

# Backend选择
export TORCHINDUCTOR_MAX_AUTOTUNE_GEMM_BACKENDS=ATEN,TRITON,CUTLASS

# 子进程autotuning
export TORCHINDUCTOR_AUTOTUNE_IN_SUBPROC=1

# 缓存设置
export TRITON_CACHE_DIR=/path/to/cache
```

### B. 关键数据结构

```python
# TritonConfig (triton包)
class Config:
    def __init__(self, kwargs, num_warps, num_stages):
        self.kwargs = kwargs  # {"BLOCK_M": 64, "BLOCK_N": 128, ...}
        self.num_warps = num_warps
        self.num_stages = num_stages

# GemmConfig (template_heuristics)
class GemmConfig:
    def __init__(self, block_m, block_n, block_k, num_stages, num_warps, group_m):
        self.block_m = block_m
        self.block_n = block_n
        self.block_k = block_k
        self.num_stages = num_stages
        self.num_warps = num_warps
        self.group_m = group_m

# ChoiceCaller (select_algorithm)
class ChoiceCaller:
    def __init__(self, choice, input_nodes, layout):
        self.choice = choice  # ExternKernelChoice or TritonTemplate
        self.input_nodes = input_nodes
        self.layout = layout

    def benchmark(self, *args):
        # 实际在GPU上运行
        pass
```

### C. 术语表

| 术语 | 解释 |
|------|------|
| **Backend** | kernel实现库 (Triton/CUTLASS/ATen/CK/CPP) |
| **Template** | 手写的kernel模板，支持参数化 |
| **Codegen** | 编译器自动生成的kernel |
| **Choice** | 一个可选的kernel实现（backend + config） |
| **Config** | Triton kernel的参数配置 |
| **Heuristic** | 基于shape的config生成规则 |
| **Launcher** | 编译后可调用的kernel对象 |
| **Size hints** | 输入张量的shape信息 |
| **Fusion** | 多个操作合并为单个kernel |
| **Epilogue fusion** | 在主kernel后追加额外操作 |

---

## 总结

PyTorch Inductor的autotuning系统是一个精心设计的**分层架构**：

1. **两个层级，一个系统**
   - Layer 1: Backend选择 (autotune_select_algorithm)
   - Layer 2: Config优化 (CachingAutotuner)

2. **Heuristic是Config生成器**
   - Runtime heuristics: 为codegen kernels生成configs
   - Template heuristics: 为template kernels生成configs

3. **max_autotune控制两方面**
   - 是否启用backend选择
   - 生成多少configs

4. **EXHAUSTIVE是搜索空间大小**
   - DEFAULT: 20个configs (快速)
   - EXHAUSTIVE: 1875个configs (极致)

5. **实践指南**
   - 开发: 关闭autotuning，快速迭代
   - 训练: 启用GEMM autotuning
   - 推理: 全局autotuning + coordinate descent
   - 生产: EXHAUSTIVE + 完整优化栈

---

**文档维护**

如有疑问或需要更新，请联系 PyTorch Inductor团队。

最后更新: 2025-01-10
