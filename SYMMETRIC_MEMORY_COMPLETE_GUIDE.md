# Symmetric Memory in torch.compile - Complete Guide

## 目录
1. [当前机制确认](#当前机制确认)
2. [用户如何触发](#用户如何触发)
3. [工作流程详解](#工作流程详解)
4. [Phase 1 实现计划](#phase-1-实现计划)
5. [性能考虑](#性能考虑)

---

## 当前机制确认

### ✅ 现状：`one_shot_all_reduce` 已在使用 Symmetric Memory

PyTorch Inductor **已经实现了** `one_shot_all_reduce` 的 symmetric memory 支持！

**文件**: `/data/users/tianren/pytorch/torch/_inductor/comm_lowering.py`

```python
# Line 196-199: all_reduce 的 lowering 注册
@register_comm_lowering(c10d.all_reduce)
def _all_reduce(inp: ir.TensorBox, reduce_op: str, group_name: str):
    if _should_lower_as_one_shot_all_reduce(inp, reduce_op, group_name):
        return _one_shot_all_reduce(inp, reduce_op, group_name)  # 使用 symmetric memory!

    # 否则使用普通的 all_reduce_
    inp = clone(inp)
    ...

# Line 159-169: one_shot_all_reduce 实现
def _one_shot_all_reduce(inp: ir.TensorBox, reduce_op, group_name):
    realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM, group_name)  # <-- 标记为 SYMM_MEM!
    return pytree.tree_map(
        ir.TensorBox.create,
        ir.FallbackKernel.create(
            torch.ops.symm_mem.one_shot_all_reduce.default,
            inp,
            reduce_op,
            group_name,
        ),
    )
```

### 触发条件

**文件**: `/data/users/tianren/pytorch/torch/_inductor/comm_lowering.py` (Line 144-156)

```python
def _should_lower_as_one_shot_all_reduce(inp: ir.TensorBox, reduce_op: str, group_name: str):
    from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group

    inp_size = inp.get_numel() * inp.get_dtype().itemsize
    return (
        config._collective.auto_select            # 条件 1: 需要开启 (默认 False)
        and is_symm_mem_enabled_for_group(group_name)  # 条件 2: symmetric memory 已启用
        and can_realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM)  # 条件 3: buffer 可以 realize
        and reduce_op == "sum"                    # 条件 4: reduce_op 必须是 "sum"
        and inp_size <= config._collective.one_shot_all_reduce_threshold_bytes  # 条件 5: 大小 <= 128KB
    )
```

---

## 用户如何触发

### ❌ 默认情况：不会自动触发

**文件**: `/data/users/tianren/pytorch/torch/_inductor/config.py` (Line 891-893)

```python
class _collective:
    auto_select: bool = False  # <-- 默认是 False！
    one_shot_all_reduce_threshold_bytes: int = 128 * 1024
```

### ✅ 需要用户手动设置两件事

#### 第 1 步：启用 `auto_select` config

```python
import torch._inductor.config as config
config._collective.auto_select = True
```

#### 第 2 步：启用 Symmetric Memory for Process Group

```python
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
import torch.distributed as dist

dist.init_process_group(backend="nccl", world_size=2, rank=rank)
enable_symm_mem_for_group("default")
```

### 完整用户代码示例

```python
import torch
import torch.distributed as dist
import torch._inductor.config as config

# 初始化分布式
dist.init_process_group(backend="nccl", world_size=2, rank=rank)

# 第 1 步：启用 auto_select
config._collective.auto_select = True

# 第 2 步：启用 symmetric memory
from torch.distributed._symmetric_memory import enable_symm_mem_for_group
enable_symm_mem_for_group("default")

# 第 3 步：使用 torch.compile
@torch.compile(backend="inductor")
def my_model(x):
    y = x * 2.0
    # 这个 all_reduce 会自动被优化为 one_shot_all_reduce (使用 symmetric memory)
    dist.all_reduce(y, op=dist.ReduceOp.SUM)
    return y

x = torch.randn(100, 100, device=f"cuda:{rank}")
output = my_model(x)
```

---

## 工作流程详解

### 完整执行流程（从用户代码到生成代码）

```
用户代码: dist.all_reduce(tensor)
    ↓
[Dynamo] 捕获 FX Graph
    ↓
    graph():
        %x : [num_users=1] = placeholder[target=x]
        %mul : [num_users=1] = call_function[target=operator.mul](args = (%x, 2.0))
        # all_reduce 作为 call_function node
        %all_reduce : [num_users=1] = call_function[
            target=torch.ops._c10d_functional.all_reduce
        ](args = (%mul, 'sum', 'default'))
        return %all_reduce
    ↓
[Inductor] 查找 lowering 函数
    ↓
    lowering_func = lowerings.get(c10d.all_reduce)
    # 找到 _all_reduce 函数
    ↓
[comm_lowering._all_reduce] 检查条件
    ↓
    if _should_lower_as_one_shot_all_reduce(...):
        return _one_shot_all_reduce(...)  # <-- 选择 symmetric memory 路径!
    ↓
[comm_lowering._one_shot_all_reduce]
    ↓
    realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM, group_name)
    # **关键：修改 buffer.layout**
    ↓
    buffer.layout = ir.CommBufferLayout(
        layout=original_layout,
        comm_buffer_type=CommBufferType.SYMM_MEM,  # <-- 标记在这里！
        group_name=group_name
    )
    ↓
[Scheduler] 遍历 buffers
    ↓
    for buf in self.buffers:
        layout = buf.get_layout()
        if isinstance(layout, ir.CommBufferLayout):
            # 生成特殊的分配代码
            self.wrapper_code.generate_comm_buffer_allocation(buf)
    ↓
[WrapperCodegen] 生成代码
    ↓
    if comm_buffer_type == ir.CommBufferType.SYMM_MEM:
        # **生成 symmetric memory 分配代码！**
        return f"{name} = empty_strided_p2p(..., group_name='default', alloc_id=12345)"
    ↓
生成的 Python 代码:
    buf0 = empty_strided_cuda((128, 128), ...)  # 普通分配
    buf1 = empty_strided_p2p((128, 128), ..., group_name="default")  # <-- Symmetric!
    buf2 = torch.ops.symm_mem.one_shot_all_reduce(buf1, "sum", "default")
```

### 核心机制：Layout 类型传递

**关键思想**：通过在 `realize_as_comm_buffer()` 中改变 buffer 的 **layout 类型**，我们在 IR 中"标记"了哪些 buffers 需要 symmetric memory。这个标记在整个编译流程中传递，最终在代码生成阶段被识别，生成 `empty_strided_p2p()` 调用。

```python
# 数据流
普通 Buffer (FlexibleLayout)
    ↓ realize_as_comm_buffer()
Symmetric Buffer (CommBufferLayout with SYMM_MEM)
    ↓ Scheduler
传递到 Codegen
    ↓ WrapperCodegen
检查 layout 类型
    ↓
生成 empty_strided_p2p() 调用
```

### 关键函数

| 函数 | 位置 | 作用 |
|------|------|------|
| `realize_as_comm_buffer()` | `comm_lowering.py:77-110` | 标记 buffer 为 symmetric memory |
| `empty_strided_p2p()` | C++ binding | 实际从 symmetric memory 分配 |
| `CommBufferLayout` | `ir.py` | 特殊 layout 类型，携带 SYMM_MEM 标志 |

---

## Phase 1 实现计划

### 目标

**当前**: 只有 `all_reduce` 会被自动优化为 `one_shot_all_reduce` + symmetric memory

**Phase 1**: 为所有 `torch.ops.symm_mem.*` 操作添加 lowering，让它们都能像 `one_shot_all_reduce` 一样自动使用 symmetric memory

### 设计原则

1. **忽略用户添加的 `torch.cuda.use_mem_pool()` context managers** - 编译器基于操作类型自动决定
2. **复用现有基础设施** - 使用已有的 `realize_as_comm_buffer()` 和 `empty_strided_p2p()`
3. **不需要改 Dynamo 层** - 所有决策在 Inductor lowering 阶段完成

### 实现步骤

#### Step 1: 通用操作检测函数

**文件**: `/data/users/tianren/pytorch/torch/_inductor/comm_lowering.py`

**位置**: 在 `_should_lower_as_one_shot_all_reduce()` 函数后添加 (约 line 157)

```python
def requires_symmetric_memory_allocation(target) -> bool:
    """
    判断一个操作是否需要 symmetric memory。

    Args:
        target: 操作的 target (torch.ops.* OpOverload)

    Returns:
        True 如果操作需要 symmetric memory
    """
    # 检查是否是 symm_mem 命名空间的操作
    if hasattr(target, "__module__"):
        module_parts = target.__module__.split(".")
        if "symm_mem" in module_parts:
            return True

    # 通过字符串表示检查 (适用于 OpOverload 对象)
    target_str = str(target)
    if "symm_mem" in target_str:
        return True

    return False
```

#### Step 2: 通用 Lowering 注册助手

**文件**: `/data/users/tianren/pytorch/torch/_inductor/comm_lowering.py`

**位置**: 在文件中部添加

```python
def create_symm_mem_lowering(op_overload, extract_group_name_fn=None):
    """
    为 symm_mem 操作创建统一的 lowering 函数。

    这个助手标准化了模式：
    1. 将所有 tensor 输入 realize 为 CommBufferType.SYMM_MEM
    2. 通过 FallbackKernel 执行操作

    Args:
        op_overload: 要注册的 torch.ops.symm_mem.* 操作
        extract_group_name_fn: 可选的函数，从 args/kwargs 提取 group_name
    """
    from .lowering import register_lowering, add_layout_constraint, constrain_to_fx_strides

    add_layout_constraint(op_overload, constrain_to_fx_strides)

    @register_lowering(op_overload)
    def _symm_mem_generic(*args, **kwargs):
        """Generic lowering for symm_mem operations"""
        # 提取 group_name
        if extract_group_name_fn:
            group_name = extract_group_name_fn(args, kwargs)
        else:
            # 默认提取逻辑
            group_name = kwargs.get('group_name', 'default')
            if not group_name and args:
                # 尝试在位置参数中找 group_name (通常是最后一个 string 参数)
                for arg in reversed(args):
                    if isinstance(arg, str) and arg:
                        group_name = arg
                        break

        # 将所有 tensor 输入 realize 为 symmetric memory buffers
        realized_args = []
        for arg in args:
            if isinstance(arg, ir.TensorBox):
                if can_realize_as_comm_buffer(arg, ir.CommBufferType.SYMM_MEM):
                    realize_as_comm_buffer(arg, ir.CommBufferType.SYMM_MEM, group_name)
                realized_args.append(arg)
            else:
                realized_args.append(arg)

        # 执行操作
        return pytree.tree_map(
            ir.TensorBox.create,
            ir.FallbackKernel.create(op_overload, *realized_args, **kwargs),
        )

    return _symm_mem_generic
```

#### Step 3: 扩展 `register_comm_lowerings()`

**文件**: `/data/users/tianren/pytorch/torch/_inductor/comm_lowering.py`

**位置**: 在 `register_comm_lowerings()` 函数末尾添加

```python
def register_comm_lowerings():
    # ... 现有代码 ...

    # 注册所有 symm_mem.* 操作
    try:
        import torch.ops.symm_mem

        # one_shot_all_reduce 已经有实现了
        @register_comm_lowering(torch.ops.symm_mem.one_shot_all_reduce)
        def _symm_mem_one_shot_all_reduce(
            inp: ir.TensorBox, reduce_op: str, group_name: str
        ) -> ir.TensorBox:
            return _one_shot_all_reduce(inp, reduce_op, group_name)

        # 为其他 symm_mem 操作注册 lowering
        # 示例：如果有 torch.ops.symm_mem.barrier
        if hasattr(torch.ops.symm_mem, 'barrier'):
            create_symm_mem_lowering(torch.ops.symm_mem.barrier)

        # 示例：如果有 torch.ops.symm_mem.all_gather
        if hasattr(torch.ops.symm_mem, 'all_gather'):
            create_symm_mem_lowering(torch.ops.symm_mem.all_gather)

        # 根据实际存在的操作继续添加...

    except (AttributeError, ImportError):
        log.info("symm_mem operations not available")
```

#### Step 4: 添加验证函数（可选）

**文件**: `/data/users/tianren/pytorch/torch/_inductor/comm_lowering.py`

```python
def validate_symmetric_memory_usage(buffer: ir.Buffer, group_name: str) -> bool:
    """
    验证 buffer 可以安全地用作 symmetric memory。

    检查：
    - Process group 是否启用了 symmetric memory
    - Buffer 大小是否在限制内
    - Buffer dtype 是否支持
    """
    try:
        from torch.distributed._symmetric_memory import is_symm_mem_enabled_for_group

        if not is_symm_mem_enabled_for_group(group_name):
            log.warning(
                f"Symmetric memory not enabled for group '{group_name}'. "
                f"Buffer {buffer.get_name()} may fail to allocate."
            )
            return False

        # 检查 buffer 大小
        buffer_size = buffer.get_numel() * buffer.get_dtype().itemsize
        max_size = config._collective.one_shot_all_reduce_threshold_bytes

        if buffer_size > max_size:
            log.info(
                f"Buffer {buffer.get_name()} size ({buffer_size} bytes) exceeds "
                f"threshold ({max_size} bytes). This may impact performance."
            )

        return True

    except (AttributeError, ImportError) as e:
        log.warning(f"Could not validate symmetric memory: {e}")
        return False
```

### 实现检查清单

- [ ] **Step 1**: 添加 `requires_symmetric_memory_allocation()` 函数
- [ ] **Step 2**: 实现 `create_symm_mem_lowering()` 助手
- [ ] **Step 3**: 扩展 `register_comm_lowerings()` 注册新操作
- [ ] **Step 4** (可选): 添加 `validate_symmetric_memory_usage()` 验证函数
- [ ] **Step 5**: 添加测试用例

### 测试示例

**文件**: `/data/users/tianren/pytorch/test/inductor/test_symmetric_memory.py` (新文件)

```python
import unittest
import torch
import torch.distributed as dist
from torch._dynamo.test_case import run_tests, TestCase
from torch.testing._internal.common_distributed import MultiProcessTestCase, skip_if_lt_x_gpu

class TestSymmetricMemoryCompile(MultiProcessTestCase):
    @property
    def world_size(self):
        return 2

    def setUp(self):
        super().setUp()
        self._spawn_processes()

    @skip_if_lt_x_gpu(2)
    def test_symm_mem_one_shot_all_reduce(self):
        """测试 one_shot_all_reduce 自动获得 symmetric memory"""
        dist.init_process_group(backend="nccl", world_size=self.world_size, rank=self.rank)

        # 启用 symmetric memory
        from torch.distributed._symmetric_memory import enable_symm_mem_for_group
        import torch._inductor.config as config

        enable_symm_mem_for_group("default")
        config._collective.auto_select = True

        @torch.compile(backend="inductor", fullgraph=True)
        def forward(x):
            y = x * 2.0
            z = y + 1.0
            result = torch.ops.symm_mem.one_shot_all_reduce(z, "sum", "default")
            return result

        x = torch.randn(128, 128, device=f"cuda:{self.rank}")
        output = forward(x)

        expected = (x * 2.0 + 1.0) * self.world_size
        torch.testing.assert_close(output, expected)

        dist.destroy_process_group()

if __name__ == "__main__":
    run_tests()
```

---

## 性能考虑

### 预期改进

1. **零拷贝 Collectives**: 无需 memcpy 进出通信缓冲区
2. **更好的内存局部性**: Symmetric buffers 放置位置最优
3. **降低延迟**: P2P 访问比通过 CPU 快

### 潜在开销

1. **分配成本**: `empty_strided_p2p()` 可能比普通分配慢（需要协调）
2. **内存碎片**: Symmetric memory 有不同的分配模式

### 缓解策略

1. **Pool 重用**: 一旦分配，symmetric buffers 可以跨迭代重用
2. **内存规划**: Inductor 的 buffer planning 优化整体内存使用
3. **基于阈值**: 只在有益时使用 symmetric memory（大小限制等）

---

## 已知限制和 Phase 2 TODO

1. **细粒度控制**: Phase 1 对 symm_mem ops 的所有 tensor 输入都分配 symmetric memory。Phase 2 可以添加逐参数注解。

2. **Buffer 重用**: 如果一个 buffer 被 symmetric 和非 symmetric ops 使用，可能需要插入拷贝。Phase 2 可以优化这个。

3. **多 Process Group**: 当前假设每个操作一个 group name。Phase 2 可以跟踪多个 groups。

4. **C++ Wrapper**: `empty_strided_p2p()` 只在 Python wrapper 模式可用。Phase 2 需要 C++ API 支持。

5. **动态 Shapes**: 有 symbolic shapes 的 buffers 不能是 symmetric（由 `realize_as_comm_buffer` 检查）。Phase 2 可以通过运行时分配支持这个。

6. **Eager 模式回退**: 当 symm_mem ops 在 eager 模式运行时，输入可能不是 symmetric。Phase 2 应在 dispatcher 层添加自动克隆。

---

## 验证方法

### 方法 1: 查看生成的代码

```python
import torch._dynamo
import torch._inductor.config

# 开启调试日志
torch._dynamo.config.verbose = True
torch._inductor.config.debug = True

# 编译后检查生成的代码
# 会输出到 /tmp/torchinductor_<user>/xxx.py
```

在生成的代码中查找：
- ✅ `empty_strided_p2p(...)` - 表示使用了 symmetric memory
- ❌ `empty_strided_cuda(...)` - 表示使用了普通内存

### 方法 2: 添加日志

在 `/data/users/tianren/pytorch/torch/_inductor/comm_lowering.py` 中：

```python
def _one_shot_all_reduce(inp: ir.TensorBox, reduce_op, group_name):
    print(f"🔥 Using one_shot_all_reduce with symmetric memory!")
    realize_as_comm_buffer(inp, ir.CommBufferType.SYMM_MEM, group_name)
    ...
```

---

## 相关代码位置

| 组件 | 文件 | 行号 | 说明 |
|------|------|------|------|
| Lowering 注册 | `comm_lowering.py` | 196-199 | all_reduce 的 lowering |
| one_shot_all_reduce | `comm_lowering.py` | 159-169 | 标记为 SYMM_MEM |
| 条件检查 | `comm_lowering.py` | 144-156 | 5 个触发条件 |
| realize_as_comm_buffer | `comm_lowering.py` | 77-110 | 标记 buffer 为 symmetric |
| Config 默认值 | `config.py` | 891-893 | auto_select = False |
| 启用函数 | `_symmetric_memory/__init__.py` | 24-48 | enable_symm_mem_for_group |
| 检查函数 | `_symmetric_memory/__init__.py` | 76-85 | is_symm_mem_enabled_for_group |
| CommBufferLayout | `ir.py` | - | 特殊 layout 类型 |
| 分配代码生成 | `wrapper.py` | ~870 | empty_strided_p2p() 生成 |

---

## 总结

### 核心要点

1. **现有机制已完善**: `one_shot_all_reduce` 已经在使用 symmetric memory，机制完整且验证过
2. **默认不触发**: 需要用户手动设置 `config._collective.auto_select = True` 和 `enable_symm_mem_for_group()`
3. **Phase 1 目标**: 扩展到所有 `symm_mem` 操作，复用现有基础设施
4. **实现简单**: 主要工作是为新操作注册 lowering，调用 `realize_as_comm_buffer()`
5. **不需要改 Dynamo**: 所有逻辑在 Inductor lowering 阶段

### 关键函数调用链

```
用户调用 torch.ops.symm_mem.*
    ↓
Inductor 查找 lowering 函数
    ↓
调用 realize_as_comm_buffer(inp, SYMM_MEM, group_name)
    ↓
修改 buffer.layout = CommBufferLayout(type=SYMM_MEM)
    ↓
Codegen 检查 layout 类型
    ↓
生成 empty_strided_p2p() 调用
    ↓
从 symmetric memory 分配！
```

---

## 参考资料

- **Existing Implementation**: `/data/users/tianren/pytorch/torch/_inductor/comm_lowering.py`
- **Allocation Code**: `/data/users/tianren/pytorch/torch/_inductor/codegen/wrapper.py`
- **Buffer Layout**: `/data/users/tianren/pytorch/torch/_inductor/ir.py`
- **Symmetric Memory API**: `/data/users/tianren/pytorch/torch/distributed/_symmetric_memory/__init__.py`
