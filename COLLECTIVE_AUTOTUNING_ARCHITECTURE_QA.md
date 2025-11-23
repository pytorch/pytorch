# Collective Op Autotuning 架构设计问答

## Q1: 现在的实现和TL原始设计一致吗？

**A: 是的，基本一致，并且已经在使用autotuning cache。**

### TL的原始设计
```
Proposal: Do lowering -> scheduler(multiTemplateBuffer gets realized) -> do ~5ms of 
benchmarking the first time we encounter a collective op of a certain shape. Use the 
cached results every time after.
```

### 当前实现

#### 1. 流程确认
```python
# custom_op.py: autotune_custom_op() 

# Step 1: Lowering - 生成multiple choices
template = SubgraphTemplate(name=name)
choices = template.generate_custom_op_choices(
    decompositions=decompositions,
    input_nodes=list(inputs),
    non_tensor_args=non_tensor_args,
)

# Step 2: Autotuning - 第一次遇到某个shape时benchmark
selected_result, winning_choice = autotune_select_algorithm(
    name=name,
    choices=choices,
    input_nodes=list(inputs),
    layout=choices[0].layout,
    input_gen_fns=input_gen_fns,
    return_choice=True,
    is_collective=is_collective,  # 检测collective ops
)

# Step 3: 后续使用cached results
# autotune_select_algorithm内部会check cache，如果hit就直接返回
```

#### 2. Caching机制
`autotune_select_algorithm` 内部使用了 inductor 的 autotuning cache：

```python
# select_algorithm.py
def autotune_select_algorithm(...):
    # 检查cache
    cached_result = check_cache(name, input_shapes, ...)
    if cached_result:
        return cached_result  # 直接返回cached choice
    
    # Cache miss: 第一次遇到这个shape，进行benchmark
    if is_collective:
        # 使用 CollectiveBenchmarker.benchmark_collective_choice()
        timings = benchmark_all_choices_with_collective_sync()
    else:
        # 使用常规benchmark
        timings = benchmark_all_choices()
    
    best_choice = select_best(timings)
    
    # 存入cache
    save_to_cache(name, input_shapes, best_choice)
    
    return best_choice
```

#### 3. Benchmark时间
- 默认每个choice做10次run (可配置: `TORCHINDUCTOR_COLLECTIVE_BENCHMARK_NRUNS`)
- 每次run有warmup + timing
- 对于简单的collective ops (如all_reduce)，10次run通常在几毫秒内完成
- **符合TL的"~5ms of benchmarking"目标**

### 验证cache是否生效

```python
# 第一次编译 - 会benchmark
model1 = torch.compile(MyModel()).to(device)
x1 = torch.randn(128, 128, device=device)
y1 = model1(x1)  # 触发benchmark，存入cache

# 第二次编译相同shape - 使用cache
model2 = torch.compile(MyModel()).to(device)
x2 = torch.randn(128, 128, device=device)  # 相同shape
y2 = model2(x2)  # 直接从cache读取，不benchmark

# 不同shape - 会重新benchmark
x3 = torch.randn(256, 256, device=device)  # 不同shape
y3 = model2(x3)  # 触发新的benchmark
```

---

## Q2: Benchmark结束后能进入inline fusion吗？

**A: 可以！这是设计的一部分。**

### Inline Fusion在哪里发生

```python
# custom_op.py: autotune_custom_op()

# Step 1: Autotuning选出最佳choice
selected_result, winning_choice = autotune_select_algorithm(
    name=name,
    choices=choices,
    input_nodes=list(inputs),
    ...
)

# Step 2: 如果winning choice有graph module，进行inline fusion
if winning_choice.gm is not None:
    log.debug("Inlining winning choice: %s", winning_choice)
    from torch._inductor.codegen.subgraph import inline_subgraph_to_ir_nodes
    
    # ✅ 这里进行inline fusion！
    return inline_subgraph_to_ir_nodes(winning_choice.gm, inputs, name)

# Step 3: 如果是fallback choice (ExternKernelChoice)，不支持inline
log.debug("Winning choice does not support inlining: %s", winning_choice)
return selected_result  # 返回extern kernel调用
```

### 什么情况能inline fusion？

#### 情况1: SubgraphTemplateChoice (可以inline) ✅
```python
# 用户提供的decomposition会被编译成graph module
def my_allreduce_impl(x):
    result = x.clone()
    return torch.ops._c10d_functional.all_reduce_(result, "sum", "default")

# 生成的choice有gm属性
choice = SubgraphTemplateChoice(
    gm=traced_graph_module,  # ✅ 有graph
    ...
)

# benchmark后，winning choice会被inline到主graph
# 可以和周围ops进行fusion (如matmul + allreduce + relu)
```

#### 情况2: ExternKernelChoice (不能inline) ❌
```python
# Fallback到default implementation
fallback_choice = ExternKernelChoice(
    kernel=op_overload,
    gm=None,  # ❌ 没有graph
    use_fallback_kernel=True,
)

# 如果这个choice赢了，会作为extern call，不能inline
```

### Inline Fusion的好处

```python
# 例子：matmul + collective + relu

# 没有inline fusion:
def forward(x, y):
    mm_out = torch.mm(x, y)           # Kernel 1
    ar_out = collective_op(mm_out)    # Kernel 2 (extern call)
    relu_out = torch.relu(ar_out)    # Kernel 3
    return relu_out
# 3个kernel launches

# 有inline fusion:
def forward(x, y):
    mm_out = torch.mm(x, y)
    # collective_op被inline后可以和relu fusion
    fused_out = fused_collective_relu(mm_out)  # Kernel 2 (fused)
    return fused_out
# 2个kernel launches (mm + fused_collective_relu)
```

---

## Q3: 完整的工作流程

```
用户代码:
  @torch.library.custom_op("mylib::my_allreduce", mutates_args=())
  def my_allreduce(x):
      ...
  
  register_custom_op_autotuning(
      my_allreduce,
      configs=[
          CustomOpConfig(impl1),
          CustomOpConfig(impl2),
      ]
  )

第一次编译 (某个shape):
  ↓
  Lowering (custom_op.py)
  ├─ 生成multiple SubgraphTemplateChoice
  ├─ 生成fallback ExternKernelChoice
  ↓
  Autotuning (select_algorithm.py)
  ├─ Check cache → Miss
  ├─ 检测collective ops → is_collective=True
  ├─ 使用CollectiveBenchmarker.benchmark_collective_choice()
  │  ├─ 每个choice: barrier + 10次benchmark + barrier
  │  ├─ all_reduce收集所有ranks的max timing
  │  └─ 返回timing
  ├─ 选出最快的choice
  ├─ Save to cache
  ↓
  Inline Fusion (custom_op.py)
  ├─ if winning_choice.gm is not None:
  │  └─ inline_subgraph_to_ir_nodes() ✅
  ├─ else:
  │  └─ return extern call (no fusion) ❌
  ↓
  Scheduler
  ├─ 已经inline的ops可以和周围ops fusion
  └─ 生成最终优化的代码

后续编译 (相同shape):
  ↓
  Lowering
  ↓
  Autotuning
  ├─ Check cache → Hit! ✅
  ├─ 直接使用cached winning choice
  └─ 跳过benchmark (节省时间)
  ↓
  Inline Fusion (和第一次一样)
  ↓
  Scheduler
```

---

## Q4: 关键设计点总结

### ✅ 符合TL设计
1. **First time**: Lowering → Benchmark (~5ms) → Cache result
2. **Subsequent times**: Lowering → Read cache → Skip benchmark
3. **Per-shape cache**: 不同shape会触发新的benchmark

### ✅ Inline Fusion支持
1. **SubgraphTemplateChoice**: 可以inline，支持fusion
2. **ExternKernelChoice**: 不能inline，作为extern call
3. **Fusion opportunities**: inline后的ops可以和周围ops fusion

### ✅ Collective ops特殊处理
1. **检测**: `_detect_collective_ops()` 自动检测
2. **Benchmarking**: 使用`CollectiveBenchmarker`进行跨rank同步
3. **Timeout**: 处理GPU资源竞争，避免hang

---

## 实际例子

```python
# 用户代码
@torch.library.custom_op("mylib::allreduce", mutates_args=())
def my_allreduce(x):
    return x.clone()

register_custom_op_autotuning(
    my_allreduce,
    configs=[
        CustomOpConfig(lambda x: torch.ops._c10d_functional.all_reduce_(x.clone(), "sum", "default")),
        CustomOpConfig(lambda x: torch.ops._c10d_functional.all_reduce_(x.clone(), "avg", "default")),
    ]
)

# 编译
model = torch.compile(MyModel())

# 第一次运行 shape (128, 128)
x1 = torch.randn(128, 128, device='cuda')
y1 = model(x1)  
# ↑ 触发benchmark，选出最快的，inline到graph，cache结果

# 第二次运行 相同shape
x2 = torch.randn(128, 128, device='cuda')
y2 = model(x2)  
# ↑ 从cache读取，直接使用之前的choice，inline到graph
# ✅ 不需要重新benchmark!

# 第三次运行 不同shape
x3 = torch.randn(256, 256, device='cuda')
y3 = model(x3)  
# ↑ Cache miss，重新benchmark新的shape
```

---

## 结论

1. ✅ **实现符合TL设计**: 第一次benchmark + cache + 后续使用cache
2. ✅ **Inline fusion支持**: benchmark结束后，winning choice会被inline (如果有graph)
3. ✅ **Performance优化**: 
   - Cache避免重复benchmark
   - Inline fusion允许和周围ops fusion
   - Collective-aware benchmarking确保准确性
4. ✅ **Timeout机制**: 处理GPU资源竞争，避免hang

整个设计既保证了性能（通过benchmark选最快的），又保证了效率（通过cache避免重复benchmark），还支持fusion优化（通过inline）。
