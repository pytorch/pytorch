# DecomposeK → MultiTemplateBuffer 流程深度分析

## 你的问题列表

1. **Internal mm.py的tuned_mm的decompose_k是怎么lower到MultiTemplateBuffer的？**
2. **如果subgraph没有precompile，到scheduler的finalize_multi_template_buffers时是否都是serialized compile（慢）？**
3. **decompose_k考虑了fusion，但为什么之前看到的decompose_k+relu的triton kernel没有fuse在一起？**
4. **如果不用MultiTemplateBuffer，到dobenchmark时直接串行compile，但max-autotune是parallel的，然后可以在后面加subgraph的inline fusion？**
5. **如果没有precompile，max-autotune是怎么被inline fusion打破的？**

---

## 核心发现：decompose_k **不使用** MultiTemplateBuffer！

### 1. decompose_k的实际Lower路径

#### 代码路径 (mm.py:1207-1208)
```python
if use_decompose_k_choice(m, n, k):
    templates_to_use.append(decompose_k_subgraph_template)
```

#### decompose_k_subgraph_template 的类型 (mm.py:1047)
```python
decompose_k_subgraph_template = DecomposeKSugraphTemplate()
```

这是一个 **SubgraphTemplate**，不是 ExternKernelChoice 或 TritonTemplate！

#### SubgraphTemplate.generate() 返回什么 (subgraph.py:217-223)
```python
def generate(...) -> SubgraphChoiceCaller:
    return SubgraphChoiceCaller(
        name=f"{name}_{next(SubgraphTemplate.index_counter)}",
        input_nodes=input_nodes,
        layout=layout,
        description=description,
        make_fx_graph=make_fx_graph,
    )
```

**关键结论：返回 `SubgraphChoiceCaller`，不是 MultiTemplateBuffer！**

---

### 2. tuned_mm 的两条返回路径

#### 路径A：MultiTemplateBuffer (仅当 return_multi_template=True)

**条件** (select_algorithm.py:2927):
```python
if return_multi_template and (config.max_autotune or config.max_autotune_gemm):
    return torch._inductor.ir.MultiTemplateBuffer(
        layout,
        input_nodes,
        get_timings,  # ← 关键：延迟benchmark的函数
        choices,
        allowed_prologue_inps,
    )
```

**触发条件**：
- `return_multi_template=True` （默认是 True）
- 且 `max_autotune=True` 或 `max_autotune_gemm=True`

**这条路径返回**：
- 一个 **延迟选择** 的 MultiTemplateBuffer
- 不立即benchmark，而是在 scheduler phase 通过 `finalize_multi_template_buffers()` 调用 `get_timings()`

#### 路径B：直接选择 (return_multi_template=False 或无 max_autotune)

**条件** (select_algorithm.py:2973):
```python
timings = do_autotuning(choices, precompile_fn)
# ... 选择最快的 choice
return min_choice.output_node()
```

**这条路径返回**：
- 立即benchmark并选择最快的实现
- 对于 SubgraphChoiceCaller，返回 `SubgraphBuffer`
- **没有** MultiTemplateBuffer 参与

---

### 3. decompose_k 如何参与 autotuning

#### 生成 Choices (mm.py:1226-1233)
```python
choices.extend(
    V.choices.get_template_configs(
        kernel_inputs,
        templates_to_use,  # ← 包含 decompose_k_subgraph_template
        "mm",
        kwarg_overrides=kwarg_overrides,
    )
)
```

这会生成：
- `[aten_mm, decompose_k_choice, mm_template_choice1, mm_template_choice2, ...]`

#### 参与 Benchmark (select_algorithm.py:2936-2938)
```python
def get_timings(hint_override: Optional[int] = None):
    timings = do_autotuning(
        filtered_choices, precompile_fn, hint_override=hint_override
    )
```

---

## 问题1：decompose_k是怎么lower到MultiTemplateBuffer的？

### ❌ 错误理解
decompose_k **不会** lower到 MultiTemplateBuffer。

### ✅ 正确流程

```
tuned_mm()
    ↓
生成 choices = [aten_mm, decompose_k, mm_template, ...]
    ↓
autotune_select_algorithm(choices, return_multi_template=True)
    ↓
┌─────────────────────────────────────────────┐
│ if return_multi_template:                   │
│   return MultiTemplateBuffer(               │
│       get_timings_fn,  # ← 所有choices包含在内│
│       choices,  # ← [aten, decompose_k, ...]│
│   )                                          │
└─────────────────────────────────────────────┘
    ↓
MultiTemplateBuffer 持有所有choices（包括decompose_k）
但不立即执行benchmark
```

**关键**：
- MultiTemplateBuffer 是一个 **容器**，持有所有 choices
- decompose_k 作为 SubgraphChoiceCaller 存在于 choices 列表中
- MultiTemplateBuffer 并不 "lower" decompose_k，而是延迟对所有 choices 的选择

---

## 问题2：如果subgraph没有precompile，是否都是serialized compile？

### 延迟Benchmark的时机

#### scheduler.py:3441 (finalize_multi_template_buffers)
```python
def finalize_multi_template_buffers(self) -> None:
    for node in self.nodes:
        if isinstance(node.node, ir.MultiTemplateBuffer):
            min_node_unfused, _ = multi_node.get_min_choice()
            # ↑ 这里触发 choice_timings()
```

#### ir.py:5344 (MultiTemplateBuffer.get_min_choice)
```python
def get_min_choice(self, hint_override: Optional[int] = None):
    timings = self.choice_timings(hint_override=hint_override)
    # ↑ 第一次调用时执行 benchmark
```

#### ir.py:5315 (choice_timings)
```python
def choice_timings(self, hint_override: Optional[int] = None):
    if hint_override not in self._choice_timings:
        self._choice_timings[hint_override] = self._choice_timings_fn(hint_override)
        # ↑ 调用 get_timings() → do_autotuning()
    return self._choice_timings[hint_override]
```

### ✅ 正确答案：是的，**serialized compile**

**流程**：
```python
# select_algorithm.py:2936
def get_timings(hint_override):
    timings = do_autotuning(
        filtered_choices, precompile_fn, hint_override=hint_override
    )

# do_autotuning 内部会：
1. 调用 precompile_fn() → 触发 precompile
2. 对每个 choice 调用 benchmark()
```

**Precompile 阶段** (select_algorithm.py:3112-3128):
```python
for c in choices:
    if hasattr(c, "precompile"):
        future = executor.submit(precompile_with_captured_stdout, c)
        # ↑ 并行 precompile
    else:
        # ↑ SubgraphChoiceCaller 没有 precompile，跳过！
        pass
```

**Benchmark 阶段** (对于 SubgraphChoiceCaller):
```python
# subgraph.py:77 (SubgraphChoiceCaller.benchmark)
def benchmark(self, *args, out):
    bm_graph_lowering = GraphLowering(...)
    bm_graph_lowering.run(*self.example_inputs)
    mod = bm_graph_lowering.compile_to_module()
    # ↑ 这里才编译！串行！
    return benchmarker.benchmark(lambda: bm_func(...))
```

**性能损失**：
- 其他 choices（TritonTemplate）：并行 precompile → 只需串行 benchmark
- SubgraphChoiceCaller：跳过 precompile → 串行 compile + benchmark

**时间差异**：
- 假设有 8 个 choices（2个 subgraph，6个 triton）
- 并行路径：max(precompile_times) + sum(benchmark_times) ≈ 2s + 1s = 3s
- 串行 subgraph：sum(subgraph_compile_times) + sum(all_benchmark_times) ≈ 5s + 1s = 6s
- **差异：2x 慢**

---

## 问题3：为什么decompose_k+relu的kernel没有fuse在一起？

### ✅ 你的观察是正确的

**原因**：decompose_k 通过 SubgraphBuffer 返回，**并不走 inline fusion**

#### SubgraphChoiceCaller.output_node() (subgraph.py:147-156)
```python
def output_node(self):
    return ir.TensorBox.create(
        ir.SubgraphBuffer(  # ← 不是 ComputedBuffer！
            layout=self.layout,
            input_nodes=self.input_nodes,
            gm=self.gm,
            example_inputs=self.example_inputs,
            subgraph_name=self.name,
        )
    )
```

**SubgraphBuffer 的特点**：
- 是一个 **opaque buffer**（不透明缓冲区）
- 不支持 epilogue fusion（无法让后续的 relu 看到内部计算）
- 必须生成一个独立的 kernel

**为什么不支持fusion？**
- SubgraphBuffer 包含一个完整的 GraphModule (self.gm)
- Scheduler 看到的是一个黑盒 operation
- Relu 后续作为一个独立的 pointwise operation

**对比：inline fusion 的实现**：
```python
# subgraph.py:27-40 (inline_subgraph_to_ir_nodes)
def inline_subgraph_to_ir_nodes(gm, inputs, name):
    from torch._inductor.lowering import process_subgraph_nodes
    return process_subgraph_nodes(gm, inputs)
    # ↑ 将 subgraph 展开成多个 ComputedBuffer
    #   这样 relu 可以 fuse 进最后一个 ComputedBuffer
```

**为什么文档说支持fusion？**
- 文档指的是 **inline fusion mode**（通过 SubgraphBuffer 的 `inline_subgraph_to_ir_nodes`）
- 但 **默认情况下不启用**，因为：
  1. 需要 `config.benchmark_epilogue_fusion=True`
  2. 且必须在 scheduler 的 fusion 阶段使用 inline 模式

**测试证据重新解释**：
```python
# test_subgraph_choice.py 测试中看到 triton_.*_fused_mm_0.run
```
这个测试可能：
1. 显式使用了 inline mode
2. 或者在 scheduler fusion 阶段手动触发了 inline

---

## 问题4：如果不用MultiTemplateBuffer，max-autotune是怎么parallel的？

### ✅ 你的理解是对的

#### 不用 MultiTemplateBuffer 的路径 (select_algorithm.py:2973)
```python
# 路径B：立即 autotune
timings = do_autotuning(choices, precompile_fn)
min_choice = min(timings, key=timings.get)
return min_choice.output_node()
```

**这条路径的优点**：
```
1. 立即调用 precompile_fn()
    ↓
2. 并行 precompile（除了 SubgraphChoiceCaller）
    ↓
3. 串行 benchmark
    ↓
4. 选择最快的 choice 并返回其 output_node()
```

**对于 SubgraphChoiceCaller 返回 SubgraphBuffer**：
```python
# subgraph.py:147
def output_node(self):
    return ir.TensorBox.create(
        ir.SubgraphBuffer(...)  # ← 不支持 fusion
    )
```

**如果想要 inline fusion**：
- 不能用 SubgraphBuffer
- 需要在 **返回之前** 调用 `inline_subgraph_to_ir_nodes()`
- 但这需要修改 SubgraphChoiceCaller 的实现

**当前限制**：
- 即使不用 MultiTemplateBuffer，SubgraphChoiceCaller 仍然返回 SubgraphBuffer
- SubgraphBuffer 仍然不支持 epilogue fusion
- 所以 decompose_k + relu 不会 fuse

---

## 问题5：如果没有precompile，max-autotune是怎么被inline fusion打破的？

### ✅ 你的理解核心是正确的

**"打破" 的含义**：
- **预期**：max-autotune 通过并行 precompile 加速编译
- **实际**：当有 SubgraphChoiceCaller 时，部分编译串行化，失去并行优势

### 具体机制

#### Precompile 阶段 (select_algorithm.py:3112)
```python
for c in choices:
    if hasattr(c, "precompile"):
        future = executor.submit(precompile_with_captured_stdout, c)
        futures[future] = c
    # else: 跳过 SubgraphChoiceCaller
```

**结果**：
- ✅ TritonTemplate, ExternKernel, CUDATemplate → 并行 precompile
- ❌ SubgraphChoiceCaller → 跳过

#### Benchmark 阶段 (串行执行)
```python
for choice in choices:
    if choice not in precompiled:
        # SubgraphChoiceCaller 在这里第一次编译
        timing = choice.benchmark(*args, out=out)
        # ↑ benchmark() 内部调用 compile_to_module()
```

**SubgraphChoiceCaller.benchmark()** (subgraph.py:77-125):
```python
def benchmark(self, *args, out):
    bm_graph_lowering = GraphLowering(...)
    bm_graph_lowering.run(*self.example_inputs)
    mod = bm_graph_lowering.compile_to_module()
    # ↑ 这里编译！block 住主线程！
    bm_func = mod.call
    return benchmarker.benchmark(lambda: bm_func(...))
```

**时间线对比**：

**没有 SubgraphChoiceCaller（只有 Triton）**：
```
Thread 1: [precompile choice1] ──────────┐
Thread 2: [precompile choice2] ──────────┤
Thread 3: [precompile choice3] ──────────┤ wait
Thread 4: [precompile choice4] ──────────┤
Thread 5: [precompile choice5] ──────────┘
                                          ↓
Main Thread:                              [benchmark all] ← 很快

Total: 2s (precompile) + 1s (benchmark) = 3s
```

**有 SubgraphChoiceCaller（2个 subgraph + 5个 Triton）**：
```
Thread 1: [precompile triton1] ──────────┐
Thread 2: [precompile triton2] ──────────┤
Thread 3: [precompile triton3] ──────────┤ wait
Thread 4: [precompile triton4] ──────────┤
Thread 5: [precompile triton5] ──────────┘
                                          ↓
Main Thread:                              [compile subgraph1 串行] [compile subgraph2 串行] [benchmark all]
                                          ↑────────── 3s ──────────↑ ↑─ 1s ─↑

Total: 2s (parallel) + 3s (serial subgraph) + 1s (benchmark) = 6s
```

**性能损失**：
- 串行编译 subgraph 的时间 **完全没有并行化**
- 如果 subgraph 编译慢（GraphLowering），总时间 **显著增加**

---

## 关键结论总结

### 1. decompose_k 的路径
```
decompose_k_subgraph_template (SubgraphTemplate)
    ↓ generate()
SubgraphChoiceCaller (作为 choice 参与 autotune)
    ↓ 加入 choices 列表
MultiTemplateBuffer (持有所有 choices，延迟选择)
    ↓ finalize_multi_template_buffers()
choice_timings() → do_autotuning()
    ↓ benchmark
SubgraphChoiceCaller.benchmark()
    ↓ 这里才编译（串行）
选择最快的 choice
    ↓
SubgraphChoiceCaller.output_node()
    ↓
SubgraphBuffer (不支持 fusion)
```

### 2. 性能瓶颈
- **瓶颈1**：SubgraphChoiceCaller 没有 precompile()，跳过并行编译
- **瓶颈2**：在 benchmark 阶段串行编译，block 主线程
- **瓶颈3**：SubgraphBuffer 不支持 epilogue fusion

### 3. 文档中的 "支持fusion"
- 指的是通过 `inline_subgraph_to_ir_nodes()` 的 **inline mode**
- 但默认 `SubgraphChoiceCaller.output_node()` 返回 SubgraphBuffer（不 inline）
- 需要额外机制触发 inline（如 `benchmark_epilogue_fusion=True`）

### 4. 为什么需要 Solution A
- 给 SubgraphChoiceCaller 添加 `precompile()` 方法
- 将 GraphLowering 编译移到 precompile 阶段
- 利用 ThreadPoolExecutor 并行编译
- **性能提升**：6s → 3s (50% faster)

### 5. 为什么需要 Solution B
- SubgraphBuffer 不支持 fusion
- 需要 MultiTemplateBuffer + inline mode
- 在 scheduler 阶段动态决定是否 inline
- 支持 epilogue/prologue fusion

---

## 你的理解准确度评估

| 问题 | 你的理解 | 实际情况 | 准确度 |
|------|---------|---------|--------|
| decompose_k → MultiTemplateBuffer | 认为有 lower 过程 | MultiTemplateBuffer 只是容器 | ⚠️ 部分正确 |
| serialized compile | 认为是串行的 | **完全正确** | ✅ 100% |
| decompose_k+relu 不 fuse | 观察到不 fuse | SubgraphBuffer 不支持 | ✅ 100% |
| 不用 MTB 可以 parallel | 认为可以 parallel | precompile 可以，但 subgraph 仍串行 | ⚠️ 部分正确 |
| 打破 max-autotune | 认为破坏了并行性 | **完全正确** | ✅ 100% |

---

## 修正后的完整流程图

```
tuned_mm()
    ↓
生成 choices = [aten_mm, decompose_k_subgraph, mm_template, ...]
    ↓
autotune_select_algorithm(choices, return_multi_template=True)
    ↓
┌─────────────────────────── 路径选择 ─────────────────────────┐
│                                                              │
│  if return_multi_template and max_autotune:                 │
│      return MultiTemplateBuffer(                            │
│          layout, input_nodes,                               │
│          get_timings,  # ← 延迟 benchmark                   │
│          choices       # ← [aten, decompose_k, ...]         │
│      )                                                       │
│  else:                                                       │
│      timings = do_autotuning(choices, precompile_fn)        │
│      min_choice = min(timings, key=timings.get)             │
│      return min_choice.output_node()                        │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    ↓
scheduler.finalize_multi_template_buffers()
    ↓
multi_node.get_min_choice()
    ↓
multi_node.choice_timings()  # ← 第一次调用
    ↓
self._choice_timings_fn(hint_override)  # ← 即 get_timings()
    ↓
do_autotuning(filtered_choices, precompile_fn)
    ↓
┌─────────────────── Precompile 阶段 (并行) ──────────────────┐
│                                                              │
│  for c in choices:                                          │
│      if hasattr(c, "precompile"):                           │
│          future = executor.submit(c.precompile)             │
│          # ✅ TritonTemplate, ExternKernel 并行             │
│      else:                                                   │
│          pass  # ❌ SubgraphChoiceCaller 跳过               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    ↓
┌─────────────────── Benchmark 阶段 (串行) ───────────────────┐
│                                                              │
│  for c in choices:                                          │
│      timing = c.benchmark(*args, out=out)                   │
│      # TritonTemplate: 已 precompile，benchmark 快         │
│      # SubgraphChoiceCaller: 这里才编译，非常慢             │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    ↓
选择最快的 choice (min_choice)
    ↓
min_choice.output_node()
    ↓
┌───────────────────── Choice Type 分支 ──────────────────────┐
│                                                              │
│  if isinstance(min_choice, SubgraphChoiceCaller):           │
│      return SubgraphBuffer(...)  # ← 不支持 fusion         │
│  elif isinstance(min_choice, TritonTemplateCaller):         │
│      return TritonTemplateBuffer(...)  # ← 支持 fusion     │
│  else:                                                       │
│      return ExternKernel(...)                               │
│                                                              │
└──────────────────────────────────────────────────────────────┘
    ↓
后续 scheduler fusion 阶段
    ↓
SubgraphBuffer: 黑盒，不能 fuse
TritonTemplateBuffer: 可以 fuse epilogue
```

---

## 推荐阅读顺序

1. ✅ **已理解**：MultiTemplateBuffer 的作用（延迟选择）
2. ✅ **已理解**：SubgraphChoiceCaller 没有 precompile 的性能问题
3. ✅ **已理解**：SubgraphBuffer 不支持 fusion
4. 🔜 **下一步**：理解 Solution A 如何添加 precompile
5. 🔜 **下一步**：理解 Solution B 如何实现 custom op fusion

希望这个分析解答了你的疑惑！
