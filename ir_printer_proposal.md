# Proposal: a compact Inductor debug printer

This proposal adds a compact, read-only debug artifact between the post-grad FX graph and generated code. It exposes the computation, iteration domains, indexing, dataflow, scheduler transformations, and kernel fusion without requiring readers to reconstruct them from native `IRNode` and scheduler dumps.

## Why this printer is useful

Inductor's detailed native dumps remain essential for debugging compiler internals. This printer complements them with a readable view of existing Inductor state; it is not a new compiler IR. Three capture points are useful in practice:

1. **Post-lowering:** raw operations and their as-lowered loop bodies.
2. **Post-scheduler:** scheduler-derived loop bodies after dependency analysis and loop reordering.
3. **Post-fusion:** generated-kernel groups, their interfaces, and the operations inside each fused region.

The post-fusion view is where the difference is clearest. Consider the softmax portion of a small attention graph. The existing dump represents one fused region containing five scheduler nodes, their dependencies, output buffers, and each complete loop body:

Abridged existing native dump:

```text
op1_op2_op3_op4_op5:
    FusedSchedulerNode(SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode)
op1_op2_op3_op4_op5.writes = [
    MemoryDep('buf1', 8*d0 + d1, {d0: 2, d1: 8}),
    MemoryDep('buf2', 8*d0 + d1, {d0: 2, d1: 8}),
    MemoryDep('buf3', 8*d0 + d1, {d0: 2, d1: 8}),
    MemoryDep('buf4', 8*d0 + d1, {d0: 2, d1: 8}),
    MemoryDep('buf5', 64*d0 + 8*d1 + d2, {d0: 2, d1: 8, d2: 8}),
]
op1_op2_op3_op4_op5.unmet_dependencies = [
    MemoryDep('buf0', 64*d0 + 8*d1 + d2, {d0: 2, d1: 8, d2: 8})
]
op1_op2_op3_op4_op5.outputs = [
    buf1: ComputedBuffer  # users: op4, op5
    buf2: ComputedBuffer  # users: op4, op5
    buf3: ComputedBuffer  # users: op4, op5
    buf4: ComputedBuffer  # user: op5
    buf5: ComputedBuffer  # user: op6
]
op1_op2_op3_op4_op5.snodes[0] =
    op1: SchedulerNode(ComputedBuffer)
    op1.writes = [MemoryDep('buf1', 8*d0 + d1, {d0: 2, d1: 8})]
    op1.unmet_dependencies = [
        MemoryDep('buf0', 64*d0 + 8*d1 + d2, {d0: 2, d1: 8, d2: 8})
    ]
    op1.met_dependencies = []
    op1.outputs = [
        buf1: ComputedBuffer  # users: op4, op5
    ]
    op1.group.device = device(type='cuda', index=0)
    op1.group.iteration = ((2, 8, 1), (8,))
    op1.sizes = ([2, 8, 1], [8])
    op1.node.data = Reduction(
        'cuda',
        torch.bool,
        def inner_fn(index, rindex):
            i0, i1, _ = index
            r0_0 = rindex
            tmp0 = ops.load(buf0, r0_0 + 8 * i1 + 64 * i0)
            tmp1 = ops.constant(0.0, torch.float32)
            tmp2 = tmp0 != tmp1
            return tmp2
        ,
        ranges=[2, 8, 1],
        reduction_ranges=[8],
        reduction_type=any,
        origin_node=any_1,
        origins=OrderedSet([any_1, ne]),
    )
op1_op2_op3_op4_op5.snodes[1] =
    op2: SchedulerNode(ComputedBuffer)
    ...
# three more scheduler nodes and their LoopBody graphs
```

New IR printer:

```text
# inputs
arg0_1: f32[2,8,16], arg1_1: f32[2,8,16], arg2_1: f32[2,8,16]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[2,8,8]
        extern_kernels.bmm(arg0_1, reinterpret_tensor(arg1_1, [2,16,8], [128,1,16], 0))

kernel k1
    inputs:   [buf0]
    outputs:  [buf5]
    internal: [buf1, buf2, buf3, buf4]

    foreach x in [0,16), reduce r in [0,8):

        op1:  buf1: b8[2,8,1] @ [x:16]  <-  buf0: f32[2,8,8] @ [x:16,r:8]      # any(not ((((buf0[r + 8*x]) * 0.25) == ((buf0[r + 8*x]) * 0.25)) and (abs((buf0[r + 8*x]) * 0.25) != inf)) over r)
            ix0 = r + 8*x
            t0 = buf0[ix0]
            t1 = t0 * 0.25
            t2 = buf0[ix0]
            t3 = t2 * 0.25
            t4 = t1 == t3
            t5 = buf0[ix0]
            t6 = t5 * 0.25
            t7 = abs(t6)
            t8 = t7 != inf
            t9 = t4 and t8
            t10 = not t9
            r0 = any(t10 over r)
            buf1[x] = r0

        op2:  buf2: f32[2,8,1] @ [x:16]  <-  buf0: f32[2,8,8] @ [x:16,r:8]      # max((buf0[r + 8*x]) * 1.0 over r)
            t0 = buf0[r + 8*x]
            t1 = t0 * 1.0
            r0 = max(t1 over r)
            buf2[x] = r0

        op3:  buf3: f32[2,8,1] @ [x:16]  <-  buf0: f32[2,8,8] @ [x:16,r:8]      # max((buf0[r + 8*x]) * 0.25 over r)
            t0 = buf0[r + 8*x]
            t1 = t0 * 0.25
            r0 = max(t1 over r)
            buf3[x] = r0

        op4:  buf4: f32[2,8,1] @ [x:16]  <-  buf1: b8[2,8,1] @ [x:16], buf0: f32[2,8,8] @ [x:16,r:8], buf2: f32[2,8,1] @ [x:16], buf3: f32[2,8,1] @ [x:16]      # sum(exp(where(not buf1[x], (((buf0[r + 8*x]) * 1.0) - buf2[x]) * 0.25, ((buf0[r + 8*x]) * 0.25) - buf3[x])) over r)
            t0 = buf1[x]
            t1 = not t0
            ix0 = r + 8*x
            t2 = buf0[ix0]
            t3 = t2 * 1.0
            t4 = buf2[x]
            t5 = t3 - t4
            t6 = t5 * 0.25
            t7 = buf0[ix0]
            t8 = t7 * 0.25
            t9 = buf3[x]
            t10 = t8 - t9
            t11 = where(t1, t6, t10)
            t12 = exp(t11)
            r0 = sum(t12 over r)
            buf4[x] = r0

        op5:  buf5: f32[2,8,8] @ [x:16,r:8]  <-  buf1: b8[2,8,1] @ [x:16], buf0: f32[2,8,8] @ [x:16,r:8], buf2: f32[2,8,1] @ [x:16], buf3: f32[2,8,1] @ [x:16], buf4: f32[2,8,1] @ [x:16]      # exp(where(not buf1[x], (((buf0[r + 8*x]) * 1.0) - buf2[x]) * 0.25, ((buf0[r + 8*x]) * 0.25) - buf3[x])) / buf4[x]
            t0 = buf1[x]
            t1 = not t0
            ix0 = r + 8*x
            t2 = buf0[ix0]
            t3 = t2 * 1.0
            t4 = buf2[x]
            t5 = t3 - t4
            t6 = t5 * 0.25
            t7 = buf0[ix0]
            t8 = t7 * 0.25
            t9 = buf3[x]
            t10 = t8 - t9
            t11 = where(t1, t6, t10)
            t12 = exp(t11)
            t13 = buf4[x]
            t14 = t12 / t13
            buf5[ix0] = t14

kernel k2
    # implementation: extern call
    inputs:   [arg2_1, buf5]
    outputs:  [buf6]

    op6:  buf6: f32[2,8,16]
        extern_kernels.bmm(buf5, arg2_1)

# outputs
return (buf6)
```

The `@ [...]` annotation appears only when the access domain is not already obvious from the logical shape and a direct row-major index. Here, `buf0` is logically `[2,8,8]` but participates as `[x:16,r:8]`, while each `[2,8,1]` reduction result participates only as `[x:16]`. The reads/writes expose the dependency structure without assigning a fusion category: `op1..op3` are independent sibling reductions over `buf0`, `op4` consumes their results, and `op5` consumes `buf4`.

The debug view makes the scheduler result explicit: one input enters one fused kernel, four former buffers are internal to it, and only `buf5` crosses the output boundary. It preserves the meaningful phases inside the kernel rather than flattening five iteration spaces into an opaque block.

A `ForeachKernelSchedulerNode` groups independent scheduler branches into one generated kernel. The debug printer gives each branch its own visible logical domain:

```text
# inputs
arg0_1: f32[1024], arg1_1: f32[2048]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0, buf1]

    parallel:

        branch 0:

            foreach x in [0,1024):

                op0:  buf0: f32[1024]  <-  arg0_1: f32[1024]      # sin(arg0_1[x])
                    t0 = arg0_1[x]
                    t1 = sin(t0)
                    buf0[x] = t1

        branch 1:

            foreach x in [0,2048):

                op1:  buf1: f32[2048]  <-  arg1_1: f32[2048]      # cos(arg1_1[x])
                    t0 = arg1_1[x]
                    t1 = cos(t0)
                    buf1[x] = t1

# outputs
return (buf0, buf1)
```

`parallel:` identifies the foreach/combo scheduler container, and each `branch N:` corresponds to one entry of `ForeachKernelSchedulerNode.snodes`. A branch may contain fused work, but the printer does not synthesize a summed top-level domain: the branches have independent logical domains, and their eventual PID, block, thread, and vector mapping belongs to backend code generation.

The printer emits a short, relevance-filtered notation legend with each capture. The proposal shows it once here and omits the repeated device/stage preamble and legend from subsequent captured blocks:

```text
#   foreach x in [0,N)    independent scalar points; backend mapping unspecified.
#                         Listed outermost to innermost; the last one varies fastest.
#   reduce  r in [0,N)    points combined; order unspecified
#   buf[expr]              expr is a FLAT element offset, not a tensor subscript
#   T[shape] @ [x:N,r:M]   non-obvious access domain; omitted for a direct match
#   rN = OP(v over r)      rN is v combined across r, one per foreach point
```

`foreach` describes scalar semantics over the whole logical point set. It does not mean a Triton program, CUDA thread or block, vector lane, or launch shape; generated code later chunks and maps these points for its backend. The bounds, flat indices, scalar operations, reduction results, and kernel membership all come from existing Inductor objects.

### Specialized mixed-order reductions

Some scheduler nodes intentionally retain node-local domains. With `triton.mix_order_reduction` enabled, these sibling reductions over opposite dimensions form a `FusedMixOrderReductions` node:

```python
def f(x):  # x: f32[8192,1024]
    return x.sum(dim=1), x.sum(dim=0)
```

The final post-fusion debug output is:

```text
# inputs
arg0_1: f32[8192,1024]

# compute

kernel k0
    # pattern: mixed-order reduction
    inputs:   [arg0_1]
    outputs:  [buf0, buf1]

    op0:  buf0: f32[8192]  <-  arg0_1: f32[8192,1024]      # sum(arg0_1[1024*p0 + p1] over p1)
        foreach p0 in [0,8192), reduce p1 in [0,1024):
            t0 = arg0_1[1024*p0 + p1]
            r0 = sum(t0 over p1)
            buf0[p0] = r0

    op1:  buf1: f32[1024,64] stride=[1,1024]  <-  arg0_1: f32[8192,1024]      # sum(arg0_1[131072*p0 + p1 + 1024*p2] over p2)
        foreach p0 in [0,64), p1 in [0,1024), reduce p2 in [0,128):
            t0 = arg0_1[131072*p0 + p1 + 1024*p2]
            r0 = sum(t0 over p2)
            buf1[1024*p0 + p1] = r0

kernel k1
    inputs:   [buf1]
    outputs:  [buf2]

    foreach x in [0,1024), reduce r in [0,64):

        op2:  buf2: f32[1024]  <-  buf1: f32[1024,64] stride=[1,1024] @ [x:1024,r:64]      # sum(buf1[1024*r + x] over r)
            t0 = buf1[1024*r + x]
            r0 = sum(t0 over r)
            buf2[x] = r0

# outputs
return (buf0, buf2)
```

`kernel k0` keeps two node-local `p*` domains because the row and column reductions have incompatible loop orders; printing one normalized domain would be false. `kernel k1` is an ordinary follow-up kernel that completes the split column reduction, so it uses the normal `x/r` form.

### Understanding transformations between compiler stages

Rendering the same graph at multiple boundaries makes compiler transformations directly comparable. The complete paired captures below show the as-lowered loop order, the scheduler-derived body after `simplify_and_reorder`, and the final fused kernel domains without inserting schematic or manually shortened output. Comparing those captures reveals dimension reordering and merging, changes to flat index expressions, and buffers that move from kernel interfaces into fused-region internals.

ExternKernelOut(
  python_kernel_name='extern_kernels.mm',
  name=buf0,
  layout=FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1]),
  inputs=[InputBuffer(name='arg0_1', layout=FixedLayout('cuda:0', torch.float32, size=[8, 64], stride=[64, 1])), ReinterpretView(
    StorageBox(
      InputBuffer(name='arg1_1', layout=FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1]))
    ),
    FixedLayout('cuda:0', torch.float32, size=[64, 32], stride=[1, 64]),
    origins=OrderedSet([mm_default, permute]),
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 347, in prog_first_medium,
        h = torch.nn.functional.linear(x, w, b),
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.mm,
  cpp_kernel_name=at::mm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.mm.out,
Each program shows the post-grad FX graph followed by paired native and formatted Inductor IR snapshots at three stages: `post_lowering` immediately before scheduler-node construction, `post_scheduler` after scheduler initialization and dependency analysis, and `post_fusion` after fusion and loop merging. Native snapshots use `DebugFormatter._write_ir`; formatted snapshots use `format_post_lowering_ir` against the state at that exact boundary. Each formatted block is exact printer output from `# inputs` through `# outputs`; only the repeated device/stage preamble and notation legend shown above are omitted.
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=mm_default,
  origins=OrderedSet([mm_default, permute]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 347, in prog_first_medium,
      h = torch.nn.functional.linear(x, w, b),
  ,
  }
)


ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.float32, size=[8], stride=[1]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0 = index
      r0_0 = rindex
      tmp0 = ops.load(arg2_1, r0_0)
      tmp1 = ops.load(buf0, r0_0 + 32 * i0)
      tmp2 = tmp0 + tmp1
      tmp3 = ops.constant(0.5, torch.float32)
      tmp4 = tmp2 * tmp3
      tmp5 = ops.load(arg2_1, r0_0)
      tmp6 = ops.load(buf0, r0_0 + 32 * i0)
      tmp7 = tmp5 + tmp6
      tmp8 = ops.constant(0.7071067811865476, torch.float32)
      tmp9 = tmp7 * tmp8
      tmp10 = ops.erf(tmp9)
      tmp11 = ops.constant(1, torch.float32)
      tmp12 = tmp10 + tmp11
      tmp13 = tmp4 * tmp12
      return tmp13
  ,
  ranges=[8],
  reduction_ranges=[32],
  reduction_type=sum,
  origin_node=None,
  origins=OrderedSet([mean, mul_2, mul, add_tensor, add, erf, m...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 349, in prog_first_medium,
      return h.mean(dim=-1),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 348, in prog_first_medium,
      h = torch.nn.functional.gelu(h),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 347, in prog_first_medium,
      h = torch.nn.functional.linear(x, w, b),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf2', layout=FixedLayout('cuda:0', torch.float32, size=[8], stride=[1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0 = index
      tmp0 = ops.load(buf1, i0)
      tmp1 = ops.index_expr(32, torch.float32)
      tmp2 = tmp0 / tmp1
      return tmp2
  ,
  ranges=[8],
  origin_node=mean,
  origins=OrderedSet([mean, mul_2, mul, add_tensor, add, erf, m...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 349, in prog_first_medium,
      return h.mean(dim=-1),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 348, in prog_first_medium,
      h = torch.nn.functional.gelu(h),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 347, in prog_first_medium,
      h = torch.nn.functional.linear(x, w, b),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[8,64], arg1_1: f32[32,64], arg2_1: f32[32]

# compute

# op0
buf0: f32[8,32]
    extern_kernels.mm(arg0_1, reinterpret_tensor(arg1_1, [64,32], [1,64], 0))

# op1  sum(((arg2_1[p1] + (buf0[32*p0 + p1])) * 0.5) * (erf((arg2_1[p1] + (buf0[32*p0 + p1])) * 0.7071067811865476) + 1.0) over p1)
buf1: f32[8]  <-  arg2_1: f32[32], buf0: f32[8,32]
    foreach p0 in [0,8), reduce p1 in [0,32):
        t0 = arg2_1[p1]
        ix0 = 32*p0 + p1
        t1 = buf0[ix0]
        t2 = t0 + t1
        t3 = t2 * 0.5
        t4 = arg2_1[p1]
        t5 = buf0[ix0]
        t6 = t4 + t5
        t7 = t6 * 0.7071067811865476
        t8 = erf(t7)
        t9 = t8 + 1.0
        t10 = t3 * t9
        r0 = sum(t10 over p1)
        buf1[p0] = r0

# op2  buf1[p0] / 32.0
buf2: f32[8]  <-  buf1: f32[8]
    foreach p0 in [0,8):
        t0 = buf1[p0]
        t1 = t0 / 32.0
        buf2[p0] = t1

# outputs
return (buf2)
```

### post_scheduler

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None), StarDep(name='arg1_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ExternKernelOut
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=False, is_weak=False)]
]
op0.node.kernel = extern_kernels.mm


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', d0, {d0: 8})]
op1.unmet_dependencies = [MemoryDep('buf0', 32*d0 + d1, {d0: 8, d1: 32})]
op1.met_dependencies = [MemoryDep('arg2_1', d1, {d0: 8, d1: 32})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=True, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (8, 32)
op1.sizes = ([8], [32])
arg2_1_layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
class op1_loop_body:
    var_ranges = {p0: 8, p1: 32}
    index0 = p1
    index1 = 32*p0 + p1
    index2 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg2_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        add = ops.add(load, load_1)
        constant = ops.constant(0.5, torch.float32)
        mul = ops.mul(add, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('arg2_1', get_index_2)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf0', get_index_3)
        add_1 = ops.add(load_2, load_3)
        constant_1 = ops.constant(0.7071067811865476, torch.float32)
        mul_1 = ops.mul(add_1, constant_1)
        erf = ops.erf(mul_1)
        constant_2 = ops.constant(1.0, torch.float32)
        add_2 = ops.add(erf, constant_2)
        mul_2 = ops.mul(mul, add_2)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', mul_2)
        get_index_4 = self.get_index('index2')
        store_reduction = ops.store_reduction('buf1', get_index_4, reduction)
        return None


op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', d0, {d0: 8})]
op2.unmet_dependencies = [MemoryDep('buf1', d0, {d0: 8})]
op2.met_dependencies = []
op2.min_input_distance = 2
op2.max_input_distance = 2
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op2.group.device = cuda:0
op2.group.iteration = (8, 1)
op2.sizes = ([8], [])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
class op2_loop_body:
    var_ranges = {p0: 8}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf1', get_index)
        constant = ops.constant(32.0, torch.float32)
        truediv = ops.truediv(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf2', get_index_1, truediv, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[8,64], arg1_1: f32[32,64], arg2_1: f32[32]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[8,32]
        extern_kernels.mm(arg0_1, reinterpret_tensor(arg1_1, [64,32], [1,64], 0))

kernel k1
    inputs:   [arg2_1, buf0]
    outputs:  [buf1]

    op1:  buf1: f32[8]  <-  arg2_1: f32[32], buf0: f32[8,32]      # sum(((arg2_1[p1] + (buf0[32*p0 + p1])) * 0.5) * (erf((arg2_1[p1] + (buf0[32*p0 + p1])) * 0.7071067811865476) + 1.0) over p1)
        foreach p0 in [0,8), reduce p1 in [0,32):
            t0 = arg2_1[p1]
            ix0 = 32*p0 + p1
            t1 = buf0[ix0]
            t2 = t0 + t1
            t3 = t2 * 0.5
            t4 = arg2_1[p1]
            t5 = buf0[ix0]
            t6 = t4 + t5
            t7 = t6 * 0.7071067811865476
            t8 = erf(t7)
            t9 = t8 + 1.0
            t10 = t3 * t9
            r0 = sum(t10 over p1)
            buf1[p0] = r0

kernel k2
    inputs:   [buf1]
    outputs:  [buf2]

    op2:  buf2: f32[8]  <-  buf1: f32[8]      # buf1[p0] / 32.0
        foreach p0 in [0,8):
            t0 = buf1[p0]
            t1 = t0 / 32.0
            buf2[p0] = t1

# outputs
return (buf2)
```

### post_fusion

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None), StarDep(name='arg1_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ExternKernelOut
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=False, is_weak=False)]
]
op0.node.kernel = extern_kernels.mm


op1_op2: FusedSchedulerNode(SchedulerNode,SchedulerNode)
op1_op2.writes = [MemoryDep('buf1', d0, {d0: 8}), MemoryDep('buf2', d0, {d0: 8})]
op1_op2.unmet_dependencies = [MemoryDep('buf0', 32*d0 + d1, {d0: 8, d1: 32})]
op1_op2.met_dependencies = [MemoryDep('arg2_1', d1, {d0: 8, d1: 32})]
op1_op2.min_input_distance = 1
op1_op2.max_input_distance = 2
op1_op2.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=True, is_weak=False)]
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op1_op2.snodes[0] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 8})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 256})]
op1.met_dependencies = [MemoryDep('arg2_1', c1, {c0: 8, c1: 32})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=True, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (8, 32)
op1.sizes = ((8,), (32,))
arg2_1_layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
class op1_loop_body:
    var_ranges = {p0: 8, p1: 32}
    index0 = p1
    index1 = 32*p0 + p1
    index2 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg2_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        add = ops.add(load, load_1)
        constant = ops.constant(0.5, torch.float32)
        mul = ops.mul(add, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('arg2_1', get_index_2)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf0', get_index_3)
        add_1 = ops.add(load_2, load_3)
        constant_1 = ops.constant(0.7071067811865476, torch.float32)
        mul_1 = ops.mul(add_1, constant_1)
        erf = ops.erf(mul_1)
        constant_2 = ops.constant(1.0, torch.float32)
        add_2 = ops.add(erf, constant_2)
        mul_2 = ops.mul(mul, add_2)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', mul_2)
        get_index_4 = self.get_index('index2')
        store_reduction = ops.store_reduction('buf1', get_index_4, reduction)
        return None
op1_op2.snodes[1] =
op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', c0, {c0: 8})]
op2.unmet_dependencies = [MemoryDep('buf1', c0, {c0: 8})]
op2.met_dependencies = []
op2.min_input_distance = 2
op2.max_input_distance = 2
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op2.group.device = cuda:0
op2.group.iteration = (8, 1)
op2.sizes = ((8,), ())
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
class op2_loop_body:
    var_ranges = {p0: 8}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf1', get_index)
        constant = ops.constant(32.0, torch.float32)
        truediv = ops.truediv(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf2', get_index_1, truediv, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[8,64], arg1_1: f32[32,64], arg2_1: f32[32]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[8,32]
        extern_kernels.mm(arg0_1, reinterpret_tensor(arg1_1, [64,32], [1,64], 0))

kernel k1
    inputs:   [arg2_1, buf0]
    outputs:  [buf2]
    internal: [buf1]

    foreach x in [0,8), reduce r in [0,32):

        op1:  buf1: f32[8]  <-  arg2_1: f32[32], buf0: f32[8,32]      # sum(((arg2_1[r] + (buf0[r + 32*x])) * 0.5) * (erf((arg2_1[r] + (buf0[r + 32*x])) * 0.7071067811865476) + 1.0) over r)
            t0 = arg2_1[r]
            ix0 = r + 32*x
            t1 = buf0[ix0]
            t2 = t0 + t1
            t3 = t2 * 0.5
            t4 = arg2_1[r]
            t5 = buf0[ix0]
            t6 = t4 + t5
            t7 = t6 * 0.7071067811865476
            t8 = erf(t7)
            t9 = t8 + 1.0
            t10 = t3 * t9
            r0 = sum(t10 over r)
            buf1[x] = r0

        op2:  buf2: f32[8]  <-  buf1: f32[8]      # buf1[x] / 32.0
            t0 = buf1[x]
            t1 = t0 / 32.0
            buf2[x] = t1

# outputs
return (buf2)
```

## Program 2: single pointwise op — x * 2 + 1

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024]"):
        # File: /tmp/ir_printer_examples.py:207 in prog_pointwise_scalar, code: return x * 2 + 1
        mul: "f32[1024]" = torch.ops.aten.mul.Tensor(arg0_1, 2);  arg0_1 = None
        add: "f32[1024]" = torch.ops.aten.add.Tensor(mul, 1);  mul = None
        return (add,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0 = index
      tmp0 = ops.load(arg0_1, i0)
      tmp1 = ops.constant(2, torch.float32)
      tmp2 = tmp0 * tmp1
      tmp3 = ops.constant(1, torch.float32)
      tmp4 = tmp2 + tmp3
      return tmp4
  ,
  ranges=[1024],
  origin_node=add,
  origins=OrderedSet([add, mul]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 207, in prog_pointwise_scalar,
      return x * 2 + 1,
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[1024]

# compute

# op0  (arg0_1[p0] * 2.0) + 1.0
buf0: f32[1024]  <-  arg0_1: f32[1024]
    foreach p0 in [0,1024):
        t0 = arg0_1[p0]
        t1 = t0 * 2.0
        t2 = t1 + 1.0
        buf0[p0] = t2

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 1024})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', d0, {d0: 1024})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (1024, 1)
op0.sizes = ([1024], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 1024}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(2.0, torch.float32)
        mul = ops.mul(load, constant)
        constant_1 = ops.constant(1.0, torch.float32)
        add = ops.add(mul, constant_1)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, add, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[1024]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[1024]  <-  arg0_1: f32[1024]      # (arg0_1[p0] * 2.0) + 1.0
        foreach p0 in [0,1024):
            t0 = arg0_1[p0]
            t1 = t0 * 2.0
            t2 = t1 + 1.0
            buf0[p0] = t2

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 1024})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 1024})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (1024, 1)
op0.sizes = ((1024,), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 1024}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(2.0, torch.float32)
        mul = ops.mul(load, constant)
        constant_1 = ops.constant(1.0, torch.float32)
        add = ops.add(mul, constant_1)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, add, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[1024]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    foreach x in [0,1024):

        op0:  buf0: f32[1024]  <-  arg0_1: f32[1024]      # (arg0_1[x] * 2.0) + 1.0
            t0 = arg0_1[x]
            t1 = t0 * 2.0
            t2 = t1 + 1.0
            buf0[x] = t2

# outputs
return (buf0)
```

## Program 3: pointwise with broadcast — x[N,M] + y[1,M]

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 64]", arg1_1: "f32[1, 64]"):
        # File: /tmp/ir_printer_examples.py:212 in prog_pointwise_broadcast, code: return x + y
        add: "f32[32, 64]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return (add,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1 = index
      tmp0 = ops.load(arg0_1, i1 + 64 * i0)
      tmp1 = ops.load(arg1_1, i1)
      tmp2 = tmp0 + tmp1
      return tmp2
  ,
  ranges=[32, 64],
  origin_node=add,
  origins=OrderedSet([add]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 212, in prog_pointwise_broadcast,
      return x + y,
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[32,64], arg1_1: f32[1,64]

# compute

# op0  (arg0_1[64*p0 + p1]) + arg1_1[p1]
buf0: f32[32,64]  <-  arg0_1: f32[32,64], arg1_1: f32[1,64]
    foreach p0 in [0,32), p1 in [0,64):
        ix0 = 64*p0 + p1
        t0 = arg0_1[ix0]
        t1 = arg1_1[p1]
        t2 = t0 + t1
        buf0[ix0] = t2

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 64*d0 + d1, {d0: 32, d1: 64})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', 64*d0 + d1, {d0: 32, d1: 64}),
        MemoryDep('arg1_1', d1, {d0: 32, d1: 64})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (2048, 1)
op0.sizes = ([32, 64], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 64], stride=[64, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
class op0_loop_body:
    var_ranges = {p0: 32, p1: 64}
    index0 = 64*p0 + p1
    index1 = p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf0', get_index_2, add, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[32,64], arg1_1: f32[1,64]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[32,64]  <-  arg0_1: f32[32,64], arg1_1: f32[1,64]      # (arg0_1[64*p0 + p1]) + arg1_1[p1]
        foreach p0 in [0,32), p1 in [0,64):
            ix0 = 64*p0 + p1
            t0 = arg0_1[ix0]
            t1 = arg1_1[p1]
            t2 = t0 + t1
            buf0[ix0] = t2

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 2048})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 2048}), MemoryDep('arg1_1', c1, {c0: 32, c1: 64})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (2048, 1)
op0.sizes = ((32, 64), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 64], stride=[64, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
class op0_loop_body:
    var_ranges = {p0: 32, p1: 64}
    index0 = 64*p0 + p1
    index1 = p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf0', get_index_2, add, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[32,64], arg1_1: f32[1,64]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    foreach x in [0,2048):

        op0:  buf0: f32[32,64] @ [x:2048]  <-  arg0_1: f32[32,64] @ [x:2048], arg1_1: f32[1,64] @ [x:2048]      # (arg0_1[64*((x//64)) + (ModularIndexing(x, 1, 64))]) + arg1_1[ModularIndexing(x, 1, 64)]
            ix0 = 64*((x//64)) + (ModularIndexing(x, 1, 64))
            t0 = arg0_1[ix0]
            t1 = arg1_1[ModularIndexing(x, 1, 64)]
            t2 = t0 + t1
            buf0[ix0] = t2

# outputs
return (buf0)
```

## Program 4: single-dim reduction — x.sum(dim=1)

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 64]"):
        # File: /tmp/ir_printer_examples.py:217 in prog_reduction_sum_dim, code: return x.sum(dim=1)
        sum_1: "f32[32]" = torch.ops.aten.sum.dim_IntList(arg0_1, [1]);  arg0_1 = None
        return (sum_1,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[32], stride=[1]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0 = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 64 * i0)
      return tmp0
  ,
  ranges=[32],
  reduction_ranges=[64],
  reduction_type=sum,
  origin_node=sum_1,
  origins=OrderedSet([sum_1]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 217, in prog_reduction_sum_dim,
      return x.sum(dim=1),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[32,64]

# compute

# op0  sum(arg0_1[64*p0 + p1] over p1)
buf0: f32[32]  <-  arg0_1: f32[32,64]
    foreach p0 in [0,32), reduce p1 in [0,64):
        t0 = arg0_1[64*p0 + p1]
        r0 = sum(t0 over p1)
        buf0[p0] = r0

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 32})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', 64*d0 + d1, {d0: 32, d1: 64})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (32, 64)
op0.sizes = ([32], [64])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 32, p1: 64}
    index0 = 64*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[32,64]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[32]  <-  arg0_1: f32[32,64]      # sum(arg0_1[64*p0 + p1] over p1)
        foreach p0 in [0,32), reduce p1 in [0,64):
            t0 = arg0_1[64*p0 + p1]
            r0 = sum(t0 over p1)
            buf0[p0] = r0

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 32})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 2048})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (32, 64)
op0.sizes = ((32,), (64,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 32, p1: 64}
    index0 = 64*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[32,64]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    foreach x in [0,32), reduce r in [0,64):

        op0:  buf0: f32[32]  <-  arg0_1: f32[32,64]      # sum(arg0_1[r + 64*x] over r)
            t0 = arg0_1[r + 64*x]
            r0 = sum(t0 over r)
            buf0[x] = r0

# outputs
return (buf0)
```

## Program 5: full reduction to scalar — x.mean()

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[4096]"):
        # File: /tmp/ir_printer_examples.py:222 in prog_reduction_mean_full, code: return x.mean()
        mean: "f32[]" = torch.ops.aten.mean.default(arg0_1);  arg0_1 = None
        return (mean,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[], stride=[]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0)
      return tmp0
  ,
  ranges=[],
  reduction_ranges=[4096],
  reduction_type=sum,
  origin_node=None,
  origins=OrderedSet([mean]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 222, in prog_reduction_mean_full,
      return x.mean(),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.float32, size=[], stride=[]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      tmp0 = ops.load(buf0, 0)
      tmp1 = ops.index_expr(4096, torch.float32)
      tmp2 = tmp0 / tmp1
      return tmp2
  ,
  ranges=[],
  origin_node=mean,
  origins=OrderedSet([mean]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 222, in prog_reduction_mean_full,
      return x.mean(),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[4096]

# compute

# op0  sum(arg0_1[p0] over p0)
buf0: f32[]  <-  arg0_1: f32[4096]
    reduce p0 in [0,4096):
        t0 = arg0_1[p0]
        r0 = sum(t0 over p0)
        buf0[0] = r0

# op1  buf0[0] / 4096.0
buf1: f32[]  <-  buf0: f32[]
    foreach (scalar):
        ix0 = 0
        t0 = buf0[ix0]
        t1 = t0 / 4096.0
        buf1[ix0] = t1

# outputs
return (buf1)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 0, {})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', d0, {d0: 4096})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (1, 4096)
op0.sizes = ([], [4096])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4096], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
class op0_loop_body:
    var_ranges = {p0: 4096}
    index0 = p0
    index1 = 0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', 0, {})]
op1.unmet_dependencies = [MemoryDep('buf0', 0, {})]
op1.met_dependencies = []
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (1, 1)
op1.sizes = ([], [])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
class op1_loop_body:
    var_ranges = {}
    index0 = 0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        constant = ops.constant(4096.0, torch.float32)
        truediv = ops.truediv(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf1', get_index_1, truediv, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[4096]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[]  <-  arg0_1: f32[4096]      # sum(arg0_1[p0] over p0)
        reduce p0 in [0,4096):
            t0 = arg0_1[p0]
            r0 = sum(t0 over p0)
            buf0[0] = r0

kernel k1
    inputs:   [buf0]
    outputs:  [buf1]

    op1:  buf1: f32[]  <-  buf0: f32[]      # buf0[0] / 4096.0
        foreach (scalar):
            ix0 = 0
            t0 = buf0[ix0]
            t1 = t0 / 4096.0
            buf1[ix0] = t1

# outputs
return (buf1)
```

### post_fusion

```text
op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
op0_op1.writes = [MemoryDep('buf0', 0, {}), MemoryDep('buf1', 0, {})]
op0_op1.unmet_dependencies = []
op0_op1.met_dependencies = [MemoryDep('arg0_1', d0, {d0: 4096})]
op0_op1.min_input_distance = 0
op0_op1.max_input_distance = 1
op0_op1.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False)]
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0_op1.snodes[0] =
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 0, {})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 4096})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (1, 4096)
op0.sizes = ((), (4096,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4096], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
class op0_loop_body:
    var_ranges = {p0: 4096}
    index0 = p0
    index1 = 0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None
op0_op1.snodes[1] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', 0, {})]
op1.unmet_dependencies = [MemoryDep('buf0', 0, {})]
op1.met_dependencies = []
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (1, 1)
op1.sizes = ((), ())
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
class op1_loop_body:
    var_ranges = {}
    index0 = 0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        constant = ops.constant(4096.0, torch.float32)
        truediv = ops.truediv(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf1', get_index_1, truediv, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[4096]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf1]
    internal: [buf0]

    reduce r in [0,4096):

        op0:  buf0: f32[] @ [scalar]  <-  arg0_1: f32[4096]      # sum(arg0_1[r] over r)
            t0 = arg0_1[r]
            r0 = sum(t0 over r)
            buf0[0] = r0

        op1:  buf1: f32[] @ [scalar]  <-  buf0: f32[] @ [scalar]      # buf0[0] / 4096.0
            ix0 = 0
            t0 = buf0[ix0]
            t1 = t0 / 4096.0
            buf1[ix0] = t1

# outputs
return (buf1)
```

## Program 6: reduction over multiple dims — x.sum(dim=(0, 2))

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 8, 16]"):
        # File: /tmp/ir_printer_examples.py:227 in prog_reduction_multi_dim, code: return x.sum(dim=(0, 2))
        sum_1: "f32[8]" = torch.ops.aten.sum.dim_IntList(arg0_1, [0, 2]);  arg0_1 = None
        return (sum_1,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[8], stride=[1]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0 = index
      r0_0, r0_1 = rindex
      tmp0 = ops.load(arg0_1, r0_1 + 16 * i0 + 128 * r0_0)
      return tmp0
  ,
  ranges=[8],
  reduction_ranges=[4, 16],
  reduction_type=sum,
  origin_node=sum_1,
  origins=OrderedSet([sum_1]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 227, in prog_reduction_multi_dim,
      return x.sum(dim=(0, 2)),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[4,8,16]

# compute

# op0  sum(arg0_1[16*p0 + 128*p1 + p2] over p1,p2)
buf0: f32[8]  <-  arg0_1: f32[4,8,16]
    foreach p0 in [0,8), reduce p1 in [0,4), p2 in [0,16):
        t0 = arg0_1[16*p0 + 128*p1 + p2]
        r0 = sum(t0 over p1,p2)
        buf0[p0] = r0

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 8})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', 16*d0 + 128*d1 + d2, {d0: 8, d1: 4, d2: 16})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (8, 64)
op0.sizes = ([8], [4, 16])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 16], stride=[128, 16, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 8, p1: 4, p2: 16}
    index0 = 16*p0 + 128*p1 + p2
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[4,8,16]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[8]  <-  arg0_1: f32[4,8,16]      # sum(arg0_1[16*p0 + 128*p1 + p2] over p1,p2)
        foreach p0 in [0,8), reduce p1 in [0,4), p2 in [0,16):
            t0 = arg0_1[16*p0 + 128*p1 + p2]
            r0 = sum(t0 over p1,p2)
            buf0[p0] = r0

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 8})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', 16*c0 + 128*c1 + c2, {c0: 8, c1: 4, c2: 16})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (8, 64)
op0.sizes = ((8,), (4, 16))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 16], stride=[128, 16, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 8, p1: 4, p2: 16}
    index0 = 16*p0 + 128*p1 + p2
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[4,8,16]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    foreach x in [0,8), reduce r in [0,64):

        op0:  buf0: f32[8]  <-  arg0_1: f32[4,8,16] @ [x:8,r:64]      # sum(arg0_1[16*x + 128*((r//16)) + (ModularIndexing(r, 1, 16))] over (r//16),ModularIndexing(r, 1, 16))
            t0 = arg0_1[16*x + 128*((r//16)) + (ModularIndexing(r, 1, 16))]
            r0 = sum(t0 over (r//16),ModularIndexing(r, 1, 16))
            buf0[x] = r0

# outputs
return (buf0)
```

## Program 7: softmax(x, dim=-1) — reduction + pointwise chain

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 64]"):
        # No stacktrace found for following nodes
        prepare_softmax_online_default = torch.ops.prims.prepare_softmax_online.default(arg0_1, -1)

        # File: /tmp/ir_printer_examples.py:232 in prog_softmax, code: return torch.softmax(x, dim=-1)
        getitem: "f32[32, 1]" = prepare_softmax_online_default[0]
        getitem_1: "f32[32, 1]" = prepare_softmax_online_default[1];  prepare_softmax_online_default = None
        sub_tensor: "f32[32, 64]" = torch.ops.aten.sub.Tensor(arg0_1, getitem);  arg0_1 = getitem = None
        exp_default: "f32[32, 64]" = torch.ops.aten.exp.default(sub_tensor);  sub_tensor = None
        div: "f32[32, 64]" = torch.ops.aten.div.Tensor(exp_default, getitem_1);  exp_default = getitem_1 = None
        return (div,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32]), data=MultiOutputReduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, _ = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 64 * i0)
      return tmp0
  ,
  ranges=[32, 1],
  reduction_ranges=[64],
  reduction_type=online_softmax_reduce,
  origin_node=getitem,
  origins=OrderedSet([prepare_softmax_online_default])
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32]), data=MultiOutputReduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, _ = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 64 * i0)
      return tmp0
  ,
  ranges=[32, 1],
  reduction_ranges=[64],
  reduction_type=online_softmax_reduce,
  origin_node=getitem_1,
  origins=OrderedSet([prepare_softmax_online_default])
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf2', layout=FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1 = index
      tmp0 = ops.load(arg0_1, i1 + 64 * i0)
      tmp1 = ops.load(buf0, i0)
      tmp2 = tmp0 - tmp1
      tmp3 = ops.exp(tmp2)
      tmp4 = ops.load(buf1, i0)
      tmp5 = tmp3 / tmp4
      return tmp5
  ,
  ranges=[32, 64],
  origin_node=div,
  origins=OrderedSet([div, exp_default, sub_tensor]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 232, in prog_softmax,
      return torch.softmax(x, dim=-1),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[32,64]

# compute

# op0  online_softmax_reduce(arg0_1[64*p0 + p1] over p1).max
buf0: f32[32,1]  <-  arg0_1: f32[32,64]
    foreach p0 in [0,32), reduce p1 in [0,64):
        t0 = arg0_1[64*p0 + p1]
        r0 = online_softmax_reduce(t0 over p1)
        buf0[p0] = r0.max

# op1  online_softmax_reduce(arg0_1[64*p0 + p1] over p1).sum
buf1: f32[32,1]  <-  arg0_1: f32[32,64]
    foreach p0 in [0,32), reduce p1 in [0,64):
        t0 = arg0_1[64*p0 + p1]
        r0 = online_softmax_reduce(t0 over p1)
        buf1[p0] = r0.sum

# op2  exp((arg0_1[64*p0 + p1]) - buf0[p0]) / buf1[p0]
buf2: f32[32,64]  <-  arg0_1: f32[32,64], buf0: f32[32,1], buf1: f32[32,1]
    foreach p0 in [0,32), p1 in [0,64):
        ix0 = 64*p0 + p1
        t0 = arg0_1[ix0]
        t1 = buf0[p0]
        t2 = t0 - t1
        t3 = exp(t2)
        t4 = buf1[p0]
        t5 = t3 / t4
        buf2[ix0] = t5

# outputs
return (buf2)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 32})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', 64*d0 + d1, {d0: 32, d1: 64})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
    buf0.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (32, 64)
op0.sizes = ([32], [64])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
class op0_loop_body:
    var_ranges = {p0: 32, p1: 64}
    index0 = 64*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'online_softmax_reduce', load)
        getitem = reduction[0]
        getitem_1 = reduction[1]
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, getitem)
        return store_reduction


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', d0, {d0: 32})]
op1.unmet_dependencies = []
op1.met_dependencies = [MemoryDep('arg0_1', 64*d0 + d1, {d0: 32, d1: 64})]
op1.min_input_distance = 0
op1.max_input_distance = 0
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (32, 64)
op1.sizes = ([32], [64])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
class op1_loop_body:
    var_ranges = {p0: 32, p1: 64}
    index0 = 64*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'online_softmax_reduce', load)
        getitem = reduction[0]
        getitem_1 = reduction[1]
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf1', get_index_1, getitem_1)
        return store_reduction


op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', 64*d0 + d1, {d0: 32, d1: 64})]
op2.unmet_dependencies = [MemoryDep('buf0', d0, {d0: 32}), MemoryDep('buf1', d0, {d0: 32})]
op2.met_dependencies = [MemoryDep('arg0_1', 64*d0 + d1, {d0: 32, d1: 64})]
op2.min_input_distance = 1
op2.max_input_distance = 1
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op2.group.device = cuda:0
op2.group.iteration = (2048, 1)
op2.sizes = ([32, 64], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
class op2_loop_body:
    var_ranges = {p0: 32, p1: 64}
    index0 = 64*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        sub = ops.sub(load, load_1)
        exp = ops.exp(sub)
        get_index_2 = self.get_index('index1')
        load_2 = ops.load('buf1', get_index_2)
        truediv = ops.truediv(exp, load_2)
        get_index_3 = self.get_index('index0')
        store = ops.store('buf2', get_index_3, truediv, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[32,64]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[32,1]  <-  arg0_1: f32[32,64]      # online_softmax_reduce(arg0_1[64*p0 + p1] over p1).max
        foreach p0 in [0,32), reduce p1 in [0,64):
            t0 = arg0_1[64*p0 + p1]
            r0 = online_softmax_reduce(t0 over p1)
            buf0[p0] = r0.max

kernel k1
    inputs:   [arg0_1]
    outputs:  [buf1]

    op1:  buf1: f32[32,1]  <-  arg0_1: f32[32,64]      # online_softmax_reduce(arg0_1[64*p0 + p1] over p1).sum
        foreach p0 in [0,32), reduce p1 in [0,64):
            t0 = arg0_1[64*p0 + p1]
            r0 = online_softmax_reduce(t0 over p1)
            buf1[p0] = r0.sum

kernel k2
    inputs:   [arg0_1, buf0, buf1]
    outputs:  [buf2]

    op2:  buf2: f32[32,64]  <-  arg0_1: f32[32,64], buf0: f32[32,1], buf1: f32[32,1]      # exp((arg0_1[64*p0 + p1]) - buf0[p0]) / buf1[p0]
        foreach p0 in [0,32), p1 in [0,64):
            ix0 = 64*p0 + p1
            t0 = arg0_1[ix0]
            t1 = buf0[p0]
            t2 = t0 - t1
            t3 = exp(t2)
            t4 = buf1[p0]
            t5 = t3 / t4
            buf2[ix0] = t5

# outputs
return (buf2)
```

### post_fusion

```text
op0_op1_op2: FusedSchedulerNode(SchedulerNode,SchedulerNode,SchedulerNode)
op0_op1_op2.writes =
    [   MemoryDep('buf0', d0, {d0: 32}),
        MemoryDep('buf1', d0, {d0: 32}),
        MemoryDep('buf2', 64*d0 + d1, {d0: 32, d1: 64})]
op0_op1_op2.unmet_dependencies = []
op0_op1_op2.met_dependencies = [MemoryDep('arg0_1', 64*d0 + d1, {d0: 32, d1: 64})]
op0_op1_op2.min_input_distance = 0
op0_op1_op2.max_input_distance = 1
op0_op1_op2.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
    buf0.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0_op1_op2.snodes[0] =
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 32})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 2048})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
    buf0.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (32, 64)
op0.sizes = ((32,), (64,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
class op0_loop_body:
    var_ranges = {p0: 32, p1: 64}
    index0 = 64*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'online_softmax_reduce', load)
        getitem = reduction[0]
        getitem_1 = reduction[1]
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, getitem)
        return store_reduction
op0_op1_op2.snodes[1] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 32})]
op1.unmet_dependencies = []
op1.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 2048})]
op1.min_input_distance = 0
op1.max_input_distance = 0
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (32, 64)
op1.sizes = ((32,), (64,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
class op1_loop_body:
    var_ranges = {p0: 32, p1: 64}
    index0 = 64*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'online_softmax_reduce', load)
        getitem = reduction[0]
        getitem_1 = reduction[1]
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf1', get_index_1, getitem_1)
        return store_reduction
op0_op1_op2.snodes[2] =
op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', c0, {c0: 2048})]
op2.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 32}), MemoryDep('buf1', c0, {c0: 32})]
op2.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 2048})]
op2.min_input_distance = 1
op2.max_input_distance = 1
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op2.group.device = cuda:0
op2.group.iteration = (2048, 1)
op2.sizes = ((32, 64), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 1], stride=[1, 32])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
class op2_loop_body:
    var_ranges = {p0: 32, p1: 64}
    index0 = 64*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        sub = ops.sub(load, load_1)
        exp = ops.exp(sub)
        get_index_2 = self.get_index('index1')
        load_2 = ops.load('buf1', get_index_2)
        truediv = ops.truediv(exp, load_2)
        get_index_3 = self.get_index('index0')
        store = ops.store('buf2', get_index_3, truediv, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[32,64]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf2]
    internal: [buf0, buf1]

    foreach x in [0,32), reduce r in [0,64):

        op0, op1:  buf0: f32[32,1] @ [x:32], buf1: f32[32,1] @ [x:32]  <-  arg0_1: f32[32,64]      # online_softmax_reduce(arg0_1[r + 64*x] over r)
            t0 = arg0_1[r + 64*x]
            r0 = online_softmax_reduce(t0 over r)
            buf0[x] = r0.max      # op0
            buf1[x] = r0.sum      # op1

        op2:  buf2: f32[32,64]  <-  arg0_1: f32[32,64], buf0: f32[32,1] @ [x:32], buf1: f32[32,1] @ [x:32]      # exp((arg0_1[r + 64*x]) - buf0[x]) / buf1[x]
            ix0 = r + 64*x
            t0 = arg0_1[ix0]
            t1 = buf0[x]
            t2 = t0 - t1
            t3 = exp(t2)
            t4 = buf1[x]
            t5 = t3 / t4
            buf2[ix0] = t5

# outputs
return (buf2)
```

## Program 8: matmul — a @ b, produces a TemplateBuffer (or extern fallback)

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64, 128]", arg1_1: "f32[128, 64]"):
        # File: /tmp/ir_printer_examples.py:237 in prog_matmul, code: return a @ b
        mm: "f32[64, 64]" = torch.ops.aten.mm.default(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        return (mm,)
```

### post_lowering

```text
ExternKernelOut(
  python_kernel_name='extern_kernels.mm',
  name=buf0,
  layout=FixedLayout('cuda:0', torch.float32, size=[64, 64], stride=[64, 1]),
  inputs=[InputBuffer(name='arg0_1', layout=FixedLayout('cuda:0', torch.float32, size=[64, 128], stride=[128, 1])), InputBuffer(name='arg1_1', layout=FixedLayout('cuda:0', torch.float32, size=[128, 64], stride=[64, 1]))],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.mm,
  cpp_kernel_name=at::mm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.mm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=mm,
  origins=OrderedSet([mm]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 237, in prog_matmul,
      return a @ b,
  ,
  }
)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[64,128], arg1_1: f32[128,64]

# compute

# op0
buf0: f32[64,64]
    extern_kernels.mm(arg0_1, arg1_1)

# outputs
return (buf0)
```

### post_scheduler

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None), StarDep(name='arg1_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ExternKernelOut
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[64, 64], stride=[64, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.node.kernel = extern_kernels.mm
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[64,128], arg1_1: f32[128,64]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[64,64]
        extern_kernels.mm(arg0_1, arg1_1)

# outputs
return (buf0)
```

### post_fusion

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None), StarDep(name='arg1_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ExternKernelOut
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[64, 64], stride=[64, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.node.kernel = extern_kernels.mm
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[64,128], arg1_1: f32[128,64]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[64,64]
        extern_kernels.mm(arg0_1, arg1_1)

# outputs
return (buf0)
```

## Program 9: torch.addmm(bias, a, b) — template + prologue-fusable epilogue

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64]", arg1_1: "f32[32, 128]", arg2_1: "f32[128, 64]"):
        # File: /tmp/ir_printer_examples.py:242 in prog_addmm, code: return torch.addmm(bias, a, b)
        addmm: "f32[32, 64]" = torch.ops.aten.addmm.default(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        return (addmm,)
```

### post_lowering

```text
ExternKernelOut(
  python_kernel_name='extern_kernels.addmm',
  name=buf0,
  layout=FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1]),
  inputs=[InputBuffer(name='arg0_1', layout=FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])), InputBuffer(name='arg1_1', layout=FixedLayout('cuda:0', torch.float32, size=[32, 128], stride=[128, 1])), InputBuffer(name='arg2_1', layout=FixedLayout('cuda:0', torch.float32, size=[128, 64], stride=[64, 1]))],
  constant_args=(),
  kwargs={'alpha': 1, 'beta': 1},
  output_view=None,
  python_kernel_name=extern_kernels.addmm,
  cpp_kernel_name=at::addmm_out,
  ordered_kwargs_for_cpp_kernel=['beta', 'alpha', 'out'],
  op_overload=aten.addmm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat1', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat1': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'beta': {'type': number, 'default_value': 1}, 'alpha': {'type': number, 'default_value': 1}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=addmm,
  origins=OrderedSet([addmm]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 242, in prog_addmm,
      return torch.addmm(bias, a, b),
  ,
  }
)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[64], arg1_1: f32[32,128], arg2_1: f32[128,64]

# compute

# op0
buf0: f32[32,64]
    extern_kernels.addmm(arg0_1, arg1_1, arg2_1, alpha=1, beta=1)

# outputs
return (buf0)
```

### post_scheduler

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   StarDep(name='arg0_1', mode=None),
        StarDep(name='arg1_1', mode=None),
        StarDep(name='arg2_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ExternKernelOut
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.node.kernel = extern_kernels.addmm
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[64], arg1_1: f32[32,128], arg2_1: f32[128,64]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1, arg1_1, arg2_1]
    outputs:  [buf0]

    op0:  buf0: f32[32,64]
        extern_kernels.addmm(arg0_1, arg1_1, arg2_1, alpha=1, beta=1)

# outputs
return (buf0)
```

### post_fusion

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   StarDep(name='arg0_1', mode=None),
        StarDep(name='arg1_1', mode=None),
        StarDep(name='arg2_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ExternKernelOut
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.node.kernel = extern_kernels.addmm
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[64], arg1_1: f32[32,128], arg2_1: f32[128,64]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1, arg1_1, arg2_1]
    outputs:  [buf0]

    op0:  buf0: f32[32,64]
        extern_kernels.addmm(arg0_1, arg1_1, arg2_1, alpha=1, beta=1)

# outputs
return (buf0)
```

## Program 10: multiple outputs — (x * 2, x.sum())

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[32, 32]"):
        # File: /tmp/ir_printer_examples.py:247 in prog_multi_output, code: return x * 2, x.sum()
        mul: "f32[32, 32]" = torch.ops.aten.mul.Tensor(arg0_1, 2)
        sum_1: "f32[]" = torch.ops.aten.sum.default(arg0_1);  arg0_1 = None
        return (mul, sum_1)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[32, 32], stride=[32, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1 = index
      tmp0 = ops.load(arg0_1, i1 + 32 * i0)
      tmp1 = ops.constant(2, torch.float32)
      tmp2 = tmp0 * tmp1
      return tmp2
  ,
  ranges=[32, 32],
  origin_node=mul,
  origins=OrderedSet([mul]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 247, in prog_multi_output,
      return x * 2, x.sum(),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.float32, size=[], stride=[]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      r0_0, r0_1 = rindex
      tmp0 = ops.load(arg0_1, r0_1 + 32 * r0_0)
      return tmp0
  ,
  ranges=[],
  reduction_ranges=[32, 32],
  reduction_type=sum,
  origin_node=sum_1,
  origins=OrderedSet([sum_1]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 247, in prog_multi_output,
      return x * 2, x.sum(),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[32,32]

# compute

# op0  (arg0_1[32*p0 + p1]) * 2.0
buf0: f32[32,32]  <-  arg0_1: f32[32,32]
    foreach p0 in [0,32), p1 in [0,32):
        ix0 = 32*p0 + p1
        t0 = arg0_1[ix0]
        t1 = t0 * 2.0
        buf0[ix0] = t1

# op1  sum(arg0_1[32*p0 + p1] over p0,p1)
buf1: f32[]  <-  arg0_1: f32[32,32]
    reduce p0 in [0,32), p1 in [0,32):
        t0 = arg0_1[32*p0 + p1]
        r0 = sum(t0 over p0,p1)
        buf1[0] = r0

# outputs
return (buf0, buf1)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 32*d0 + d1, {d0: 32, d1: 32})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', 32*d0 + d1, {d0: 32, d1: 32})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32, 32], stride=[32, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (1024, 1)
op0.sizes = ([32, 32], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 32], stride=[32, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32, 32], stride=[32, 1])
class op0_loop_body:
    var_ranges = {p0: 32, p1: 32}
    index0 = 32*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(2.0, torch.float32)
        mul = ops.mul(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, mul, None)
        return store


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', 0, {})]
op1.unmet_dependencies = []
op1.met_dependencies = [MemoryDep('arg0_1', 32*d0 + d1, {d0: 32, d1: 32})]
op1.min_input_distance = 0
op1.max_input_distance = 0
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (1, 1024)
op1.sizes = ([], [32, 32])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 32], stride=[32, 1])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
class op1_loop_body:
    var_ranges = {p0: 32, p1: 32}
    index0 = 32*p0 + p1
    index1 = 0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf1', get_index_1, reduction)
        return None
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[32,32]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[32,32]  <-  arg0_1: f32[32,32]      # (arg0_1[32*p0 + p1]) * 2.0
        foreach p0 in [0,32), p1 in [0,32):
            ix0 = 32*p0 + p1
            t0 = arg0_1[ix0]
            t1 = t0 * 2.0
            buf0[ix0] = t1

kernel k1
    inputs:   [arg0_1]
    outputs:  [buf1]

    op1:  buf1: f32[]  <-  arg0_1: f32[32,32]      # sum(arg0_1[32*p0 + p1] over p0,p1)
        reduce p0 in [0,32), p1 in [0,32):
            t0 = arg0_1[32*p0 + p1]
            r0 = sum(t0 over p0,p1)
            buf1[0] = r0

# outputs
return (buf0, buf1)
```

### post_fusion

```text
op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
op0_op1.writes = [MemoryDep('buf0', 32*d0 + d1, {d0: 32, d1: 32}), MemoryDep('buf1', 0, {})]
op0_op1.unmet_dependencies = []
op0_op1.met_dependencies = [MemoryDep('arg0_1', 32*d0 + d1, {d0: 32, d1: 32})]
op0_op1.min_input_distance = 0
op0_op1.max_input_distance = 0
op0_op1.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32, 32], stride=[32, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0_op1.snodes[0] =
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 1024})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 1024})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32, 32], stride=[32, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (1024, 1)
op0.sizes = ((1024,), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 32], stride=[32, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32, 32], stride=[32, 1])
class op0_loop_body:
    var_ranges = {p0: 1024}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(2.0, torch.float32)
        mul = ops.mul(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, mul, None)
        return store
op0_op1.snodes[1] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', 0, {})]
op1.unmet_dependencies = []
op1.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 1024})]
op1.min_input_distance = 0
op1.max_input_distance = 0
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (1, 1024)
op1.sizes = ((), (1024,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[32, 32], stride=[32, 1])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[], stride=[])
class op1_loop_body:
    var_ranges = {p0: 1024}
    index0 = p0
    index1 = 0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf1', get_index_1, reduction)
        return None
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[32,32]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0, buf1]

    reduce r in [0,1024):

        op0:  buf0: f32[32,32] @ [r:1024]  <-  arg0_1: f32[32,32] @ [r:1024]      # arg0_1[r] * 2.0
            t0 = arg0_1[r]
            t1 = t0 * 2.0
            buf0[r] = t1

        op1:  buf1: f32[] @ [scalar]  <-  arg0_1: f32[32,32] @ [r:1024]      # sum(arg0_1[r] over r)
            t0 = arg0_1[r]
            r0 = sum(t0 over r)
            buf1[0] = r0

# outputs
return (buf0, buf1)
```

## Program 11: pure dtype cast — x.to(torch.float16)

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024]"):
        # File: /tmp/ir_printer_examples.py:252 in prog_dtype_cast, code: return x.to(torch.float16)
        convert_element_type: "f16[1024]" = torch.ops.prims.convert_element_type.default(arg0_1, torch.float16);  arg0_1 = None
        return (convert_element_type,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float16, size=[1024], stride=[1]), data=Pointwise(
  'cuda',
  torch.float16,
  def inner_fn(index):
      i0 = index
      tmp0 = ops.load(arg0_1, i0)
      tmp1 = ops.to_dtype(tmp0, torch.float16, src_dtype=torch.float32, use_compute_types=True)
      return tmp1
  ,
  ranges=[1024],
  origin_node=convert_element_type,
  origins=OrderedSet([convert_element_type]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 252, in prog_dtype_cast,
      return x.to(torch.float16),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[1024]

# compute

# op0  to_dtype(arg0_1[p0], torch.float16)
buf0: f16[1024]  <-  arg0_1: f32[1024]
    foreach p0 in [0,1024):
        t0 = arg0_1[p0]
        t1 = to_dtype(t0, f16)
        buf0[p0] = t1

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 1024})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', d0, {d0: 1024})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float16, size=[1024], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (1024, 1)
op0.sizes = ([1024], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float16, size=[1024], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 1024}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        to_dtype = ops.to_dtype(load, torch.float16, src_dtype = torch.float32, use_compute_types = True)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, to_dtype, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[1024]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f16[1024]  <-  arg0_1: f32[1024]      # to_dtype(arg0_1[p0], torch.float16)
        foreach p0 in [0,1024):
            t0 = arg0_1[p0]
            t1 = to_dtype(t0, f16)
            buf0[p0] = t1

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 1024})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 1024})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float16, size=[1024], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (1024, 1)
op0.sizes = ((1024,), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float16, size=[1024], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 1024}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        to_dtype = ops.to_dtype(load, torch.float16, src_dtype = torch.float32, use_compute_types = True)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, to_dtype, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[1024]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    foreach x in [0,1024):

        op0:  buf0: f16[1024]  <-  arg0_1: f32[1024]      # to_dtype(arg0_1[x], torch.float16)
            t0 = arg0_1[x]
            t1 = to_dtype(t0, f16)
            buf0[x] = t1

# outputs
return (buf0)
```

## Program 12: torch.cat([a, b], dim=0) — exercises ConcatKernel/aliases

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 8]", arg1_1: "f32[2, 8]"):
        # File: /tmp/ir_printer_examples.py:257 in prog_cat, code: return torch.cat([a, b], dim=0)
        cat: "f32[6, 8]" = torch.ops.aten.cat.default([arg0_1, arg1_1]);  arg0_1 = arg1_1 = None
        return (cat,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[6, 8], stride=[8, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1 = index
      tmp0 = ops.index_expr(i0, torch.int64)
      tmp1 = ops.constant(0, torch.int64)
      tmp2 = ops.index_expr(4, torch.int64)
      tmp3 = tmp0 >= tmp1
      tmp4 = tmp0 < tmp2
      tmp5 = ops.load(arg0_1, i1 + 8 * Identity(i0))
      tmp6 = ops.masked(tmp4, tmp5, 0.0)
      tmp7 = ops.index_expr(4, torch.int64)
      tmp8 = ops.index_expr(6, torch.int64)
      tmp9 = tmp0 >= tmp7
      tmp10 = tmp0 < tmp8
      tmp11 = ops.load(arg1_1, i1 + 8 * Identity(-4 + i0))
      tmp12 = ops.masked(tmp9, tmp11, 0.0)
      tmp13 = ops.where(tmp4, tmp6, tmp12)
      return tmp13
  ,
  ranges=[6, 8],
  origin_node=cat,
  origins=OrderedSet([cat]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 257, in prog_cat,
      return torch.cat([a, b], dim=0),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[4,8], arg1_1: f32[2,8]

# compute

# op0
buf0: f32[6,8]
    foreach p0 in [0,6), p1 in [0,8):
        t0 = index(p0)
        t1 = t0 >= 0
        t2 = index(p0)
        t3 = t2 < 4
        t4 = masked_subblock1(t3, 0.0)
        t5 = index(p0)
        t6 = t5 >= 4
        t7 = index(p0)
        t8 = t7 < 6
        t9 = masked_subblock2(t6, 0.0)
        t10 = where(t3, t4, t9)
        buf0[8*p0 + p1] = t10

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 8*d0 + d1, {d0: 6, d1: 8})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', d1 + 8*((d0)), {d0: 6, d1: 8}),
        MemoryDep('arg1_1', d1 + 8*((d0 - 4)), {d0: 6, d1: 8})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[6, 8], stride=[8, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (48, 1)
op0.sizes = ([6, 8], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8], stride=[8, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8], stride=[8, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[6, 8], stride=[8, 1])
class op0_loop_body:
    var_ranges = {p0: 6, p1: 8}
    index0 = p0
    index1 = p1 + 8*((p0))
    index2 = p1 + 8*((p0 - 4))
    index3 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        index_expr = ops.index_expr(get_index, torch.int64)
        constant = ops.constant(0, torch.int64)
        ge = ops.ge(index_expr, constant)
        get_index_1 = self.get_index('index0')
        index_expr_1 = ops.index_expr(get_index_1, torch.int64)
        constant_1 = ops.constant(4, torch.int64)
        lt = ops.lt(index_expr_1, constant_1)
        masked_subblock1 = self.masked_subblock1(lt, 0.0)
        get_index_2 = self.get_index('index0')
        index_expr_2 = ops.index_expr(get_index_2, torch.int64)
        constant_2 = ops.constant(4, torch.int64)
        ge_1 = ops.ge(index_expr_2, constant_2)
        get_index_3 = self.get_index('index0')
        index_expr_3 = ops.index_expr(get_index_3, torch.int64)
        constant_3 = ops.constant(6, torch.int64)
        lt_1 = ops.lt(index_expr_3, constant_3)
        masked_subblock2 = self.masked_subblock2(ge_1, 0.0)
        where = ops.where(lt, masked_subblock1, masked_subblock2)
        get_index_4 = self.get_index('index3')
        store = ops.store('buf0', get_index_4, where, None)
        return store
    def masked_subblock1(self, ops):
        get_index = self.get_index('index1')
        load = ops.load('arg0_1', get_index)
        return load
    def masked_subblock2(self, ops):
        get_index = self.get_index('index2')
        load = ops.load('arg1_1', get_index)
        return load
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[4,8], arg1_1: f32[2,8]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[6,8]
        foreach p0 in [0,6), p1 in [0,8):
            t0 = index(p0)
            t1 = t0 >= 0
            t2 = index(p0)
            t3 = t2 < 4
            t4 = masked_subblock1(t3, 0.0)
            t5 = index(p0)
            t6 = t5 >= 4
            t7 = index(p0)
            t8 = t7 < 6
            t9 = masked_subblock2(t6, 0.0)
            t10 = where(t3, t4, t9)
            buf0[8*p0 + p1] = t10

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 48})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', c1 + 8*((c0)), {c0: 6, c1: 8}),
        MemoryDep('arg1_1', c1 + 8*((c0 - 4)), {c0: 6, c1: 8})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[6, 8], stride=[8, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (48, 1)
op0.sizes = ((6, 8), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8], stride=[8, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8], stride=[8, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[6, 8], stride=[8, 1])
class op0_loop_body:
    var_ranges = {p0: 6, p1: 8}
    index0 = p0
    index1 = p1 + 8*((p0))
    index2 = p1 + 8*((p0 - 4))
    index3 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        index_expr = ops.index_expr(get_index, torch.int64)
        constant = ops.constant(0, torch.int64)
        ge = ops.ge(index_expr, constant)
        get_index_1 = self.get_index('index0')
        index_expr_1 = ops.index_expr(get_index_1, torch.int64)
        constant_1 = ops.constant(4, torch.int64)
        lt = ops.lt(index_expr_1, constant_1)
        masked_subblock1 = self.masked_subblock1(lt, 0.0)
        get_index_2 = self.get_index('index0')
        index_expr_2 = ops.index_expr(get_index_2, torch.int64)
        constant_2 = ops.constant(4, torch.int64)
        ge_1 = ops.ge(index_expr_2, constant_2)
        get_index_3 = self.get_index('index0')
        index_expr_3 = ops.index_expr(get_index_3, torch.int64)
        constant_3 = ops.constant(6, torch.int64)
        lt_1 = ops.lt(index_expr_3, constant_3)
        masked_subblock2 = self.masked_subblock2(ge_1, 0.0)
        where = ops.where(lt, masked_subblock1, masked_subblock2)
        get_index_4 = self.get_index('index3')
        store = ops.store('buf0', get_index_4, where, None)
        return store
    def masked_subblock1(self, ops):
        get_index = self.get_index('index1')
        load = ops.load('arg0_1', get_index)
        return load
    def masked_subblock2(self, ops):
        get_index = self.get_index('index2')
        load = ops.load('arg1_1', get_index)
        return load
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[4,8], arg1_1: f32[2,8]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    foreach x in [0,48):

        op0:  buf0: f32[6,8] @ [x:48]
            ix0 = (x//8)
            t0 = index(ix0)
            t1 = t0 >= 0
            t2 = index(ix0)
            t3 = t2 < 4
            t4 = masked_subblock1(t3, 0.0)
            t5 = index(ix0)
            t6 = t5 >= 4
            t7 = index(ix0)
            t8 = t7 < 6
            t9 = masked_subblock2(t6, 0.0)
            t10 = where(t3, t4, t9)
            buf0[8*((x//8)) + (ModularIndexing(x, 1, 8))] = t10

# outputs
return (buf0)
```

## Program 13: torch.stack([a, b], dim=0) — cat + unsqueeze

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[8]", arg1_1: "f32[8]"):
        # File: /tmp/ir_printer_examples.py:262 in prog_stack, code: return torch.stack([a, b], dim=0)
        cat: "f32[16]" = torch.ops.aten.cat.default([arg0_1, arg1_1]);  arg0_1 = arg1_1 = None
        view: "f32[2, 8]" = torch.ops.aten.reshape.default(cat, [2, 8]);  cat = None
        return (view,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[16], stride=[1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0 = index
      tmp0 = ops.index_expr(i0, torch.int64)
      tmp1 = ops.constant(0, torch.int64)
      tmp2 = ops.index_expr(8, torch.int64)
      tmp3 = tmp0 >= tmp1
      tmp4 = tmp0 < tmp2
      tmp5 = ops.load(arg0_1, Identity(i0))
      tmp6 = ops.masked(tmp4, tmp5, 0.0)
      tmp7 = ops.index_expr(8, torch.int64)
      tmp8 = ops.index_expr(16, torch.int64)
      tmp9 = tmp0 >= tmp7
      tmp10 = tmp0 < tmp8
      tmp11 = ops.load(arg1_1, Identity(-8 + i0))
      tmp12 = ops.masked(tmp9, tmp11, 0.0)
      tmp13 = ops.where(tmp4, tmp6, tmp12)
      return tmp13
  ,
  ranges=[16],
  origin_node=cat,
  origins=OrderedSet([cat]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 262, in prog_stack,
      return torch.stack([a, b], dim=0),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[8], arg1_1: f32[8]

# compute

# op0
buf0: f32[16]
    foreach p0 in [0,16):
        t0 = index(p0)
        t1 = t0 >= 0
        t2 = index(p0)
        t3 = t2 < 8
        t4 = masked_subblock1(t3, 0.0)
        t5 = index(p0)
        t6 = t5 >= 8
        t7 = index(p0)
        t8 = t7 < 16
        t9 = masked_subblock2(t6, 0.0)
        t10 = where(t3, t4, t9)
        buf0[p0] = t10

# outputs
return (reinterpret_tensor(buf0, [2,8], [8,1], 0))
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 16})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', (d0), {d0: 16}), MemoryDep('arg1_1', (d0 - 8), {d0: 16})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (16, 1)
op0.sizes = ([16], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 16}
    index0 = p0
    index1 = (p0)
    index2 = (p0 - 8)
    def body(self, ops):
        get_index = self.get_index('index0')
        index_expr = ops.index_expr(get_index, torch.int64)
        constant = ops.constant(0, torch.int64)
        ge = ops.ge(index_expr, constant)
        get_index_1 = self.get_index('index0')
        index_expr_1 = ops.index_expr(get_index_1, torch.int64)
        constant_1 = ops.constant(8, torch.int64)
        lt = ops.lt(index_expr_1, constant_1)
        masked_subblock1 = self.masked_subblock1(lt, 0.0)
        get_index_2 = self.get_index('index0')
        index_expr_2 = ops.index_expr(get_index_2, torch.int64)
        constant_2 = ops.constant(8, torch.int64)
        ge_1 = ops.ge(index_expr_2, constant_2)
        get_index_3 = self.get_index('index0')
        index_expr_3 = ops.index_expr(get_index_3, torch.int64)
        constant_3 = ops.constant(16, torch.int64)
        lt_1 = ops.lt(index_expr_3, constant_3)
        masked_subblock2 = self.masked_subblock2(ge_1, 0.0)
        where = ops.where(lt, masked_subblock1, masked_subblock2)
        get_index_4 = self.get_index('index0')
        store = ops.store('buf0', get_index_4, where, None)
        return store
    def masked_subblock1(self, ops):
        get_index = self.get_index('index1')
        load = ops.load('arg0_1', get_index)
        return load
    def masked_subblock2(self, ops):
        get_index = self.get_index('index2')
        load = ops.load('arg1_1', get_index)
        return load
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[8], arg1_1: f32[8]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    internal: [buf0]

    op0:  buf0: f32[16]
        foreach p0 in [0,16):
            t0 = index(p0)
            t1 = t0 >= 0
            t2 = index(p0)
            t3 = t2 < 8
            t4 = masked_subblock1(t3, 0.0)
            t5 = index(p0)
            t6 = t5 >= 8
            t7 = index(p0)
            t8 = t7 < 16
            t9 = masked_subblock2(t6, 0.0)
            t10 = where(t3, t4, t9)
            buf0[p0] = t10

# outputs
return (reinterpret_tensor(buf0, [2,8], [8,1], 0))
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 16})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', (c0), {c0: 16}), MemoryDep('arg1_1', (c0 - 8), {c0: 16})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (16, 1)
op0.sizes = ((16,), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[8], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 16}
    index0 = p0
    index1 = (p0)
    index2 = (p0 - 8)
    def body(self, ops):
        get_index = self.get_index('index0')
        index_expr = ops.index_expr(get_index, torch.int64)
        constant = ops.constant(0, torch.int64)
        ge = ops.ge(index_expr, constant)
        get_index_1 = self.get_index('index0')
        index_expr_1 = ops.index_expr(get_index_1, torch.int64)
        constant_1 = ops.constant(8, torch.int64)
        lt = ops.lt(index_expr_1, constant_1)
        masked_subblock1 = self.masked_subblock1(lt, 0.0)
        get_index_2 = self.get_index('index0')
        index_expr_2 = ops.index_expr(get_index_2, torch.int64)
        constant_2 = ops.constant(8, torch.int64)
        ge_1 = ops.ge(index_expr_2, constant_2)
        get_index_3 = self.get_index('index0')
        index_expr_3 = ops.index_expr(get_index_3, torch.int64)
        constant_3 = ops.constant(16, torch.int64)
        lt_1 = ops.lt(index_expr_3, constant_3)
        masked_subblock2 = self.masked_subblock2(ge_1, 0.0)
        where = ops.where(lt, masked_subblock1, masked_subblock2)
        get_index_4 = self.get_index('index0')
        store = ops.store('buf0', get_index_4, where, None)
        return store
    def masked_subblock1(self, ops):
        get_index = self.get_index('index1')
        load = ops.load('arg0_1', get_index)
        return load
    def masked_subblock2(self, ops):
        get_index = self.get_index('index2')
        load = ops.load('arg1_1', get_index)
        return load
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[8], arg1_1: f32[8]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    internal: [buf0]

    foreach x in [0,16):

        op0:  buf0: f32[16]
            t0 = index(x)
            t1 = t0 >= 0
            t2 = index(x)
            t3 = t2 < 8
            t4 = masked_subblock1(t3, 0.0)
            t5 = index(x)
            t6 = t5 >= 8
            t7 = index(x)
            t8 = t7 < 16
            t9 = masked_subblock2(t6, 0.0)
            t10 = where(t3, t4, t9)
            buf0[x] = t10

# outputs
return (reinterpret_tensor(buf0, [2,8], [8,1], 0))
```

## Program 14: indirect indexing — x[idx], data-dependent load

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[100, 8]", arg1_1: "i64[16]"):
        # File: /tmp/ir_printer_examples.py:267 in prog_indirect_index, code: return x[idx]
        index: "f32[16, 8]" = torch.ops.aten.index.Tensor(arg0_1, [arg1_1]);  arg0_1 = arg1_1 = None
        return (index,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[16, 8], stride=[8, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1 = index
      tmp0 = ops.load(arg1_1, i0)
      tmp1 = ops.load(arg0_1, i1 + 8 * tmp0)
      return tmp1
  ,
  ranges=[16, 8],
  origin_node=index,
  origins=OrderedSet([index]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 267, in prog_indirect_index,
      return x[idx],
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[100,8], arg1_1: i64[16]

# compute

# op0  indirect arg0_1[8*i0 + p1]
buf0: f32[16,8]  <-  arg1_1: i64[16], arg0_1: f32[100,8]
    foreach p0 in [0,16), p1 in [0,8):
        t0 = arg1_1[p0]
        i0 = wrap_index(t0, 100)
        t1 = arg0_1[8*i0 + p1]
        buf0[8*p0 + p1] = t1

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 8*d0 + d1, {d0: 16, d1: 8})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', d1 + 8*tmp0, {d0: 16, d1: 8}),
        MemoryDep('arg1_1', d0, {d0: 16})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[16, 8], stride=[8, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (128, 1)
op0.sizes = ([16, 8], [])
arg1_1_layout = FixedLayout('cuda:0', torch.int64, size=[16], stride=[1])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[100, 8], stride=[8, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[16, 8], stride=[8, 1])
class op0_loop_body:
    var_ranges = {p0: 16, p1: 8}
    index0 = p0
    index1 = 8*indirect0 + p1
    index2 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg1_1', get_index)
        set_indirect0 = self.set_indirect0(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg0_1', get_index_1)
        get_index_2 = self.get_index('index2')
        store = ops.store('buf0', get_index_2, load_1, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[100,8], arg1_1: i64[16]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[16,8]  <-  arg1_1: i64[16], arg0_1: f32[100,8]      # indirect arg0_1[8*i0 + p1]
        foreach p0 in [0,16), p1 in [0,8):
            t0 = arg1_1[p0]
            i0 = wrap_index(t0, 100)
            t1 = arg0_1[8*i0 + p1]
            buf0[8*p0 + p1] = t1

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 128})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', c1 + 8*tmp0, {c0: 16, c1: 8}),
        MemoryDep('arg1_1', c0, {c0: 16})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[16, 8], stride=[8, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (128, 1)
op0.sizes = ((16, 8), ())
arg1_1_layout = FixedLayout('cuda:0', torch.int64, size=[16], stride=[1])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[100, 8], stride=[8, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[16, 8], stride=[8, 1])
class op0_loop_body:
    var_ranges = {p0: 16, p1: 8}
    index0 = p0
    index1 = 8*indirect0 + p1
    index2 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg1_1', get_index)
        set_indirect0 = self.set_indirect0(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg0_1', get_index_1)
        get_index_2 = self.get_index('index2')
        store = ops.store('buf0', get_index_2, load_1, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[100,8], arg1_1: i64[16]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    foreach x in [0,128):

        op0:  buf0: f32[16,8] @ [x:128]  <-  arg1_1: i64[16] @ [x:128], arg0_1: f32[100,8]      # indirect arg0_1[8*i0 + (ModularIndexing(x, 1, 8))]
            t0 = arg1_1[(x//8)]
            i0 = wrap_index(t0, 100)
            t1 = arg0_1[8*i0 + (ModularIndexing(x, 1, 8))]
            buf0[8*((x//8)) + (ModularIndexing(x, 1, 8))] = t1

# outputs
return (buf0)
```

## Program 15: embedding lookup — F.embedding(idx, weight)

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "i64[4, 8]", arg1_1: "f32[50, 16]"):
        # File: /tmp/ir_printer_examples.py:272 in prog_embedding, code: return torch.nn.functional.embedding(idx, weight)
        embedding: "f32[4, 8, 16]" = torch.ops.aten.embedding.default(arg1_1, arg0_1);  arg1_1 = arg0_1 = None
        return (embedding,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[4, 8, 16], stride=[128, 16, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1, i2 = index
      tmp0 = ops.load(arg0_1, i1 + 8 * i0)
      tmp1 = ops.load(arg1_1, i2 + 16 * tmp0)
      return tmp1
  ,
  ranges=[4, 8, 16],
  origin_node=embedding,
  origins=OrderedSet([embedding]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 272, in prog_embedding,
      return torch.nn.functional.embedding(idx, weight),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: i64[4,8], arg1_1: f32[50,16]

# compute

# op0  indirect arg1_1[16*i0 + p2]
buf0: f32[4,8,16]  <-  arg0_1: i64[4,8], arg1_1: f32[50,16]
    foreach p0 in [0,4), p1 in [0,8), p2 in [0,16):
        t0 = arg0_1[8*p0 + p1]
        i0 = index(t0, 50)
        t1 = arg1_1[16*i0 + p2]
        buf0[128*p0 + 16*p1 + p2] = t1

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 128*d0 + 16*d1 + d2, {d0: 4, d1: 8, d2: 16})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', 8*d0 + d1, {d0: 4, d1: 8}),
        MemoryDep('arg1_1', d2 + 16*tmp0, {d0: 4, d1: 8, d2: 16})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 16], stride=[128, 16, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (512, 1)
op0.sizes = ([4, 8, 16], [])
arg0_1_layout = FixedLayout('cuda:0', torch.int64, size=[4, 8], stride=[8, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[50, 16], stride=[16, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 16], stride=[128, 16, 1])
class op0_loop_body:
    var_ranges = {p0: 4, p1: 8, p2: 16}
    index0 = 8*p0 + p1
    index1 = 16*indirect0 + p2
    index2 = 128*p0 + 16*p1 + p2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        set_indirect0 = self.set_indirect0(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        get_index_2 = self.get_index('index2')
        store = ops.store('buf0', get_index_2, load_1, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: i64[4,8], arg1_1: f32[50,16]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[4,8,16]  <-  arg0_1: i64[4,8], arg1_1: f32[50,16]      # indirect arg1_1[16*i0 + p2]
        foreach p0 in [0,4), p1 in [0,8), p2 in [0,16):
            t0 = arg0_1[8*p0 + p1]
            i0 = index(t0, 50)
            t1 = arg1_1[16*i0 + p2]
            buf0[128*p0 + 16*p1 + p2] = t1

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 512})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', c0, {c0: 32}),
        MemoryDep('arg1_1', c1 + 16*tmp0, {c0: 32, c1: 16})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 16], stride=[128, 16, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (512, 1)
op0.sizes = ((32, 16), ())
arg0_1_layout = FixedLayout('cuda:0', torch.int64, size=[4, 8], stride=[8, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[50, 16], stride=[16, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 16], stride=[128, 16, 1])
class op0_loop_body:
    var_ranges = {p0: 32, p1: 16}
    index0 = p0
    index1 = 16*indirect0 + p1
    index2 = 16*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        set_indirect0 = self.set_indirect0(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        get_index_2 = self.get_index('index2')
        store = ops.store('buf0', get_index_2, load_1, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: i64[4,8], arg1_1: f32[50,16]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    foreach x in [0,512):

        op0:  buf0: f32[4,8,16] @ [x:512]  <-  arg0_1: i64[4,8] @ [x:512], arg1_1: f32[50,16]      # indirect arg1_1[16*i0 + (ModularIndexing(x, 1, 16))]
            t0 = arg0_1[(x//16)]
            i0 = index(t0, 50)
            t1 = arg1_1[16*i0 + (ModularIndexing(x, 1, 16))]
            buf0[16*((x//16)) + (ModularIndexing(x, 1, 16))] = t1

# outputs
return (buf0)
```

## Program 16: torch.where(cond, x, y) — masked select-like pointwise

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "b8[64]", arg1_1: "f32[64]", arg2_1: "f32[64]"):
        # File: /tmp/ir_printer_examples.py:277 in prog_where, code: return torch.where(cond, x, y)
        where: "f32[64]" = torch.ops.aten.where.self(arg0_1, arg1_1, arg2_1);  arg0_1 = arg1_1 = arg2_1 = None
        return (where,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[64], stride=[1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0 = index
      tmp0 = ops.load(arg0_1, i0)
      tmp1 = ops.load(arg1_1, i0)
      tmp2 = ops.load(arg2_1, i0)
      tmp3 = ops.where(tmp0, tmp1, tmp2)
      return tmp3
  ,
  ranges=[64],
  origin_node=where,
  origins=OrderedSet([where]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 277, in prog_where,
      return torch.where(cond, x, y),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: b8[64], arg1_1: f32[64], arg2_1: f32[64]

# compute

# op0  where(arg0_1[p0], arg1_1[p0], arg2_1[p0])
buf0: f32[64]  <-  arg0_1: b8[64], arg1_1: f32[64], arg2_1: f32[64]
    foreach p0 in [0,64):
        t0 = arg0_1[p0]
        t1 = arg1_1[p0]
        t2 = arg2_1[p0]
        t3 = where(t0, t1, t2)
        buf0[p0] = t3

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 64})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', d0, {d0: 64}),
        MemoryDep('arg1_1', d0, {d0: 64}),
        MemoryDep('arg2_1', d0, {d0: 64})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (64, 1)
op0.sizes = ([64], [])
arg0_1_layout = FixedLayout('cuda:0', torch.bool, size=[64], stride=[1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
arg2_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 64}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg1_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('arg2_1', get_index_2)
        where = ops.where(load, load_1, load_2)
        get_index_3 = self.get_index('index0')
        store = ops.store('buf0', get_index_3, where, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: b8[64], arg1_1: f32[64], arg2_1: f32[64]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1, arg2_1]
    outputs:  [buf0]

    op0:  buf0: f32[64]  <-  arg0_1: b8[64], arg1_1: f32[64], arg2_1: f32[64]      # where(arg0_1[p0], arg1_1[p0], arg2_1[p0])
        foreach p0 in [0,64):
            t0 = arg0_1[p0]
            t1 = arg1_1[p0]
            t2 = arg2_1[p0]
            t3 = where(t0, t1, t2)
            buf0[p0] = t3

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 64})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', c0, {c0: 64}),
        MemoryDep('arg1_1', c0, {c0: 64}),
        MemoryDep('arg2_1', c0, {c0: 64})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (64, 1)
op0.sizes = ((64,), ())
arg0_1_layout = FixedLayout('cuda:0', torch.bool, size=[64], stride=[1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
arg2_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 64}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg1_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('arg2_1', get_index_2)
        where = ops.where(load, load_1, load_2)
        get_index_3 = self.get_index('index0')
        store = ops.store('buf0', get_index_3, where, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: b8[64], arg1_1: f32[64], arg2_1: f32[64]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1, arg2_1]
    outputs:  [buf0]

    foreach x in [0,64):

        op0:  buf0: f32[64]  <-  arg0_1: b8[64], arg1_1: f32[64], arg2_1: f32[64]      # where(arg0_1[x], arg1_1[x], arg2_1[x])
            t0 = arg0_1[x]
            t1 = arg1_1[x]
            t2 = arg2_1[x]
            t3 = where(t0, t1, t2)
            buf0[x] = t3

# outputs
return (buf0)
```

## Program 17: view + pointwise — x.view(-1) + 1, exercises ReinterpretView

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 8]"):
        # File: /tmp/ir_printer_examples.py:282 in prog_view_then_pointwise, code: return x.view(-1) + 1
        view: "f32[32]" = torch.ops.aten.reshape.default(arg0_1, [-1]);  arg0_1 = None
        add: "f32[32]" = torch.ops.aten.add.Tensor(view, 1);  view = None
        return (add,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[32], stride=[1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0 = index
      tmp0 = ops.load(arg0_1, i0)
      tmp1 = ops.constant(1, torch.float32)
      tmp2 = tmp0 + tmp1
      return tmp2
  ,
  ranges=[32],
  origin_node=add,
  origins=OrderedSet([add, view]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 282, in prog_view_then_pointwise,
      return x.view(-1) + 1,
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[4,8]

# compute

# op0  arg0_1[p0] + 1.0
buf0: f32[32]  <-  arg0_1: f32[4,8]
    foreach p0 in [0,32):
        t0 = arg0_1[p0]
        t1 = t0 + 1.0
        buf0[p0] = t1

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 32})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', d0, {d0: 32})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (32, 1)
op0.sizes = ([32], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8], stride=[8, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 32}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(1.0, torch.float32)
        add = ops.add(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, add, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[4,8]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[32]  <-  arg0_1: f32[4,8]      # arg0_1[p0] + 1.0
        foreach p0 in [0,32):
            t0 = arg0_1[p0]
            t1 = t0 + 1.0
            buf0[p0] = t1

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 32})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 32})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (32, 1)
op0.sizes = ((32,), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8], stride=[8, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 32}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(1.0, torch.float32)
        add = ops.add(load, constant)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, add, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[4,8]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    foreach x in [0,32):

        op0:  buf0: f32[32]  <-  arg0_1: f32[4,8] @ [x:32]      # arg0_1[x] + 1.0
            t0 = arg0_1[x]
            t1 = t0 + 1.0
            buf0[x] = t1

# outputs
return (buf0)
```

## Program 18: permute + add — x.permute(1, 0) + y, shows non-contiguous strides

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 8]", arg1_1: "f32[8, 4]"):
        # File: /tmp/ir_printer_examples.py:287 in prog_permute_then_add, code: return x.permute(1, 0) + y
        permute: "f32[8, 4]" = torch.ops.aten.permute.default(arg0_1, [1, 0]);  arg0_1 = None
        add: "f32[8, 4]" = torch.ops.aten.add.Tensor(permute, arg1_1);  permute = arg1_1 = None
        return (add,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[8, 4], stride=[1, 8]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1 = index
      tmp0 = ops.load(arg0_1, i0 + 8 * i1)
      tmp1 = ops.load(arg1_1, i1 + 4 * i0)
      tmp2 = tmp0 + tmp1
      return tmp2
  ,
  ranges=[8, 4],
  origin_node=add,
  origins=OrderedSet([add, permute]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 287, in prog_permute_then_add,
      return x.permute(1, 0) + y,
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[4,8], arg1_1: f32[8,4]

# compute

# op0  (arg0_1[p0 + 8*p1]) + (arg1_1[4*p0 + p1])
buf0: f32[8,4] stride=[1,8]  <-  arg0_1: f32[4,8], arg1_1: f32[8,4]
    foreach p0 in [0,8), p1 in [0,4):
        ix0 = p0 + 8*p1
        t0 = arg0_1[ix0]
        t1 = arg1_1[4*p0 + p1]
        t2 = t0 + t1
        buf0[ix0] = t2

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 8*d0 + d1, {d0: 4, d1: 8})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', 8*d0 + d1, {d0: 4, d1: 8}),
        MemoryDep('arg1_1', d0 + 4*d1, {d0: 4, d1: 8})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[8, 4], stride=[1, 8])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (32, 1)
op0.sizes = ([4, 8], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8], stride=[8, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[8, 4], stride=[4, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[8, 4], stride=[1, 8])
class op0_loop_body:
    var_ranges = {p0: 4, p1: 8}
    index0 = 8*p0 + p1
    index1 = p0 + 4*p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf0', get_index_2, add, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[4,8], arg1_1: f32[8,4]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[8,4] stride=[1,8]  <-  arg0_1: f32[4,8], arg1_1: f32[8,4]      # (arg0_1[8*p0 + p1]) + (arg1_1[p0 + 4*p1])
        foreach p0 in [0,4), p1 in [0,8):
            ix0 = 8*p0 + p1
            t0 = arg0_1[ix0]
            t1 = arg1_1[p0 + 4*p1]
            t2 = t0 + t1
            buf0[ix0] = t2

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 32})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', c0, {c0: 32}),
        MemoryDep('arg1_1', c0 + 4*c1, {c0: 4, c1: 8})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[8, 4], stride=[1, 8])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (32, 1)
op0.sizes = ((4, 8), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8], stride=[8, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[8, 4], stride=[4, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[8, 4], stride=[1, 8])
class op0_loop_body:
    var_ranges = {p0: 4, p1: 8}
    index0 = 8*p0 + p1
    index1 = p0 + 4*p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf0', get_index_2, add, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[4,8], arg1_1: f32[8,4]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    foreach y in [0,4), foreach x in [0,8):

        op0:  buf0: f32[8,4] stride=[1,8] @ [y:4,x:8]  <-  arg0_1: f32[4,8], arg1_1: f32[8,4] @ [y:4,x:8]      # (arg0_1[x + 8*y]) + (arg1_1[4*x + y])
            ix0 = x + 8*y
            t0 = arg0_1[ix0]
            t1 = arg1_1[4*x + y]
            t2 = t0 + t1
            buf0[ix0] = t2

# outputs
return (buf0)
```

## Program 19: RMSNorm — x / sqrt(mean(x**2) + eps), fuses reduction into pointwise

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 128]"):
        # File: /tmp/ir_printer_examples.py:292 in prog_rmsnorm, code: return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-5)
        mul: "f32[4, 128]" = torch.ops.aten.mul.Tensor(arg0_1, arg0_1)
        mean: "f32[4, 1]" = torch.ops.aten.mean.dim(mul, [-1], True);  mul = None
        add: "f32[4, 1]" = torch.ops.aten.add.Tensor(mean, 1e-05);  mean = None
        rsqrt: "f32[4, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul_1: "f32[4, 128]" = torch.ops.aten.mul.Tensor(arg0_1, rsqrt);  arg0_1 = rsqrt = None
        return (mul_1,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, _ = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 128 * i0)
      tmp1 = ops.load(arg0_1, r0_0 + 128 * i0)
      tmp2 = tmp0 * tmp1
      return tmp2
  ,
  ranges=[4, 1],
  reduction_ranges=[128],
  reduction_type=sum,
  origin_node=None,
  origins=OrderedSet([mean, mul]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 292, in prog_rmsnorm,
      return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-5),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1 = index
      tmp0 = ops.load(arg0_1, i1 + 128 * i0)
      tmp1 = ops.load(buf0, i0)
      tmp2 = ops.index_expr(128, torch.float32)
      tmp3 = tmp1 / tmp2
      tmp4 = ops.constant(1e-05, torch.float32)
      tmp5 = tmp3 + tmp4
      tmp6 = ops.rsqrt(tmp5)
      tmp7 = tmp0 * tmp6
      return tmp7
  ,
  ranges=[4, 128],
  origin_node=mul_1,
  origins=OrderedSet([mul_1, rsqrt, add, mean, mul]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 292, in prog_rmsnorm,
      return x * torch.rsqrt((x * x).mean(dim=-1, keepdim=True) + 1e-5),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[4,128]

# compute

# op0  sum((arg0_1[128*p0 + p1]) * (arg0_1[128*p0 + p1]) over p1)
buf0: f32[4,1]  <-  arg0_1: f32[4,128]
    foreach p0 in [0,4), reduce p1 in [0,128):
        ix0 = 128*p0 + p1
        t0 = arg0_1[ix0]
        t1 = arg0_1[ix0]
        t2 = t0 * t1
        r0 = sum(t2 over p1)
        buf0[p0] = r0

# op1  (arg0_1[128*p0 + p1]) * rsqrt((buf0[p0] / 128.0) + 1e-05)
buf1: f32[4,128]  <-  arg0_1: f32[4,128], buf0: f32[4,1]
    foreach p0 in [0,4), p1 in [0,128):
        ix0 = 128*p0 + p1
        t0 = arg0_1[ix0]
        t1 = buf0[p0]
        t2 = t1 / 128.0
        t3 = t2 + 1e-05
        t4 = rsqrt(t3)
        t5 = t0 * t4
        buf1[ix0] = t5

# outputs
return (buf1)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 4})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', 128*d0 + d1, {d0: 4, d1: 128})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (4, 128)
op0.sizes = ([4], [128])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
class op0_loop_body:
    var_ranges = {p0: 4, p1: 128}
    index0 = 128*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg0_1', get_index_1)
        mul = ops.mul(load, load_1)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', mul)
        get_index_2 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_2, reduction)
        return None


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', 128*d0 + d1, {d0: 4, d1: 128})]
op1.unmet_dependencies = [MemoryDep('buf0', d0, {d0: 4})]
op1.met_dependencies = [MemoryDep('arg0_1', 128*d0 + d1, {d0: 4, d1: 128})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (512, 1)
op1.sizes = ([4, 128], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
class op1_loop_body:
    var_ranges = {p0: 4, p1: 128}
    index0 = 128*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(128.0, torch.float32)
        truediv = ops.truediv(load_1, constant)
        constant_1 = ops.constant(1e-05, torch.float32)
        add = ops.add(truediv, constant_1)
        rsqrt = ops.rsqrt(add)
        mul = ops.mul(load, rsqrt)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf1', get_index_2, mul, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[4,128]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[4,1]  <-  arg0_1: f32[4,128]      # sum((arg0_1[128*p0 + p1]) * (arg0_1[128*p0 + p1]) over p1)
        foreach p0 in [0,4), reduce p1 in [0,128):
            ix0 = 128*p0 + p1
            t0 = arg0_1[ix0]
            t1 = arg0_1[ix0]
            t2 = t0 * t1
            r0 = sum(t2 over p1)
            buf0[p0] = r0

kernel k1
    inputs:   [arg0_1, buf0]
    outputs:  [buf1]

    op1:  buf1: f32[4,128]  <-  arg0_1: f32[4,128], buf0: f32[4,1]      # (arg0_1[128*p0 + p1]) * rsqrt((buf0[p0] / 128.0) + 1e-05)
        foreach p0 in [0,4), p1 in [0,128):
            ix0 = 128*p0 + p1
            t0 = arg0_1[ix0]
            t1 = buf0[p0]
            t2 = t1 / 128.0
            t3 = t2 + 1e-05
            t4 = rsqrt(t3)
            t5 = t0 * t4
            buf1[ix0] = t5

# outputs
return (buf1)
```

### post_fusion

```text
op0_op1: FusedSchedulerNode(SchedulerNode,SchedulerNode)
op0_op1.writes =
    [   MemoryDep('buf0', d0, {d0: 4}),
        MemoryDep('buf1', 128*d0 + d1, {d0: 4, d1: 128})]
op0_op1.unmet_dependencies = []
op0_op1.met_dependencies = [MemoryDep('arg0_1', 128*d0 + d1, {d0: 4, d1: 128})]
op0_op1.min_input_distance = 0
op0_op1.max_input_distance = 1
op0_op1.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=False, is_weak=False)]
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0_op1.snodes[0] =
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 4})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 512})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (4, 128)
op0.sizes = ((4,), (128,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
class op0_loop_body:
    var_ranges = {p0: 4, p1: 128}
    index0 = 128*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg0_1', get_index_1)
        mul = ops.mul(load, load_1)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', mul)
        get_index_2 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_2, reduction)
        return None
op0_op1.snodes[1] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 512})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 4})]
op1.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 512})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (512, 1)
op1.sizes = ((4, 128), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
class op1_loop_body:
    var_ranges = {p0: 4, p1: 128}
    index0 = 128*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(128.0, torch.float32)
        truediv = ops.truediv(load_1, constant)
        constant_1 = ops.constant(1e-05, torch.float32)
        add = ops.add(truediv, constant_1)
        rsqrt = ops.rsqrt(add)
        mul = ops.mul(load, rsqrt)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf1', get_index_2, mul, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[4,128]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf1]
    internal: [buf0]

    foreach x in [0,4), reduce r in [0,128):

        op0:  buf0: f32[4,1] @ [x:4]  <-  arg0_1: f32[4,128]      # sum((arg0_1[r + 128*x]) * (arg0_1[r + 128*x]) over r)
            ix0 = r + 128*x
            t0 = arg0_1[ix0]
            t1 = arg0_1[ix0]
            t2 = t0 * t1
            r0 = sum(t2 over r)
            buf0[x] = r0

        op1:  buf1: f32[4,128]  <-  arg0_1: f32[4,128], buf0: f32[4,1] @ [x:4]      # (arg0_1[r + 128*x]) * rsqrt((buf0[x] / 128.0) + 1e-05)
            ix0 = r + 128*x
            t0 = arg0_1[ix0]
            t1 = buf0[x]
            t2 = t1 / 128.0
            t3 = t2 + 1e-05
            t4 = rsqrt(t3)
            t5 = t0 * t4
            buf1[ix0] = t5

# outputs
return (buf1)
```

## Program 20: LayerNorm body — (x - mean) / sqrt(var + eps), two reductions

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 128]"):
        # File: /tmp/ir_printer_examples.py:297 in prog_layernorm, code: m = x.mean(dim=-1, keepdim=True)
        mean: "f32[4, 1]" = torch.ops.aten.mean.dim(arg0_1, [-1], True)

        # File: /tmp/ir_printer_examples.py:299 in prog_layernorm, code: return (x - m) / torch.sqrt(v + 1e-5)
        sub_1: "f32[4, 128]" = torch.ops.aten.sub.Tensor(arg0_1, mean)

        # File: /tmp/ir_printer_examples.py:298 in prog_layernorm, code: v = ((x - m) ** 2).mean(dim=-1, keepdim=True)
        sub: "f32[4, 128]" = torch.ops.aten.sub.Tensor(arg0_1, mean);  arg0_1 = mean = None
        pow_1: "f32[4, 128]" = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
        mean_1: "f32[4, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /tmp/ir_printer_examples.py:299 in prog_layernorm, code: return (x - m) / torch.sqrt(v + 1e-5)
        add: "f32[4, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        sqrt: "f32[4, 1]" = torch.ops.aten.sqrt.default(add);  add = None
        div: "f32[4, 128]" = torch.ops.aten.div.Tensor(sub_1, sqrt);  sub_1 = sqrt = None
        return (div,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, _ = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 128 * i0)
      return tmp0
  ,
  ranges=[4, 1],
  reduction_ranges=[128],
  reduction_type=sum,
  origin_node=None,
  origins=OrderedSet([mean]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 297, in prog_layernorm,
      m = x.mean(dim=-1, keepdim=True),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, _ = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 128 * i0)
      tmp1 = ops.load(buf0, i0)
      tmp2 = ops.index_expr(128, torch.float32)
      tmp3 = tmp1 / tmp2
      tmp4 = tmp0 - tmp3
      tmp5 = tmp4 * tmp4
      return tmp5
  ,
  ranges=[4, 1],
  reduction_ranges=[128],
  reduction_type=sum,
  origin_node=None,
  origins=OrderedSet([mean_1, pow_1, sub, mean]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 298, in prog_layernorm,
      v = ((x - m) ** 2).mean(dim=-1, keepdim=True),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 297, in prog_layernorm,
      m = x.mean(dim=-1, keepdim=True),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf2', layout=FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1 = index
      tmp0 = ops.load(arg0_1, i1 + 128 * i0)
      tmp1 = ops.load(buf0, i0)
      tmp2 = ops.index_expr(128, torch.float32)
      tmp3 = tmp1 / tmp2
      tmp4 = tmp0 - tmp3
      tmp5 = ops.load(buf1, i0)
      tmp6 = ops.index_expr(128, torch.float32)
      tmp7 = tmp5 / tmp6
      tmp8 = ops.constant(1e-05, torch.float32)
      tmp9 = tmp7 + tmp8
      tmp10 = ops.sqrt(tmp9)
      tmp11 = tmp4 / tmp10
      return tmp11
  ,
  ranges=[4, 128],
  origin_node=div,
  origins=OrderedSet([div, sub_1, mean, sqrt, add, mean_1, pow_...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 299, in prog_layernorm,
      return (x - m) / torch.sqrt(v + 1e-5),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 297, in prog_layernorm,
      m = x.mean(dim=-1, keepdim=True),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 298, in prog_layernorm,
      v = ((x - m) ** 2).mean(dim=-1, keepdim=True),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[4,128]

# compute

# op0  sum(arg0_1[128*p0 + p1] over p1)
buf0: f32[4,1]  <-  arg0_1: f32[4,128]
    foreach p0 in [0,4), reduce p1 in [0,128):
        t0 = arg0_1[128*p0 + p1]
        r0 = sum(t0 over p1)
        buf0[p0] = r0

# op1  sum(((arg0_1[128*p0 + p1]) - (buf0[p0] / 128.0)) * ((arg0_1[128*p0 + p1]) - (buf0[p0] / 128.0)) over p1)
buf1: f32[4,1]  <-  arg0_1: f32[4,128], buf0: f32[4,1]
    foreach p0 in [0,4), reduce p1 in [0,128):
        t0 = arg0_1[128*p0 + p1]
        t1 = buf0[p0]
        t2 = t1 / 128.0
        t3 = t0 - t2
        t4 = t3 * t3
        r0 = sum(t4 over p1)
        buf1[p0] = r0

# op2  ((arg0_1[128*p0 + p1]) - (buf0[p0] / 128.0)) / sqrt((buf1[p0] / 128.0) + 1e-05)
buf2: f32[4,128]  <-  arg0_1: f32[4,128], buf0: f32[4,1], buf1: f32[4,1]
    foreach p0 in [0,4), p1 in [0,128):
        ix0 = 128*p0 + p1
        t0 = arg0_1[ix0]
        t1 = buf0[p0]
        t2 = t1 / 128.0
        t3 = t0 - t2
        t4 = buf1[p0]
        t5 = t4 / 128.0
        t6 = t5 + 1e-05
        t7 = sqrt(t6)
        t8 = t3 / t7
        buf2[ix0] = t8

# outputs
return (buf2)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 4})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', 128*d0 + d1, {d0: 4, d1: 128})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
    buf0.users = [
        NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False),
    ]
]
op0.group.device = cuda:0
op0.group.iteration = (4, 128)
op0.sizes = ([4], [128])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
class op0_loop_body:
    var_ranges = {p0: 4, p1: 128}
    index0 = 128*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', d0, {d0: 4})]
op1.unmet_dependencies = [MemoryDep('buf0', d0, {d0: 4})]
op1.met_dependencies = [MemoryDep('arg0_1', 128*d0 + d1, {d0: 4, d1: 128})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (4, 128)
op1.sizes = ([4], [128])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
class op1_loop_body:
    var_ranges = {p0: 4, p1: 128}
    index0 = 128*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(128.0, torch.float32)
        truediv = ops.truediv(load_1, constant)
        sub = ops.sub(load, truediv)
        mul = ops.mul(sub, sub)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', mul)
        get_index_2 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf1', get_index_2, reduction)
        return None


op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', 128*d0 + d1, {d0: 4, d1: 128})]
op2.unmet_dependencies = [MemoryDep('buf0', d0, {d0: 4}), MemoryDep('buf1', d0, {d0: 4})]
op2.met_dependencies = [MemoryDep('arg0_1', 128*d0 + d1, {d0: 4, d1: 128})]
op2.min_input_distance = 1
op2.max_input_distance = 2
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op2.group.device = cuda:0
op2.group.iteration = (512, 1)
op2.sizes = ([4, 128], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
class op2_loop_body:
    var_ranges = {p0: 4, p1: 128}
    index0 = 128*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(128.0, torch.float32)
        truediv = ops.truediv(load_1, constant)
        sub = ops.sub(load, truediv)
        get_index_2 = self.get_index('index1')
        load_2 = ops.load('buf1', get_index_2)
        constant_1 = ops.constant(128.0, torch.float32)
        truediv_1 = ops.truediv(load_2, constant_1)
        constant_2 = ops.constant(1e-05, torch.float32)
        add = ops.add(truediv_1, constant_2)
        sqrt = ops.sqrt(add)
        truediv_2 = ops.truediv(sub, sqrt)
        get_index_3 = self.get_index('index0')
        store = ops.store('buf2', get_index_3, truediv_2, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[4,128]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[4,1]  <-  arg0_1: f32[4,128]      # sum(arg0_1[128*p0 + p1] over p1)
        foreach p0 in [0,4), reduce p1 in [0,128):
            t0 = arg0_1[128*p0 + p1]
            r0 = sum(t0 over p1)
            buf0[p0] = r0

kernel k1
    inputs:   [arg0_1, buf0]
    outputs:  [buf1]

    op1:  buf1: f32[4,1]  <-  arg0_1: f32[4,128], buf0: f32[4,1]      # sum(((arg0_1[128*p0 + p1]) - (buf0[p0] / 128.0)) * ((arg0_1[128*p0 + p1]) - (buf0[p0] / 128.0)) over p1)
        foreach p0 in [0,4), reduce p1 in [0,128):
            t0 = arg0_1[128*p0 + p1]
            t1 = buf0[p0]
            t2 = t1 / 128.0
            t3 = t0 - t2
            t4 = t3 * t3
            r0 = sum(t4 over p1)
            buf1[p0] = r0

kernel k2
    inputs:   [arg0_1, buf0, buf1]
    outputs:  [buf2]

    op2:  buf2: f32[4,128]  <-  arg0_1: f32[4,128], buf0: f32[4,1], buf1: f32[4,1]      # ((arg0_1[128*p0 + p1]) - (buf0[p0] / 128.0)) / sqrt((buf1[p0] / 128.0) + 1e-05)
        foreach p0 in [0,4), p1 in [0,128):
            ix0 = 128*p0 + p1
            t0 = arg0_1[ix0]
            t1 = buf0[p0]
            t2 = t1 / 128.0
            t3 = t0 - t2
            t4 = buf1[p0]
            t5 = t4 / 128.0
            t6 = t5 + 1e-05
            t7 = sqrt(t6)
            t8 = t3 / t7
            buf2[ix0] = t8

# outputs
return (buf2)
```

### post_fusion

```text
op0_op1_op2: FusedSchedulerNode(SchedulerNode,SchedulerNode,SchedulerNode)
op0_op1_op2.writes =
    [   MemoryDep('buf0', d0, {d0: 4}),
        MemoryDep('buf1', d0, {d0: 4}),
        MemoryDep('buf2', 128*d0 + d1, {d0: 4, d1: 128})]
op0_op1_op2.unmet_dependencies = []
op0_op1_op2.met_dependencies = [MemoryDep('arg0_1', 128*d0 + d1, {d0: 4, d1: 128})]
op0_op1_op2.min_input_distance = 0
op0_op1_op2.max_input_distance = 2
op0_op1_op2.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
    buf0.users = [
        NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False),
    ]
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0_op1_op2.snodes[0] =
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 4})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 512})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
    buf0.users = [
        NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False),
    ]
]
op0.group.device = cuda:0
op0.group.iteration = (4, 128)
op0.sizes = ((4,), (128,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
class op0_loop_body:
    var_ranges = {p0: 4, p1: 128}
    index0 = 128*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None
op0_op1_op2.snodes[1] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 4})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 4})]
op1.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 512})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (4, 128)
op1.sizes = ((4,), (128,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
class op1_loop_body:
    var_ranges = {p0: 4, p1: 128}
    index0 = 128*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(128.0, torch.float32)
        truediv = ops.truediv(load_1, constant)
        sub = ops.sub(load, truediv)
        mul = ops.mul(sub, sub)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', mul)
        get_index_2 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf1', get_index_2, reduction)
        return None
op0_op1_op2.snodes[2] =
op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', c0, {c0: 512})]
op2.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 4}), MemoryDep('buf1', c0, {c0: 4})]
op2.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 512})]
op2.min_input_distance = 1
op2.max_input_distance = 2
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op2.group.device = cuda:0
op2.group.iteration = (512, 1)
op2.sizes = ((4, 128), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 1], stride=[1, 4])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[4, 128], stride=[128, 1])
class op2_loop_body:
    var_ranges = {p0: 4, p1: 128}
    index0 = 128*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(128.0, torch.float32)
        truediv = ops.truediv(load_1, constant)
        sub = ops.sub(load, truediv)
        get_index_2 = self.get_index('index1')
        load_2 = ops.load('buf1', get_index_2)
        constant_1 = ops.constant(128.0, torch.float32)
        truediv_1 = ops.truediv(load_2, constant_1)
        constant_2 = ops.constant(1e-05, torch.float32)
        add = ops.add(truediv_1, constant_2)
        sqrt = ops.sqrt(add)
        truediv_2 = ops.truediv(sub, sqrt)
        get_index_3 = self.get_index('index0')
        store = ops.store('buf2', get_index_3, truediv_2, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[4,128]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf2]
    internal: [buf0, buf1]

    foreach x in [0,4), reduce r in [0,128):

        op0:  buf0: f32[4,1] @ [x:4]  <-  arg0_1: f32[4,128]      # sum(arg0_1[r + 128*x] over r)
            t0 = arg0_1[r + 128*x]
            r0 = sum(t0 over r)
            buf0[x] = r0

        op1:  buf1: f32[4,1] @ [x:4]  <-  arg0_1: f32[4,128], buf0: f32[4,1] @ [x:4]      # sum(((arg0_1[r + 128*x]) - (buf0[x] / 128.0)) * ((arg0_1[r + 128*x]) - (buf0[x] / 128.0)) over r)
            t0 = arg0_1[r + 128*x]
            t1 = buf0[x]
            t2 = t1 / 128.0
            t3 = t0 - t2
            t4 = t3 * t3
            r0 = sum(t4 over r)
            buf1[x] = r0

        op2:  buf2: f32[4,128]  <-  arg0_1: f32[4,128], buf0: f32[4,1] @ [x:4], buf1: f32[4,1] @ [x:4]      # ((arg0_1[r + 128*x]) - (buf0[x] / 128.0)) / sqrt((buf1[x] / 128.0) + 1e-05)
            ix0 = r + 128*x
            t0 = arg0_1[ix0]
            t1 = buf0[x]
            t2 = t1 / 128.0
            t3 = t0 - t2
            t4 = buf1[x]
            t5 = t4 / 128.0
            t6 = t5 + 1e-05
            t7 = sqrt(t6)
            t8 = t3 / t7
            buf2[ix0] = t8

# outputs
return (buf2)
```

## Program 21: torch.argmax(x, dim=-1) — reduction returning int64 indices

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 32]"):
        # File: /tmp/ir_printer_examples.py:304 in prog_argmax, code: return torch.argmax(x, dim=-1)
        argmax: "i64[8]" = torch.ops.aten.argmax.default(arg0_1, -1);  arg0_1 = None
        return (argmax,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.int64, size=[8], stride=[1]), data=Reduction(
  'cuda',
  torch.int64,
  def inner_fn(index, rindex):
      i0 = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 32 * i0)
      return tmp0
  ,
  ranges=[8],
  reduction_ranges=[32],
  reduction_type=argmax,
  origin_node=argmax,
  origins=OrderedSet([argmax]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 304, in prog_argmax,
      return torch.argmax(x, dim=-1),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[8,32]

# compute

# op0  argmax(arg0_1[32*p0 + p1] over p1)
buf0: i64[8]  <-  arg0_1: f32[8,32]
    foreach p0 in [0,8), reduce p1 in [0,32):
        t0 = arg0_1[32*p0 + p1]
        r0 = argmax(t0 over p1) acc=f32
        buf0[p0] = r0

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 8})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', 32*d0 + d1, {d0: 8, d1: 32})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.int64, size=[8], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (8, 32)
op0.sizes = ([8], [32])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
buf0_layout = FixedLayout('cuda:0', torch.int64, size=[8], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 8, p1: 32}
    index0 = 32*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.int64, torch.float32, 'argmax', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[8,32]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: i64[8]  <-  arg0_1: f32[8,32]      # argmax(arg0_1[32*p0 + p1] over p1)
        foreach p0 in [0,8), reduce p1 in [0,32):
            t0 = arg0_1[32*p0 + p1]
            r0 = argmax(t0 over p1) acc=f32
            buf0[p0] = r0

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 8})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.int64, size=[8], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (8, 32)
op0.sizes = ((8,), (32,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
buf0_layout = FixedLayout('cuda:0', torch.int64, size=[8], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 8, p1: 32}
    index0 = 32*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.int64, torch.float32, 'argmax', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[8,32]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    foreach x in [0,8), reduce r in [0,32):

        op0:  buf0: i64[8]  <-  arg0_1: f32[8,32]      # argmax(arg0_1[r + 32*x] over r)
            t0 = arg0_1[r + 32*x]
            r0 = argmax(t0 over r) acc=f32
            buf0[x] = r0

# outputs
return (buf0)
```

## Program 22: torch.topk(x, k=4, dim=-1) — extern kernel, tuple output

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 32]"):
        # File: /tmp/ir_printer_examples.py:309 in prog_topk, code: vals, idxs = torch.topk(x, k=4, dim=-1)
        topk = torch.ops.aten.topk.default(arg0_1, 4);  arg0_1 = None
        getitem: "f32[8, 4]" = topk[0]
        getitem_1: "i64[8, 4]" = topk[1];  topk = None
        return (getitem, getitem_1)
```

### post_lowering

```text
FallbackKernel(
  python_kernel_name='torch.ops.aten.topk.default',
  name=buf0,
  layout=MultiOutputLayout(device=device(type='cuda', index=0)),
  inputs=[InputBuffer(name='arg0_1', layout=FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1]))],
  constant_args=(4, -1, True, True),
  kwargs={},
  output_view=None,
  python_kernel_name=torch.ops.aten.topk.default,
  cpp_kernel_name=None,
  ordered_kwargs_for_cpp_kernel=[],
  op_overload=aten.topk.default,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'k', 'type': int, 'default_value': None}, {'name': 'dim', 'type': int, 'default_value': -1}, {'name': 'largest', 'type': bool, 'default_value': True}, {'name': 'sorted', 'type': bool, 'default_value': True}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'k': {'type': int, 'default_value': None}, 'dim': {'type': int, 'default_value': -1}, 'largest': {'type': bool, 'default_value': True}, 'sorted': {'type': bool, 'default_value': True}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=None,
  origins=OrderedSet([topk]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 309, in prog_topk,
      vals, idxs = torch.topk(x, k=4, dim=-1),
  ,
  }
)


MultiOutput(
  python_kernel_name=None,
  name=buf1,
  layout=FixedLayout('cuda:0', torch.float32, size=[8, 4], stride=[4, 1]),
  inputs=[FallbackKernel(
    python_kernel_name='torch.ops.aten.topk.default',
    name=buf0,
    layout=MultiOutputLayout(device=device(type='cuda', index=0)),
    inputs=[InputBuffer(name='arg0_1', layout=FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1]))],
    constant_args=(4, -1, True, True),
    kwargs={},
    output_view=None,
    python_kernel_name=torch.ops.aten.topk.default,
    cpp_kernel_name=None,
    ordered_kwargs_for_cpp_kernel=[],
    op_overload=aten.topk.default,
    arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'k', 'type': int, 'default_value': None}, {'name': 'dim', 'type': int, 'default_value': -1}, {'name': 'largest', 'type': bool, 'default_value': True}, {'name': 'sorted', 'type': bool, 'default_value': True}],
    allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'k': {'type': int, 'default_value': None}, 'dim': {'type': int, 'default_value': -1}, 'largest': {'type': bool, 'default_value': True}, 'sorted': {'type': bool, 'default_value': True}},
    kwarg_properties=None,
    unbacked_bindings={},
    mutation_outputs=[],
    origin_node=None,
    origins=OrderedSet([topk]),
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 309, in prog_topk,
        vals, idxs = torch.topk(x, k=4, dim=-1),
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=None,
  cpp_kernel_name=None,
  ordered_kwargs_for_cpp_kernel=(),
  op_overload=None,
  arg_properties=[{}],
  allarg_properties={},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=getitem,
  origins=OrderedSet([topk]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 309, in prog_topk,
      vals, idxs = torch.topk(x, k=4, dim=-1),
  ,
  }
)


MultiOutput(
  python_kernel_name=None,
  name=buf2,
  layout=FixedLayout('cuda:0', torch.int64, size=[8, 4], stride=[4, 1]),
  inputs=[FallbackKernel(
    python_kernel_name='torch.ops.aten.topk.default',
    name=buf0,
    layout=MultiOutputLayout(device=device(type='cuda', index=0)),
    inputs=[InputBuffer(name='arg0_1', layout=FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1]))],
    constant_args=(4, -1, True, True),
    kwargs={},
    output_view=None,
    python_kernel_name=torch.ops.aten.topk.default,
    cpp_kernel_name=None,
    ordered_kwargs_for_cpp_kernel=[],
    op_overload=aten.topk.default,
    arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'k', 'type': int, 'default_value': None}, {'name': 'dim', 'type': int, 'default_value': -1}, {'name': 'largest', 'type': bool, 'default_value': True}, {'name': 'sorted', 'type': bool, 'default_value': True}],
    allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'k': {'type': int, 'default_value': None}, 'dim': {'type': int, 'default_value': -1}, 'largest': {'type': bool, 'default_value': True}, 'sorted': {'type': bool, 'default_value': True}},
    kwarg_properties=None,
    unbacked_bindings={},
    mutation_outputs=[],
    origin_node=None,
    origins=OrderedSet([topk]),
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 309, in prog_topk,
        vals, idxs = torch.topk(x, k=4, dim=-1),
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=None,
  cpp_kernel_name=None,
  ordered_kwargs_for_cpp_kernel=(),
  op_overload=None,
  arg_properties=[{}],
  allarg_properties={},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=getitem_1,
  origins=OrderedSet([topk]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 309, in prog_topk,
      vals, idxs = torch.topk(x, k=4, dim=-1),
  ,
  }
)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[8,32]

# compute

# op0, op1, op2
buf1, buf2 = torch.ops.aten.topk.default(arg0_1, 4, -1, True, True)
    buf1: f32[8,4]
    buf2: i64[8,4]



# outputs
return (buf1, buf2)
```

### post_scheduler

```text
op0: ExternKernelSchedulerNode(FallbackKernel)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: FallbackKernel
    buf0.layout = MultiOutputLayout(device=device(type='cuda', index=0))
    buf0.users = [
        NodeUser(node=ExternKernelSchedulerNode(name='op1'), can_inplace=False, is_weak=False),
        NodeUser(node=ExternKernelSchedulerNode(name='op2'), can_inplace=False, is_weak=False),
    ]
]
op0.node.kernel = torch.ops.aten.topk.default


op1: ExternKernelSchedulerNode(MultiOutput)
op1.writes = [MemoryDep('buf1', 4*d0 + d1, {d0: 8, d1: 4})]
op1.unmet_dependencies = [StarDep(name='buf0', mode=None)]
op1.met_dependencies = []
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: MultiOutput
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[8, 4], stride=[4, 1])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op1.node.kernel = None


op2: ExternKernelSchedulerNode(MultiOutput)
op2.writes = [MemoryDep('buf2', 4*d0 + d1, {d0: 8, d1: 4})]
op2.unmet_dependencies = [StarDep(name='buf0', mode=None)]
op2.met_dependencies = []
op2.min_input_distance = 1
op2.max_input_distance = 1
op2.outputs = [
    buf2: MultiOutput
    buf2.layout = FixedLayout('cuda:0', torch.int64, size=[8, 4], stride=[4, 1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op2.node.kernel = None
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[8,32]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf1, buf2 = torch.ops.aten.topk.default(arg0_1, 4, -1, True, True)      # , op1, op2
        buf1: f32[8,4]
        buf2: i64[8,4]

kernel k1
    # implementation: extern call
    inputs:   [buf0]
    outputs:  [buf1]

    op1:

kernel k2
    # implementation: extern call
    inputs:   [buf0]
    outputs:  [buf2]

    op2:

# outputs
return (buf1, buf2)
```

### post_fusion

```text
op0: ExternKernelSchedulerNode(FallbackKernel)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: FallbackKernel
    buf0.layout = MultiOutputLayout(device=device(type='cuda', index=0))
    buf0.users = [
        NodeUser(node=ExternKernelSchedulerNode(name='op1'), can_inplace=False, is_weak=False),
        NodeUser(node=ExternKernelSchedulerNode(name='op2'), can_inplace=False, is_weak=False),
    ]
]
op0.node.kernel = torch.ops.aten.topk.default


op1: ExternKernelSchedulerNode(MultiOutput)
op1.writes = [MemoryDep('buf1', 4*d0 + d1, {d0: 8, d1: 4})]
op1.unmet_dependencies = [StarDep(name='buf0', mode=None)]
op1.met_dependencies = []
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: MultiOutput
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[8, 4], stride=[4, 1])
    buf1.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op1.node.kernel = None


op2: ExternKernelSchedulerNode(MultiOutput)
op2.writes = [MemoryDep('buf2', 4*d0 + d1, {d0: 8, d1: 4})]
op2.unmet_dependencies = [StarDep(name='buf0', mode=None)]
op2.met_dependencies = []
op2.min_input_distance = 1
op2.max_input_distance = 1
op2.outputs = [
    buf2: MultiOutput
    buf2.layout = FixedLayout('cuda:0', torch.int64, size=[8, 4], stride=[4, 1])
    buf2.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op2.node.kernel = None
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[8,32]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf1, buf2 = torch.ops.aten.topk.default(arg0_1, 4, -1, True, True)      # , op1, op2
        buf1: f32[8,4]
        buf2: i64[8,4]

kernel k1
    # implementation: extern call
    inputs:   [buf0]
    outputs:  [buf1]

    op1:

kernel k2
    # implementation: extern call
    inputs:   [buf0]
    outputs:  [buf2]

    op2:

# outputs
return (buf1, buf2)
```

## Program 23: small attention — softmax(q @ k^T / sqrt(d)) @ v

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 8, 16]", arg1_1: "f32[2, 8, 16]", arg2_1: "f32[2, 8, 16]"):
        # File: /tmp/ir_printer_examples.py:316 in prog_scaled_dot_product, code: scores = q @ k.transpose(-2, -1) / (d**0.5)
        expand: "f32[2, 8, 16]" = torch.ops.aten.expand.default(arg0_1, [2, 8, 16]);  arg0_1 = None
        permute: "f32[2, 16, 8]" = torch.ops.aten.permute.default(arg1_1, [0, 2, 1]);  arg1_1 = None
        expand_1: "f32[2, 16, 8]" = torch.ops.aten.expand.default(permute, [2, 16, 8]);  permute = None
        bmm: "f32[2, 8, 8]" = torch.ops.aten.bmm.default(expand, expand_1);  expand = expand_1 = None

        # No stacktrace found for following nodes
        div_tensor: "f32[2, 8, 8]" = torch.ops.aten.div.Tensor(bmm, 4.0)
        eq_tensor: "b8[2, 8, 8]" = torch.ops.aten.eq.Tensor(div_tensor, div_tensor)
        abs_default: "f32[2, 8, 8]" = torch.ops.aten.abs.default(div_tensor)
        ne_scalar: "b8[2, 8, 8]" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        mul_tensor_1: "b8[2, 8, 8]" = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default: "b8[2, 8, 8]" = torch.ops.aten.logical_not.default(mul_tensor_1);  mul_tensor_1 = None
        any_dims: "b8[2, 8, 1]" = torch.ops.aten.any.dims(logical_not_default, [-1], True);  logical_not_default = None
        logical_not_default_1: "b8[2, 8, 1]" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None
        mul_tensor: "f32[2, 8, 8]" = torch.ops.aten.mul.Tensor(bmm, 1);  bmm = None
        amax_default: "f32[2, 8, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[2, 8, 8]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        div_tensor_1: "f32[2, 8, 8]" = torch.ops.aten.div.Tensor(sub_tensor, 4.0);  sub_tensor = None
        amax_default_1: "f32[2, 8, 1]" = torch.ops.aten.amax.default(div_tensor, [-1], True)
        sub_tensor_1: "f32[2, 8, 8]" = torch.ops.aten.sub.Tensor(div_tensor, amax_default_1);  div_tensor = amax_default_1 = None

        # File: /tmp/ir_printer_examples.py:317 in prog_scaled_dot_product, code: return torch.softmax(scores, dim=-1) @ v
        where_self: "f32[2, 8, 8]" = torch.ops.aten.where.self(logical_not_default_1, div_tensor_1, sub_tensor_1);  logical_not_default_1 = div_tensor_1 = sub_tensor_1 = None
        exp: "f32[2, 8, 8]" = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_1: "f32[2, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[2, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None
        expand_2: "f32[2, 8, 8]" = torch.ops.aten.expand.default(div_1, [2, 8, 8]);  div_1 = None
        expand_3: "f32[2, 8, 16]" = torch.ops.aten.expand.default(arg2_1, [2, 8, 16]);  arg2_1 = None
        bmm_1: "f32[2, 8, 16]" = torch.ops.aten.bmm.default(expand_2, expand_3);  expand_2 = expand_3 = None
        return (bmm_1,)
```

### post_lowering

```text
ExternKernelOut(
  python_kernel_name='extern_kernels.bmm',
  name=buf0,
  layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1]),
  inputs=[InputBuffer(name='arg0_1', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])), ReinterpretView(
    StorageBox(
      InputBuffer(name='arg1_1', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1]))
    ),
    FixedLayout('cuda:0', torch.float32, size=[2, 16, 8], stride=[128, 1, 16]),
    origins=OrderedSet([bmm, permute]),
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 316, in prog_scaled_dot_product,
        scores = q @ k.transpose(-2, -1) / (d**0.5),
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.bmm,
  cpp_kernel_name=at::bmm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.bmm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=bmm,
  origins=OrderedSet([bmm, permute]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 316, in prog_scaled_dot_product,
      scores = q @ k.transpose(-2, -1) / (d**0.5),
  ,
  }
)


ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.bool, size=[2, 8, 1], stride=[8, 1, 16]), data=Reduction(
  'cuda',
  torch.bool,
  def inner_fn(index, rindex):
      i0, i1, _ = index
      r0_0 = rindex
      tmp0 = ops.load(buf0, r0_0 + 8 * i1 + 64 * i0)
      tmp1 = ops.constant(0.25, torch.float32)
      tmp2 = tmp0 * tmp1
      tmp3 = ops.load(buf0, r0_0 + 8 * i1 + 64 * i0)
      tmp4 = ops.constant(0.25, torch.float32)
      tmp5 = tmp3 * tmp4
      tmp6 = tmp2 == tmp5
      tmp7 = ops.load(buf0, r0_0 + 8 * i1 + 64 * i0)
      tmp8 = ops.constant(0.25, torch.float32)
      tmp9 = tmp7 * tmp8
      tmp10 = ops.abs(tmp9)
      tmp11 = ops.constant(inf, torch.float32)
      tmp12 = tmp10 != tmp11
      tmp13 = ops.logical_and(tmp6, tmp12)
      tmp14 = ops.logical_not(tmp13)
      return tmp14
  ,
  ranges=[2, 8, 1],
  reduction_ranges=[8],
  reduction_type=any,
  origin_node=any_dims,
  origins=OrderedSet([any_dims, logical_not_default, mul_tensor...
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf2', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, i1, _ = index
      r0_0 = rindex
      tmp0 = ops.load(buf0, r0_0 + 8 * i1 + 64 * i0)
      tmp1 = ops.constant(1, torch.float32)
      tmp2 = tmp0 * tmp1
      return tmp2
  ,
  ranges=[2, 8, 1],
  reduction_ranges=[8],
  reduction_type=max,
  origin_node=amax_default,
  origins=OrderedSet([amax_default, mul_tensor])
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf3', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, i1, _ = index
      r0_0 = rindex
      tmp0 = ops.load(buf0, r0_0 + 8 * i1 + 64 * i0)
      tmp1 = ops.constant(0.25, torch.float32)
      tmp2 = tmp0 * tmp1
      return tmp2
  ,
  ranges=[2, 8, 1],
  reduction_ranges=[8],
  reduction_type=max,
  origin_node=amax_default_1,
  origins=OrderedSet([amax_default_1, div_tensor])
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf4', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, i1, _ = index
      r0_0 = rindex
      tmp0 = ops.load(buf1, i1 + 8 * i0)
      tmp1 = ops.logical_not(tmp0)
      tmp2 = ops.load(buf0, r0_0 + 8 * i1 + 64 * i0)
      tmp3 = ops.constant(1, torch.float32)
      tmp4 = tmp2 * tmp3
      tmp5 = ops.load(buf2, i1 + 8 * i0)
      tmp6 = tmp4 - tmp5
      tmp7 = ops.constant(0.25, torch.float32)
      tmp8 = tmp6 * tmp7
      tmp9 = ops.load(buf0, r0_0 + 8 * i1 + 64 * i0)
      tmp10 = ops.constant(0.25, torch.float32)
      tmp11 = tmp9 * tmp10
      tmp12 = ops.load(buf3, i1 + 8 * i0)
      tmp13 = tmp11 - tmp12
      tmp14 = ops.where(tmp1, tmp8, tmp13)
      tmp15 = ops.exp(tmp14)
      return tmp15
  ,
  ranges=[2, 8, 1],
  reduction_ranges=[8],
  reduction_type=sum,
  origin_node=sum_1,
  origins=OrderedSet([sum_1, exp, where_self, logical_not_defau...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 317, in prog_scaled_dot_product,
      return torch.softmax(scores, dim=-1) @ v,
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf5', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1, i2 = index
      tmp0 = ops.load(buf1, i1 + 8 * i0)
      tmp1 = ops.logical_not(tmp0)
      tmp2 = ops.load(buf0, i2 + 8 * i1 + 64 * i0)
      tmp3 = ops.constant(1, torch.float32)
      tmp4 = tmp2 * tmp3
      tmp5 = ops.load(buf2, i1 + 8 * i0)
      tmp6 = tmp4 - tmp5
      tmp7 = ops.constant(0.25, torch.float32)
      tmp8 = tmp6 * tmp7
      tmp9 = ops.load(buf0, i2 + 8 * i1 + 64 * i0)
      tmp10 = ops.constant(0.25, torch.float32)
      tmp11 = tmp9 * tmp10
      tmp12 = ops.load(buf3, i1 + 8 * i0)
      tmp13 = tmp11 - tmp12
      tmp14 = ops.where(tmp1, tmp8, tmp13)
      tmp15 = ops.exp(tmp14)
      tmp16 = ops.load(buf4, i1 + 8 * i0)
      tmp17 = tmp15 / tmp16
      return tmp17
  ,
  ranges=[2, 8, 8],
  origin_node=expand_2,
  origins=OrderedSet([div_1, exp, where_self, logical_not_defau...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 317, in prog_scaled_dot_product,
      return torch.softmax(scores, dim=-1) @ v,
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ExternKernelOut(
  python_kernel_name='extern_kernels.bmm',
  name=buf6,
  layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1]),
  inputs=[ComputedBuffer(name='buf5', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1]), data=Pointwise(
    'cuda',
    torch.float32,
    def inner_fn(index):
        i0, i1, i2 = index
        tmp0 = ops.load(buf1, i1 + 8 * i0)
        tmp1 = ops.logical_not(tmp0)
        tmp2 = ops.load(buf0, i2 + 8 * i1 + 64 * i0)
        tmp3 = ops.constant(1, torch.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = ops.load(buf2, i1 + 8 * i0)
        tmp6 = tmp4 - tmp5
        tmp7 = ops.constant(0.25, torch.float32)
        tmp8 = tmp6 * tmp7
        tmp9 = ops.load(buf0, i2 + 8 * i1 + 64 * i0)
        tmp10 = ops.constant(0.25, torch.float32)
        tmp11 = tmp9 * tmp10
        tmp12 = ops.load(buf3, i1 + 8 * i0)
        tmp13 = tmp11 - tmp12
        tmp14 = ops.where(tmp1, tmp8, tmp13)
        tmp15 = ops.exp(tmp14)
        tmp16 = ops.load(buf4, i1 + 8 * i0)
        tmp17 = tmp15 / tmp16
        return tmp17
    ,
    ranges=[2, 8, 8],
    origin_node=expand_2,
    origins=OrderedSet([div_1, exp, where_self, logical_not_defau...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 317, in prog_scaled_dot_product,
        return torch.softmax(scores, dim=-1) @ v,
    ,
    }
  ), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None), InputBuffer(name='arg2_1', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1]))],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.bmm,
  cpp_kernel_name=at::bmm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.bmm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=bmm_1,
  origins=OrderedSet([bmm_1, div_1, exp, where_self, logical_no...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 317, in prog_scaled_dot_product,
      return torch.softmax(scores, dim=-1) @ v,
  ,
  }
)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[2,8,16], arg1_1: f32[2,8,16], arg2_1: f32[2,8,16]

# compute

# op0
buf0: f32[2,8,8]
    extern_kernels.bmm(arg0_1, reinterpret_tensor(arg1_1, [2,16,8], [128,1,16], 0))

# op1  any(not ((((buf0[64*p0 + 8*p1 + p2]) * 0.25) == ((buf0[64*p0 + 8*p1 + p2]) * 0.25)) and (abs((buf0[64*p0 + 8*p1 + p2]) * 0.25) != inf)) over p2)
buf1: b8[2,8,1]  <-  buf0: f32[2,8,8]
    foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,8):
        ix0 = 64*p0 + 8*p1 + p2
        t0 = buf0[ix0]
        t1 = t0 * 0.25
        t2 = buf0[ix0]
        t3 = t2 * 0.25
        t4 = t1 == t3
        t5 = buf0[ix0]
        t6 = t5 * 0.25
        t7 = abs(t6)
        t8 = t7 != inf
        t9 = t4 and t8
        t10 = not t9
        r0 = any(t10 over p2)
        buf1[8*p0 + p1] = r0

# op2  max((buf0[64*p0 + 8*p1 + p2]) * 1.0 over p2)
buf2: f32[2,8,1]  <-  buf0: f32[2,8,8]
    foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,8):
        t0 = buf0[64*p0 + 8*p1 + p2]
        t1 = t0 * 1.0
        r0 = max(t1 over p2)
        buf2[8*p0 + p1] = r0

# op3  max((buf0[64*p0 + 8*p1 + p2]) * 0.25 over p2)
buf3: f32[2,8,1]  <-  buf0: f32[2,8,8]
    foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,8):
        t0 = buf0[64*p0 + 8*p1 + p2]
        t1 = t0 * 0.25
        r0 = max(t1 over p2)
        buf3[8*p0 + p1] = r0

# op4  sum(exp(where(not (buf1[8*p0 + p1]), (((buf0[64*p0 + 8*p1 + p2]) * 1.0) - (buf2[8*p0 + p1])) * 0.25, ((buf0[64*p0 + 8*p1 + p2]) * 0.25) - (buf3[8*p0 + p1]))) over p2)
buf4: f32[2,8,1]  <-  buf1: b8[2,8,1], buf0: f32[2,8,8], buf2: f32[2,8,1], buf3: f32[2,8,1]
    foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,8):
        ix0 = 8*p0 + p1
        t0 = buf1[ix0]
        t1 = not t0
        ix1 = 64*p0 + 8*p1 + p2
        t2 = buf0[ix1]
        t3 = t2 * 1.0
        t4 = buf2[ix0]
        t5 = t3 - t4
        t6 = t5 * 0.25
        t7 = buf0[ix1]
        t8 = t7 * 0.25
        t9 = buf3[ix0]
        t10 = t8 - t9
        t11 = where(t1, t6, t10)
        t12 = exp(t11)
        r0 = sum(t12 over p2)
        buf4[ix0] = r0

# op5  exp(where(not (buf1[8*p0 + p1]), (((buf0[64*p0 + 8*p1 + p2]) * 1.0) - (buf2[8*p0 + p1])) * 0.25, ((buf0[64*p0 + 8*p1 + p2]) * 0.25) - (buf3[8*p0 + p1]))) / (buf4[8*p0 + p1])
buf5: f32[2,8,8]  <-  buf1: b8[2,8,1], buf0: f32[2,8,8], buf2: f32[2,8,1], buf3: f32[2,8,1], buf4: f32[2,8,1]
    foreach p0 in [0,2), p1 in [0,8), p2 in [0,8):
        ix0 = 8*p0 + p1
        t0 = buf1[ix0]
        t1 = not t0
        ix1 = 64*p0 + 8*p1 + p2
        t2 = buf0[ix1]
        t3 = t2 * 1.0
        t4 = buf2[ix0]
        t5 = t3 - t4
        t6 = t5 * 0.25
        t7 = buf0[ix1]
        t8 = t7 * 0.25
        t9 = buf3[ix0]
        t10 = t8 - t9
        t11 = where(t1, t6, t10)
        t12 = exp(t11)
        t13 = buf4[ix0]
        t14 = t12 / t13
        buf5[ix1] = t14

# op6
buf6: f32[2,8,16]
    extern_kernels.bmm(buf5, arg2_1)

# outputs
return (buf6)
```

### post_scheduler

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None), StarDep(name='arg1_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ExternKernelOut
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
    buf0.users = [
        NodeUser(node=SchedulerNode(name='op1'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op3'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=True, is_weak=False),
    ]
]
op0.node.kernel = extern_kernels.bmm


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', 8*d0 + d1, {d0: 2, d1: 8})]
op1.unmet_dependencies = [MemoryDep('buf0', 64*d0 + 8*d1 + d2, {d0: 2, d1: 8, d2: 8})]
op1.met_dependencies = []
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.bool, size=[2, 8, 1], stride=[8, 1, 16])
    buf1.users = [
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False),
    ]
]
op1.group.device = cuda:0
op1.group.iteration = (16, 8)
op1.sizes = ([2, 8], [8])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
buf1_layout = FixedLayout('cuda:0', torch.bool, size=[2, 8, 1], stride=[8, 1, 16])
class op1_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 8}
    index0 = 64*p0 + 8*p1 + p2
    index1 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        constant = ops.constant(0.25, torch.float32)
        mul = ops.mul(load, constant)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('buf0', get_index_1)
        constant_1 = ops.constant(0.25, torch.float32)
        mul_1 = ops.mul(load_1, constant_1)
        eq = ops.eq(mul, mul_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf0', get_index_2)
        constant_2 = ops.constant(0.25, torch.float32)
        mul_2 = ops.mul(load_2, constant_2)
        abs_1 = ops.abs(mul_2)
        constant_3 = ops.constant(inf, torch.float32)
        ne = ops.ne(abs_1, constant_3)
        logical_and = ops.logical_and(eq, ne)
        logical_not = ops.logical_not(logical_and)
        reduction = ops.reduction(torch.bool, torch.bool, 'any', logical_not)
        get_index_3 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf1', get_index_3, reduction)
        return None


op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', 8*d0 + d1, {d0: 2, d1: 8})]
op2.unmet_dependencies = [MemoryDep('buf0', 64*d0 + 8*d1 + d2, {d0: 2, d1: 8, d2: 8})]
op2.met_dependencies = []
op2.min_input_distance = 1
op2.max_input_distance = 1
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf2.users = [
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False),
    ]
]
op2.group.device = cuda:0
op2.group.iteration = (16, 8)
op2.sizes = ([2, 8], [8])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op2_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 8}
    index0 = 64*p0 + 8*p1 + p2
    index1 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load, constant)
        reduction = ops.reduction(torch.float32, torch.float32, 'max', mul)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf2', get_index_1, reduction)
        return None


op3: SchedulerNode(ComputedBuffer)
op3.writes = [MemoryDep('buf3', 8*d0 + d1, {d0: 2, d1: 8})]
op3.unmet_dependencies = [MemoryDep('buf0', 64*d0 + 8*d1 + d2, {d0: 2, d1: 8, d2: 8})]
op3.met_dependencies = []
op3.min_input_distance = 1
op3.max_input_distance = 1
op3.outputs = [
    buf3: ComputedBuffer
    buf3.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf3.users = [
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False),
    ]
]
op3.group.device = cuda:0
op3.group.iteration = (16, 8)
op3.sizes = ([2, 8], [8])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op3_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 8}
    index0 = 64*p0 + 8*p1 + p2
    index1 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        constant = ops.constant(0.25, torch.float32)
        mul = ops.mul(load, constant)
        reduction = ops.reduction(torch.float32, torch.float32, 'max', mul)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf3', get_index_1, reduction)
        return None


op4: SchedulerNode(ComputedBuffer)
op4.writes = [MemoryDep('buf4', 8*d0 + d1, {d0: 2, d1: 8})]
op4.unmet_dependencies =
    [   MemoryDep('buf0', 64*d0 + 8*d1 + d2, {d0: 2, d1: 8, d2: 8}),
        MemoryDep('buf1', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf2', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf3', 8*d0 + d1, {d0: 2, d1: 8})]
op4.met_dependencies = []
op4.min_input_distance = 1
op4.max_input_distance = 2
op4.outputs = [
    buf4: ComputedBuffer
    buf4.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf4.users = [NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False)]
]
op4.group.device = cuda:0
op4.group.iteration = (16, 8)
op4.sizes = ([2, 8], [8])
buf1_layout = FixedLayout('cuda:0', torch.bool, size=[2, 8, 1], stride=[8, 1, 16])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf4_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op4_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 8}
    index0 = 8*p0 + p1
    index1 = 64*p0 + 8*p1 + p2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf1', get_index)
        logical_not = ops.logical_not(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load_1, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf2', get_index_2)
        sub = ops.sub(mul, load_2)
        constant_1 = ops.constant(0.25, torch.float32)
        mul_1 = ops.mul(sub, constant_1)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf0', get_index_3)
        constant_2 = ops.constant(0.25, torch.float32)
        mul_2 = ops.mul(load_3, constant_2)
        get_index_4 = self.get_index('index0')
        load_4 = ops.load('buf3', get_index_4)
        sub_1 = ops.sub(mul_2, load_4)
        where = ops.where(logical_not, mul_1, sub_1)
        exp = ops.exp(where)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', exp)
        get_index_5 = self.get_index('index0')
        store_reduction = ops.store_reduction('buf4', get_index_5, reduction)
        return None


op5: SchedulerNode(ComputedBuffer)
op5.writes = [MemoryDep('buf5', 64*d0 + 8*d1 + d2, {d0: 2, d1: 8, d2: 8})]
op5.unmet_dependencies =
    [   MemoryDep('buf0', 64*d0 + 8*d1 + d2, {d0: 2, d1: 8, d2: 8}),
        MemoryDep('buf1', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf2', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf3', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf4', 8*d0 + d1, {d0: 2, d1: 8})]
op5.met_dependencies = []
op5.min_input_distance = 1
op5.max_input_distance = 3
op5.outputs = [
    buf5: ComputedBuffer
    buf5.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
    buf5.users = [NodeUser(node=ExternKernelSchedulerNode(name='op6'), can_inplace=False, is_weak=False)]
]
op5.group.device = cuda:0
op5.group.iteration = (128, 1)
op5.sizes = ([2, 8, 8], [])
buf1_layout = FixedLayout('cuda:0', torch.bool, size=[2, 8, 1], stride=[8, 1, 16])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf4_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf5_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
class op5_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 8}
    index0 = 8*p0 + p1
    index1 = 64*p0 + 8*p1 + p2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf1', get_index)
        logical_not = ops.logical_not(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load_1, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf2', get_index_2)
        sub = ops.sub(mul, load_2)
        constant_1 = ops.constant(0.25, torch.float32)
        mul_1 = ops.mul(sub, constant_1)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf0', get_index_3)
        constant_2 = ops.constant(0.25, torch.float32)
        mul_2 = ops.mul(load_3, constant_2)
        get_index_4 = self.get_index('index0')
        load_4 = ops.load('buf3', get_index_4)
        sub_1 = ops.sub(mul_2, load_4)
        where = ops.where(logical_not, mul_1, sub_1)
        exp = ops.exp(where)
        get_index_5 = self.get_index('index0')
        load_5 = ops.load('buf4', get_index_5)
        truediv = ops.truediv(exp, load_5)
        get_index_6 = self.get_index('index1')
        store = ops.store('buf5', get_index_6, truediv, None)
        return store


op6: ExternKernelSchedulerNode(ExternKernelOut)
op6.writes = [StarDep(name='buf6', mode=None)]
op6.unmet_dependencies = [StarDep(name='buf5', mode=None)]
op6.met_dependencies = [StarDep(name='arg2_1', mode=None)]
op6.min_input_distance = 2
op6.max_input_distance = 4
op6.outputs = [
    buf6: ExternKernelOut
    buf6.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
    buf6.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op6.node.kernel = extern_kernels.bmm
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[2,8,16], arg1_1: f32[2,8,16], arg2_1: f32[2,8,16]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[2,8,8]
        extern_kernels.bmm(arg0_1, reinterpret_tensor(arg1_1, [2,16,8], [128,1,16], 0))

kernel k1
    inputs:   [buf0]
    outputs:  [buf1]

    op1:  buf1: b8[2,8,1]  <-  buf0: f32[2,8,8]      # any(not ((((buf0[64*p0 + 8*p1 + p2]) * 0.25) == ((buf0[64*p0 + 8*p1 + p2]) * 0.25)) and (abs((buf0[64*p0 + 8*p1 + p2]) * 0.25) != inf)) over p2)
        foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,8):
            ix0 = 64*p0 + 8*p1 + p2
            t0 = buf0[ix0]
            t1 = t0 * 0.25
            t2 = buf0[ix0]
            t3 = t2 * 0.25
            t4 = t1 == t3
            t5 = buf0[ix0]
            t6 = t5 * 0.25
            t7 = abs(t6)
            t8 = t7 != inf
            t9 = t4 and t8
            t10 = not t9
            r0 = any(t10 over p2)
            buf1[8*p0 + p1] = r0

kernel k2
    inputs:   [buf0]
    outputs:  [buf2]

    op2:  buf2: f32[2,8,1]  <-  buf0: f32[2,8,8]      # max((buf0[64*p0 + 8*p1 + p2]) * 1.0 over p2)
        foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,8):
            t0 = buf0[64*p0 + 8*p1 + p2]
            t1 = t0 * 1.0
            r0 = max(t1 over p2)
            buf2[8*p0 + p1] = r0

kernel k3
    inputs:   [buf0]
    outputs:  [buf3]

    op3:  buf3: f32[2,8,1]  <-  buf0: f32[2,8,8]      # max((buf0[64*p0 + 8*p1 + p2]) * 0.25 over p2)
        foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,8):
            t0 = buf0[64*p0 + 8*p1 + p2]
            t1 = t0 * 0.25
            r0 = max(t1 over p2)
            buf3[8*p0 + p1] = r0

kernel k4
    inputs:   [buf0, buf1, buf2, buf3]
    outputs:  [buf4]

    op4:  buf4: f32[2,8,1]  <-  buf1: b8[2,8,1], buf0: f32[2,8,8], buf2: f32[2,8,1], buf3: f32[2,8,1]      # sum(exp(where(not (buf1[8*p0 + p1]), (((buf0[64*p0 + 8*p1 + p2]) * 1.0) - (buf2[8*p0 + p1])) * 0.25, ((buf0[64*p0 + 8*p1 + p2]) * 0.25) - (buf3[8*p0 + p1]))) over p2)
        foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,8):
            ix0 = 8*p0 + p1
            t0 = buf1[ix0]
            t1 = not t0
            ix1 = 64*p0 + 8*p1 + p2
            t2 = buf0[ix1]
            t3 = t2 * 1.0
            t4 = buf2[ix0]
            t5 = t3 - t4
            t6 = t5 * 0.25
            t7 = buf0[ix1]
            t8 = t7 * 0.25
            t9 = buf3[ix0]
            t10 = t8 - t9
            t11 = where(t1, t6, t10)
            t12 = exp(t11)
            r0 = sum(t12 over p2)
            buf4[ix0] = r0

kernel k5
    inputs:   [buf0, buf1, buf2, buf3, buf4]
    outputs:  [buf5]

    op5:  buf5: f32[2,8,8]  <-  buf1: b8[2,8,1], buf0: f32[2,8,8], buf2: f32[2,8,1], buf3: f32[2,8,1], buf4: f32[2,8,1]      # exp(where(not (buf1[8*p0 + p1]), (((buf0[64*p0 + 8*p1 + p2]) * 1.0) - (buf2[8*p0 + p1])) * 0.25, ((buf0[64*p0 + 8*p1 + p2]) * 0.25) - (buf3[8*p0 + p1]))) / (buf4[8*p0 + p1])
        foreach p0 in [0,2), p1 in [0,8), p2 in [0,8):
            ix0 = 8*p0 + p1
            t0 = buf1[ix0]
            t1 = not t0
            ix1 = 64*p0 + 8*p1 + p2
            t2 = buf0[ix1]
            t3 = t2 * 1.0
            t4 = buf2[ix0]
            t5 = t3 - t4
            t6 = t5 * 0.25
            t7 = buf0[ix1]
            t8 = t7 * 0.25
            t9 = buf3[ix0]
            t10 = t8 - t9
            t11 = where(t1, t6, t10)
            t12 = exp(t11)
            t13 = buf4[ix0]
            t14 = t12 / t13
            buf5[ix1] = t14

kernel k6
    # implementation: extern call
    inputs:   [arg2_1, buf5]
    outputs:  [buf6]

    op6:  buf6: f32[2,8,16]
        extern_kernels.bmm(buf5, arg2_1)

# outputs
return (buf6)
```

### post_fusion

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None), StarDep(name='arg1_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ExternKernelOut
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
    buf0.users = [
        NodeUser(node=SchedulerNode(name='op1'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op3'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=True, is_weak=False),
    ]
]
op0.node.kernel = extern_kernels.bmm


op1_op2_op3_op4_op5: FusedSchedulerNode(SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode)
op1_op2_op3_op4_op5.writes =
    [   MemoryDep('buf1', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf2', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf3', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf4', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf5', 64*d0 + 8*d1 + d2, {d0: 2, d1: 8, d2: 8})]
op1_op2_op3_op4_op5.unmet_dependencies = [MemoryDep('buf0', 64*d0 + 8*d1 + d2, {d0: 2, d1: 8, d2: 8})]
op1_op2_op3_op4_op5.met_dependencies = []
op1_op2_op3_op4_op5.min_input_distance = 1
op1_op2_op3_op4_op5.max_input_distance = 3
op1_op2_op3_op4_op5.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.bool, size=[2, 8, 1], stride=[8, 1, 16])
    buf1.users = [
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False),
    ]
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf2.users = [
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False),
    ]
    buf3: ComputedBuffer
    buf3.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf3.users = [
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False),
    ]
    buf4: ComputedBuffer
    buf4.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf4.users = [NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False)]
    buf5: ComputedBuffer
    buf5.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
    buf5.users = [NodeUser(node=ExternKernelSchedulerNode(name='op6'), can_inplace=False, is_weak=False)]
]
op1_op2_op3_op4_op5.snodes[0] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 16})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 128})]
op1.met_dependencies = []
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.bool, size=[2, 8, 1], stride=[8, 1, 16])
    buf1.users = [
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False),
    ]
]
op1.group.device = cuda:0
op1.group.iteration = (16, 8)
op1.sizes = ((16,), (8,))
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
buf1_layout = FixedLayout('cuda:0', torch.bool, size=[2, 8, 1], stride=[8, 1, 16])
class op1_loop_body:
    var_ranges = {p0: 16, p1: 8}
    index0 = 8*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        constant = ops.constant(0.25, torch.float32)
        mul = ops.mul(load, constant)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('buf0', get_index_1)
        constant_1 = ops.constant(0.25, torch.float32)
        mul_1 = ops.mul(load_1, constant_1)
        eq = ops.eq(mul, mul_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf0', get_index_2)
        constant_2 = ops.constant(0.25, torch.float32)
        mul_2 = ops.mul(load_2, constant_2)
        abs_1 = ops.abs(mul_2)
        constant_3 = ops.constant(inf, torch.float32)
        ne = ops.ne(abs_1, constant_3)
        logical_and = ops.logical_and(eq, ne)
        logical_not = ops.logical_not(logical_and)
        reduction = ops.reduction(torch.bool, torch.bool, 'any', logical_not)
        get_index_3 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf1', get_index_3, reduction)
        return None
op1_op2_op3_op4_op5.snodes[1] =
op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', c0, {c0: 16})]
op2.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 128})]
op2.met_dependencies = []
op2.min_input_distance = 1
op2.max_input_distance = 1
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf2.users = [
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False),
    ]
]
op2.group.device = cuda:0
op2.group.iteration = (16, 8)
op2.sizes = ((16,), (8,))
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op2_loop_body:
    var_ranges = {p0: 16, p1: 8}
    index0 = 8*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load, constant)
        reduction = ops.reduction(torch.float32, torch.float32, 'max', mul)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf2', get_index_1, reduction)
        return None
op1_op2_op3_op4_op5.snodes[2] =
op3: SchedulerNode(ComputedBuffer)
op3.writes = [MemoryDep('buf3', c0, {c0: 16})]
op3.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 128})]
op3.met_dependencies = []
op3.min_input_distance = 1
op3.max_input_distance = 1
op3.outputs = [
    buf3: ComputedBuffer
    buf3.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf3.users = [
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False),
    ]
]
op3.group.device = cuda:0
op3.group.iteration = (16, 8)
op3.sizes = ((16,), (8,))
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op3_loop_body:
    var_ranges = {p0: 16, p1: 8}
    index0 = 8*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf0', get_index)
        constant = ops.constant(0.25, torch.float32)
        mul = ops.mul(load, constant)
        reduction = ops.reduction(torch.float32, torch.float32, 'max', mul)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf3', get_index_1, reduction)
        return None
op1_op2_op3_op4_op5.snodes[3] =
op4: SchedulerNode(ComputedBuffer)
op4.writes = [MemoryDep('buf4', c0, {c0: 16})]
op4.unmet_dependencies =
    [   MemoryDep('buf0', c0, {c0: 128}),
        MemoryDep('buf1', c0, {c0: 16}),
        MemoryDep('buf2', c0, {c0: 16}),
        MemoryDep('buf3', c0, {c0: 16})]
op4.met_dependencies = []
op4.min_input_distance = 1
op4.max_input_distance = 2
op4.outputs = [
    buf4: ComputedBuffer
    buf4.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf4.users = [NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False)]
]
op4.group.device = cuda:0
op4.group.iteration = (16, 8)
op4.sizes = ((16,), (8,))
buf1_layout = FixedLayout('cuda:0', torch.bool, size=[2, 8, 1], stride=[8, 1, 16])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf4_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op4_loop_body:
    var_ranges = {p0: 16, p1: 8}
    index0 = p0
    index1 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf1', get_index)
        logical_not = ops.logical_not(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load_1, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf2', get_index_2)
        sub = ops.sub(mul, load_2)
        constant_1 = ops.constant(0.25, torch.float32)
        mul_1 = ops.mul(sub, constant_1)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf0', get_index_3)
        constant_2 = ops.constant(0.25, torch.float32)
        mul_2 = ops.mul(load_3, constant_2)
        get_index_4 = self.get_index('index0')
        load_4 = ops.load('buf3', get_index_4)
        sub_1 = ops.sub(mul_2, load_4)
        where = ops.where(logical_not, mul_1, sub_1)
        exp = ops.exp(where)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', exp)
        get_index_5 = self.get_index('index0')
        store_reduction = ops.store_reduction('buf4', get_index_5, reduction)
        return None
op1_op2_op3_op4_op5.snodes[4] =
op5: SchedulerNode(ComputedBuffer)
op5.writes = [MemoryDep('buf5', c0, {c0: 128})]
op5.unmet_dependencies =
    [   MemoryDep('buf0', c0, {c0: 128}),
        MemoryDep('buf1', c0, {c0: 16}),
        MemoryDep('buf2', c0, {c0: 16}),
        MemoryDep('buf3', c0, {c0: 16}),
        MemoryDep('buf4', c0, {c0: 16})]
op5.met_dependencies = []
op5.min_input_distance = 1
op5.max_input_distance = 3
op5.outputs = [
    buf5: ComputedBuffer
    buf5.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
    buf5.users = [NodeUser(node=ExternKernelSchedulerNode(name='op6'), can_inplace=False, is_weak=False)]
]
op5.group.device = cuda:0
op5.group.iteration = (128, 1)
op5.sizes = ((16, 8), ())
buf1_layout = FixedLayout('cuda:0', torch.bool, size=[2, 8, 1], stride=[8, 1, 16])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf4_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf5_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 8], stride=[64, 8, 1])
class op5_loop_body:
    var_ranges = {p0: 16, p1: 8}
    index0 = p0
    index1 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf1', get_index)
        logical_not = ops.logical_not(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load_1, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf2', get_index_2)
        sub = ops.sub(mul, load_2)
        constant_1 = ops.constant(0.25, torch.float32)
        mul_1 = ops.mul(sub, constant_1)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf0', get_index_3)
        constant_2 = ops.constant(0.25, torch.float32)
        mul_2 = ops.mul(load_3, constant_2)
        get_index_4 = self.get_index('index0')
        load_4 = ops.load('buf3', get_index_4)
        sub_1 = ops.sub(mul_2, load_4)
        where = ops.where(logical_not, mul_1, sub_1)
        exp = ops.exp(where)
        get_index_5 = self.get_index('index0')
        load_5 = ops.load('buf4', get_index_5)
        truediv = ops.truediv(exp, load_5)
        get_index_6 = self.get_index('index1')
        store = ops.store('buf5', get_index_6, truediv, None)
        return store


op6: ExternKernelSchedulerNode(ExternKernelOut)
op6.writes = [StarDep(name='buf6', mode=None)]
op6.unmet_dependencies = [StarDep(name='buf5', mode=None)]
op6.met_dependencies = [StarDep(name='arg2_1', mode=None)]
op6.min_input_distance = 2
op6.max_input_distance = 4
op6.outputs = [
    buf6: ExternKernelOut
    buf6.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
    buf6.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op6.node.kernel = extern_kernels.bmm
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[2,8,16], arg1_1: f32[2,8,16], arg2_1: f32[2,8,16]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[2,8,8]
        extern_kernels.bmm(arg0_1, reinterpret_tensor(arg1_1, [2,16,8], [128,1,16], 0))

kernel k1
    inputs:   [buf0]
    outputs:  [buf5]
    internal: [buf1, buf2, buf3, buf4]

    foreach x in [0,16), reduce r in [0,8):

        op1:  buf1: b8[2,8,1] @ [x:16]  <-  buf0: f32[2,8,8] @ [x:16,r:8]      # any(not ((((buf0[r + 8*x]) * 0.25) == ((buf0[r + 8*x]) * 0.25)) and (abs((buf0[r + 8*x]) * 0.25) != inf)) over r)
            ix0 = r + 8*x
            t0 = buf0[ix0]
            t1 = t0 * 0.25
            t2 = buf0[ix0]
            t3 = t2 * 0.25
            t4 = t1 == t3
            t5 = buf0[ix0]
            t6 = t5 * 0.25
            t7 = abs(t6)
            t8 = t7 != inf
            t9 = t4 and t8
            t10 = not t9
            r0 = any(t10 over r)
            buf1[x] = r0

        op2:  buf2: f32[2,8,1] @ [x:16]  <-  buf0: f32[2,8,8] @ [x:16,r:8]      # max((buf0[r + 8*x]) * 1.0 over r)
            t0 = buf0[r + 8*x]
            t1 = t0 * 1.0
            r0 = max(t1 over r)
            buf2[x] = r0

        op3:  buf3: f32[2,8,1] @ [x:16]  <-  buf0: f32[2,8,8] @ [x:16,r:8]      # max((buf0[r + 8*x]) * 0.25 over r)
            t0 = buf0[r + 8*x]
            t1 = t0 * 0.25
            r0 = max(t1 over r)
            buf3[x] = r0

        op4:  buf4: f32[2,8,1] @ [x:16]  <-  buf1: b8[2,8,1] @ [x:16], buf0: f32[2,8,8] @ [x:16,r:8], buf2: f32[2,8,1] @ [x:16], buf3: f32[2,8,1] @ [x:16]      # sum(exp(where(not buf1[x], (((buf0[r + 8*x]) * 1.0) - buf2[x]) * 0.25, ((buf0[r + 8*x]) * 0.25) - buf3[x])) over r)
            t0 = buf1[x]
            t1 = not t0
            ix0 = r + 8*x
            t2 = buf0[ix0]
            t3 = t2 * 1.0
            t4 = buf2[x]
            t5 = t3 - t4
            t6 = t5 * 0.25
            t7 = buf0[ix0]
            t8 = t7 * 0.25
            t9 = buf3[x]
            t10 = t8 - t9
            t11 = where(t1, t6, t10)
            t12 = exp(t11)
            r0 = sum(t12 over r)
            buf4[x] = r0

        op5:  buf5: f32[2,8,8] @ [x:16,r:8]  <-  buf1: b8[2,8,1] @ [x:16], buf0: f32[2,8,8] @ [x:16,r:8], buf2: f32[2,8,1] @ [x:16], buf3: f32[2,8,1] @ [x:16], buf4: f32[2,8,1] @ [x:16]      # exp(where(not buf1[x], (((buf0[r + 8*x]) * 1.0) - buf2[x]) * 0.25, ((buf0[r + 8*x]) * 0.25) - buf3[x])) / buf4[x]
            t0 = buf1[x]
            t1 = not t0
            ix0 = r + 8*x
            t2 = buf0[ix0]
            t3 = t2 * 1.0
            t4 = buf2[x]
            t5 = t3 - t4
            t6 = t5 * 0.25
            t7 = buf0[ix0]
            t8 = t7 * 0.25
            t9 = buf3[x]
            t10 = t8 - t9
            t11 = where(t1, t6, t10)
            t12 = exp(t11)
            t13 = buf4[x]
            t14 = t12 / t13
            buf5[ix0] = t14

kernel k2
    # implementation: extern call
    inputs:   [arg2_1, buf5]
    outputs:  [buf6]

    op6:  buf6: f32[2,8,16]
        extern_kernels.bmm(buf5, arg2_1)

# outputs
return (buf6)
```

## Program 24: GELU — 0.5 * x * (1 + erf(x / sqrt(2))), dense pointwise chain

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[1024]"):
        # File: /tmp/ir_printer_examples.py:322 in prog_gelu, code: return 0.5 * x * (1.0 + torch.erf(x / (2.0**0.5)))
        mul: "f32[1024]" = torch.ops.aten.mul.Tensor(arg0_1, 0.5)
        div: "f32[1024]" = torch.ops.aten.div.Tensor(arg0_1, 1.4142135623730951);  arg0_1 = None
        erf: "f32[1024]" = torch.ops.aten.erf.default(div);  div = None
        add: "f32[1024]" = torch.ops.aten.add.Tensor(erf, 1.0);  erf = None
        mul_1: "f32[1024]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None
        return (mul_1,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0 = index
      tmp0 = ops.load(arg0_1, i0)
      tmp1 = ops.constant(0.5, torch.float32)
      tmp2 = tmp0 * tmp1
      tmp3 = ops.load(arg0_1, i0)
      tmp4 = ops.constant(0.7071067811865475, torch.float32)
      tmp5 = tmp3 * tmp4
      tmp6 = ops.erf(tmp5)
      tmp7 = ops.constant(1.0, torch.float32)
      tmp8 = tmp6 + tmp7
      tmp9 = tmp2 * tmp8
      return tmp9
  ,
  ranges=[1024],
  origin_node=mul_1,
  origins=OrderedSet([mul_1, mul, add, erf, div]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 322, in prog_gelu,
      return 0.5 * x * (1.0 + torch.erf(x / (2.0**0.5))),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[1024]

# compute

# op0  (arg0_1[p0] * 0.5) * (erf(arg0_1[p0] * 0.7071067811865475) + 1.0)
buf0: f32[1024]  <-  arg0_1: f32[1024]
    foreach p0 in [0,1024):
        t0 = arg0_1[p0]
        t1 = t0 * 0.5
        t2 = arg0_1[p0]
        t3 = t2 * 0.7071067811865475
        t4 = erf(t3)
        t5 = t4 + 1.0
        t6 = t1 * t5
        buf0[p0] = t6

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 1024})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', d0, {d0: 1024})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (1024, 1)
op0.sizes = ([1024], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 1024}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(0.5, torch.float32)
        mul = ops.mul(load, constant)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg0_1', get_index_1)
        constant_1 = ops.constant(0.7071067811865475, torch.float32)
        mul_1 = ops.mul(load_1, constant_1)
        erf = ops.erf(mul_1)
        constant_2 = ops.constant(1.0, torch.float32)
        add = ops.add(erf, constant_2)
        mul_2 = ops.mul(mul, add)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf0', get_index_2, mul_2, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[1024]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[1024]  <-  arg0_1: f32[1024]      # (arg0_1[p0] * 0.5) * (erf(arg0_1[p0] * 0.7071067811865475) + 1.0)
        foreach p0 in [0,1024):
            t0 = arg0_1[p0]
            t1 = t0 * 0.5
            t2 = arg0_1[p0]
            t3 = t2 * 0.7071067811865475
            t4 = erf(t3)
            t5 = t4 + 1.0
            t6 = t1 * t5
            buf0[p0] = t6

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 1024})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 1024})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (1024, 1)
op0.sizes = ((1024,), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[1024], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 1024}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(0.5, torch.float32)
        mul = ops.mul(load, constant)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg0_1', get_index_1)
        constant_1 = ops.constant(0.7071067811865475, torch.float32)
        mul_1 = ops.mul(load_1, constant_1)
        erf = ops.erf(mul_1)
        constant_2 = ops.constant(1.0, torch.float32)
        add = ops.add(erf, constant_2)
        mul_2 = ops.mul(mul, add)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf0', get_index_2, mul_2, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[1024]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    foreach x in [0,1024):

        op0:  buf0: f32[1024]  <-  arg0_1: f32[1024]      # (arg0_1[x] * 0.5) * (erf(arg0_1[x] * 0.7071067811865475) + 1.0)
            t0 = arg0_1[x]
            t1 = t0 * 0.5
            t2 = arg0_1[x]
            t3 = t2 * 0.7071067811865475
            t4 = erf(t3)
            t5 = t4 + 1.0
            t6 = t1 * t5
            buf0[x] = t6

# outputs
return (buf0)
```

## Program 25: x.masked_fill(mask, -inf) — conditional pointwise store

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[8, 32]", arg1_1: "b8[8, 32]"):
        # File: /tmp/ir_printer_examples.py:327 in prog_masked_fill, code: return x.masked_fill(mask, float("-inf"))
        full_default: "f32[]" = torch.ops.aten.full.default([], -inf, dtype = torch.float32, layout = torch.strided, device = device(type='cuda', index=0), pin_memory = False)
        where: "f32[8, 32]" = torch.ops.aten.where.self(arg1_1, full_default, arg0_1);  arg1_1 = full_default = arg0_1 = None
        return (where,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1 = index
      tmp0 = ops.load(arg1_1, i1 + 32 * i0)
      tmp1 = ops.constant(-inf, torch.float32)
      tmp2 = ops.load(arg0_1, i1 + 32 * i0)
      tmp3 = ops.where(tmp0, tmp1, tmp2)
      return tmp3
  ,
  ranges=[8, 32],
  origin_node=where,
  origins=OrderedSet([where, full_default]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 327, in prog_masked_fill,
      return x.masked_fill(mask, float("-inf")),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[8,32], arg1_1: b8[8,32]

# compute

# op0  where(arg1_1[32*p0 + p1], -inf, arg0_1[32*p0 + p1])
buf0: f32[8,32]  <-  arg1_1: b8[8,32], arg0_1: f32[8,32]
    foreach p0 in [0,8), p1 in [0,32):
        ix0 = 32*p0 + p1
        t0 = arg1_1[ix0]
        t1 = arg0_1[ix0]
        t2 = where(t0, -inf, t1)
        buf0[ix0] = t2

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 32*d0 + d1, {d0: 8, d1: 32})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', 32*d0 + d1, {d0: 8, d1: 32}),
        MemoryDep('arg1_1', 32*d0 + d1, {d0: 8, d1: 32})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (256, 1)
op0.sizes = ([8, 32], [])
arg1_1_layout = FixedLayout('cuda:0', torch.bool, size=[8, 32], stride=[32, 1])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
class op0_loop_body:
    var_ranges = {p0: 8, p1: 32}
    index0 = 32*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg1_1', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg0_1', get_index_1)
        constant = ops.constant(-inf, torch.float32)
        where = ops.where(load, constant, load_1)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf0', get_index_2, where, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[8,32], arg1_1: b8[8,32]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[8,32]  <-  arg1_1: b8[8,32], arg0_1: f32[8,32]      # where(arg1_1[32*p0 + p1], -inf, arg0_1[32*p0 + p1])
        foreach p0 in [0,8), p1 in [0,32):
            ix0 = 32*p0 + p1
            t0 = arg1_1[ix0]
            t1 = arg0_1[ix0]
            t2 = where(t0, -inf, t1)
            buf0[ix0] = t2

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 256})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256}), MemoryDep('arg1_1', c0, {c0: 256})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (256, 1)
op0.sizes = ((256,), ())
arg1_1_layout = FixedLayout('cuda:0', torch.bool, size=[8, 32], stride=[32, 1])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[8, 32], stride=[32, 1])
class op0_loop_body:
    var_ranges = {p0: 256}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg1_1', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg0_1', get_index_1)
        constant = ops.constant(-inf, torch.float32)
        where = ops.where(load, constant, load_1)
        get_index_2 = self.get_index('index0')
        store = ops.store('buf0', get_index_2, where, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[8,32], arg1_1: b8[8,32]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    foreach x in [0,256):

        op0:  buf0: f32[8,32] @ [x:256]  <-  arg1_1: b8[8,32] @ [x:256], arg0_1: f32[8,32] @ [x:256]      # where(arg1_1[x], -inf, arg0_1[x])
            t0 = arg1_1[x]
            t1 = arg0_1[x]
            t2 = where(t0, -inf, t1)
            buf0[x] = t2

# outputs
return (buf0)
```

## Program 26: torch.cumsum(x, dim=-1) — scan, typically extern

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 64]"):
        # File: /tmp/ir_printer_examples.py:332 in prog_cumsum, code: return torch.cumsum(x, dim=-1)
        cumsum: "f32[4, 64]" = torch.ops.aten.cumsum.default(arg0_1, -1);  arg0_1 = None
        return (cumsum,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1]), data=Scan(device=device(type='cuda', index=0), dtype=torch.float32, inner_fn=<function Buffer.make_loader.<locals>.loader at 0x7f37802c5900>, ranges=[4], scan_ranges=[64], size=[4, 64], combine_fn=<function cumsum.<locals>.combine_fn at 0x7f37802c7c70>, reindex=<function Scan.create.<locals>.reindex at 0x7f378e80ec20>, reduction_hint=<ReductionHint.INNER: 0>, output_index=0, dtypes=(torch.float32,), inner_fns=(<function Buffer.make_loader.<locals>.loader at 0x7f37802c5900>,)), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[4,64]

# compute

# op0
buf0: f32[4,64]  <-  arg0_1: f32[4,64]
    foreach p0 in [0,4), reduce p1 in [0,64):
        ix0 = 64*p0 + p1
        t0 = arg0_1[ix0]
        t1 = scan1((f32,), (t0,))
        t2 = t1[0]
        buf0[ix0] = t2

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 64*d0 + d1, {d0: 4, d1: 64})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', 64*d0 + d1, {d0: 4, d1: 64})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (4, 64)
op0.sizes = ([4], [64])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
class op0_loop_body:
    var_ranges = {p0: 4, p1: 64}
    index0 = 64*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        scan1 = self.scan1((torch.float32,), (load,))
        getitem = scan1[0]
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, getitem, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[4,64]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[4,64]  <-  arg0_1: f32[4,64]
        foreach p0 in [0,4), reduce p1 in [0,64):
            ix0 = 64*p0 + p1
            t0 = arg0_1[ix0]
            t1 = scan1((f32,), (t0,))
            t2 = t1[0]
            buf0[ix0] = t2

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 256})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (4, 64)
op0.sizes = ((4,), (64,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
class op0_loop_body:
    var_ranges = {p0: 4, p1: 64}
    index0 = 64*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        scan1 = self.scan1((torch.float32,), (load,))
        getitem = scan1[0]
        get_index_1 = self.get_index('index0')
        store = ops.store('buf0', get_index_1, getitem, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[4,64]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    foreach x in [0,4), reduce r in [0,64):

        op0:  buf0: f32[4,64]  <-  arg0_1: f32[4,64]
            ix0 = r + 64*x
            t0 = arg0_1[ix0]
            t1 = scan1((f32,), (t0,))
            t2 = t1[0]
            buf0[ix0] = t2

# outputs
return (buf0)
```

## Program 27: fusion candidate — relu(x + y) + z

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[64]", arg1_1: "f32[64]", arg2_1: "f32[64]"):
        # File: /tmp/ir_printer_examples.py:337 in prog_add_relu_add, code: return torch.relu(x + y) + z
        add: "f32[64]" = torch.ops.aten.add.Tensor(arg0_1, arg1_1);  arg0_1 = arg1_1 = None
        relu: "f32[64]" = torch.ops.aten.relu.default(add);  add = None
        add_1: "f32[64]" = torch.ops.aten.add.Tensor(relu, arg2_1);  relu = arg2_1 = None
        return (add_1,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[64], stride=[1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0 = index
      tmp0 = ops.load(arg0_1, i0)
      tmp1 = ops.load(arg1_1, i0)
      tmp2 = tmp0 + tmp1
      tmp3 = ops.relu(tmp2)
      tmp4 = ops.load(arg2_1, i0)
      tmp5 = tmp3 + tmp4
      return tmp5
  ,
  ranges=[64],
  origin_node=add_1,
  origins=OrderedSet([add_1, relu, add]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 337, in prog_add_relu_add,
      return torch.relu(x + y) + z,
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[64], arg1_1: f32[64], arg2_1: f32[64]

# compute

# op0  relu(arg0_1[p0] + arg1_1[p0]) + arg2_1[p0]
buf0: f32[64]  <-  arg0_1: f32[64], arg1_1: f32[64], arg2_1: f32[64]
    foreach p0 in [0,64):
        t0 = arg0_1[p0]
        t1 = arg1_1[p0]
        t2 = t0 + t1
        t3 = relu(t2)
        t4 = arg2_1[p0]
        t5 = t3 + t4
        buf0[p0] = t5

# outputs
return (buf0)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', d0, {d0: 64})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', d0, {d0: 64}),
        MemoryDep('arg1_1', d0, {d0: 64}),
        MemoryDep('arg2_1', d0, {d0: 64})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (64, 1)
op0.sizes = ([64], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
arg2_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 64}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(load, load_1)
        relu = ops.relu(add)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('arg2_1', get_index_2)
        add_1 = ops.add(relu, load_2)
        get_index_3 = self.get_index('index0')
        store = ops.store('buf0', get_index_3, add_1, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[64], arg1_1: f32[64], arg2_1: f32[64]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1, arg2_1]
    outputs:  [buf0]

    op0:  buf0: f32[64]  <-  arg0_1: f32[64], arg1_1: f32[64], arg2_1: f32[64]      # relu(arg0_1[p0] + arg1_1[p0]) + arg2_1[p0]
        foreach p0 in [0,64):
            t0 = arg0_1[p0]
            t1 = arg1_1[p0]
            t2 = t0 + t1
            t3 = relu(t2)
            t4 = arg2_1[p0]
            t5 = t3 + t4
            buf0[p0] = t5

# outputs
return (buf0)
```

### post_fusion

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 64})]
op0.unmet_dependencies = []
op0.met_dependencies =
    [   MemoryDep('arg0_1', c0, {c0: 64}),
        MemoryDep('arg1_1', c0, {c0: 64}),
        MemoryDep('arg2_1', c0, {c0: 64})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
    buf0.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0.group.device = cuda:0
op0.group.iteration = (64, 1)
op0.sizes = ((64,), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
arg2_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
class op0_loop_body:
    var_ranges = {p0: 64}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(load, load_1)
        relu = ops.relu(add)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('arg2_1', get_index_2)
        add_1 = ops.add(relu, load_2)
        get_index_3 = self.get_index('index0')
        store = ops.store('buf0', get_index_3, add_1, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[64], arg1_1: f32[64], arg2_1: f32[64]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1, arg2_1]
    outputs:  [buf0]

    foreach x in [0,64):

        op0:  buf0: f32[64]  <-  arg0_1: f32[64], arg1_1: f32[64], arg2_1: f32[64]      # relu(arg0_1[x] + arg1_1[x]) + arg2_1[x]
            t0 = arg0_1[x]
            t1 = arg1_1[x]
            t2 = t0 + t1
            t3 = relu(t2)
            t4 = arg2_1[x]
            t5 = t3 + t4
            buf0[x] = t5

# outputs
return (buf0)
```

## Program 28: medium/real-world — MLP block: Linear -> GELU -> Linear -> residual

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[4, 32]", arg1_1: "f32[64, 32]", arg2_1: "f32[64]", arg3_1: "f32[32, 64]", arg4_1: "f32[32]"):
        # File: /tmp/ir_printer_examples.py:354 in prog_mlp_block, code: h = torch.nn.functional.linear(x, w1, b1)
        permute: "f32[32, 64]" = torch.ops.aten.permute.default(arg1_1, [1, 0]);  arg1_1 = None
        mm_default_1: "f32[4, 64]" = torch.ops.aten.mm.default(arg0_1, permute);  permute = None
        add_tensor_1: "f32[4, 64]" = torch.ops.aten.add.Tensor(arg2_1, mm_default_1);  arg2_1 = mm_default_1 = None

        # File: /tmp/ir_printer_examples.py:355 in prog_mlp_block, code: h = torch.nn.functional.gelu(h)
        mul: "f32[4, 64]" = torch.ops.aten.mul.Tensor(add_tensor_1, 0.5)
        mul_1: "f32[4, 64]" = torch.ops.aten.mul.Tensor(add_tensor_1, 0.7071067811865476);  add_tensor_1 = None
        erf: "f32[4, 64]" = torch.ops.aten.erf.default(mul_1);  mul_1 = None
        add: "f32[4, 64]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_2: "f32[4, 64]" = torch.ops.aten.mul.Tensor(mul, add);  mul = add = None

        # File: /tmp/ir_printer_examples.py:356 in prog_mlp_block, code: h = torch.nn.functional.linear(h, w2, b2)
        permute_1: "f32[64, 32]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm_default: "f32[4, 32]" = torch.ops.aten.mm.default(mul_2, permute_1);  mul_2 = permute_1 = None
        add_tensor: "f32[4, 32]" = torch.ops.aten.add.Tensor(arg4_1, mm_default);  arg4_1 = mm_default = None

        # File: /tmp/ir_printer_examples.py:357 in prog_mlp_block, code: return x + h
        add_1: "f32[4, 32]" = torch.ops.aten.add.Tensor(arg0_1, add_tensor);  arg0_1 = add_tensor = None
        return (add_1,)
```

### post_lowering

```text
ExternKernelOut(
  python_kernel_name='extern_kernels.mm',
  name=buf0,
  layout=FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1]),
  inputs=[InputBuffer(name='arg0_1', layout=FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1])), ReinterpretView(
    StorageBox(
      InputBuffer(name='arg1_1', layout=FixedLayout('cuda:0', torch.float32, size=[64, 32], stride=[32, 1]))
    ),
    FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[1, 32]),
    origins=OrderedSet([mm_default_1, permute]),
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 354, in prog_mlp_block,
        h = torch.nn.functional.linear(x, w1, b1),
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.mm,
  cpp_kernel_name=at::mm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.mm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=mm_default_1,
  origins=OrderedSet([mm_default_1, permute]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 354, in prog_mlp_block,
      h = torch.nn.functional.linear(x, w1, b1),
  ,
  }
)


ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1 = index
      tmp0 = ops.load(arg2_1, i1)
      tmp1 = ops.load(buf0, i1 + 64 * i0)
      tmp2 = tmp0 + tmp1
      tmp3 = ops.constant(0.5, torch.float32)
      tmp4 = tmp2 * tmp3
      tmp5 = ops.load(arg2_1, i1)
      tmp6 = ops.load(buf0, i1 + 64 * i0)
      tmp7 = tmp5 + tmp6
      tmp8 = ops.constant(0.7071067811865476, torch.float32)
      tmp9 = tmp7 * tmp8
      tmp10 = ops.erf(tmp9)
      tmp11 = ops.constant(1, torch.float32)
      tmp12 = tmp10 + tmp11
      tmp13 = tmp4 * tmp12
      return tmp13
  ,
  ranges=[4, 64],
  origin_node=mul_2,
  origins=OrderedSet([mul_2, mul, add_tensor_1, add, erf, mul_1]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 355, in prog_mlp_block,
      h = torch.nn.functional.gelu(h),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 354, in prog_mlp_block,
      h = torch.nn.functional.linear(x, w1, b1),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ExternKernelOut(
  python_kernel_name='extern_kernels.mm',
  name=buf2,
  layout=FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1]),
  inputs=[ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1]), data=Pointwise(
    'cuda',
    torch.float32,
    def inner_fn(index):
        i0, i1 = index
        tmp0 = ops.load(arg2_1, i1)
        tmp1 = ops.load(buf0, i1 + 64 * i0)
        tmp2 = tmp0 + tmp1
        tmp3 = ops.constant(0.5, torch.float32)
        tmp4 = tmp2 * tmp3
        tmp5 = ops.load(arg2_1, i1)
        tmp6 = ops.load(buf0, i1 + 64 * i0)
        tmp7 = tmp5 + tmp6
        tmp8 = ops.constant(0.7071067811865476, torch.float32)
        tmp9 = tmp7 * tmp8
        tmp10 = ops.erf(tmp9)
        tmp11 = ops.constant(1, torch.float32)
        tmp12 = tmp10 + tmp11
        tmp13 = tmp4 * tmp12
        return tmp13
    ,
    ranges=[4, 64],
    origin_node=mul_2,
    origins=OrderedSet([mul_2, mul, add_tensor_1, add, erf, mul_1]),
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 355, in prog_mlp_block,
        h = torch.nn.functional.gelu(h),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 354, in prog_mlp_block,
        h = torch.nn.functional.linear(x, w1, b1),
    ,
    }
  ), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None), ReinterpretView(
    StorageBox(
      InputBuffer(name='arg3_1', layout=FixedLayout('cuda:0', torch.float32, size=[32, 64], stride=[64, 1]))
    ),
    FixedLayout('cuda:0', torch.float32, size=[64, 32], stride=[1, 64]),
    origins=OrderedSet([mm_default, mul_2, mul, add_tensor_1, add...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 356, in prog_mlp_block,
        h = torch.nn.functional.linear(h, w2, b2),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 355, in prog_mlp_block,
        h = torch.nn.functional.gelu(h),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 354, in prog_mlp_block,
        h = torch.nn.functional.linear(x, w1, b1),
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.mm,
  cpp_kernel_name=at::mm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.mm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=mm_default,
  origins=OrderedSet([mm_default, mul_2, mul, add_tensor_1, add...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 356, in prog_mlp_block,
      h = torch.nn.functional.linear(h, w2, b2),
  ,
  }
)


ComputedBuffer(name='buf3', layout=FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1 = index
      tmp0 = ops.load(arg0_1, i1 + 32 * i0)
      tmp1 = ops.load(arg4_1, i1)
      tmp2 = ops.load(buf2, i1 + 32 * i0)
      tmp3 = tmp1 + tmp2
      tmp4 = tmp0 + tmp3
      return tmp4
  ,
  ranges=[4, 32],
  origin_node=add_1,
  origins=OrderedSet([add_1, add_tensor]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 357, in prog_mlp_block,
      return x + h,
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 356, in prog_mlp_block,
      h = torch.nn.functional.linear(h, w2, b2),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[4,32], arg1_1: f32[64,32], arg2_1: f32[64], arg3_1: f32[32,64], arg4_1: f32[32]

# compute

# op0
buf0: f32[4,64]
    extern_kernels.mm(arg0_1, reinterpret_tensor(arg1_1, [32,64], [1,32], 0))

# op1  ((arg2_1[p1] + (buf0[64*p0 + p1])) * 0.5) * (erf((arg2_1[p1] + (buf0[64*p0 + p1])) * 0.7071067811865476) + 1.0)
buf1: f32[4,64]  <-  arg2_1: f32[64], buf0: f32[4,64]
    foreach p0 in [0,4), p1 in [0,64):
        t0 = arg2_1[p1]
        ix0 = 64*p0 + p1
        t1 = buf0[ix0]
        t2 = t0 + t1
        t3 = t2 * 0.5
        t4 = arg2_1[p1]
        t5 = buf0[ix0]
        t6 = t4 + t5
        t7 = t6 * 0.7071067811865476
        t8 = erf(t7)
        t9 = t8 + 1.0
        t10 = t3 * t9
        buf1[ix0] = t10

# op2
buf2: f32[4,32]
    extern_kernels.mm(buf1, reinterpret_tensor(arg3_1, [64,32], [1,64], 0))

# op3  (arg0_1[32*p0 + p1]) + (arg4_1[p1] + (buf2[32*p0 + p1]))
buf3: f32[4,32]  <-  arg0_1: f32[4,32], arg4_1: f32[32], buf2: f32[4,32]
    foreach p0 in [0,4), p1 in [0,32):
        ix0 = 32*p0 + p1
        t0 = arg0_1[ix0]
        t1 = arg4_1[p1]
        t2 = buf2[ix0]
        t3 = t1 + t2
        t4 = t0 + t3
        buf3[ix0] = t4

# outputs
return (buf3)
```

### post_scheduler

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None), StarDep(name='arg1_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ExternKernelOut
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False)]
]
op0.node.kernel = extern_kernels.mm


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', 64*d0 + d1, {d0: 4, d1: 64})]
op1.unmet_dependencies = [MemoryDep('buf0', 64*d0 + d1, {d0: 4, d1: 64})]
op1.met_dependencies = [MemoryDep('arg2_1', d1, {d0: 4, d1: 64})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
    buf1.users = [NodeUser(node=ExternKernelSchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (256, 1)
op1.sizes = ([4, 64], [])
arg2_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
class op1_loop_body:
    var_ranges = {p0: 4, p1: 64}
    index0 = p1
    index1 = 64*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg2_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        add = ops.add(load, load_1)
        constant = ops.constant(0.5, torch.float32)
        mul = ops.mul(add, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('arg2_1', get_index_2)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf0', get_index_3)
        add_1 = ops.add(load_2, load_3)
        constant_1 = ops.constant(0.7071067811865476, torch.float32)
        mul_1 = ops.mul(add_1, constant_1)
        erf = ops.erf(mul_1)
        constant_2 = ops.constant(1.0, torch.float32)
        add_2 = ops.add(erf, constant_2)
        mul_2 = ops.mul(mul, add_2)
        get_index_4 = self.get_index('index1')
        store = ops.store('buf1', get_index_4, mul_2, None)
        return store


op2: ExternKernelSchedulerNode(ExternKernelOut)
op2.writes = [StarDep(name='buf2', mode=None)]
op2.unmet_dependencies = [StarDep(name='buf1', mode=None)]
op2.met_dependencies = [StarDep(name='arg3_1', mode=None)]
op2.min_input_distance = 2
op2.max_input_distance = 2
op2.outputs = [
    buf2: ExternKernelOut
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1])
    buf2.users = [NodeUser(node=SchedulerNode(name='op3'), can_inplace=True, is_weak=False)]
]
op2.node.kernel = extern_kernels.mm


op3: SchedulerNode(ComputedBuffer)
op3.writes = [MemoryDep('buf3', 32*d0 + d1, {d0: 4, d1: 32})]
op3.unmet_dependencies = [MemoryDep('buf2', 32*d0 + d1, {d0: 4, d1: 32})]
op3.met_dependencies =
    [   MemoryDep('arg0_1', 32*d0 + d1, {d0: 4, d1: 32}),
        MemoryDep('arg4_1', d1, {d0: 4, d1: 32})]
op3.min_input_distance = 3
op3.max_input_distance = 3
op3.outputs = [
    buf3: ComputedBuffer
    buf3.layout = FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1])
    buf3.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op3.group.device = cuda:0
op3.group.iteration = (128, 1)
op3.sizes = ([4, 32], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1])
arg4_1_layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1])
class op3_loop_body:
    var_ranges = {p0: 4, p1: 32}
    index0 = 32*p0 + p1
    index1 = p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg4_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf2', get_index_2)
        add = ops.add(load_1, load_2)
        add_1 = ops.add(load, add)
        get_index_3 = self.get_index('index0')
        store = ops.store('buf3', get_index_3, add_1, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[4,32], arg1_1: f32[64,32], arg2_1: f32[64], arg3_1: f32[32,64], arg4_1: f32[32]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[4,64]
        extern_kernels.mm(arg0_1, reinterpret_tensor(arg1_1, [32,64], [1,32], 0))

kernel k1
    inputs:   [arg2_1, buf0]
    outputs:  [buf1]

    op1:  buf1: f32[4,64]  <-  arg2_1: f32[64], buf0: f32[4,64]      # ((arg2_1[p1] + (buf0[64*p0 + p1])) * 0.5) * (erf((arg2_1[p1] + (buf0[64*p0 + p1])) * 0.7071067811865476) + 1.0)
        foreach p0 in [0,4), p1 in [0,64):
            t0 = arg2_1[p1]
            ix0 = 64*p0 + p1
            t1 = buf0[ix0]
            t2 = t0 + t1
            t3 = t2 * 0.5
            t4 = arg2_1[p1]
            t5 = buf0[ix0]
            t6 = t4 + t5
            t7 = t6 * 0.7071067811865476
            t8 = erf(t7)
            t9 = t8 + 1.0
            t10 = t3 * t9
            buf1[ix0] = t10

kernel k2
    # implementation: extern call
    inputs:   [arg3_1, buf1]
    outputs:  [buf2]

    op2:  buf2: f32[4,32]
        extern_kernels.mm(buf1, reinterpret_tensor(arg3_1, [64,32], [1,64], 0))

kernel k3
    inputs:   [arg0_1, arg4_1, buf2]
    outputs:  [buf3]

    op3:  buf3: f32[4,32]  <-  arg0_1: f32[4,32], arg4_1: f32[32], buf2: f32[4,32]      # (arg0_1[32*p0 + p1]) + (arg4_1[p1] + (buf2[32*p0 + p1]))
        foreach p0 in [0,4), p1 in [0,32):
            ix0 = 32*p0 + p1
            t0 = arg0_1[ix0]
            t1 = arg4_1[p1]
            t2 = buf2[ix0]
            t3 = t1 + t2
            t4 = t0 + t3
            buf3[ix0] = t4

# outputs
return (buf3)
```

### post_fusion

```text
op0: ExternKernelSchedulerNode(ExternKernelOut)
op0.writes = [StarDep(name='buf0', mode=None)]
op0.unmet_dependencies = []
op0.met_dependencies = [StarDep(name='arg0_1', mode=None), StarDep(name='arg1_1', mode=None)]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ExternKernelOut
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
    buf0.users = [NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False)]
]
op0.node.kernel = extern_kernels.mm


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 256})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 256})]
op1.met_dependencies = [MemoryDep('arg2_1', c1, {c0: 4, c1: 64})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
    buf1.users = [NodeUser(node=ExternKernelSchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (256, 1)
op1.sizes = ((4, 64), ())
arg2_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 64], stride=[64, 1])
class op1_loop_body:
    var_ranges = {p0: 4, p1: 64}
    index0 = p1
    index1 = 64*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg2_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        add = ops.add(load, load_1)
        constant = ops.constant(0.5, torch.float32)
        mul = ops.mul(add, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('arg2_1', get_index_2)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf0', get_index_3)
        add_1 = ops.add(load_2, load_3)
        constant_1 = ops.constant(0.7071067811865476, torch.float32)
        mul_1 = ops.mul(add_1, constant_1)
        erf = ops.erf(mul_1)
        constant_2 = ops.constant(1.0, torch.float32)
        add_2 = ops.add(erf, constant_2)
        mul_2 = ops.mul(mul, add_2)
        get_index_4 = self.get_index('index1')
        store = ops.store('buf1', get_index_4, mul_2, None)
        return store


op2: ExternKernelSchedulerNode(ExternKernelOut)
op2.writes = [StarDep(name='buf2', mode=None)]
op2.unmet_dependencies = [StarDep(name='buf1', mode=None)]
op2.met_dependencies = [StarDep(name='arg3_1', mode=None)]
op2.min_input_distance = 2
op2.max_input_distance = 2
op2.outputs = [
    buf2: ExternKernelOut
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1])
    buf2.users = [NodeUser(node=SchedulerNode(name='op3'), can_inplace=True, is_weak=False)]
]
op2.node.kernel = extern_kernels.mm


op3: SchedulerNode(ComputedBuffer)
op3.writes = [MemoryDep('buf3', c0, {c0: 128})]
op3.unmet_dependencies = [MemoryDep('buf2', c0, {c0: 128})]
op3.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 128}), MemoryDep('arg4_1', c1, {c0: 4, c1: 32})]
op3.min_input_distance = 3
op3.max_input_distance = 3
op3.outputs = [
    buf3: ComputedBuffer
    buf3.layout = FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1])
    buf3.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op3.group.device = cuda:0
op3.group.iteration = (128, 1)
op3.sizes = ((4, 32), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1])
arg4_1_layout = FixedLayout('cuda:0', torch.float32, size=[32], stride=[1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[4, 32], stride=[32, 1])
class op3_loop_body:
    var_ranges = {p0: 4, p1: 32}
    index0 = 32*p0 + p1
    index1 = p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg4_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf2', get_index_2)
        add = ops.add(load_1, load_2)
        add_1 = ops.add(load, add)
        get_index_3 = self.get_index('index0')
        store = ops.store('buf3', get_index_3, add_1, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[4,32], arg1_1: f32[64,32], arg2_1: f32[64], arg3_1: f32[32,64], arg4_1: f32[32]

# compute

kernel k0
    # implementation: extern call
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[4,64]
        extern_kernels.mm(arg0_1, reinterpret_tensor(arg1_1, [32,64], [1,32], 0))

kernel k1
    inputs:   [arg2_1, buf0]
    outputs:  [buf1]

    foreach x in [0,256):

        op1:  buf1: f32[4,64] @ [x:256]  <-  arg2_1: f32[64] @ [x:256], buf0: f32[4,64] @ [x:256]      # ((arg2_1[ModularIndexing(x, 1, 64)] + (buf0[64*((x//64)) + (ModularIndexing(x, 1, 64))])) * 0.5) * (erf((arg2_1[ModularIndexing(x, 1, 64)] + (buf0[64*((x//64)) + (ModularIndexing(x, 1, 64))])) * 0.7071067811865476) + 1.0)
            ix0 = ModularIndexing(x, 1, 64)
            t0 = arg2_1[ix0]
            ix1 = 64*((x//64)) + (ModularIndexing(x, 1, 64))
            t1 = buf0[ix1]
            t2 = t0 + t1
            t3 = t2 * 0.5
            t4 = arg2_1[ix0]
            t5 = buf0[ix1]
            t6 = t4 + t5
            t7 = t6 * 0.7071067811865476
            t8 = erf(t7)
            t9 = t8 + 1.0
            t10 = t3 * t9
            buf1[ix1] = t10

kernel k2
    # implementation: extern call
    inputs:   [arg3_1, buf1]
    outputs:  [buf2]

    op2:  buf2: f32[4,32]
        extern_kernels.mm(buf1, reinterpret_tensor(arg3_1, [64,32], [1,64], 0))

kernel k3
    inputs:   [arg0_1, arg4_1, buf2]
    outputs:  [buf3]

    foreach x in [0,128):

        op3:  buf3: f32[4,32] @ [x:128]  <-  arg0_1: f32[4,32] @ [x:128], arg4_1: f32[32] @ [x:128], buf2: f32[4,32] @ [x:128]      # (arg0_1[32*((x//32)) + (ModularIndexing(x, 1, 32))]) + (arg4_1[ModularIndexing(x, 1, 32)] + (buf2[32*((x//32)) + (ModularIndexing(x, 1, 32))]))
            ix0 = 32*((x//32)) + (ModularIndexing(x, 1, 32))
            t0 = arg0_1[ix0]
            t1 = arg4_1[ModularIndexing(x, 1, 32)]
            t2 = buf2[ix0]
            t3 = t1 + t2
            t4 = t0 + t3
            buf3[ix0] = t4

# outputs
return (buf3)
```

## Program 29: complex — a full pre-norm transformer block: LN -> MHA -> residual -> LN -> MLP -> residual.

### post_grad_fx

```python
class GraphModule(torch.nn.Module):
    def forward(self, arg0_1: "f32[2, 8, 16]", arg1_1: "f32[16]", arg2_1: "f32[16]", arg3_1: "f32[48, 16]", arg4_1: "f32[48]", arg5_1: "f32[16, 16]", arg6_1: "f32[16]", arg7_1: "f32[16]", arg8_1: "f32[16]", arg9_1: "f32[64, 16]", arg10_1: "f32[64]", arg11_1: "f32[16, 64]", arg12_1: "f32[16]"):
        # File: /tmp/ir_printer_examples.py:389 in prog_transformer_block, code: m = x.mean(dim=-1, keepdim=True)
        mean: "f32[2, 8, 1]" = torch.ops.aten.mean.dim(arg0_1, [-1], True)

        # File: /tmp/ir_printer_examples.py:391 in prog_transformer_block, code: y = (x - m) * torch.rsqrt(v + 1e-5) * ln1_w + ln1_b
        sub_1: "f32[2, 8, 16]" = torch.ops.aten.sub.Tensor(arg0_1, mean)

        # File: /tmp/ir_printer_examples.py:390 in prog_transformer_block, code: v = ((x - m) ** 2).mean(dim=-1, keepdim=True)
        sub: "f32[2, 8, 16]" = torch.ops.aten.sub.Tensor(arg0_1, mean);  mean = None
        pow_1: "f32[2, 8, 16]" = torch.ops.aten.pow.Tensor_Scalar(sub, 2);  sub = None
        mean_1: "f32[2, 8, 1]" = torch.ops.aten.mean.dim(pow_1, [-1], True);  pow_1 = None

        # File: /tmp/ir_printer_examples.py:391 in prog_transformer_block, code: y = (x - m) * torch.rsqrt(v + 1e-5) * ln1_w + ln1_b
        add: "f32[2, 8, 1]" = torch.ops.aten.add.Tensor(mean_1, 1e-05);  mean_1 = None
        rsqrt: "f32[2, 8, 1]" = torch.ops.aten.rsqrt.default(add);  add = None
        mul: "f32[2, 8, 16]" = torch.ops.aten.mul.Tensor(sub_1, rsqrt);  sub_1 = rsqrt = None
        mul_1: "f32[2, 8, 16]" = torch.ops.aten.mul.Tensor(mul, arg1_1);  mul = arg1_1 = None
        add_1: "f32[2, 8, 16]" = torch.ops.aten.add.Tensor(mul_1, arg2_1);  mul_1 = arg2_1 = None

        # File: /tmp/ir_printer_examples.py:393 in prog_transformer_block, code: qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D]
        view: "f32[16, 16]" = torch.ops.aten.reshape.default(add_1, [16, 16]);  add_1 = None
        permute: "f32[16, 48]" = torch.ops.aten.permute.default(arg3_1, [1, 0]);  arg3_1 = None
        mm_default_3: "f32[16, 48]" = torch.ops.aten.mm.default(view, permute);  view = permute = None
        add_tensor_3: "f32[16, 48]" = torch.ops.aten.add.Tensor(arg4_1, mm_default_3);  arg4_1 = mm_default_3 = None
        view_1: "f32[2, 8, 48]" = torch.ops.aten.reshape.default(add_tensor_3, [2, 8, 48]);  add_tensor_3 = None

        # File: /tmp/ir_printer_examples.py:394 in prog_transformer_block, code: q, k, v = qkv.chunk(3, dim=-1)
        split = torch.ops.aten.split.Tensor(view_1, 16, -1);  view_1 = None
        getitem: "f32[2, 8, 16]" = split[0]
        getitem_1: "f32[2, 8, 16]" = split[1]
        getitem_2: "f32[2, 8, 16]" = split[2];  split = None

        # File: /tmp/ir_printer_examples.py:395 in prog_transformer_block, code: q = q.view(b, s, n_heads, head_d).permute(0, 2, 1, 3)  # [B, H, S, Dh]
        view_2: "f32[2, 8, 2, 8]" = torch.ops.aten.reshape.default(getitem, [2, 8, 2, 8]);  getitem = None
        permute_1: "f32[2, 2, 8, 8]" = torch.ops.aten.permute.default(view_2, [0, 2, 1, 3]);  view_2 = None

        # File: /tmp/ir_printer_examples.py:399 in prog_transformer_block, code: scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S]
        expand: "f32[2, 2, 8, 8]" = torch.ops.aten.expand.default(permute_1, [2, 2, 8, 8]);  permute_1 = None
        clone: "f32[2, 2, 8, 8]" = torch.ops.aten.clone.default(expand, memory_format = torch.contiguous_format);  expand = None
        view_5: "f32[4, 8, 8]" = torch.ops.aten.reshape.default(clone, [4, 8, 8]);  clone = None

        # File: /tmp/ir_printer_examples.py:396 in prog_transformer_block, code: k = k.view(b, s, n_heads, head_d).permute(0, 2, 1, 3)
        view_3: "f32[2, 8, 2, 8]" = torch.ops.aten.reshape.default(getitem_1, [2, 8, 2, 8]);  getitem_1 = None
        permute_2: "f32[2, 2, 8, 8]" = torch.ops.aten.permute.default(view_3, [0, 2, 1, 3]);  view_3 = None

        # File: /tmp/ir_printer_examples.py:399 in prog_transformer_block, code: scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S]
        permute_4: "f32[2, 2, 8, 8]" = torch.ops.aten.permute.default(permute_2, [0, 1, 3, 2]);  permute_2 = None
        expand_1: "f32[2, 2, 8, 8]" = torch.ops.aten.expand.default(permute_4, [2, 2, 8, 8]);  permute_4 = None
        clone_1: "f32[2, 2, 8, 8]" = torch.ops.aten.clone.default(expand_1, memory_format = torch.contiguous_format);  expand_1 = None
        view_6: "f32[4, 8, 8]" = torch.ops.aten.reshape.default(clone_1, [4, 8, 8]);  clone_1 = None
        bmm: "f32[4, 8, 8]" = torch.ops.aten.bmm.default(view_5, view_6);  view_5 = view_6 = None
        view_7: "f32[2, 2, 8, 8]" = torch.ops.aten.reshape.default(bmm, [2, 2, 8, 8]);  bmm = None

        # No stacktrace found for following nodes
        div_tensor: "f32[2, 2, 8, 8]" = torch.ops.aten.div.Tensor(view_7, 2.8284271247461903)
        eq_tensor: "b8[2, 2, 8, 8]" = torch.ops.aten.eq.Tensor(div_tensor, div_tensor)
        abs_default: "f32[2, 2, 8, 8]" = torch.ops.aten.abs.default(div_tensor)
        ne_scalar: "b8[2, 2, 8, 8]" = torch.ops.aten.ne.Scalar(abs_default, inf);  abs_default = None
        mul_tensor_1: "b8[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(eq_tensor, ne_scalar);  eq_tensor = ne_scalar = None
        logical_not_default: "b8[2, 2, 8, 8]" = torch.ops.aten.logical_not.default(mul_tensor_1);  mul_tensor_1 = None
        any_dims: "b8[2, 2, 8, 1]" = torch.ops.aten.any.dims(logical_not_default, [-1], True);  logical_not_default = None
        logical_not_default_1: "b8[2, 2, 8, 1]" = torch.ops.aten.logical_not.default(any_dims);  any_dims = None
        mul_tensor: "f32[2, 2, 8, 8]" = torch.ops.aten.mul.Tensor(view_7, 1);  view_7 = None
        amax_default: "f32[2, 2, 8, 1]" = torch.ops.aten.amax.default(mul_tensor, [-1], True)
        sub_tensor: "f32[2, 2, 8, 8]" = torch.ops.aten.sub.Tensor(mul_tensor, amax_default);  mul_tensor = amax_default = None
        div_tensor_1: "f32[2, 2, 8, 8]" = torch.ops.aten.div.Tensor(sub_tensor, 2.8284271247461903);  sub_tensor = None
        amax_default_1: "f32[2, 2, 8, 1]" = torch.ops.aten.amax.default(div_tensor, [-1], True)
        sub_tensor_1: "f32[2, 2, 8, 8]" = torch.ops.aten.sub.Tensor(div_tensor, amax_default_1);  div_tensor = amax_default_1 = None

        # File: /tmp/ir_printer_examples.py:400 in prog_transformer_block, code: attn = torch.softmax(scores, dim=-1)
        where_self: "f32[2, 2, 8, 8]" = torch.ops.aten.where.self(logical_not_default_1, div_tensor_1, sub_tensor_1);  logical_not_default_1 = div_tensor_1 = sub_tensor_1 = None
        exp: "f32[2, 2, 8, 8]" = torch.ops.aten.exp.default(where_self);  where_self = None
        sum_1: "f32[2, 2, 8, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)
        div_1: "f32[2, 2, 8, 8]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /tmp/ir_printer_examples.py:401 in prog_transformer_block, code: ctx = (attn @ v).permute(0, 2, 1, 3).contiguous().view(b, s, d)  # [B, S, D]
        expand_2: "f32[2, 2, 8, 8]" = torch.ops.aten.expand.default(div_1, [2, 2, 8, 8]);  div_1 = None
        view_8: "f32[4, 8, 8]" = torch.ops.aten.reshape.default(expand_2, [4, 8, 8]);  expand_2 = None

        # File: /tmp/ir_printer_examples.py:397 in prog_transformer_block, code: v = v.view(b, s, n_heads, head_d).permute(0, 2, 1, 3)
        view_4: "f32[2, 8, 2, 8]" = torch.ops.aten.reshape.default(getitem_2, [2, 8, 2, 8]);  getitem_2 = None
        permute_3: "f32[2, 2, 8, 8]" = torch.ops.aten.permute.default(view_4, [0, 2, 1, 3]);  view_4 = None

        # File: /tmp/ir_printer_examples.py:401 in prog_transformer_block, code: ctx = (attn @ v).permute(0, 2, 1, 3).contiguous().view(b, s, d)  # [B, S, D]
        expand_3: "f32[2, 2, 8, 8]" = torch.ops.aten.expand.default(permute_3, [2, 2, 8, 8]);  permute_3 = None
        clone_2: "f32[2, 2, 8, 8]" = torch.ops.aten.clone.default(expand_3, memory_format = torch.contiguous_format);  expand_3 = None
        view_9: "f32[4, 8, 8]" = torch.ops.aten.reshape.default(clone_2, [4, 8, 8]);  clone_2 = None
        bmm_1: "f32[4, 8, 8]" = torch.ops.aten.bmm.default(view_8, view_9);  view_8 = view_9 = None
        view_10: "f32[2, 2, 8, 8]" = torch.ops.aten.reshape.default(bmm_1, [2, 2, 8, 8]);  bmm_1 = None
        permute_5: "f32[2, 8, 2, 8]" = torch.ops.aten.permute.default(view_10, [0, 2, 1, 3]);  view_10 = None
        clone_3: "f32[2, 8, 2, 8]" = torch.ops.aten.clone.default(permute_5, memory_format = torch.contiguous_format);  permute_5 = None
        view_11: "f32[2, 8, 16]" = torch.ops.aten.reshape.default(clone_3, [2, 8, 16]);  clone_3 = None

        # File: /tmp/ir_printer_examples.py:402 in prog_transformer_block, code: x = x + torch.nn.functional.linear(ctx, out_w, out_b)
        view_12: "f32[16, 16]" = torch.ops.aten.reshape.default(view_11, [16, 16]);  view_11 = None
        permute_6: "f32[16, 16]" = torch.ops.aten.permute.default(arg5_1, [1, 0]);  arg5_1 = None
        mm_default_2: "f32[16, 16]" = torch.ops.aten.mm.default(view_12, permute_6);  view_12 = permute_6 = None
        add_tensor_2: "f32[16, 16]" = torch.ops.aten.add.Tensor(arg6_1, mm_default_2);  arg6_1 = mm_default_2 = None
        view_13: "f32[2, 8, 16]" = torch.ops.aten.reshape.default(add_tensor_2, [2, 8, 16]);  add_tensor_2 = None
        add_2: "f32[2, 8, 16]" = torch.ops.aten.add.Tensor(arg0_1, view_13);  arg0_1 = view_13 = None

        # File: /tmp/ir_printer_examples.py:405 in prog_transformer_block, code: m2 = x.mean(dim=-1, keepdim=True)
        mean_2: "f32[2, 8, 1]" = torch.ops.aten.mean.dim(add_2, [-1], True)

        # File: /tmp/ir_printer_examples.py:407 in prog_transformer_block, code: z = (x - m2) * torch.rsqrt(v2 + 1e-5) * ln2_w + ln2_b
        sub_4: "f32[2, 8, 16]" = torch.ops.aten.sub.Tensor(add_2, mean_2)

        # File: /tmp/ir_printer_examples.py:406 in prog_transformer_block, code: v2 = ((x - m2) ** 2).mean(dim=-1, keepdim=True)
        sub_3: "f32[2, 8, 16]" = torch.ops.aten.sub.Tensor(add_2, mean_2);  mean_2 = None
        pow_2: "f32[2, 8, 16]" = torch.ops.aten.pow.Tensor_Scalar(sub_3, 2);  sub_3 = None
        mean_3: "f32[2, 8, 1]" = torch.ops.aten.mean.dim(pow_2, [-1], True);  pow_2 = None

        # File: /tmp/ir_printer_examples.py:407 in prog_transformer_block, code: z = (x - m2) * torch.rsqrt(v2 + 1e-5) * ln2_w + ln2_b
        add_3: "f32[2, 8, 1]" = torch.ops.aten.add.Tensor(mean_3, 1e-05);  mean_3 = None
        rsqrt_1: "f32[2, 8, 1]" = torch.ops.aten.rsqrt.default(add_3);  add_3 = None
        mul_2: "f32[2, 8, 16]" = torch.ops.aten.mul.Tensor(sub_4, rsqrt_1);  sub_4 = rsqrt_1 = None
        mul_3: "f32[2, 8, 16]" = torch.ops.aten.mul.Tensor(mul_2, arg7_1);  mul_2 = arg7_1 = None
        add_4: "f32[2, 8, 16]" = torch.ops.aten.add.Tensor(mul_3, arg8_1);  mul_3 = arg8_1 = None

        # File: /tmp/ir_printer_examples.py:408 in prog_transformer_block, code: h = torch.nn.functional.gelu(torch.nn.functional.linear(z, ff1_w, ff1_b))
        view_14: "f32[16, 16]" = torch.ops.aten.reshape.default(add_4, [16, 16]);  add_4 = None
        permute_7: "f32[16, 64]" = torch.ops.aten.permute.default(arg9_1, [1, 0]);  arg9_1 = None
        mm_default_1: "f32[16, 64]" = torch.ops.aten.mm.default(view_14, permute_7);  view_14 = permute_7 = None
        add_tensor_1: "f32[16, 64]" = torch.ops.aten.add.Tensor(arg10_1, mm_default_1);  arg10_1 = mm_default_1 = None
        view_15: "f32[2, 8, 64]" = torch.ops.aten.reshape.default(add_tensor_1, [2, 8, 64]);  add_tensor_1 = None
        mul_4: "f32[2, 8, 64]" = torch.ops.aten.mul.Tensor(view_15, 0.5)
        mul_5: "f32[2, 8, 64]" = torch.ops.aten.mul.Tensor(view_15, 0.7071067811865476);  view_15 = None
        erf: "f32[2, 8, 64]" = torch.ops.aten.erf.default(mul_5);  mul_5 = None
        add_5: "f32[2, 8, 64]" = torch.ops.aten.add.Tensor(erf, 1);  erf = None
        mul_6: "f32[2, 8, 64]" = torch.ops.aten.mul.Tensor(mul_4, add_5);  mul_4 = add_5 = None

        # File: /tmp/ir_printer_examples.py:409 in prog_transformer_block, code: x = x + torch.nn.functional.linear(h, ff2_w, ff2_b)
        view_16: "f32[16, 64]" = torch.ops.aten.reshape.default(mul_6, [16, 64]);  mul_6 = None
        permute_8: "f32[64, 16]" = torch.ops.aten.permute.default(arg11_1, [1, 0]);  arg11_1 = None
        mm_default: "f32[16, 16]" = torch.ops.aten.mm.default(view_16, permute_8);  view_16 = permute_8 = None
        add_tensor: "f32[16, 16]" = torch.ops.aten.add.Tensor(arg12_1, mm_default);  arg12_1 = mm_default = None
        view_17: "f32[2, 8, 16]" = torch.ops.aten.reshape.default(add_tensor, [2, 8, 16]);  add_tensor = None
        add_6: "f32[2, 8, 16]" = torch.ops.aten.add.Tensor(add_2, view_17);  add_2 = view_17 = None
        return (add_6,)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, i1, _ = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 16 * i1 + 128 * i0)
      return tmp0
  ,
  ranges=[2, 8, 1],
  reduction_ranges=[16],
  reduction_type=sum,
  origin_node=None,
  origins=OrderedSet([mean]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 389, in prog_transformer_block,
      m = x.mean(dim=-1, keepdim=True),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, i1, _ = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 16 * i1 + 128 * i0)
      tmp1 = ops.load(buf0, i1 + 8 * i0)
      tmp2 = ops.index_expr(16, torch.float32)
      tmp3 = tmp1 / tmp2
      tmp4 = tmp0 - tmp3
      tmp5 = tmp4 * tmp4
      return tmp5
  ,
  ranges=[2, 8, 1],
  reduction_ranges=[16],
  reduction_type=sum,
  origin_node=None,
  origins=OrderedSet([mean_1, pow_1, sub, mean]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 390, in prog_transformer_block,
      v = ((x - m) ** 2).mean(dim=-1, keepdim=True),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 389, in prog_transformer_block,
      m = x.mean(dim=-1, keepdim=True),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf2', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1, i2 = index
      tmp0 = ops.load(arg0_1, i2 + 16 * i1 + 128 * i0)
      tmp1 = ops.load(buf0, i1 + 8 * i0)
      tmp2 = ops.index_expr(16, torch.float32)
      tmp3 = tmp1 / tmp2
      tmp4 = tmp0 - tmp3
      tmp5 = ops.load(buf1, i1 + 8 * i0)
      tmp6 = ops.index_expr(16, torch.float32)
      tmp7 = tmp5 / tmp6
      tmp8 = ops.constant(1e-05, torch.float32)
      tmp9 = tmp7 + tmp8
      tmp10 = ops.rsqrt(tmp9)
      tmp11 = tmp4 * tmp10
      tmp12 = ops.load(arg1_1, i2)
      tmp13 = tmp11 * tmp12
      tmp14 = ops.load(arg2_1, i2)
      tmp15 = tmp13 + tmp14
      return tmp15
  ,
  ranges=[2, 8, 16],
  origin_node=add_1,
  origins=OrderedSet([add_1, mul_1, mul, sub_1, mean, rsqrt, ad...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 391, in prog_transformer_block,
      y = (x - m) * torch.rsqrt(v + 1e-5) * ln1_w + ln1_b,
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 389, in prog_transformer_block,
      m = x.mean(dim=-1, keepdim=True),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 390, in prog_transformer_block,
      v = ((x - m) ** 2).mean(dim=-1, keepdim=True),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ExternKernelOut(
  python_kernel_name='extern_kernels.mm',
  name=buf3,
  layout=FixedLayout('cuda:0', torch.float32, size=[16, 48], stride=[48, 1]),
  inputs=[ReinterpretView(
    StorageBox(
      ComputedBuffer(name='buf2', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1]), data=Pointwise(
        'cuda',
        torch.float32,
        def inner_fn(index):
            i0, i1, i2 = index
            tmp0 = ops.load(arg0_1, i2 + 16 * i1 + 128 * i0)
            tmp1 = ops.load(buf0, i1 + 8 * i0)
            tmp2 = ops.index_expr(16, torch.float32)
            tmp3 = tmp1 / tmp2
            tmp4 = tmp0 - tmp3
            tmp5 = ops.load(buf1, i1 + 8 * i0)
            tmp6 = ops.index_expr(16, torch.float32)
            tmp7 = tmp5 / tmp6
            tmp8 = ops.constant(1e-05, torch.float32)
            tmp9 = tmp7 + tmp8
            tmp10 = ops.rsqrt(tmp9)
            tmp11 = tmp4 * tmp10
            tmp12 = ops.load(arg1_1, i2)
            tmp13 = tmp11 * tmp12
            tmp14 = ops.load(arg2_1, i2)
            tmp15 = tmp13 + tmp14
            return tmp15
        ,
        ranges=[2, 8, 16],
        origin_node=add_1,
        origins=OrderedSet([add_1, mul_1, mul, sub_1, mean, rsqrt, ad...,
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 391, in prog_transformer_block,
            y = (x - m) * torch.rsqrt(v + 1e-5) * ln1_w + ln1_b,
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 389, in prog_transformer_block,
            m = x.mean(dim=-1, keepdim=True),
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 390, in prog_transformer_block,
            v = ((x - m) ** 2).mean(dim=-1, keepdim=True),
        ,
        }
      ), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
    ),
    FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1]),
    origins=OrderedSet([mm_default_3, view, add_1, mul_1, mul, su...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
        qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 391, in prog_transformer_block,
        y = (x - m) * torch.rsqrt(v + 1e-5) * ln1_w + ln1_b,
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 389, in prog_transformer_block,
        m = x.mean(dim=-1, keepdim=True),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 390, in prog_transformer_block,
        v = ((x - m) ** 2).mean(dim=-1, keepdim=True),
    ,
    }
  ), ReinterpretView(
    StorageBox(
      InputBuffer(name='arg3_1', layout=FixedLayout('cuda:0', torch.float32, size=[48, 16], stride=[16, 1]))
    ),
    FixedLayout('cuda:0', torch.float32, size=[16, 48], stride=[1, 16]),
    origins=OrderedSet([mm_default_3, view, add_1, mul_1, mul, su...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
        qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 391, in prog_transformer_block,
        y = (x - m) * torch.rsqrt(v + 1e-5) * ln1_w + ln1_b,
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 389, in prog_transformer_block,
        m = x.mean(dim=-1, keepdim=True),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 390, in prog_transformer_block,
        v = ((x - m) ** 2).mean(dim=-1, keepdim=True),
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.mm,
  cpp_kernel_name=at::mm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.mm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=mm_default_3,
  origins=OrderedSet([mm_default_3, view, add_1, mul_1, mul, su...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
      qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
  ,
  }
)


ComputedBuffer(name='buf4', layout=FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1, i2, i3 = index
      tmp0 = ops.load(arg4_1, i3 + 8 * i1)
      tmp1 = ops.load(buf3, i3 + 8 * i1 + 48 * i2 + 384 * i0)
      tmp2 = tmp0 + tmp1
      return tmp2
  ,
  ranges=[2, 2, 8, 8],
  origin_node=clone,
  origins=OrderedSet([clone, permute_1, view_2, split, view_1, ...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
      scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 395, in prog_transformer_block,
      q = q.view(b, s, n_heads, head_d).permute(0, 2, 1, 3)  # [B, H, S, Dh],
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 394, in prog_transformer_block,
      q, k, v = qkv.chunk(3, dim=-1),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
      qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf5', layout=FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1, i2, i3 = index
      tmp0 = ops.load(arg4_1, 16 + i2 + 8 * i1)
      tmp1 = ops.load(buf3, 16 + i2 + 8 * i1 + 48 * i3 + 384 * i0)
      tmp2 = tmp0 + tmp1
      return tmp2
  ,
  ranges=[2, 2, 8, 8],
  origin_node=clone_1,
  origins=OrderedSet([clone_1, permute_4, permute_2, view_3, sp...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
      scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 396, in prog_transformer_block,
      k = k.view(b, s, n_heads, head_d).permute(0, 2, 1, 3),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 394, in prog_transformer_block,
      q, k, v = qkv.chunk(3, dim=-1),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
      qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ExternKernelOut(
  python_kernel_name='extern_kernels.bmm',
  name=buf6,
  layout=FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1]),
  inputs=[ReinterpretView(
    StorageBox(
      ComputedBuffer(name='buf4', layout=FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1]), data=Pointwise(
        'cuda',
        torch.float32,
        def inner_fn(index):
            i0, i1, i2, i3 = index
            tmp0 = ops.load(arg4_1, i3 + 8 * i1)
            tmp1 = ops.load(buf3, i3 + 8 * i1 + 48 * i2 + 384 * i0)
            tmp2 = tmp0 + tmp1
            return tmp2
        ,
        ranges=[2, 2, 8, 8],
        origin_node=clone,
        origins=OrderedSet([clone, permute_1, view_2, split, view_1, ...,
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
            scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 395, in prog_transformer_block,
            q = q.view(b, s, n_heads, head_d).permute(0, 2, 1, 3)  # [B, H, S, Dh],
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 394, in prog_transformer_block,
            q, k, v = qkv.chunk(3, dim=-1),
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
            qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
        ,
        }
      ), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
    ),
    FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1]),
    origins=OrderedSet([bmm, view_5, clone, permute_1, view_2, sp...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
        scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 395, in prog_transformer_block,
        q = q.view(b, s, n_heads, head_d).permute(0, 2, 1, 3)  # [B, H, S, Dh],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 394, in prog_transformer_block,
        q, k, v = qkv.chunk(3, dim=-1),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
        qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 396, in prog_transformer_block,
        k = k.view(b, s, n_heads, head_d).permute(0, 2, 1, 3),
    ,
    }
  ), ReinterpretView(
    StorageBox(
      ComputedBuffer(name='buf5', layout=FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1]), data=Pointwise(
        'cuda',
        torch.float32,
        def inner_fn(index):
            i0, i1, i2, i3 = index
            tmp0 = ops.load(arg4_1, 16 + i2 + 8 * i1)
            tmp1 = ops.load(buf3, 16 + i2 + 8 * i1 + 48 * i3 + 384 * i0)
            tmp2 = tmp0 + tmp1
            return tmp2
        ,
        ranges=[2, 2, 8, 8],
        origin_node=clone_1,
        origins=OrderedSet([clone_1, permute_4, permute_2, view_3, sp...,
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
            scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 396, in prog_transformer_block,
            k = k.view(b, s, n_heads, head_d).permute(0, 2, 1, 3),
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 394, in prog_transformer_block,
            q, k, v = qkv.chunk(3, dim=-1),
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
            qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
        ,
        }
      ), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
    ),
    FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1]),
    origins=OrderedSet([bmm, view_5, clone, permute_1, view_2, sp...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
        scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 395, in prog_transformer_block,
        q = q.view(b, s, n_heads, head_d).permute(0, 2, 1, 3)  # [B, H, S, Dh],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 394, in prog_transformer_block,
        q, k, v = qkv.chunk(3, dim=-1),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
        qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 396, in prog_transformer_block,
        k = k.view(b, s, n_heads, head_d).permute(0, 2, 1, 3),
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.bmm,
  cpp_kernel_name=at::bmm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.bmm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=bmm,
  origins=OrderedSet([bmm, view_5, clone, permute_1, view_2, sp...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
      scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
  ,
  }
)


ComputedBuffer(name='buf7', layout=FixedLayout('cuda:0', torch.bool, size=[2, 2, 8, 1], stride=[16, 8, 1, 32]), data=Reduction(
  'cuda',
  torch.bool,
  def inner_fn(index, rindex):
      i0, i1, i2, _ = index
      r0_0 = rindex
      tmp0 = ops.load(buf6, r0_0 + 8 * i2 + 64 * i1 + 128 * i0)
      tmp1 = ops.constant(0.35355339059327373, torch.float32)
      tmp2 = tmp0 * tmp1
      tmp3 = ops.load(buf6, r0_0 + 8 * i2 + 64 * i1 + 128 * i0)
      tmp4 = ops.constant(0.35355339059327373, torch.float32)
      tmp5 = tmp3 * tmp4
      tmp6 = tmp2 == tmp5
      tmp7 = ops.load(buf6, r0_0 + 8 * i2 + 64 * i1 + 128 * i0)
      tmp8 = ops.constant(0.35355339059327373, torch.float32)
      tmp9 = tmp7 * tmp8
      tmp10 = ops.abs(tmp9)
      tmp11 = ops.constant(inf, torch.float32)
      tmp12 = tmp10 != tmp11
      tmp13 = ops.logical_and(tmp6, tmp12)
      tmp14 = ops.logical_not(tmp13)
      return tmp14
  ,
  ranges=[2, 2, 8, 1],
  reduction_ranges=[8],
  reduction_type=any,
  origin_node=any_dims,
  origins=OrderedSet([any_dims, logical_not_default, mul_tensor...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
      scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf8', layout=FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, i1, i2, _ = index
      r0_0 = rindex
      tmp0 = ops.load(buf6, r0_0 + 8 * i2 + 64 * i1 + 128 * i0)
      tmp1 = ops.constant(1, torch.float32)
      tmp2 = tmp0 * tmp1
      return tmp2
  ,
  ranges=[2, 2, 8, 1],
  reduction_ranges=[8],
  reduction_type=max,
  origin_node=amax_default,
  origins=OrderedSet([amax_default, mul_tensor, view_7]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
      scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf9', layout=FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, i1, i2, _ = index
      r0_0 = rindex
      tmp0 = ops.load(buf6, r0_0 + 8 * i2 + 64 * i1 + 128 * i0)
      tmp1 = ops.constant(0.35355339059327373, torch.float32)
      tmp2 = tmp0 * tmp1
      return tmp2
  ,
  ranges=[2, 2, 8, 1],
  reduction_ranges=[8],
  reduction_type=max,
  origin_node=amax_default_1,
  origins=OrderedSet([amax_default_1, div_tensor, view_7]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
      scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf10', layout=FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, i1, i2, _ = index
      r0_0 = rindex
      tmp0 = ops.load(buf7, i2 + 8 * i1 + 16 * i0)
      tmp1 = ops.logical_not(tmp0)
      tmp2 = ops.load(buf6, r0_0 + 8 * i2 + 64 * i1 + 128 * i0)
      tmp3 = ops.constant(1, torch.float32)
      tmp4 = tmp2 * tmp3
      tmp5 = ops.load(buf8, i2 + 8 * i1 + 16 * i0)
      tmp6 = tmp4 - tmp5
      tmp7 = ops.constant(0.35355339059327373, torch.float32)
      tmp8 = tmp6 * tmp7
      tmp9 = ops.load(buf6, r0_0 + 8 * i2 + 64 * i1 + 128 * i0)
      tmp10 = ops.constant(0.35355339059327373, torch.float32)
      tmp11 = tmp9 * tmp10
      tmp12 = ops.load(buf9, i2 + 8 * i1 + 16 * i0)
      tmp13 = tmp11 - tmp12
      tmp14 = ops.where(tmp1, tmp8, tmp13)
      tmp15 = ops.exp(tmp14)
      return tmp15
  ,
  ranges=[2, 2, 8, 1],
  reduction_ranges=[8],
  reduction_type=sum,
  origin_node=sum_1,
  origins=OrderedSet([sum_1, exp, where_self, logical_not_defau...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 400, in prog_transformer_block,
      attn = torch.softmax(scores, dim=-1),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
      scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf11', layout=FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1, i2, i3 = index
      tmp0 = ops.load(buf7, i2 + 8 * i1 + 16 * i0)
      tmp1 = ops.logical_not(tmp0)
      tmp2 = ops.load(buf6, i3 + 8 * i2 + 64 * i1 + 128 * i0)
      tmp3 = ops.constant(1, torch.float32)
      tmp4 = tmp2 * tmp3
      tmp5 = ops.load(buf8, i2 + 8 * i1 + 16 * i0)
      tmp6 = tmp4 - tmp5
      tmp7 = ops.constant(0.35355339059327373, torch.float32)
      tmp8 = tmp6 * tmp7
      tmp9 = ops.load(buf6, i3 + 8 * i2 + 64 * i1 + 128 * i0)
      tmp10 = ops.constant(0.35355339059327373, torch.float32)
      tmp11 = tmp9 * tmp10
      tmp12 = ops.load(buf9, i2 + 8 * i1 + 16 * i0)
      tmp13 = tmp11 - tmp12
      tmp14 = ops.where(tmp1, tmp8, tmp13)
      tmp15 = ops.exp(tmp14)
      tmp16 = ops.load(buf10, i2 + 8 * i1 + 16 * i0)
      tmp17 = tmp15 / tmp16
      return tmp17
  ,
  ranges=[2, 2, 8, 8],
  origin_node=expand_2,
  origins=OrderedSet([div_1, exp, where_self, logical_not_defau...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 400, in prog_transformer_block,
      attn = torch.softmax(scores, dim=-1),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
      scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf12', layout=FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1, i2, i3 = index
      tmp0 = ops.load(arg4_1, 32 + i3 + 8 * i1)
      tmp1 = ops.load(buf3, 32 + i3 + 8 * i1 + 48 * i2 + 384 * i0)
      tmp2 = tmp0 + tmp1
      return tmp2
  ,
  ranges=[2, 2, 8, 8],
  origin_node=clone_2,
  origins=OrderedSet([clone_2, permute_3, view_4, split, view_1...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 401, in prog_transformer_block,
      ctx = (attn @ v).permute(0, 2, 1, 3).contiguous().view(b, s, d)  # [B, S, D],
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 397, in prog_transformer_block,
      v = v.view(b, s, n_heads, head_d).permute(0, 2, 1, 3),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 394, in prog_transformer_block,
      q, k, v = qkv.chunk(3, dim=-1),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
      qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ExternKernelOut(
  python_kernel_name='extern_kernels.bmm',
  name=buf13,
  layout=FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1]),
  inputs=[ReinterpretView(
    StorageBox(
      ComputedBuffer(name='buf11', layout=FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1]), data=Pointwise(
        'cuda',
        torch.float32,
        def inner_fn(index):
            i0, i1, i2, i3 = index
            tmp0 = ops.load(buf7, i2 + 8 * i1 + 16 * i0)
            tmp1 = ops.logical_not(tmp0)
            tmp2 = ops.load(buf6, i3 + 8 * i2 + 64 * i1 + 128 * i0)
            tmp3 = ops.constant(1, torch.float32)
            tmp4 = tmp2 * tmp3
            tmp5 = ops.load(buf8, i2 + 8 * i1 + 16 * i0)
            tmp6 = tmp4 - tmp5
            tmp7 = ops.constant(0.35355339059327373, torch.float32)
            tmp8 = tmp6 * tmp7
            tmp9 = ops.load(buf6, i3 + 8 * i2 + 64 * i1 + 128 * i0)
            tmp10 = ops.constant(0.35355339059327373, torch.float32)
            tmp11 = tmp9 * tmp10
            tmp12 = ops.load(buf9, i2 + 8 * i1 + 16 * i0)
            tmp13 = tmp11 - tmp12
            tmp14 = ops.where(tmp1, tmp8, tmp13)
            tmp15 = ops.exp(tmp14)
            tmp16 = ops.load(buf10, i2 + 8 * i1 + 16 * i0)
            tmp17 = tmp15 / tmp16
            return tmp17
        ,
        ranges=[2, 2, 8, 8],
        origin_node=expand_2,
        origins=OrderedSet([div_1, exp, where_self, logical_not_defau...,
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 400, in prog_transformer_block,
            attn = torch.softmax(scores, dim=-1),
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
            scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
        ,
        }
      ), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
    ),
    FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1]),
    origins=OrderedSet([bmm_1, view_8, div_1, exp, where_self, lo...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 401, in prog_transformer_block,
        ctx = (attn @ v).permute(0, 2, 1, 3).contiguous().view(b, s, d)  # [B, S, D],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 400, in prog_transformer_block,
        attn = torch.softmax(scores, dim=-1),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
        scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 397, in prog_transformer_block,
        v = v.view(b, s, n_heads, head_d).permute(0, 2, 1, 3),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 394, in prog_transformer_block,
        q, k, v = qkv.chunk(3, dim=-1),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
        qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
    ,
    }
  ), ReinterpretView(
    StorageBox(
      ComputedBuffer(name='buf12', layout=FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1]), data=Pointwise(
        'cuda',
        torch.float32,
        def inner_fn(index):
            i0, i1, i2, i3 = index
            tmp0 = ops.load(arg4_1, 32 + i3 + 8 * i1)
            tmp1 = ops.load(buf3, 32 + i3 + 8 * i1 + 48 * i2 + 384 * i0)
            tmp2 = tmp0 + tmp1
            return tmp2
        ,
        ranges=[2, 2, 8, 8],
        origin_node=clone_2,
        origins=OrderedSet([clone_2, permute_3, view_4, split, view_1...,
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 401, in prog_transformer_block,
            ctx = (attn @ v).permute(0, 2, 1, 3).contiguous().view(b, s, d)  # [B, S, D],
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 397, in prog_transformer_block,
            v = v.view(b, s, n_heads, head_d).permute(0, 2, 1, 3),
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 394, in prog_transformer_block,
            q, k, v = qkv.chunk(3, dim=-1),
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
            qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
        ,
        }
      ), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
    ),
    FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1]),
    origins=OrderedSet([bmm_1, view_8, div_1, exp, where_self, lo...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 401, in prog_transformer_block,
        ctx = (attn @ v).permute(0, 2, 1, 3).contiguous().view(b, s, d)  # [B, S, D],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 400, in prog_transformer_block,
        attn = torch.softmax(scores, dim=-1),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 399, in prog_transformer_block,
        scores = q @ k.transpose(-2, -1) / (head_d**0.5)  # [B, H, S, S],
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 397, in prog_transformer_block,
        v = v.view(b, s, n_heads, head_d).permute(0, 2, 1, 3),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 394, in prog_transformer_block,
        q, k, v = qkv.chunk(3, dim=-1),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 393, in prog_transformer_block,
        qkv = torch.nn.functional.linear(y, qkv_w, qkv_b)  # [B, S, 3D],
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.bmm,
  cpp_kernel_name=at::bmm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.bmm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=bmm_1,
  origins=OrderedSet([bmm_1, view_8, div_1, exp, where_self, lo...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 401, in prog_transformer_block,
      ctx = (attn @ v).permute(0, 2, 1, 3).contiguous().view(b, s, d)  # [B, S, D],
  ,
  }
)


ComputedBuffer(name='buf14', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 2, 8], stride=[128, 16, 8, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1, i2, i3 = index
      tmp0 = ops.load(buf13, i3 + 8 * i1 + 64 * i2 + 128 * i0)
      return tmp0
  ,
  ranges=[2, 8, 2, 8],
  origin_node=clone_3,
  origins=OrderedSet([clone_3, permute_5, view_10]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 401, in prog_transformer_block,
      ctx = (attn @ v).permute(0, 2, 1, 3).contiguous().view(b, s, d)  # [B, S, D],
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ExternKernelOut(
  python_kernel_name='extern_kernels.mm',
  name=buf15,
  layout=FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1]),
  inputs=[ReinterpretView(
    StorageBox(
      ComputedBuffer(name='buf14', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 2, 8], stride=[128, 16, 8, 1]), data=Pointwise(
        'cuda',
        torch.float32,
        def inner_fn(index):
            i0, i1, i2, i3 = index
            tmp0 = ops.load(buf13, i3 + 8 * i1 + 64 * i2 + 128 * i0)
            return tmp0
        ,
        ranges=[2, 8, 2, 8],
        origin_node=clone_3,
        origins=OrderedSet([clone_3, permute_5, view_10]),
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 401, in prog_transformer_block,
            ctx = (attn @ v).permute(0, 2, 1, 3).contiguous().view(b, s, d)  # [B, S, D],
        ,
        }
      ), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
    ),
    FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1]),
    origins=OrderedSet([mm_default_2, view_12, view_11, clone_3, ...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 402, in prog_transformer_block,
        x = x + torch.nn.functional.linear(ctx, out_w, out_b),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 401, in prog_transformer_block,
        ctx = (attn @ v).permute(0, 2, 1, 3).contiguous().view(b, s, d)  # [B, S, D],
    ,
    }
  ), ReinterpretView(
    StorageBox(
      InputBuffer(name='arg5_1', layout=FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1]))
    ),
    FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[1, 16]),
    origins=OrderedSet([mm_default_2, view_12, view_11, clone_3, ...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 402, in prog_transformer_block,
        x = x + torch.nn.functional.linear(ctx, out_w, out_b),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 401, in prog_transformer_block,
        ctx = (attn @ v).permute(0, 2, 1, 3).contiguous().view(b, s, d)  # [B, S, D],
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.mm,
  cpp_kernel_name=at::mm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.mm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=mm_default_2,
  origins=OrderedSet([mm_default_2, view_12, view_11, clone_3, ...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 402, in prog_transformer_block,
      x = x + torch.nn.functional.linear(ctx, out_w, out_b),
  ,
  }
)


ComputedBuffer(name='buf16', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, i1, _ = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 16 * i1 + 128 * i0)
      tmp1 = ops.load(arg6_1, r0_0)
      tmp2 = ops.load(buf15, r0_0 + 16 * i1 + 128 * i0)
      tmp3 = tmp1 + tmp2
      tmp4 = tmp0 + tmp3
      return tmp4
  ,
  ranges=[2, 8, 1],
  reduction_ranges=[16],
  reduction_type=sum,
  origin_node=None,
  origins=OrderedSet([mean_2, add_2, view_13, add_tensor_2]),
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 405, in prog_transformer_block,
      m2 = x.mean(dim=-1, keepdim=True),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 402, in prog_transformer_block,
      x = x + torch.nn.functional.linear(ctx, out_w, out_b),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf17', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      i0, i1, _ = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 16 * i1 + 128 * i0)
      tmp1 = ops.load(arg6_1, r0_0)
      tmp2 = ops.load(buf15, r0_0 + 16 * i1 + 128 * i0)
      tmp3 = tmp1 + tmp2
      tmp4 = tmp0 + tmp3
      tmp5 = ops.load(buf16, i1 + 8 * i0)
      tmp6 = ops.index_expr(16, torch.float32)
      tmp7 = tmp5 / tmp6
      tmp8 = tmp4 - tmp7
      tmp9 = tmp8 * tmp8
      return tmp9
  ,
  ranges=[2, 8, 1],
  reduction_ranges=[16],
  reduction_type=sum,
  origin_node=None,
  origins=OrderedSet([mean_3, pow_2, sub_3, add_2, view_13, add...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 406, in prog_transformer_block,
      v2 = ((x - m2) ** 2).mean(dim=-1, keepdim=True),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 402, in prog_transformer_block,
      x = x + torch.nn.functional.linear(ctx, out_w, out_b),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 405, in prog_transformer_block,
      m2 = x.mean(dim=-1, keepdim=True),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf18', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1, i2 = index
      tmp0 = ops.load(arg0_1, i2 + 16 * i1 + 128 * i0)
      tmp1 = ops.load(arg6_1, i2)
      tmp2 = ops.load(buf15, i2 + 16 * i1 + 128 * i0)
      tmp3 = tmp1 + tmp2
      tmp4 = tmp0 + tmp3
      tmp5 = ops.load(buf16, i1 + 8 * i0)
      tmp6 = ops.index_expr(16, torch.float32)
      tmp7 = tmp5 / tmp6
      tmp8 = tmp4 - tmp7
      tmp9 = ops.load(buf17, i1 + 8 * i0)
      tmp10 = ops.index_expr(16, torch.float32)
      tmp11 = tmp9 / tmp10
      tmp12 = ops.constant(1e-05, torch.float32)
      tmp13 = tmp11 + tmp12
      tmp14 = ops.rsqrt(tmp13)
      tmp15 = tmp8 * tmp14
      tmp16 = ops.load(arg7_1, i2)
      tmp17 = tmp15 * tmp16
      tmp18 = ops.load(arg8_1, i2)
      tmp19 = tmp17 + tmp18
      return tmp19
  ,
  ranges=[2, 8, 16],
  origin_node=add_4,
  origins=OrderedSet([add_4, mul_3, mul_2, sub_4, add_2, view_1...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 407, in prog_transformer_block,
      z = (x - m2) * torch.rsqrt(v2 + 1e-5) * ln2_w + ln2_b,
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 402, in prog_transformer_block,
      x = x + torch.nn.functional.linear(ctx, out_w, out_b),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 405, in prog_transformer_block,
      m2 = x.mean(dim=-1, keepdim=True),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 406, in prog_transformer_block,
      v2 = ((x - m2) ** 2).mean(dim=-1, keepdim=True),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ExternKernelOut(
  python_kernel_name='extern_kernels.mm',
  name=buf19,
  layout=FixedLayout('cuda:0', torch.float32, size=[16, 64], stride=[64, 1]),
  inputs=[ReinterpretView(
    StorageBox(
      ComputedBuffer(name='buf18', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1]), data=Pointwise(
        'cuda',
        torch.float32,
        def inner_fn(index):
            i0, i1, i2 = index
            tmp0 = ops.load(arg0_1, i2 + 16 * i1 + 128 * i0)
            tmp1 = ops.load(arg6_1, i2)
            tmp2 = ops.load(buf15, i2 + 16 * i1 + 128 * i0)
            tmp3 = tmp1 + tmp2
            tmp4 = tmp0 + tmp3
            tmp5 = ops.load(buf16, i1 + 8 * i0)
            tmp6 = ops.index_expr(16, torch.float32)
            tmp7 = tmp5 / tmp6
            tmp8 = tmp4 - tmp7
            tmp9 = ops.load(buf17, i1 + 8 * i0)
            tmp10 = ops.index_expr(16, torch.float32)
            tmp11 = tmp9 / tmp10
            tmp12 = ops.constant(1e-05, torch.float32)
            tmp13 = tmp11 + tmp12
            tmp14 = ops.rsqrt(tmp13)
            tmp15 = tmp8 * tmp14
            tmp16 = ops.load(arg7_1, i2)
            tmp17 = tmp15 * tmp16
            tmp18 = ops.load(arg8_1, i2)
            tmp19 = tmp17 + tmp18
            return tmp19
        ,
        ranges=[2, 8, 16],
        origin_node=add_4,
        origins=OrderedSet([add_4, mul_3, mul_2, sub_4, add_2, view_1...,
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 407, in prog_transformer_block,
            z = (x - m2) * torch.rsqrt(v2 + 1e-5) * ln2_w + ln2_b,
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 402, in prog_transformer_block,
            x = x + torch.nn.functional.linear(ctx, out_w, out_b),
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 405, in prog_transformer_block,
            m2 = x.mean(dim=-1, keepdim=True),
        ,
        },
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 406, in prog_transformer_block,
            v2 = ((x - m2) ** 2).mean(dim=-1, keepdim=True),
        ,
        }
      ), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
    ),
    FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1]),
    origins=OrderedSet([mm_default_1, view_14, add_4, mul_3, mul_...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 408, in prog_transformer_block,
        h = torch.nn.functional.gelu(torch.nn.functional.linear(z, ff1_w, ff1_b)),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 407, in prog_transformer_block,
        z = (x - m2) * torch.rsqrt(v2 + 1e-5) * ln2_w + ln2_b,
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 402, in prog_transformer_block,
        x = x + torch.nn.functional.linear(ctx, out_w, out_b),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 405, in prog_transformer_block,
        m2 = x.mean(dim=-1, keepdim=True),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 406, in prog_transformer_block,
        v2 = ((x - m2) ** 2).mean(dim=-1, keepdim=True),
    ,
    }
  ), ReinterpretView(
    StorageBox(
      InputBuffer(name='arg9_1', layout=FixedLayout('cuda:0', torch.float32, size=[64, 16], stride=[16, 1]))
    ),
    FixedLayout('cuda:0', torch.float32, size=[16, 64], stride=[1, 16]),
    origins=OrderedSet([mm_default_1, view_14, add_4, mul_3, mul_...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 408, in prog_transformer_block,
        h = torch.nn.functional.gelu(torch.nn.functional.linear(z, ff1_w, ff1_b)),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 407, in prog_transformer_block,
        z = (x - m2) * torch.rsqrt(v2 + 1e-5) * ln2_w + ln2_b,
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 402, in prog_transformer_block,
        x = x + torch.nn.functional.linear(ctx, out_w, out_b),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 405, in prog_transformer_block,
        m2 = x.mean(dim=-1, keepdim=True),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 406, in prog_transformer_block,
        v2 = ((x - m2) ** 2).mean(dim=-1, keepdim=True),
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.mm,
  cpp_kernel_name=at::mm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.mm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=mm_default_1,
  origins=OrderedSet([mm_default_1, view_14, add_4, mul_3, mul_...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 408, in prog_transformer_block,
      h = torch.nn.functional.gelu(torch.nn.functional.linear(z, ff1_w, ff1_b)),
  ,
  }
)


ComputedBuffer(name='buf20', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 64], stride=[512, 64, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1, i2 = index
      tmp0 = ops.load(arg10_1, i2)
      tmp1 = ops.load(buf19, i2 + 64 * i1 + 512 * i0)
      tmp2 = tmp0 + tmp1
      tmp3 = ops.constant(0.5, torch.float32)
      tmp4 = tmp2 * tmp3
      tmp5 = ops.load(arg10_1, i2)
      tmp6 = ops.load(buf19, i2 + 64 * i1 + 512 * i0)
      tmp7 = tmp5 + tmp6
      tmp8 = ops.constant(0.7071067811865476, torch.float32)
      tmp9 = tmp7 * tmp8
      tmp10 = ops.erf(tmp9)
      tmp11 = ops.constant(1, torch.float32)
      tmp12 = tmp10 + tmp11
      tmp13 = tmp4 * tmp12
      return tmp13
  ,
  ranges=[2, 8, 64],
  origin_node=mul_6,
  origins=OrderedSet([mul_6, mul_4, view_15, add_tensor_1, add_...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 408, in prog_transformer_block,
      h = torch.nn.functional.gelu(torch.nn.functional.linear(z, ff1_w, ff1_b)),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ExternKernelOut(
  python_kernel_name='extern_kernels.mm',
  name=buf21,
  layout=FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1]),
  inputs=[ReinterpretView(
    StorageBox(
      ComputedBuffer(name='buf20', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 64], stride=[512, 64, 1]), data=Pointwise(
        'cuda',
        torch.float32,
        def inner_fn(index):
            i0, i1, i2 = index
            tmp0 = ops.load(arg10_1, i2)
            tmp1 = ops.load(buf19, i2 + 64 * i1 + 512 * i0)
            tmp2 = tmp0 + tmp1
            tmp3 = ops.constant(0.5, torch.float32)
            tmp4 = tmp2 * tmp3
            tmp5 = ops.load(arg10_1, i2)
            tmp6 = ops.load(buf19, i2 + 64 * i1 + 512 * i0)
            tmp7 = tmp5 + tmp6
            tmp8 = ops.constant(0.7071067811865476, torch.float32)
            tmp9 = tmp7 * tmp8
            tmp10 = ops.erf(tmp9)
            tmp11 = ops.constant(1, torch.float32)
            tmp12 = tmp10 + tmp11
            tmp13 = tmp4 * tmp12
            return tmp13
        ,
        ranges=[2, 8, 64],
        origin_node=mul_6,
        origins=OrderedSet([mul_6, mul_4, view_15, add_tensor_1, add_...,
        stack_traces = {,
          File "/tmp/ir_printer_examples.py", line 408, in prog_transformer_block,
            h = torch.nn.functional.gelu(torch.nn.functional.linear(z, ff1_w, ff1_b)),
        ,
        }
      ), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
    ),
    FixedLayout('cuda:0', torch.float32, size=[16, 64], stride=[64, 1]),
    origins=OrderedSet([mm_default, view_16, mul_6, mul_4, view_1...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 409, in prog_transformer_block,
        x = x + torch.nn.functional.linear(h, ff2_w, ff2_b),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 408, in prog_transformer_block,
        h = torch.nn.functional.gelu(torch.nn.functional.linear(z, ff1_w, ff1_b)),
    ,
    }
  ), ReinterpretView(
    StorageBox(
      InputBuffer(name='arg11_1', layout=FixedLayout('cuda:0', torch.float32, size=[16, 64], stride=[64, 1]))
    ),
    FixedLayout('cuda:0', torch.float32, size=[64, 16], stride=[1, 64]),
    origins=OrderedSet([mm_default, view_16, mul_6, mul_4, view_1...,
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 409, in prog_transformer_block,
        x = x + torch.nn.functional.linear(h, ff2_w, ff2_b),
    ,
    },
    stack_traces = {,
      File "/tmp/ir_printer_examples.py", line 408, in prog_transformer_block,
        h = torch.nn.functional.gelu(torch.nn.functional.linear(z, ff1_w, ff1_b)),
    ,
    }
  )],
  constant_args=(),
  kwargs={},
  output_view=None,
  python_kernel_name=extern_kernels.mm,
  cpp_kernel_name=at::mm_out,
  ordered_kwargs_for_cpp_kernel=['out'],
  op_overload=aten.mm.out,
  arg_properties=[{'name': 'self', 'type': Tensor, 'default_value': None}, {'name': 'mat2', 'type': Tensor, 'default_value': None}],
  allarg_properties={'self': {'type': Tensor, 'default_value': None}, 'mat2': {'type': Tensor, 'default_value': None}, 'out': {'type': Tensor, 'default_value': None}},
  kwarg_properties=None,
  unbacked_bindings={},
  mutation_outputs=[],
  origin_node=mm_default,
  origins=OrderedSet([mm_default, view_16, mul_6, mul_4, view_1...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 409, in prog_transformer_block,
      x = x + torch.nn.functional.linear(h, ff2_w, ff2_b),
  ,
  }
)


ComputedBuffer(name='buf22', layout=FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      i0, i1, i2 = index
      tmp0 = ops.load(arg0_1, i2 + 16 * i1 + 128 * i0)
      tmp1 = ops.load(arg6_1, i2)
      tmp2 = ops.load(buf15, i2 + 16 * i1 + 128 * i0)
      tmp3 = tmp1 + tmp2
      tmp4 = tmp0 + tmp3
      tmp5 = ops.load(arg12_1, i2)
      tmp6 = ops.load(buf21, i2 + 16 * i1 + 128 * i0)
      tmp7 = tmp5 + tmp6
      tmp8 = tmp4 + tmp7
      return tmp8
  ,
  ranges=[2, 8, 16],
  origin_node=add_6,
  origins=OrderedSet([add_6, add_2, view_13, add_tensor_2, view...,
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 409, in prog_transformer_block,
      x = x + torch.nn.functional.linear(h, ff2_w, ff2_b),
  ,
  },
  stack_traces = {,
    File "/tmp/ir_printer_examples.py", line 402, in prog_transformer_block,
      x = x + torch.nn.functional.linear(ctx, out_w, out_b),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# inputs
arg0_1: f32[2,8,16], arg1_1: f32[16], arg2_1: f32[16], arg3_1: f32[48,16], arg4_1: f32[48],
arg5_1: f32[16,16], arg6_1: f32[16], arg7_1: f32[16], arg8_1: f32[16], arg9_1: f32[64,16],
arg10_1: f32[64], arg11_1: f32[16,64], arg12_1: f32[16]

# compute

# op0  sum(arg0_1[128*p0 + 16*p1 + p2] over p2)
buf0: f32[2,8,1]  <-  arg0_1: f32[2,8,16]
    foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,16):
        t0 = arg0_1[128*p0 + 16*p1 + p2]
        r0 = sum(t0 over p2)
        buf0[8*p0 + p1] = r0

# op1  sum(((arg0_1[128*p0 + 16*p1 + p2]) - ((buf0[8*p0 + p1]) / 16.0)) * ((arg0_1[128*p0 + 16*p1 + p2]) - ((buf0[8*p0 + p1]) / 16.0)) over p2)
buf1: f32[2,8,1]  <-  arg0_1: f32[2,8,16], buf0: f32[2,8,1]
    foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,16):
        t0 = arg0_1[128*p0 + 16*p1 + p2]
        ix0 = 8*p0 + p1
        t1 = buf0[ix0]
        t2 = t1 / 16.0
        t3 = t0 - t2
        t4 = t3 * t3
        r0 = sum(t4 over p2)
        buf1[ix0] = r0

# op2  ((((arg0_1[128*p0 + 16*p1 + p2]) - ((buf0[8*p0 + p1]) / 16.0)) * rsqrt(((buf1[8*p0 + p1]) / 16.0) + 1e-05)) * arg1_1[p2]) + arg2_1[p2]
buf2: f32[2,8,16]  <-  arg0_1: f32[2,8,16], buf0: f32[2,8,1], buf1: f32[2,8,1], arg1_1: f32[16], arg2_1: f32[16]
    foreach p0 in [0,2), p1 in [0,8), p2 in [0,16):
        ix0 = 128*p0 + 16*p1 + p2
        t0 = arg0_1[ix0]
        ix1 = 8*p0 + p1
        t1 = buf0[ix1]
        t2 = t1 / 16.0
        t3 = t0 - t2
        t4 = buf1[ix1]
        t5 = t4 / 16.0
        t6 = t5 + 1e-05
        t7 = rsqrt(t6)
        t8 = t3 * t7
        t9 = arg1_1[p2]
        t10 = t8 * t9
        t11 = arg2_1[p2]
        t12 = t10 + t11
        buf2[ix0] = t12

# op3
buf3: f32[16,48]
    extern_kernels.mm(reinterpret_tensor(buf2, [16,16], [16,1], 0),
        reinterpret_tensor(arg3_1, [16,48], [1,16], 0))

# op4  (arg4_1[8*p1 + p3]) + (buf3[384*p0 + 8*p1 + 48*p2 + p3])
buf4: f32[2,2,8,8]  <-  arg4_1: f32[48], buf3: f32[16,48]
    foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), p3 in [0,8):
        t0 = arg4_1[8*p1 + p3]
        t1 = buf3[384*p0 + 8*p1 + 48*p2 + p3]
        t2 = t0 + t1
        buf4[128*p0 + 64*p1 + 8*p2 + p3] = t2

# op5  (arg4_1[8*p1 + p2 + 16]) + (buf3[384*p0 + 8*p1 + p2 + 48*p3 + 16])
buf5: f32[2,2,8,8]  <-  arg4_1: f32[48], buf3: f32[16,48]
    foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), p3 in [0,8):
        t0 = arg4_1[8*p1 + p2 + 16]
        t1 = buf3[384*p0 + 8*p1 + p2 + 48*p3 + 16]
        t2 = t0 + t1
        buf5[128*p0 + 64*p1 + 8*p2 + p3] = t2

# op6
buf6: f32[4,8,8]
    extern_kernels.bmm(reinterpret_tensor(buf4, [4,8,8], [64,8,1], 0),
        reinterpret_tensor(buf5, [4,8,8], [64,8,1], 0))

# op7  any(not ((((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373) == ((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373)) and (abs((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373) != inf)) over p3)
buf7: b8[2,2,8,1]  <-  buf6: f32[4,8,8]
    foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), reduce p3 in [0,8):
        ix0 = 128*p0 + 64*p1 + 8*p2 + p3
        t0 = buf6[ix0]
        t1 = t0 * 0.35355339059327373
        t2 = buf6[ix0]
        t3 = t2 * 0.35355339059327373
        t4 = t1 == t3
        t5 = buf6[ix0]
        t6 = t5 * 0.35355339059327373
        t7 = abs(t6)
        t8 = t7 != inf
        t9 = t4 and t8
        t10 = not t9
        r0 = any(t10 over p3)
        buf7[16*p0 + 8*p1 + p2] = r0

# op8  max((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 1.0 over p3)
buf8: f32[2,2,8,1]  <-  buf6: f32[4,8,8]
    foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), reduce p3 in [0,8):
        t0 = buf6[128*p0 + 64*p1 + 8*p2 + p3]
        t1 = t0 * 1.0
        r0 = max(t1 over p3)
        buf8[16*p0 + 8*p1 + p2] = r0

# op9  max((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373 over p3)
buf9: f32[2,2,8,1]  <-  buf6: f32[4,8,8]
    foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), reduce p3 in [0,8):
        t0 = buf6[128*p0 + 64*p1 + 8*p2 + p3]
        t1 = t0 * 0.35355339059327373
        r0 = max(t1 over p3)
        buf9[16*p0 + 8*p1 + p2] = r0

# op10  sum(exp(where(not (buf7[16*p0 + 8*p1 + p2]), (((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 1.0) - (buf8[16*p0 + 8*p1 + p2])) * 0.35355339059327373, ((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373) - (buf9[16*p0 + 8*p1 + p2]))) over p3)
buf10: f32[2,2,8,1]  <-  buf7: b8[2,2,8,1], buf6: f32[4,8,8], buf8: f32[2,2,8,1], buf9: f32[2,2,8,1]
    foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), reduce p3 in [0,8):
        ix0 = 16*p0 + 8*p1 + p2
        t0 = buf7[ix0]
        t1 = not t0
        ix1 = 128*p0 + 64*p1 + 8*p2 + p3
        t2 = buf6[ix1]
        t3 = t2 * 1.0
        t4 = buf8[ix0]
        t5 = t3 - t4
        t6 = t5 * 0.35355339059327373
        t7 = buf6[ix1]
        t8 = t7 * 0.35355339059327373
        t9 = buf9[ix0]
        t10 = t8 - t9
        t11 = where(t1, t6, t10)
        t12 = exp(t11)
        r0 = sum(t12 over p3)
        buf10[ix0] = r0

# op11  exp(where(not (buf7[16*p0 + 8*p1 + p2]), (((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 1.0) - (buf8[16*p0 + 8*p1 + p2])) * 0.35355339059327373, ((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373) - (buf9[16*p0 + 8*p1 + p2]))) / (buf10[16*p0 + 8*p1 + p2])
buf11: f32[2,2,8,8]  <-  buf7: b8[2,2,8,1], buf6: f32[4,8,8], buf8: f32[2,2,8,1], buf9: f32[2,2,8,1], buf10: f32[2,2,8,1]
    foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), p3 in [0,8):
        ix0 = 16*p0 + 8*p1 + p2
        t0 = buf7[ix0]
        t1 = not t0
        ix1 = 128*p0 + 64*p1 + 8*p2 + p3
        t2 = buf6[ix1]
        t3 = t2 * 1.0
        t4 = buf8[ix0]
        t5 = t3 - t4
        t6 = t5 * 0.35355339059327373
        t7 = buf6[ix1]
        t8 = t7 * 0.35355339059327373
        t9 = buf9[ix0]
        t10 = t8 - t9
        t11 = where(t1, t6, t10)
        t12 = exp(t11)
        t13 = buf10[ix0]
        t14 = t12 / t13
        buf11[ix1] = t14

# op12  (arg4_1[8*p1 + p3 + 32]) + (buf3[384*p0 + 8*p1 + 48*p2 + p3 + 32])
buf12: f32[2,2,8,8]  <-  arg4_1: f32[48], buf3: f32[16,48]
    foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), p3 in [0,8):
        t0 = arg4_1[8*p1 + p3 + 32]
        t1 = buf3[384*p0 + 8*p1 + 48*p2 + p3 + 32]
        t2 = t0 + t1
        buf12[128*p0 + 64*p1 + 8*p2 + p3] = t2

# op13
buf13: f32[4,8,8]
    extern_kernels.bmm(reinterpret_tensor(buf11, [4,8,8], [64,8,1], 0),
        reinterpret_tensor(buf12, [4,8,8], [64,8,1], 0))

# op14  copy
buf14: f32[2,8,2,8]  <-  buf13: f32[4,8,8]
    foreach p0 in [0,2), p1 in [0,8), p2 in [0,2), p3 in [0,8):
        t0 = buf13[128*p0 + 8*p1 + 64*p2 + p3]
        buf14[128*p0 + 16*p1 + 8*p2 + p3] = t0

# op15
buf15: f32[16,16]
    extern_kernels.mm(reinterpret_tensor(buf14, [16,16], [16,1], 0),
        reinterpret_tensor(arg5_1, [16,16], [1,16], 0))

# op16  sum((arg0_1[128*p0 + 16*p1 + p2]) + (arg6_1[p2] + (buf15[128*p0 + 16*p1 + p2])) over p2)
buf16: f32[2,8,1]  <-  arg0_1: f32[2,8,16], arg6_1: f32[16], buf15: f32[16,16]
    foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,16):
        ix0 = 128*p0 + 16*p1 + p2
        t0 = arg0_1[ix0]
        t1 = arg6_1[p2]
        t2 = buf15[ix0]
        t3 = t1 + t2
        t4 = t0 + t3
        r0 = sum(t4 over p2)
        buf16[8*p0 + p1] = r0

# op17  sum((((arg0_1[128*p0 + 16*p1 + p2]) + (arg6_1[p2] + (buf15[128*p0 + 16*p1 + p2]))) - ((buf16[8*p0 + p1]) / 16.0)) * (((arg0_1[128*p0 + 16*p1 + p2]) + (arg6_1[p2] + (buf15[128*p0 + 16*p1 + p2]))) - ((buf16[8*p0 + p1]) / 16.0)) over p2)
buf17: f32[2,8,1]  <-  arg0_1: f32[2,8,16], arg6_1: f32[16], buf15: f32[16,16], buf16: f32[2,8,1]
    foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,16):
        ix0 = 128*p0 + 16*p1 + p2
        t0 = arg0_1[ix0]
        t1 = arg6_1[p2]
        t2 = buf15[ix0]
        t3 = t1 + t2
        t4 = t0 + t3
        ix1 = 8*p0 + p1
        t5 = buf16[ix1]
        t6 = t5 / 16.0
        t7 = t4 - t6
        t8 = t7 * t7
        r0 = sum(t8 over p2)
        buf17[ix1] = r0

# op18  (((((arg0_1[128*p0 + 16*p1 + p2]) + (arg6_1[p2] + (buf15[128*p0 + 16*p1 + p2]))) - ((buf16[8*p0 + p1]) / 16.0)) * rsqrt(((buf17[8*p0 + p1]) / 16.0) + 1e-05)) * arg7_1[p2]) + arg8_1[p2]
buf18: f32[2,8,16]  <-  arg0_1: f32[2,8,16], arg6_1: f32[16], buf15: f32[16,16], buf16: f32[2,8,1], buf17: f32[2,8,1], arg7_1: f32[16], arg8_1: f32[16]
    foreach p0 in [0,2), p1 in [0,8), p2 in [0,16):
        ix0 = 128*p0 + 16*p1 + p2
        t0 = arg0_1[ix0]
        t1 = arg6_1[p2]
        t2 = buf15[ix0]
        t3 = t1 + t2
        t4 = t0 + t3
        ix1 = 8*p0 + p1
        t5 = buf16[ix1]
        t6 = t5 / 16.0
        t7 = t4 - t6
        t8 = buf17[ix1]
        t9 = t8 / 16.0
        t10 = t9 + 1e-05
        t11 = rsqrt(t10)
        t12 = t7 * t11
        t13 = arg7_1[p2]
        t14 = t12 * t13
        t15 = arg8_1[p2]
        t16 = t14 + t15
        buf18[ix0] = t16

# op19
buf19: f32[16,64]
    extern_kernels.mm(reinterpret_tensor(buf18, [16,16], [16,1], 0),
        reinterpret_tensor(arg9_1, [16,64], [1,16], 0))

# op20  ((arg10_1[p2] + (buf19[512*p0 + 64*p1 + p2])) * 0.5) * (erf((arg10_1[p2] + (buf19[512*p0 + 64*p1 + p2])) * 0.7071067811865476) + 1.0)
buf20: f32[2,8,64]  <-  arg10_1: f32[64], buf19: f32[16,64]
    foreach p0 in [0,2), p1 in [0,8), p2 in [0,64):
        t0 = arg10_1[p2]
        ix0 = 512*p0 + 64*p1 + p2
        t1 = buf19[ix0]
        t2 = t0 + t1
        t3 = t2 * 0.5
        t4 = arg10_1[p2]
        t5 = buf19[ix0]
        t6 = t4 + t5
        t7 = t6 * 0.7071067811865476
        t8 = erf(t7)
        t9 = t8 + 1.0
        t10 = t3 * t9
        buf20[ix0] = t10

# op21
buf21: f32[16,16]
    extern_kernels.mm(reinterpret_tensor(buf20, [16,64], [64,1], 0),
        reinterpret_tensor(arg11_1, [64,16], [1,64], 0))

# op22  ((arg0_1[128*p0 + 16*p1 + p2]) + (arg6_1[p2] + (buf15[128*p0 + 16*p1 + p2]))) + (arg12_1[p2] + (buf21[128*p0 + 16*p1 + p2]))
buf22: f32[2,8,16]  <-  arg0_1: f32[2,8,16], arg6_1: f32[16], buf15: f32[16,16], arg12_1: f32[16], buf21: f32[16,16]
    foreach p0 in [0,2), p1 in [0,8), p2 in [0,16):
        ix0 = 128*p0 + 16*p1 + p2
        t0 = arg0_1[ix0]
        t1 = arg6_1[p2]
        t2 = buf15[ix0]
        t3 = t1 + t2
        t4 = t0 + t3
        t5 = arg12_1[p2]
        t6 = buf21[ix0]
        t7 = t5 + t6
        t8 = t4 + t7
        buf22[ix0] = t8

# outputs
return (buf22)
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 8*d0 + d1, {d0: 2, d1: 8})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf0.users = [
        NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False),
    ]
]
op0.group.device = cuda:0
op0.group.iteration = (16, 16)
op0.sizes = ([2, 8], [16])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op0_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 16}
    index0 = 128*p0 + 16*p1 + p2
    index1 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', 8*d0 + d1, {d0: 2, d1: 8})]
op1.unmet_dependencies = [MemoryDep('buf0', 8*d0 + d1, {d0: 2, d1: 8})]
op1.met_dependencies = [MemoryDep('arg0_1', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (16, 16)
op1.sizes = ([2, 8], [16])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op1_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 16}
    index0 = 128*p0 + 16*p1 + p2
    index1 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(16.0, torch.float32)
        truediv = ops.truediv(load_1, constant)
        sub = ops.sub(load, truediv)
        mul = ops.mul(sub, sub)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', mul)
        get_index_2 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf1', get_index_2, reduction)
        return None


op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16})]
op2.unmet_dependencies =
    [   MemoryDep('buf0', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf1', 8*d0 + d1, {d0: 2, d1: 8})]
op2.met_dependencies =
    [   MemoryDep('arg0_1', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg1_1', d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg2_1', d2, {d0: 2, d1: 8, d2: 16})]
op2.min_input_distance = 1
op2.max_input_distance = 2
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
    buf2.users = [NodeUser(node=ExternKernelSchedulerNode(name='op3'), can_inplace=False, is_weak=False)]
]
op2.group.device = cuda:0
op2.group.iteration = (256, 1)
op2.sizes = ([2, 8, 16], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
arg2_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
class op2_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 16}
    index0 = 128*p0 + 16*p1 + p2
    index1 = 8*p0 + p1
    index2 = p2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(16.0, torch.float32)
        truediv = ops.truediv(load_1, constant)
        sub = ops.sub(load, truediv)
        get_index_2 = self.get_index('index1')
        load_2 = ops.load('buf1', get_index_2)
        constant_1 = ops.constant(16.0, torch.float32)
        truediv_1 = ops.truediv(load_2, constant_1)
        constant_2 = ops.constant(1e-05, torch.float32)
        add = ops.add(truediv_1, constant_2)
        rsqrt = ops.rsqrt(add)
        mul = ops.mul(sub, rsqrt)
        get_index_3 = self.get_index('index2')
        load_3 = ops.load('arg1_1', get_index_3)
        mul_1 = ops.mul(mul, load_3)
        get_index_4 = self.get_index('index2')
        load_4 = ops.load('arg2_1', get_index_4)
        add_1 = ops.add(mul_1, load_4)
        get_index_5 = self.get_index('index0')
        store = ops.store('buf2', get_index_5, add_1, None)
        return store


op3: ExternKernelSchedulerNode(ExternKernelOut)
op3.writes = [StarDep(name='buf3', mode=None)]
op3.unmet_dependencies = [StarDep(name='buf2', mode=None)]
op3.met_dependencies = [StarDep(name='arg3_1', mode=None)]
op3.min_input_distance = 2
op3.max_input_distance = 3
op3.outputs = [
    buf3: ExternKernelOut
    buf3.layout = FixedLayout('cuda:0', torch.float32, size=[16, 48], stride=[48, 1])
    buf3.users = [
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op12'), can_inplace=False, is_weak=False),
    ]
]
op3.node.kernel = extern_kernels.mm


op4: SchedulerNode(ComputedBuffer)
op4.writes = [MemoryDep('buf4', 128*d0 + 64*d1 + 8*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8})]
op4.unmet_dependencies = [MemoryDep('buf3', 384*d0 + 8*d1 + 48*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8})]
op4.met_dependencies = [MemoryDep('arg4_1', 8*d1 + d3, {d0: 2, d1: 2, d2: 8, d3: 8})]
op4.min_input_distance = 3
op4.max_input_distance = 4
op4.outputs = [
    buf4: ComputedBuffer
    buf4.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
    buf4.users = [NodeUser(node=ExternKernelSchedulerNode(name='op6'), can_inplace=False, is_weak=False)]
]
op4.group.device = cuda:0
op4.group.iteration = (256, 1)
op4.sizes = ([2, 2, 8, 8], [])
arg4_1_layout = FixedLayout('cuda:0', torch.float32, size=[48], stride=[1])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[16, 48], stride=[48, 1])
buf4_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
class op4_loop_body:
    var_ranges = {p0: 2, p1: 2, p2: 8, p3: 8}
    index0 = 8*p1 + p3
    index1 = 384*p0 + 8*p1 + 48*p2 + p3
    index2 = 128*p0 + 64*p1 + 8*p2 + p3
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg4_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf3', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index2')
        store = ops.store('buf4', get_index_2, add, None)
        return store


op5: SchedulerNode(ComputedBuffer)
op5.writes = [MemoryDep('buf5', 128*d0 + 64*d1 + 8*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8})]
op5.unmet_dependencies = [   MemoryDep('buf3', 384*d0 + 8*d1 + d2 + 48*d3 + 16, {d0: 2, d1: 2, d2: 8, d3: 8})]
op5.met_dependencies = [MemoryDep('arg4_1', 8*d1 + d2 + 16, {d0: 2, d1: 2, d2: 8})]
op5.min_input_distance = 3
op5.max_input_distance = 4
op5.outputs = [
    buf5: ComputedBuffer
    buf5.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
    buf5.users = [NodeUser(node=ExternKernelSchedulerNode(name='op6'), can_inplace=False, is_weak=False)]
]
op5.group.device = cuda:0
op5.group.iteration = (256, 1)
op5.sizes = ([2, 2, 8, 8], [])
arg4_1_layout = FixedLayout('cuda:0', torch.float32, size=[48], stride=[1])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[16, 48], stride=[48, 1])
buf5_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
class op5_loop_body:
    var_ranges = {p0: 2, p1: 2, p2: 8, p3: 8}
    index0 = 8*p1 + p2 + 16
    index1 = 384*p0 + 8*p1 + p2 + 48*p3 + 16
    index2 = 128*p0 + 64*p1 + 8*p2 + p3
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg4_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf3', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index2')
        store = ops.store('buf5', get_index_2, add, None)
        return store


op6: ExternKernelSchedulerNode(ExternKernelOut)
op6.writes = [StarDep(name='buf6', mode=None)]
op6.unmet_dependencies = [StarDep(name='buf4', mode=None), StarDep(name='buf5', mode=None)]
op6.met_dependencies = []
op6.min_input_distance = 4
op6.max_input_distance = 5
op6.outputs = [
    buf6: ExternKernelOut
    buf6.layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
    buf6.users = [
        NodeUser(node=SchedulerNode(name='op7'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op8'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op9'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op10'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op11'), can_inplace=True, is_weak=False),
    ]
]
op6.node.kernel = extern_kernels.bmm


op7: SchedulerNode(ComputedBuffer)
op7.writes = [MemoryDep('buf7', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8})]
op7.unmet_dependencies = [MemoryDep('buf6', 128*d0 + 64*d1 + 8*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8})]
op7.met_dependencies = []
op7.min_input_distance = 5
op7.max_input_distance = 6
op7.outputs = [
    buf7: ComputedBuffer
    buf7.layout = FixedLayout('cuda:0', torch.bool, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf7.users = [
        NodeUser(node=SchedulerNode(name='op10'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False),
    ]
]
op7.group.device = cuda:0
op7.group.iteration = (32, 8)
op7.sizes = ([2, 2, 8], [8])
buf6_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf7_layout = FixedLayout('cuda:0', torch.bool, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
class op7_loop_body:
    var_ranges = {p0: 2, p1: 2, p2: 8, p3: 8}
    index0 = 128*p0 + 64*p1 + 8*p2 + p3
    index1 = 16*p0 + 8*p1 + p2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf6', get_index)
        constant = ops.constant(0.35355339059327373, torch.float32)
        mul = ops.mul(load, constant)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('buf6', get_index_1)
        constant_1 = ops.constant(0.35355339059327373, torch.float32)
        mul_1 = ops.mul(load_1, constant_1)
        eq = ops.eq(mul, mul_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf6', get_index_2)
        constant_2 = ops.constant(0.35355339059327373, torch.float32)
        mul_2 = ops.mul(load_2, constant_2)
        abs_1 = ops.abs(mul_2)
        constant_3 = ops.constant(inf, torch.float32)
        ne = ops.ne(abs_1, constant_3)
        logical_and = ops.logical_and(eq, ne)
        logical_not = ops.logical_not(logical_and)
        reduction = ops.reduction(torch.bool, torch.bool, 'any', logical_not)
        get_index_3 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf7', get_index_3, reduction)
        return None


op8: SchedulerNode(ComputedBuffer)
op8.writes = [MemoryDep('buf8', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8})]
op8.unmet_dependencies = [MemoryDep('buf6', 128*d0 + 64*d1 + 8*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8})]
op8.met_dependencies = []
op8.min_input_distance = 5
op8.max_input_distance = 6
op8.outputs = [
    buf8: ComputedBuffer
    buf8.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf8.users = [
        NodeUser(node=SchedulerNode(name='op10'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False),
    ]
]
op8.group.device = cuda:0
op8.group.iteration = (32, 8)
op8.sizes = ([2, 2, 8], [8])
buf6_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf8_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
class op8_loop_body:
    var_ranges = {p0: 2, p1: 2, p2: 8, p3: 8}
    index0 = 128*p0 + 64*p1 + 8*p2 + p3
    index1 = 16*p0 + 8*p1 + p2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf6', get_index)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load, constant)
        reduction = ops.reduction(torch.float32, torch.float32, 'max', mul)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf8', get_index_1, reduction)
        return None


op9: SchedulerNode(ComputedBuffer)
op9.writes = [MemoryDep('buf9', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8})]
op9.unmet_dependencies = [MemoryDep('buf6', 128*d0 + 64*d1 + 8*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8})]
op9.met_dependencies = []
op9.min_input_distance = 5
op9.max_input_distance = 6
op9.outputs = [
    buf9: ComputedBuffer
    buf9.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf9.users = [
        NodeUser(node=SchedulerNode(name='op10'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False),
    ]
]
op9.group.device = cuda:0
op9.group.iteration = (32, 8)
op9.sizes = ([2, 2, 8], [8])
buf6_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf9_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
class op9_loop_body:
    var_ranges = {p0: 2, p1: 2, p2: 8, p3: 8}
    index0 = 128*p0 + 64*p1 + 8*p2 + p3
    index1 = 16*p0 + 8*p1 + p2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf6', get_index)
        constant = ops.constant(0.35355339059327373, torch.float32)
        mul = ops.mul(load, constant)
        reduction = ops.reduction(torch.float32, torch.float32, 'max', mul)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf9', get_index_1, reduction)
        return None


op10: SchedulerNode(ComputedBuffer)
op10.writes = [MemoryDep('buf10', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8})]
op10.unmet_dependencies =
    [   MemoryDep('buf6', 128*d0 + 64*d1 + 8*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8}),
        MemoryDep('buf7', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8}),
        MemoryDep('buf8', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8}),
        MemoryDep('buf9', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8})]
op10.met_dependencies = []
op10.min_input_distance = 5
op10.max_input_distance = 7
op10.outputs = [
    buf10: ComputedBuffer
    buf10.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf10.users = [NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False)]
]
op10.group.device = cuda:0
op10.group.iteration = (32, 8)
op10.sizes = ([2, 2, 8], [8])
buf7_layout = FixedLayout('cuda:0', torch.bool, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf6_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf8_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf9_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf10_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
class op10_loop_body:
    var_ranges = {p0: 2, p1: 2, p2: 8, p3: 8}
    index0 = 16*p0 + 8*p1 + p2
    index1 = 128*p0 + 64*p1 + 8*p2 + p3
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf7', get_index)
        logical_not = ops.logical_not(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf6', get_index_1)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load_1, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf8', get_index_2)
        sub = ops.sub(mul, load_2)
        constant_1 = ops.constant(0.35355339059327373, torch.float32)
        mul_1 = ops.mul(sub, constant_1)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf6', get_index_3)
        constant_2 = ops.constant(0.35355339059327373, torch.float32)
        mul_2 = ops.mul(load_3, constant_2)
        get_index_4 = self.get_index('index0')
        load_4 = ops.load('buf9', get_index_4)
        sub_1 = ops.sub(mul_2, load_4)
        where = ops.where(logical_not, mul_1, sub_1)
        exp = ops.exp(where)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', exp)
        get_index_5 = self.get_index('index0')
        store_reduction = ops.store_reduction('buf10', get_index_5, reduction)
        return None


op11: SchedulerNode(ComputedBuffer)
op11.writes = [MemoryDep('buf11', 128*d0 + 64*d1 + 8*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8})]
op11.unmet_dependencies =
    [   MemoryDep('buf10', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8}),
        MemoryDep('buf6', 128*d0 + 64*d1 + 8*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8}),
        MemoryDep('buf7', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8}),
        MemoryDep('buf8', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8}),
        MemoryDep('buf9', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8})]
op11.met_dependencies = []
op11.min_input_distance = 5
op11.max_input_distance = 8
op11.outputs = [
    buf11: ComputedBuffer
    buf11.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
    buf11.users = [NodeUser(node=ExternKernelSchedulerNode(name='op13'), can_inplace=False, is_weak=False)]
]
op11.group.device = cuda:0
op11.group.iteration = (256, 1)
op11.sizes = ([2, 2, 8, 8], [])
buf7_layout = FixedLayout('cuda:0', torch.bool, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf6_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf8_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf9_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf10_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf11_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
class op11_loop_body:
    var_ranges = {p0: 2, p1: 2, p2: 8, p3: 8}
    index0 = 16*p0 + 8*p1 + p2
    index1 = 128*p0 + 64*p1 + 8*p2 + p3
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf7', get_index)
        logical_not = ops.logical_not(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf6', get_index_1)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load_1, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf8', get_index_2)
        sub = ops.sub(mul, load_2)
        constant_1 = ops.constant(0.35355339059327373, torch.float32)
        mul_1 = ops.mul(sub, constant_1)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf6', get_index_3)
        constant_2 = ops.constant(0.35355339059327373, torch.float32)
        mul_2 = ops.mul(load_3, constant_2)
        get_index_4 = self.get_index('index0')
        load_4 = ops.load('buf9', get_index_4)
        sub_1 = ops.sub(mul_2, load_4)
        where = ops.where(logical_not, mul_1, sub_1)
        exp = ops.exp(where)
        get_index_5 = self.get_index('index0')
        load_5 = ops.load('buf10', get_index_5)
        truediv = ops.truediv(exp, load_5)
        get_index_6 = self.get_index('index1')
        store = ops.store('buf11', get_index_6, truediv, None)
        return store


op12: SchedulerNode(ComputedBuffer)
op12.writes = [MemoryDep('buf12', 128*d0 + 64*d1 + 8*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8})]
op12.unmet_dependencies = [   MemoryDep('buf3', 384*d0 + 8*d1 + 48*d2 + d3 + 32, {d0: 2, d1: 2, d2: 8, d3: 8})]
op12.met_dependencies = [MemoryDep('arg4_1', 8*d1 + d3 + 32, {d0: 2, d1: 2, d2: 8, d3: 8})]
op12.min_input_distance = 3
op12.max_input_distance = 4
op12.outputs = [
    buf12: ComputedBuffer
    buf12.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
    buf12.users = [NodeUser(node=ExternKernelSchedulerNode(name='op13'), can_inplace=False, is_weak=False)]
]
op12.group.device = cuda:0
op12.group.iteration = (256, 1)
op12.sizes = ([2, 2, 8, 8], [])
arg4_1_layout = FixedLayout('cuda:0', torch.float32, size=[48], stride=[1])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[16, 48], stride=[48, 1])
buf12_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
class op12_loop_body:
    var_ranges = {p0: 2, p1: 2, p2: 8, p3: 8}
    index0 = 8*p1 + p3 + 32
    index1 = 384*p0 + 8*p1 + 48*p2 + p3 + 32
    index2 = 128*p0 + 64*p1 + 8*p2 + p3
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg4_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf3', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index2')
        store = ops.store('buf12', get_index_2, add, None)
        return store


op13: ExternKernelSchedulerNode(ExternKernelOut)
op13.writes = [StarDep(name='buf13', mode=None)]
op13.unmet_dependencies = [StarDep(name='buf11', mode=None), StarDep(name='buf12', mode=None)]
op13.met_dependencies = []
op13.min_input_distance = 4
op13.max_input_distance = 9
op13.outputs = [
    buf13: ExternKernelOut
    buf13.layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
    buf13.users = [NodeUser(node=SchedulerNode(name='op14'), can_inplace=False, is_weak=False)]
]
op13.node.kernel = extern_kernels.bmm


op14: SchedulerNode(ComputedBuffer)
op14.writes = [MemoryDep('buf14', 128*d0 + 16*d1 + 8*d2 + d3, {d0: 2, d1: 8, d2: 2, d3: 8})]
op14.unmet_dependencies = [MemoryDep('buf13', 128*d0 + 8*d1 + 64*d2 + d3, {d0: 2, d1: 8, d2: 2, d3: 8})]
op14.met_dependencies = []
op14.min_input_distance = 5
op14.max_input_distance = 10
op14.outputs = [
    buf14: ComputedBuffer
    buf14.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 2, 8], stride=[128, 16, 8, 1])
    buf14.users = [NodeUser(node=ExternKernelSchedulerNode(name='op15'), can_inplace=False, is_weak=False)]
]
op14.group.device = cuda:0
op14.group.iteration = (256, 1)
op14.sizes = ([2, 8, 2, 8], [])
buf13_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf14_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 2, 8], stride=[128, 16, 8, 1])
class op14_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 2, p3: 8}
    index0 = 128*p0 + 8*p1 + 64*p2 + p3
    index1 = 128*p0 + 16*p1 + 8*p2 + p3
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf13', get_index)
        get_index_1 = self.get_index('index1')
        store = ops.store('buf14', get_index_1, load, None)
        return store


op15: ExternKernelSchedulerNode(ExternKernelOut)
op15.writes = [StarDep(name='buf15', mode=None)]
op15.unmet_dependencies = [StarDep(name='buf14', mode=None)]
op15.met_dependencies = [StarDep(name='arg5_1', mode=None)]
op15.min_input_distance = 6
op15.max_input_distance = 11
op15.outputs = [
    buf15: ExternKernelOut
    buf15.layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
    buf15.users = [
        NodeUser(node=SchedulerNode(name='op16'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op17'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op18'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op22'), can_inplace=True, is_weak=False),
    ]
]
op15.node.kernel = extern_kernels.mm


op16: SchedulerNode(ComputedBuffer)
op16.writes = [MemoryDep('buf16', 8*d0 + d1, {d0: 2, d1: 8})]
op16.unmet_dependencies = [MemoryDep('buf15', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16})]
op16.met_dependencies =
    [   MemoryDep('arg0_1', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg6_1', d2, {d0: 2, d1: 8, d2: 16})]
op16.min_input_distance = 7
op16.max_input_distance = 12
op16.outputs = [
    buf16: ComputedBuffer
    buf16.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf16.users = [
        NodeUser(node=SchedulerNode(name='op17'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op18'), can_inplace=False, is_weak=False),
    ]
]
op16.group.device = cuda:0
op16.group.iteration = (16, 16)
op16.sizes = ([2, 8], [16])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
arg6_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf15_layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
buf16_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op16_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 16}
    index0 = 128*p0 + 16*p1 + p2
    index1 = p2
    index2 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg6_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf15', get_index_2)
        add = ops.add(load_1, load_2)
        add_1 = ops.add(load, add)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', add_1)
        get_index_3 = self.get_index('index2')
        store_reduction = ops.store_reduction('buf16', get_index_3, reduction)
        return None


op17: SchedulerNode(ComputedBuffer)
op17.writes = [MemoryDep('buf17', 8*d0 + d1, {d0: 2, d1: 8})]
op17.unmet_dependencies =
    [   MemoryDep('buf15', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('buf16', 8*d0 + d1, {d0: 2, d1: 8})]
op17.met_dependencies =
    [   MemoryDep('arg0_1', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg6_1', d2, {d0: 2, d1: 8, d2: 16})]
op17.min_input_distance = 7
op17.max_input_distance = 13
op17.outputs = [
    buf17: ComputedBuffer
    buf17.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf17.users = [NodeUser(node=SchedulerNode(name='op18'), can_inplace=False, is_weak=False)]
]
op17.group.device = cuda:0
op17.group.iteration = (16, 16)
op17.sizes = ([2, 8], [16])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
arg6_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf15_layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
buf16_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf17_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op17_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 16}
    index0 = 128*p0 + 16*p1 + p2
    index1 = p2
    index2 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg6_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf15', get_index_2)
        add = ops.add(load_1, load_2)
        add_1 = ops.add(load, add)
        get_index_3 = self.get_index('index2')
        load_3 = ops.load('buf16', get_index_3)
        constant = ops.constant(16.0, torch.float32)
        truediv = ops.truediv(load_3, constant)
        sub = ops.sub(add_1, truediv)
        mul = ops.mul(sub, sub)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', mul)
        get_index_4 = self.get_index('index2')
        store_reduction = ops.store_reduction('buf17', get_index_4, reduction)
        return None


op18: SchedulerNode(ComputedBuffer)
op18.writes = [MemoryDep('buf18', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16})]
op18.unmet_dependencies =
    [   MemoryDep('buf15', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('buf16', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf17', 8*d0 + d1, {d0: 2, d1: 8})]
op18.met_dependencies =
    [   MemoryDep('arg0_1', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg6_1', d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg7_1', d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg8_1', d2, {d0: 2, d1: 8, d2: 16})]
op18.min_input_distance = 7
op18.max_input_distance = 14
op18.outputs = [
    buf18: ComputedBuffer
    buf18.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
    buf18.users = [NodeUser(node=ExternKernelSchedulerNode(name='op19'), can_inplace=False, is_weak=False)]
]
op18.group.device = cuda:0
op18.group.iteration = (256, 1)
op18.sizes = ([2, 8, 16], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
arg6_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf15_layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
buf16_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf17_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
arg7_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
arg8_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf18_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
class op18_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 16}
    index0 = 128*p0 + 16*p1 + p2
    index1 = p2
    index2 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg6_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf15', get_index_2)
        add = ops.add(load_1, load_2)
        add_1 = ops.add(load, add)
        get_index_3 = self.get_index('index2')
        load_3 = ops.load('buf16', get_index_3)
        constant = ops.constant(16.0, torch.float32)
        truediv = ops.truediv(load_3, constant)
        sub = ops.sub(add_1, truediv)
        get_index_4 = self.get_index('index2')
        load_4 = ops.load('buf17', get_index_4)
        constant_1 = ops.constant(16.0, torch.float32)
        truediv_1 = ops.truediv(load_4, constant_1)
        constant_2 = ops.constant(1e-05, torch.float32)
        add_2 = ops.add(truediv_1, constant_2)
        rsqrt = ops.rsqrt(add_2)
        mul = ops.mul(sub, rsqrt)
        get_index_5 = self.get_index('index1')
        load_5 = ops.load('arg7_1', get_index_5)
        mul_1 = ops.mul(mul, load_5)
        get_index_6 = self.get_index('index1')
        load_6 = ops.load('arg8_1', get_index_6)
        add_3 = ops.add(mul_1, load_6)
        get_index_7 = self.get_index('index0')
        store = ops.store('buf18', get_index_7, add_3, None)
        return store


op19: ExternKernelSchedulerNode(ExternKernelOut)
op19.writes = [StarDep(name='buf19', mode=None)]
op19.unmet_dependencies = [StarDep(name='buf18', mode=None)]
op19.met_dependencies = [StarDep(name='arg9_1', mode=None)]
op19.min_input_distance = 8
op19.max_input_distance = 15
op19.outputs = [
    buf19: ExternKernelOut
    buf19.layout = FixedLayout('cuda:0', torch.float32, size=[16, 64], stride=[64, 1])
    buf19.users = [NodeUser(node=SchedulerNode(name='op20'), can_inplace=True, is_weak=False)]
]
op19.node.kernel = extern_kernels.mm


op20: SchedulerNode(ComputedBuffer)
op20.writes = [MemoryDep('buf20', 512*d0 + 64*d1 + d2, {d0: 2, d1: 8, d2: 64})]
op20.unmet_dependencies = [MemoryDep('buf19', 512*d0 + 64*d1 + d2, {d0: 2, d1: 8, d2: 64})]
op20.met_dependencies = [MemoryDep('arg10_1', d2, {d0: 2, d1: 8, d2: 64})]
op20.min_input_distance = 9
op20.max_input_distance = 16
op20.outputs = [
    buf20: ComputedBuffer
    buf20.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 64], stride=[512, 64, 1])
    buf20.users = [NodeUser(node=ExternKernelSchedulerNode(name='op21'), can_inplace=False, is_weak=False)]
]
op20.group.device = cuda:0
op20.group.iteration = (1024, 1)
op20.sizes = ([2, 8, 64], [])
arg10_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
buf19_layout = FixedLayout('cuda:0', torch.float32, size=[16, 64], stride=[64, 1])
buf20_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 64], stride=[512, 64, 1])
class op20_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 64}
    index0 = p2
    index1 = 512*p0 + 64*p1 + p2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg10_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf19', get_index_1)
        add = ops.add(load, load_1)
        constant = ops.constant(0.5, torch.float32)
        mul = ops.mul(add, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('arg10_1', get_index_2)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf19', get_index_3)
        add_1 = ops.add(load_2, load_3)
        constant_1 = ops.constant(0.7071067811865476, torch.float32)
        mul_1 = ops.mul(add_1, constant_1)
        erf = ops.erf(mul_1)
        constant_2 = ops.constant(1.0, torch.float32)
        add_2 = ops.add(erf, constant_2)
        mul_2 = ops.mul(mul, add_2)
        get_index_4 = self.get_index('index1')
        store = ops.store('buf20', get_index_4, mul_2, None)
        return store


op21: ExternKernelSchedulerNode(ExternKernelOut)
op21.writes = [StarDep(name='buf21', mode=None)]
op21.unmet_dependencies = [StarDep(name='buf20', mode=None)]
op21.met_dependencies = [StarDep(name='arg11_1', mode=None)]
op21.min_input_distance = 10
op21.max_input_distance = 17
op21.outputs = [
    buf21: ExternKernelOut
    buf21.layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
    buf21.users = [NodeUser(node=SchedulerNode(name='op22'), can_inplace=True, is_weak=False)]
]
op21.node.kernel = extern_kernels.mm


op22: SchedulerNode(ComputedBuffer)
op22.writes = [MemoryDep('buf22', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16})]
op22.unmet_dependencies =
    [   MemoryDep('buf15', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('buf21', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16})]
op22.met_dependencies =
    [   MemoryDep('arg0_1', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg12_1', d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg6_1', d2, {d0: 2, d1: 8, d2: 16})]
op22.min_input_distance = 7
op22.max_input_distance = 18
op22.outputs = [
    buf22: ComputedBuffer
    buf22.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
    buf22.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op22.group.device = cuda:0
op22.group.iteration = (256, 1)
op22.sizes = ([2, 8, 16], [])
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
arg6_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf15_layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
arg12_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf21_layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
buf22_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
class op22_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 16}
    index0 = 128*p0 + 16*p1 + p2
    index1 = p2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg6_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf15', get_index_2)
        add = ops.add(load_1, load_2)
        add_1 = ops.add(load, add)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('arg12_1', get_index_3)
        get_index_4 = self.get_index('index0')
        load_4 = ops.load('buf21', get_index_4)
        add_2 = ops.add(load_3, load_4)
        add_3 = ops.add(add_1, add_2)
        get_index_5 = self.get_index('index0')
        store = ops.store('buf22', get_index_5, add_3, None)
        return store
```

### formatted_post_scheduler

```text
# inputs
arg0_1: f32[2,8,16], arg1_1: f32[16], arg2_1: f32[16], arg3_1: f32[48,16], arg4_1: f32[48],
arg5_1: f32[16,16], arg6_1: f32[16], arg7_1: f32[16], arg8_1: f32[16], arg9_1: f32[64,16],
arg10_1: f32[64], arg11_1: f32[16,64], arg12_1: f32[16]

# compute

kernel k0
    inputs:   [arg0_1]
    outputs:  [buf0]

    op0:  buf0: f32[2,8,1]  <-  arg0_1: f32[2,8,16]      # sum(arg0_1[128*p0 + 16*p1 + p2] over p2)
        foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,16):
            t0 = arg0_1[128*p0 + 16*p1 + p2]
            r0 = sum(t0 over p2)
            buf0[8*p0 + p1] = r0

kernel k1
    inputs:   [arg0_1, buf0]
    outputs:  [buf1]

    op1:  buf1: f32[2,8,1]  <-  arg0_1: f32[2,8,16], buf0: f32[2,8,1]      # sum(((arg0_1[128*p0 + 16*p1 + p2]) - ((buf0[8*p0 + p1]) / 16.0)) * ((arg0_1[128*p0 + 16*p1 + p2]) - ((buf0[8*p0 + p1]) / 16.0)) over p2)
        foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,16):
            t0 = arg0_1[128*p0 + 16*p1 + p2]
            ix0 = 8*p0 + p1
            t1 = buf0[ix0]
            t2 = t1 / 16.0
            t3 = t0 - t2
            t4 = t3 * t3
            r0 = sum(t4 over p2)
            buf1[ix0] = r0

kernel k2
    inputs:   [arg0_1, arg1_1, arg2_1, buf0, buf1]
    outputs:  [buf2]

    op2:  buf2: f32[2,8,16]  <-  arg0_1: f32[2,8,16], buf0: f32[2,8,1], buf1: f32[2,8,1], arg1_1: f32[16], arg2_1: f32[16]      # ((((arg0_1[128*p0 + 16*p1 + p2]) - ((buf0[8*p0 + p1]) / 16.0)) * rsqrt(((buf1[8*p0 + p1]) / 16.0) + 1e-05)) * arg1_1[p2]) + arg2_1[p2]
        foreach p0 in [0,2), p1 in [0,8), p2 in [0,16):
            ix0 = 128*p0 + 16*p1 + p2
            t0 = arg0_1[ix0]
            ix1 = 8*p0 + p1
            t1 = buf0[ix1]
            t2 = t1 / 16.0
            t3 = t0 - t2
            t4 = buf1[ix1]
            t5 = t4 / 16.0
            t6 = t5 + 1e-05
            t7 = rsqrt(t6)
            t8 = t3 * t7
            t9 = arg1_1[p2]
            t10 = t8 * t9
            t11 = arg2_1[p2]
            t12 = t10 + t11
            buf2[ix0] = t12

kernel k3
    # implementation: extern call
    inputs:   [arg3_1, buf2]
    outputs:  [buf3]

    op3:  buf3: f32[16,48]
        extern_kernels.mm(reinterpret_tensor(buf2, [16,16], [16,1], 0),
            reinterpret_tensor(arg3_1, [16,48], [1,16], 0))

kernel k4
    inputs:   [arg4_1, buf3]
    outputs:  [buf4]

    op4:  buf4: f32[2,2,8,8]  <-  arg4_1: f32[48], buf3: f32[16,48]      # (arg4_1[8*p1 + p3]) + (buf3[384*p0 + 8*p1 + 48*p2 + p3])
        foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), p3 in [0,8):
            t0 = arg4_1[8*p1 + p3]
            t1 = buf3[384*p0 + 8*p1 + 48*p2 + p3]
            t2 = t0 + t1
            buf4[128*p0 + 64*p1 + 8*p2 + p3] = t2

kernel k5
    inputs:   [arg4_1, buf3]
    outputs:  [buf5]

    op5:  buf5: f32[2,2,8,8]  <-  arg4_1: f32[48], buf3: f32[16,48]      # (arg4_1[8*p1 + p2 + 16]) + (buf3[384*p0 + 8*p1 + p2 + 48*p3 + 16])
        foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), p3 in [0,8):
            t0 = arg4_1[8*p1 + p2 + 16]
            t1 = buf3[384*p0 + 8*p1 + p2 + 48*p3 + 16]
            t2 = t0 + t1
            buf5[128*p0 + 64*p1 + 8*p2 + p3] = t2

kernel k6
    # implementation: extern call
    inputs:   [buf4, buf5]
    outputs:  [buf6]

    op6:  buf6: f32[4,8,8]
        extern_kernels.bmm(reinterpret_tensor(buf4, [4,8,8], [64,8,1], 0),
            reinterpret_tensor(buf5, [4,8,8], [64,8,1], 0))

kernel k7
    inputs:   [buf6]
    outputs:  [buf7]

    op7:  buf7: b8[2,2,8,1]  <-  buf6: f32[4,8,8]      # any(not ((((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373) == ((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373)) and (abs((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373) != inf)) over p3)
        foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), reduce p3 in [0,8):
            ix0 = 128*p0 + 64*p1 + 8*p2 + p3
            t0 = buf6[ix0]
            t1 = t0 * 0.35355339059327373
            t2 = buf6[ix0]
            t3 = t2 * 0.35355339059327373
            t4 = t1 == t3
            t5 = buf6[ix0]
            t6 = t5 * 0.35355339059327373
            t7 = abs(t6)
            t8 = t7 != inf
            t9 = t4 and t8
            t10 = not t9
            r0 = any(t10 over p3)
            buf7[16*p0 + 8*p1 + p2] = r0

kernel k8
    inputs:   [buf6]
    outputs:  [buf8]

    op8:  buf8: f32[2,2,8,1]  <-  buf6: f32[4,8,8]      # max((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 1.0 over p3)
        foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), reduce p3 in [0,8):
            t0 = buf6[128*p0 + 64*p1 + 8*p2 + p3]
            t1 = t0 * 1.0
            r0 = max(t1 over p3)
            buf8[16*p0 + 8*p1 + p2] = r0

kernel k9
    inputs:   [buf6]
    outputs:  [buf9]

    op9:  buf9: f32[2,2,8,1]  <-  buf6: f32[4,8,8]      # max((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373 over p3)
        foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), reduce p3 in [0,8):
            t0 = buf6[128*p0 + 64*p1 + 8*p2 + p3]
            t1 = t0 * 0.35355339059327373
            r0 = max(t1 over p3)
            buf9[16*p0 + 8*p1 + p2] = r0

kernel k10
    inputs:   [buf6, buf7, buf8, buf9]
    outputs:  [buf10]

    op10:  buf10: f32[2,2,8,1]  <-  buf7: b8[2,2,8,1], buf6: f32[4,8,8], buf8: f32[2,2,8,1], buf9: f32[2,2,8,1]      # sum(exp(where(not (buf7[16*p0 + 8*p1 + p2]), (((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 1.0) - (buf8[16*p0 + 8*p1 + p2])) * 0.35355339059327373, ((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373) - (buf9[16*p0 + 8*p1 + p2]))) over p3)
        foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), reduce p3 in [0,8):
            ix0 = 16*p0 + 8*p1 + p2
            t0 = buf7[ix0]
            t1 = not t0
            ix1 = 128*p0 + 64*p1 + 8*p2 + p3
            t2 = buf6[ix1]
            t3 = t2 * 1.0
            t4 = buf8[ix0]
            t5 = t3 - t4
            t6 = t5 * 0.35355339059327373
            t7 = buf6[ix1]
            t8 = t7 * 0.35355339059327373
            t9 = buf9[ix0]
            t10 = t8 - t9
            t11 = where(t1, t6, t10)
            t12 = exp(t11)
            r0 = sum(t12 over p3)
            buf10[ix0] = r0

kernel k11
    inputs:   [buf6, buf7, buf8, buf9, buf10]
    outputs:  [buf11]

    op11:  buf11: f32[2,2,8,8]  <-  buf7: b8[2,2,8,1], buf6: f32[4,8,8], buf8: f32[2,2,8,1], buf9: f32[2,2,8,1], buf10: f32[2,2,8,1]      # exp(where(not (buf7[16*p0 + 8*p1 + p2]), (((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 1.0) - (buf8[16*p0 + 8*p1 + p2])) * 0.35355339059327373, ((buf6[128*p0 + 64*p1 + 8*p2 + p3]) * 0.35355339059327373) - (buf9[16*p0 + 8*p1 + p2]))) / (buf10[16*p0 + 8*p1 + p2])
        foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), p3 in [0,8):
            ix0 = 16*p0 + 8*p1 + p2
            t0 = buf7[ix0]
            t1 = not t0
            ix1 = 128*p0 + 64*p1 + 8*p2 + p3
            t2 = buf6[ix1]
            t3 = t2 * 1.0
            t4 = buf8[ix0]
            t5 = t3 - t4
            t6 = t5 * 0.35355339059327373
            t7 = buf6[ix1]
            t8 = t7 * 0.35355339059327373
            t9 = buf9[ix0]
            t10 = t8 - t9
            t11 = where(t1, t6, t10)
            t12 = exp(t11)
            t13 = buf10[ix0]
            t14 = t12 / t13
            buf11[ix1] = t14

kernel k12
    inputs:   [arg4_1, buf3]
    outputs:  [buf12]

    op12:  buf12: f32[2,2,8,8]  <-  arg4_1: f32[48], buf3: f32[16,48]      # (arg4_1[8*p1 + p3 + 32]) + (buf3[384*p0 + 8*p1 + 48*p2 + p3 + 32])
        foreach p0 in [0,2), p1 in [0,2), p2 in [0,8), p3 in [0,8):
            t0 = arg4_1[8*p1 + p3 + 32]
            t1 = buf3[384*p0 + 8*p1 + 48*p2 + p3 + 32]
            t2 = t0 + t1
            buf12[128*p0 + 64*p1 + 8*p2 + p3] = t2

kernel k13
    # implementation: extern call
    inputs:   [buf11, buf12]
    outputs:  [buf13]

    op13:  buf13: f32[4,8,8]
        extern_kernels.bmm(reinterpret_tensor(buf11, [4,8,8], [64,8,1], 0),
            reinterpret_tensor(buf12, [4,8,8], [64,8,1], 0))

kernel k14
    inputs:   [buf13]
    outputs:  [buf14]

    op14:  buf14: f32[2,8,2,8]  <-  buf13: f32[4,8,8]      # copy
        foreach p0 in [0,2), p1 in [0,8), p2 in [0,2), p3 in [0,8):
            t0 = buf13[128*p0 + 8*p1 + 64*p2 + p3]
            buf14[128*p0 + 16*p1 + 8*p2 + p3] = t0

kernel k15
    # implementation: extern call
    inputs:   [arg5_1, buf14]
    outputs:  [buf15]

    op15:  buf15: f32[16,16]
        extern_kernels.mm(reinterpret_tensor(buf14, [16,16], [16,1], 0),
            reinterpret_tensor(arg5_1, [16,16], [1,16], 0))

kernel k16
    inputs:   [arg0_1, arg6_1, buf15]
    outputs:  [buf16]

    op16:  buf16: f32[2,8,1]  <-  arg0_1: f32[2,8,16], arg6_1: f32[16], buf15: f32[16,16]      # sum((arg0_1[128*p0 + 16*p1 + p2]) + (arg6_1[p2] + (buf15[128*p0 + 16*p1 + p2])) over p2)
        foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,16):
            ix0 = 128*p0 + 16*p1 + p2
            t0 = arg0_1[ix0]
            t1 = arg6_1[p2]
            t2 = buf15[ix0]
            t3 = t1 + t2
            t4 = t0 + t3
            r0 = sum(t4 over p2)
            buf16[8*p0 + p1] = r0

kernel k17
    inputs:   [arg0_1, arg6_1, buf15, buf16]
    outputs:  [buf17]

    op17:  buf17: f32[2,8,1]  <-  arg0_1: f32[2,8,16], arg6_1: f32[16], buf15: f32[16,16], buf16: f32[2,8,1]      # sum((((arg0_1[128*p0 + 16*p1 + p2]) + (arg6_1[p2] + (buf15[128*p0 + 16*p1 + p2]))) - ((buf16[8*p0 + p1]) / 16.0)) * (((arg0_1[128*p0 + 16*p1 + p2]) + (arg6_1[p2] + (buf15[128*p0 + 16*p1 + p2]))) - ((buf16[8*p0 + p1]) / 16.0)) over p2)
        foreach p0 in [0,2), p1 in [0,8), reduce p2 in [0,16):
            ix0 = 128*p0 + 16*p1 + p2
            t0 = arg0_1[ix0]
            t1 = arg6_1[p2]
            t2 = buf15[ix0]
            t3 = t1 + t2
            t4 = t0 + t3
            ix1 = 8*p0 + p1
            t5 = buf16[ix1]
            t6 = t5 / 16.0
            t7 = t4 - t6
            t8 = t7 * t7
            r0 = sum(t8 over p2)
            buf17[ix1] = r0

kernel k18
    inputs:   [arg0_1, arg6_1, arg7_1, arg8_1, buf15, buf16, buf17]
    outputs:  [buf18]

    op18:  buf18: f32[2,8,16]  <-  arg0_1: f32[2,8,16], arg6_1: f32[16], buf15: f32[16,16], buf16: f32[2,8,1], buf17: f32[2,8,1], arg7_1: f32[16], arg8_1: f32[16]      # (((((arg0_1[128*p0 + 16*p1 + p2]) + (arg6_1[p2] + (buf15[128*p0 + 16*p1 + p2]))) - ((buf16[8*p0 + p1]) / 16.0)) * rsqrt(((buf17[8*p0 + p1]) / 16.0) + 1e-05)) * arg7_1[p2]) + arg8_1[p2]
        foreach p0 in [0,2), p1 in [0,8), p2 in [0,16):
            ix0 = 128*p0 + 16*p1 + p2
            t0 = arg0_1[ix0]
            t1 = arg6_1[p2]
            t2 = buf15[ix0]
            t3 = t1 + t2
            t4 = t0 + t3
            ix1 = 8*p0 + p1
            t5 = buf16[ix1]
            t6 = t5 / 16.0
            t7 = t4 - t6
            t8 = buf17[ix1]
            t9 = t8 / 16.0
            t10 = t9 + 1e-05
            t11 = rsqrt(t10)
            t12 = t7 * t11
            t13 = arg7_1[p2]
            t14 = t12 * t13
            t15 = arg8_1[p2]
            t16 = t14 + t15
            buf18[ix0] = t16

kernel k19
    # implementation: extern call
    inputs:   [arg9_1, buf18]
    outputs:  [buf19]

    op19:  buf19: f32[16,64]
        extern_kernels.mm(reinterpret_tensor(buf18, [16,16], [16,1], 0),
            reinterpret_tensor(arg9_1, [16,64], [1,16], 0))

kernel k20
    inputs:   [arg10_1, buf19]
    outputs:  [buf20]

    op20:  buf20: f32[2,8,64]  <-  arg10_1: f32[64], buf19: f32[16,64]      # ((arg10_1[p2] + (buf19[512*p0 + 64*p1 + p2])) * 0.5) * (erf((arg10_1[p2] + (buf19[512*p0 + 64*p1 + p2])) * 0.7071067811865476) + 1.0)
        foreach p0 in [0,2), p1 in [0,8), p2 in [0,64):
            t0 = arg10_1[p2]
            ix0 = 512*p0 + 64*p1 + p2
            t1 = buf19[ix0]
            t2 = t0 + t1
            t3 = t2 * 0.5
            t4 = arg10_1[p2]
            t5 = buf19[ix0]
            t6 = t4 + t5
            t7 = t6 * 0.7071067811865476
            t8 = erf(t7)
            t9 = t8 + 1.0
            t10 = t3 * t9
            buf20[ix0] = t10

kernel k21
    # implementation: extern call
    inputs:   [arg11_1, buf20]
    outputs:  [buf21]

    op21:  buf21: f32[16,16]
        extern_kernels.mm(reinterpret_tensor(buf20, [16,64], [64,1], 0),
            reinterpret_tensor(arg11_1, [64,16], [1,64], 0))

kernel k22
    inputs:   [arg0_1, arg12_1, arg6_1, buf15, buf21]
    outputs:  [buf22]

    op22:  buf22: f32[2,8,16]  <-  arg0_1: f32[2,8,16], arg6_1: f32[16], buf15: f32[16,16], arg12_1: f32[16], buf21: f32[16,16]      # ((arg0_1[128*p0 + 16*p1 + p2]) + (arg6_1[p2] + (buf15[128*p0 + 16*p1 + p2]))) + (arg12_1[p2] + (buf21[128*p0 + 16*p1 + p2]))
        foreach p0 in [0,2), p1 in [0,8), p2 in [0,16):
            ix0 = 128*p0 + 16*p1 + p2
            t0 = arg0_1[ix0]
            t1 = arg6_1[p2]
            t2 = buf15[ix0]
            t3 = t1 + t2
            t4 = t0 + t3
            t5 = arg12_1[p2]
            t6 = buf21[ix0]
            t7 = t5 + t6
            t8 = t4 + t7
            buf22[ix0] = t8

# outputs
return (buf22)
```

### post_fusion

```text
op0_op1_op2: FusedSchedulerNode(SchedulerNode,SchedulerNode,SchedulerNode)
op0_op1_op2.writes =
    [   MemoryDep('buf0', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf1', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf2', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16})]
op0_op1_op2.unmet_dependencies = []
op0_op1_op2.met_dependencies =
    [   MemoryDep('arg0_1', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg1_1', d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg2_1', d2, {d0: 2, d1: 8, d2: 16})]
op0_op1_op2.min_input_distance = 0
op0_op1_op2.max_input_distance = 2
op0_op1_op2.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf0.users = [
        NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False),
    ]
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
    buf2.users = [NodeUser(node=ExternKernelSchedulerNode(name='op3'), can_inplace=False, is_weak=False)]
]
op0_op1_op2.snodes[0] =
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 16})]
op0.unmet_dependencies = []
op0.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf0.users = [
        NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False),
    ]
]
op0.group.device = cuda:0
op0.group.iteration = (16, 16)
op0.sizes = ((16,), (16,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op0_loop_body:
    var_ranges = {p0: 16, p1: 16}
    index0 = 16*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', load)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf0', get_index_1, reduction)
        return None
op0_op1_op2.snodes[1] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 16})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 16})]
op1.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (16, 16)
op1.sizes = ((16,), (16,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op1_loop_body:
    var_ranges = {p0: 16, p1: 16}
    index0 = 16*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(16.0, torch.float32)
        truediv = ops.truediv(load_1, constant)
        sub = ops.sub(load, truediv)
        mul = ops.mul(sub, sub)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', mul)
        get_index_2 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf1', get_index_2, reduction)
        return None
op0_op1_op2.snodes[2] =
op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', c0, {c0: 256})]
op2.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 16}), MemoryDep('buf1', c0, {c0: 16})]
op2.met_dependencies =
    [   MemoryDep('arg0_1', c0, {c0: 256}),
        MemoryDep('arg1_1', c1, {c0: 16, c1: 16}),
        MemoryDep('arg2_1', c1, {c0: 16, c1: 16})]
op2.min_input_distance = 1
op2.max_input_distance = 2
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
    buf2.users = [NodeUser(node=ExternKernelSchedulerNode(name='op3'), can_inplace=False, is_weak=False)]
]
op2.group.device = cuda:0
op2.group.iteration = (256, 1)
op2.sizes = ((16, 16), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
arg2_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
class op2_loop_body:
    var_ranges = {p0: 16, p1: 16}
    index0 = 16*p0 + p1
    index1 = p0
    index2 = p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf0', get_index_1)
        constant = ops.constant(16.0, torch.float32)
        truediv = ops.truediv(load_1, constant)
        sub = ops.sub(load, truediv)
        get_index_2 = self.get_index('index1')
        load_2 = ops.load('buf1', get_index_2)
        constant_1 = ops.constant(16.0, torch.float32)
        truediv_1 = ops.truediv(load_2, constant_1)
        constant_2 = ops.constant(1e-05, torch.float32)
        add = ops.add(truediv_1, constant_2)
        rsqrt = ops.rsqrt(add)
        mul = ops.mul(sub, rsqrt)
        get_index_3 = self.get_index('index2')
        load_3 = ops.load('arg1_1', get_index_3)
        mul_1 = ops.mul(mul, load_3)
        get_index_4 = self.get_index('index2')
        load_4 = ops.load('arg2_1', get_index_4)
        add_1 = ops.add(mul_1, load_4)
        get_index_5 = self.get_index('index0')
        store = ops.store('buf2', get_index_5, add_1, None)
        return store


op3: ExternKernelSchedulerNode(ExternKernelOut)
op3.writes = [StarDep(name='buf3', mode=None)]
op3.unmet_dependencies = [StarDep(name='buf2', mode=None)]
op3.met_dependencies = [StarDep(name='arg3_1', mode=None)]
op3.min_input_distance = 2
op3.max_input_distance = 3
op3.outputs = [
    buf3: ExternKernelOut
    buf3.layout = FixedLayout('cuda:0', torch.float32, size=[16, 48], stride=[48, 1])
    buf3.users = [
        NodeUser(node=SchedulerNode(name='op4'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op5'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op12'), can_inplace=False, is_weak=False),
    ]
]
op3.node.kernel = extern_kernels.mm


op4: SchedulerNode(ComputedBuffer)
op4.writes = [MemoryDep('buf4', c0, {c0: 256})]
op4.unmet_dependencies = [MemoryDep('buf3', 384*c0 + 8*c1 + 48*c2 + c3, {c0: 2, c1: 2, c2: 8, c3: 8})]
op4.met_dependencies = [MemoryDep('arg4_1', 8*c1 + c3, {c0: 2, c1: 2, c2: 8, c3: 8})]
op4.min_input_distance = 3
op4.max_input_distance = 4
op4.outputs = [
    buf4: ComputedBuffer
    buf4.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
    buf4.users = [NodeUser(node=ExternKernelSchedulerNode(name='op6'), can_inplace=False, is_weak=False)]
]
op4.group.device = cuda:0
op4.group.iteration = (256, 1)
op4.sizes = ((2, 2, 8, 8), ())
arg4_1_layout = FixedLayout('cuda:0', torch.float32, size=[48], stride=[1])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[16, 48], stride=[48, 1])
buf4_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
class op4_loop_body:
    var_ranges = {p0: 2, p1: 2, p2: 8, p3: 8}
    index0 = 8*p1 + p3
    index1 = 384*p0 + 8*p1 + 48*p2 + p3
    index2 = 128*p0 + 64*p1 + 8*p2 + p3
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg4_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf3', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index2')
        store = ops.store('buf4', get_index_2, add, None)
        return store


op5: SchedulerNode(ComputedBuffer)
op5.writes = [MemoryDep('buf5', c0, {c0: 256})]
op5.unmet_dependencies = [MemoryDep('buf3', 384*c0 + c1 + 48*c2 + 16, {c0: 2, c1: 16, c2: 8})]
op5.met_dependencies = [MemoryDep('arg4_1', c1 + 16, {c0: 2, c1: 16})]
op5.min_input_distance = 3
op5.max_input_distance = 4
op5.outputs = [
    buf5: ComputedBuffer
    buf5.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
    buf5.users = [NodeUser(node=ExternKernelSchedulerNode(name='op6'), can_inplace=False, is_weak=False)]
]
op5.group.device = cuda:0
op5.group.iteration = (256, 1)
op5.sizes = ((2, 16, 8), ())
arg4_1_layout = FixedLayout('cuda:0', torch.float32, size=[48], stride=[1])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[16, 48], stride=[48, 1])
buf5_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
class op5_loop_body:
    var_ranges = {p0: 2, p1: 16, p2: 8}
    index0 = p1 + 16
    index1 = 384*p0 + p1 + 48*p2 + 16
    index2 = 128*p0 + 8*p1 + p2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg4_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf3', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index2')
        store = ops.store('buf5', get_index_2, add, None)
        return store


op6: ExternKernelSchedulerNode(ExternKernelOut)
op6.writes = [StarDep(name='buf6', mode=None)]
op6.unmet_dependencies = [StarDep(name='buf4', mode=None), StarDep(name='buf5', mode=None)]
op6.met_dependencies = []
op6.min_input_distance = 4
op6.max_input_distance = 5
op6.outputs = [
    buf6: ExternKernelOut
    buf6.layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
    buf6.users = [
        NodeUser(node=SchedulerNode(name='op7'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op8'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op9'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op10'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op11'), can_inplace=True, is_weak=False),
    ]
]
op6.node.kernel = extern_kernels.bmm


op7_op8_op9_op10_op11: FusedSchedulerNode(SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode)
op7_op8_op9_op10_op11.writes =
    [   MemoryDep('buf10', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8}),
        MemoryDep('buf11', 128*d0 + 64*d1 + 8*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8}),
        MemoryDep('buf7', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8}),
        MemoryDep('buf8', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8}),
        MemoryDep('buf9', 16*d0 + 8*d1 + d2, {d0: 2, d1: 2, d2: 8})]
op7_op8_op9_op10_op11.unmet_dependencies = [MemoryDep('buf6', 128*d0 + 64*d1 + 8*d2 + d3, {d0: 2, d1: 2, d2: 8, d3: 8})]
op7_op8_op9_op10_op11.met_dependencies = []
op7_op8_op9_op10_op11.min_input_distance = 5
op7_op8_op9_op10_op11.max_input_distance = 8
op7_op8_op9_op10_op11.outputs = [
    buf7: ComputedBuffer
    buf7.layout = FixedLayout('cuda:0', torch.bool, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf7.users = [
        NodeUser(node=SchedulerNode(name='op10'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False),
    ]
    buf8: ComputedBuffer
    buf8.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf8.users = [
        NodeUser(node=SchedulerNode(name='op10'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False),
    ]
    buf9: ComputedBuffer
    buf9.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf9.users = [
        NodeUser(node=SchedulerNode(name='op10'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False),
    ]
    buf10: ComputedBuffer
    buf10.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf10.users = [NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False)]
    buf11: ComputedBuffer
    buf11.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
    buf11.users = [NodeUser(node=ExternKernelSchedulerNode(name='op13'), can_inplace=False, is_weak=False)]
]
op7_op8_op9_op10_op11.snodes[0] =
op7: SchedulerNode(ComputedBuffer)
op7.writes = [MemoryDep('buf7', c0, {c0: 32})]
op7.unmet_dependencies = [MemoryDep('buf6', c0, {c0: 256})]
op7.met_dependencies = []
op7.min_input_distance = 5
op7.max_input_distance = 6
op7.outputs = [
    buf7: ComputedBuffer
    buf7.layout = FixedLayout('cuda:0', torch.bool, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf7.users = [
        NodeUser(node=SchedulerNode(name='op10'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False),
    ]
]
op7.group.device = cuda:0
op7.group.iteration = (32, 8)
op7.sizes = ((32,), (8,))
buf6_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf7_layout = FixedLayout('cuda:0', torch.bool, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
class op7_loop_body:
    var_ranges = {p0: 32, p1: 8}
    index0 = 8*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf6', get_index)
        constant = ops.constant(0.35355339059327373, torch.float32)
        mul = ops.mul(load, constant)
        get_index_1 = self.get_index('index0')
        load_1 = ops.load('buf6', get_index_1)
        constant_1 = ops.constant(0.35355339059327373, torch.float32)
        mul_1 = ops.mul(load_1, constant_1)
        eq = ops.eq(mul, mul_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf6', get_index_2)
        constant_2 = ops.constant(0.35355339059327373, torch.float32)
        mul_2 = ops.mul(load_2, constant_2)
        abs_1 = ops.abs(mul_2)
        constant_3 = ops.constant(inf, torch.float32)
        ne = ops.ne(abs_1, constant_3)
        logical_and = ops.logical_and(eq, ne)
        logical_not = ops.logical_not(logical_and)
        reduction = ops.reduction(torch.bool, torch.bool, 'any', logical_not)
        get_index_3 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf7', get_index_3, reduction)
        return None
op7_op8_op9_op10_op11.snodes[1] =
op8: SchedulerNode(ComputedBuffer)
op8.writes = [MemoryDep('buf8', c0, {c0: 32})]
op8.unmet_dependencies = [MemoryDep('buf6', c0, {c0: 256})]
op8.met_dependencies = []
op8.min_input_distance = 5
op8.max_input_distance = 6
op8.outputs = [
    buf8: ComputedBuffer
    buf8.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf8.users = [
        NodeUser(node=SchedulerNode(name='op10'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False),
    ]
]
op8.group.device = cuda:0
op8.group.iteration = (32, 8)
op8.sizes = ((32,), (8,))
buf6_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf8_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
class op8_loop_body:
    var_ranges = {p0: 32, p1: 8}
    index0 = 8*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf6', get_index)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load, constant)
        reduction = ops.reduction(torch.float32, torch.float32, 'max', mul)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf8', get_index_1, reduction)
        return None
op7_op8_op9_op10_op11.snodes[2] =
op9: SchedulerNode(ComputedBuffer)
op9.writes = [MemoryDep('buf9', c0, {c0: 32})]
op9.unmet_dependencies = [MemoryDep('buf6', c0, {c0: 256})]
op9.met_dependencies = []
op9.min_input_distance = 5
op9.max_input_distance = 6
op9.outputs = [
    buf9: ComputedBuffer
    buf9.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf9.users = [
        NodeUser(node=SchedulerNode(name='op10'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False),
    ]
]
op9.group.device = cuda:0
op9.group.iteration = (32, 8)
op9.sizes = ((32,), (8,))
buf6_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf9_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
class op9_loop_body:
    var_ranges = {p0: 32, p1: 8}
    index0 = 8*p0 + p1
    index1 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf6', get_index)
        constant = ops.constant(0.35355339059327373, torch.float32)
        mul = ops.mul(load, constant)
        reduction = ops.reduction(torch.float32, torch.float32, 'max', mul)
        get_index_1 = self.get_index('index1')
        store_reduction = ops.store_reduction('buf9', get_index_1, reduction)
        return None
op7_op8_op9_op10_op11.snodes[3] =
op10: SchedulerNode(ComputedBuffer)
op10.writes = [MemoryDep('buf10', c0, {c0: 32})]
op10.unmet_dependencies =
    [   MemoryDep('buf6', c0, {c0: 256}),
        MemoryDep('buf7', c0, {c0: 32}),
        MemoryDep('buf8', c0, {c0: 32}),
        MemoryDep('buf9', c0, {c0: 32})]
op10.met_dependencies = []
op10.min_input_distance = 5
op10.max_input_distance = 7
op10.outputs = [
    buf10: ComputedBuffer
    buf10.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
    buf10.users = [NodeUser(node=SchedulerNode(name='op11'), can_inplace=False, is_weak=False)]
]
op10.group.device = cuda:0
op10.group.iteration = (32, 8)
op10.sizes = ((32,), (8,))
buf7_layout = FixedLayout('cuda:0', torch.bool, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf6_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf8_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf9_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf10_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
class op10_loop_body:
    var_ranges = {p0: 32, p1: 8}
    index0 = p0
    index1 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf7', get_index)
        logical_not = ops.logical_not(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf6', get_index_1)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load_1, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf8', get_index_2)
        sub = ops.sub(mul, load_2)
        constant_1 = ops.constant(0.35355339059327373, torch.float32)
        mul_1 = ops.mul(sub, constant_1)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf6', get_index_3)
        constant_2 = ops.constant(0.35355339059327373, torch.float32)
        mul_2 = ops.mul(load_3, constant_2)
        get_index_4 = self.get_index('index0')
        load_4 = ops.load('buf9', get_index_4)
        sub_1 = ops.sub(mul_2, load_4)
        where = ops.where(logical_not, mul_1, sub_1)
        exp = ops.exp(where)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', exp)
        get_index_5 = self.get_index('index0')
        store_reduction = ops.store_reduction('buf10', get_index_5, reduction)
        return None
op7_op8_op9_op10_op11.snodes[4] =
op11: SchedulerNode(ComputedBuffer)
op11.writes = [MemoryDep('buf11', c0, {c0: 256})]
op11.unmet_dependencies =
    [   MemoryDep('buf10', c0, {c0: 32}),
        MemoryDep('buf6', c0, {c0: 256}),
        MemoryDep('buf7', c0, {c0: 32}),
        MemoryDep('buf8', c0, {c0: 32}),
        MemoryDep('buf9', c0, {c0: 32})]
op11.met_dependencies = []
op11.min_input_distance = 5
op11.max_input_distance = 8
op11.outputs = [
    buf11: ComputedBuffer
    buf11.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
    buf11.users = [NodeUser(node=ExternKernelSchedulerNode(name='op13'), can_inplace=False, is_weak=False)]
]
op11.group.device = cuda:0
op11.group.iteration = (256, 1)
op11.sizes = ((32, 8), ())
buf7_layout = FixedLayout('cuda:0', torch.bool, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf6_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf8_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf9_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf10_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 1], stride=[16, 8, 1, 32])
buf11_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
class op11_loop_body:
    var_ranges = {p0: 32, p1: 8}
    index0 = p0
    index1 = 8*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf7', get_index)
        logical_not = ops.logical_not(load)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf6', get_index_1)
        constant = ops.constant(1.0, torch.float32)
        mul = ops.mul(load_1, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf8', get_index_2)
        sub = ops.sub(mul, load_2)
        constant_1 = ops.constant(0.35355339059327373, torch.float32)
        mul_1 = ops.mul(sub, constant_1)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf6', get_index_3)
        constant_2 = ops.constant(0.35355339059327373, torch.float32)
        mul_2 = ops.mul(load_3, constant_2)
        get_index_4 = self.get_index('index0')
        load_4 = ops.load('buf9', get_index_4)
        sub_1 = ops.sub(mul_2, load_4)
        where = ops.where(logical_not, mul_1, sub_1)
        exp = ops.exp(where)
        get_index_5 = self.get_index('index0')
        load_5 = ops.load('buf10', get_index_5)
        truediv = ops.truediv(exp, load_5)
        get_index_6 = self.get_index('index1')
        store = ops.store('buf11', get_index_6, truediv, None)
        return store


op12: SchedulerNode(ComputedBuffer)
op12.writes = [MemoryDep('buf12', c0, {c0: 256})]
op12.unmet_dependencies = [   MemoryDep('buf3', 384*c0 + 8*c1 + 48*c2 + c3 + 32, {c0: 2, c1: 2, c2: 8, c3: 8})]
op12.met_dependencies = [MemoryDep('arg4_1', 8*c1 + c3 + 32, {c0: 2, c1: 2, c2: 8, c3: 8})]
op12.min_input_distance = 3
op12.max_input_distance = 4
op12.outputs = [
    buf12: ComputedBuffer
    buf12.layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
    buf12.users = [NodeUser(node=ExternKernelSchedulerNode(name='op13'), can_inplace=False, is_weak=False)]
]
op12.group.device = cuda:0
op12.group.iteration = (256, 1)
op12.sizes = ((2, 2, 8, 8), ())
arg4_1_layout = FixedLayout('cuda:0', torch.float32, size=[48], stride=[1])
buf3_layout = FixedLayout('cuda:0', torch.float32, size=[16, 48], stride=[48, 1])
buf12_layout = FixedLayout('cuda:0', torch.float32, size=[2, 2, 8, 8], stride=[128, 64, 8, 1])
class op12_loop_body:
    var_ranges = {p0: 2, p1: 2, p2: 8, p3: 8}
    index0 = 8*p1 + p3 + 32
    index1 = 384*p0 + 8*p1 + 48*p2 + p3 + 32
    index2 = 128*p0 + 64*p1 + 8*p2 + p3
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg4_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf3', get_index_1)
        add = ops.add(load, load_1)
        get_index_2 = self.get_index('index2')
        store = ops.store('buf12', get_index_2, add, None)
        return store


op13: ExternKernelSchedulerNode(ExternKernelOut)
op13.writes = [StarDep(name='buf13', mode=None)]
op13.unmet_dependencies = [StarDep(name='buf11', mode=None), StarDep(name='buf12', mode=None)]
op13.met_dependencies = []
op13.min_input_distance = 4
op13.max_input_distance = 9
op13.outputs = [
    buf13: ExternKernelOut
    buf13.layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
    buf13.users = [NodeUser(node=SchedulerNode(name='op14'), can_inplace=False, is_weak=False)]
]
op13.node.kernel = extern_kernels.bmm


op14: SchedulerNode(ComputedBuffer)
op14.writes = [MemoryDep('buf14', c0, {c0: 256})]
op14.unmet_dependencies = [MemoryDep('buf13', 128*c0 + 8*c1 + 64*c2 + c3, {c0: 2, c1: 8, c2: 2, c3: 8})]
op14.met_dependencies = []
op14.min_input_distance = 5
op14.max_input_distance = 10
op14.outputs = [
    buf14: ComputedBuffer
    buf14.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 2, 8], stride=[128, 16, 8, 1])
    buf14.users = [NodeUser(node=ExternKernelSchedulerNode(name='op15'), can_inplace=False, is_weak=False)]
]
op14.group.device = cuda:0
op14.group.iteration = (256, 1)
op14.sizes = ((2, 8, 2, 8), ())
buf13_layout = FixedLayout('cuda:0', torch.float32, size=[4, 8, 8], stride=[64, 8, 1])
buf14_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 2, 8], stride=[128, 16, 8, 1])
class op14_loop_body:
    var_ranges = {p0: 2, p1: 8, p2: 2, p3: 8}
    index0 = 128*p0 + 8*p1 + 64*p2 + p3
    index1 = 128*p0 + 16*p1 + 8*p2 + p3
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf13', get_index)
        get_index_1 = self.get_index('index1')
        store = ops.store('buf14', get_index_1, load, None)
        return store


op15: ExternKernelSchedulerNode(ExternKernelOut)
op15.writes = [StarDep(name='buf15', mode=None)]
op15.unmet_dependencies = [StarDep(name='buf14', mode=None)]
op15.met_dependencies = [StarDep(name='arg5_1', mode=None)]
op15.min_input_distance = 6
op15.max_input_distance = 11
op15.outputs = [
    buf15: ExternKernelOut
    buf15.layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
    buf15.users = [
        NodeUser(node=SchedulerNode(name='op16'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op17'), can_inplace=False, is_weak=False),
        NodeUser(node=SchedulerNode(name='op18'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op22'), can_inplace=True, is_weak=False),
    ]
]
op15.node.kernel = extern_kernels.mm


op16_op17_op18: FusedSchedulerNode(SchedulerNode,SchedulerNode,SchedulerNode)
op16_op17_op18.writes =
    [   MemoryDep('buf16', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf17', 8*d0 + d1, {d0: 2, d1: 8}),
        MemoryDep('buf18', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16})]
op16_op17_op18.unmet_dependencies = [MemoryDep('buf15', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16})]
op16_op17_op18.met_dependencies =
    [   MemoryDep('arg0_1', 128*d0 + 16*d1 + d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg6_1', d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg7_1', d2, {d0: 2, d1: 8, d2: 16}),
        MemoryDep('arg8_1', d2, {d0: 2, d1: 8, d2: 16})]
op16_op17_op18.min_input_distance = 7
op16_op17_op18.max_input_distance = 14
op16_op17_op18.outputs = [
    buf16: ComputedBuffer
    buf16.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf16.users = [
        NodeUser(node=SchedulerNode(name='op17'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op18'), can_inplace=False, is_weak=False),
    ]
    buf17: ComputedBuffer
    buf17.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf17.users = [NodeUser(node=SchedulerNode(name='op18'), can_inplace=False, is_weak=False)]
    buf18: ComputedBuffer
    buf18.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
    buf18.users = [NodeUser(node=ExternKernelSchedulerNode(name='op19'), can_inplace=False, is_weak=False)]
]
op16_op17_op18.snodes[0] =
op16: SchedulerNode(ComputedBuffer)
op16.writes = [MemoryDep('buf16', c0, {c0: 16})]
op16.unmet_dependencies = [MemoryDep('buf15', c0, {c0: 256})]
op16.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256}), MemoryDep('arg6_1', c1, {c0: 16, c1: 16})]
op16.min_input_distance = 7
op16.max_input_distance = 12
op16.outputs = [
    buf16: ComputedBuffer
    buf16.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf16.users = [
        NodeUser(node=SchedulerNode(name='op17'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op18'), can_inplace=False, is_weak=False),
    ]
]
op16.group.device = cuda:0
op16.group.iteration = (16, 16)
op16.sizes = ((16,), (16,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
arg6_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf15_layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
buf16_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op16_loop_body:
    var_ranges = {p0: 16, p1: 16}
    index0 = 16*p0 + p1
    index1 = p1
    index2 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg6_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf15', get_index_2)
        add = ops.add(load_1, load_2)
        add_1 = ops.add(load, add)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', add_1)
        get_index_3 = self.get_index('index2')
        store_reduction = ops.store_reduction('buf16', get_index_3, reduction)
        return None
op16_op17_op18.snodes[1] =
op17: SchedulerNode(ComputedBuffer)
op17.writes = [MemoryDep('buf17', c0, {c0: 16})]
op17.unmet_dependencies = [MemoryDep('buf15', c0, {c0: 256}), MemoryDep('buf16', c0, {c0: 16})]
op17.met_dependencies = [MemoryDep('arg0_1', c0, {c0: 256}), MemoryDep('arg6_1', c1, {c0: 16, c1: 16})]
op17.min_input_distance = 7
op17.max_input_distance = 13
op17.outputs = [
    buf17: ComputedBuffer
    buf17.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
    buf17.users = [NodeUser(node=SchedulerNode(name='op18'), can_inplace=False, is_weak=False)]
]
op17.group.device = cuda:0
op17.group.iteration = (16, 16)
op17.sizes = ((16,), (16,))
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
arg6_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf15_layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
buf16_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf17_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
class op17_loop_body:
    var_ranges = {p0: 16, p1: 16}
    index0 = 16*p0 + p1
    index1 = p1
    index2 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg6_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf15', get_index_2)
        add = ops.add(load_1, load_2)
        add_1 = ops.add(load, add)
        get_index_3 = self.get_index('index2')
        load_3 = ops.load('buf16', get_index_3)
        constant = ops.constant(16.0, torch.float32)
        truediv = ops.truediv(load_3, constant)
        sub = ops.sub(add_1, truediv)
        mul = ops.mul(sub, sub)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', mul)
        get_index_4 = self.get_index('index2')
        store_reduction = ops.store_reduction('buf17', get_index_4, reduction)
        return None
op16_op17_op18.snodes[2] =
op18: SchedulerNode(ComputedBuffer)
op18.writes = [MemoryDep('buf18', c0, {c0: 256})]
op18.unmet_dependencies =
    [   MemoryDep('buf15', c0, {c0: 256}),
        MemoryDep('buf16', c0, {c0: 16}),
        MemoryDep('buf17', c0, {c0: 16})]
op18.met_dependencies =
    [   MemoryDep('arg0_1', c0, {c0: 256}),
        MemoryDep('arg6_1', c1, {c0: 16, c1: 16}),
        MemoryDep('arg7_1', c1, {c0: 16, c1: 16}),
        MemoryDep('arg8_1', c1, {c0: 16, c1: 16})]
op18.min_input_distance = 7
op18.max_input_distance = 14
op18.outputs = [
    buf18: ComputedBuffer
    buf18.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
    buf18.users = [NodeUser(node=ExternKernelSchedulerNode(name='op19'), can_inplace=False, is_weak=False)]
]
op18.group.device = cuda:0
op18.group.iteration = (256, 1)
op18.sizes = ((16, 16), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
arg6_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf15_layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
buf16_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
buf17_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 1], stride=[8, 1, 16])
arg7_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
arg8_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf18_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
class op18_loop_body:
    var_ranges = {p0: 16, p1: 16}
    index0 = 16*p0 + p1
    index1 = p1
    index2 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg6_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf15', get_index_2)
        add = ops.add(load_1, load_2)
        add_1 = ops.add(load, add)
        get_index_3 = self.get_index('index2')
        load_3 = ops.load('buf16', get_index_3)
        constant = ops.constant(16.0, torch.float32)
        truediv = ops.truediv(load_3, constant)
        sub = ops.sub(add_1, truediv)
        get_index_4 = self.get_index('index2')
        load_4 = ops.load('buf17', get_index_4)
        constant_1 = ops.constant(16.0, torch.float32)
        truediv_1 = ops.truediv(load_4, constant_1)
        constant_2 = ops.constant(1e-05, torch.float32)
        add_2 = ops.add(truediv_1, constant_2)
        rsqrt = ops.rsqrt(add_2)
        mul = ops.mul(sub, rsqrt)
        get_index_5 = self.get_index('index1')
        load_5 = ops.load('arg7_1', get_index_5)
        mul_1 = ops.mul(mul, load_5)
        get_index_6 = self.get_index('index1')
        load_6 = ops.load('arg8_1', get_index_6)
        add_3 = ops.add(mul_1, load_6)
        get_index_7 = self.get_index('index0')
        store = ops.store('buf18', get_index_7, add_3, None)
        return store


op19: ExternKernelSchedulerNode(ExternKernelOut)
op19.writes = [StarDep(name='buf19', mode=None)]
op19.unmet_dependencies = [StarDep(name='buf18', mode=None)]
op19.met_dependencies = [StarDep(name='arg9_1', mode=None)]
op19.min_input_distance = 8
op19.max_input_distance = 15
op19.outputs = [
    buf19: ExternKernelOut
    buf19.layout = FixedLayout('cuda:0', torch.float32, size=[16, 64], stride=[64, 1])
    buf19.users = [NodeUser(node=SchedulerNode(name='op20'), can_inplace=True, is_weak=False)]
]
op19.node.kernel = extern_kernels.mm


op20: SchedulerNode(ComputedBuffer)
op20.writes = [MemoryDep('buf20', c0, {c0: 1024})]
op20.unmet_dependencies = [MemoryDep('buf19', c0, {c0: 1024})]
op20.met_dependencies = [MemoryDep('arg10_1', c1, {c0: 16, c1: 64})]
op20.min_input_distance = 9
op20.max_input_distance = 16
op20.outputs = [
    buf20: ComputedBuffer
    buf20.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 64], stride=[512, 64, 1])
    buf20.users = [NodeUser(node=ExternKernelSchedulerNode(name='op21'), can_inplace=False, is_weak=False)]
]
op20.group.device = cuda:0
op20.group.iteration = (1024, 1)
op20.sizes = ((16, 64), ())
arg10_1_layout = FixedLayout('cuda:0', torch.float32, size=[64], stride=[1])
buf19_layout = FixedLayout('cuda:0', torch.float32, size=[16, 64], stride=[64, 1])
buf20_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 64], stride=[512, 64, 1])
class op20_loop_body:
    var_ranges = {p0: 16, p1: 64}
    index0 = p1
    index1 = 64*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg10_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('buf19', get_index_1)
        add = ops.add(load, load_1)
        constant = ops.constant(0.5, torch.float32)
        mul = ops.mul(add, constant)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('arg10_1', get_index_2)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('buf19', get_index_3)
        add_1 = ops.add(load_2, load_3)
        constant_1 = ops.constant(0.7071067811865476, torch.float32)
        mul_1 = ops.mul(add_1, constant_1)
        erf = ops.erf(mul_1)
        constant_2 = ops.constant(1.0, torch.float32)
        add_2 = ops.add(erf, constant_2)
        mul_2 = ops.mul(mul, add_2)
        get_index_4 = self.get_index('index1')
        store = ops.store('buf20', get_index_4, mul_2, None)
        return store


op21: ExternKernelSchedulerNode(ExternKernelOut)
op21.writes = [StarDep(name='buf21', mode=None)]
op21.unmet_dependencies = [StarDep(name='buf20', mode=None)]
op21.met_dependencies = [StarDep(name='arg11_1', mode=None)]
op21.min_input_distance = 10
op21.max_input_distance = 17
op21.outputs = [
    buf21: ExternKernelOut
    buf21.layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
    buf21.users = [NodeUser(node=SchedulerNode(name='op22'), can_inplace=True, is_weak=False)]
]
op21.node.kernel = extern_kernels.mm


op22: SchedulerNode(ComputedBuffer)
op22.writes = [MemoryDep('buf22', c0, {c0: 256})]
op22.unmet_dependencies = [MemoryDep('buf15', c0, {c0: 256}), MemoryDep('buf21', c0, {c0: 256})]
op22.met_dependencies =
    [   MemoryDep('arg0_1', c0, {c0: 256}),
        MemoryDep('arg12_1', c1, {c0: 16, c1: 16}),
        MemoryDep('arg6_1', c1, {c0: 16, c1: 16})]
op22.min_input_distance = 7
op22.max_input_distance = 18
op22.outputs = [
    buf22: ComputedBuffer
    buf22.layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
    buf22.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op22.group.device = cuda:0
op22.group.iteration = (256, 1)
op22.sizes = ((16, 16), ())
arg0_1_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
arg6_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf15_layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
arg12_1_layout = FixedLayout('cuda:0', torch.float32, size=[16], stride=[1])
buf21_layout = FixedLayout('cuda:0', torch.float32, size=[16, 16], stride=[16, 1])
buf22_layout = FixedLayout('cuda:0', torch.float32, size=[2, 8, 16], stride=[128, 16, 1])
class op22_loop_body:
    var_ranges = {p0: 16, p1: 16}
    index0 = 16*p0 + p1
    index1 = p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg6_1', get_index_1)
        get_index_2 = self.get_index('index0')
        load_2 = ops.load('buf15', get_index_2)
        add = ops.add(load_1, load_2)
        add_1 = ops.add(load, add)
        get_index_3 = self.get_index('index1')
        load_3 = ops.load('arg12_1', get_index_3)
        get_index_4 = self.get_index('index0')
        load_4 = ops.load('buf21', get_index_4)
        add_2 = ops.add(load_3, load_4)
        add_3 = ops.add(add_1, add_2)
        get_index_5 = self.get_index('index0')
        store = ops.store('buf22', get_index_5, add_3, None)
        return store
```

### formatted_post_fusion

```text
# inputs
arg0_1: f32[2,8,16], arg1_1: f32[16], arg2_1: f32[16], arg3_1: f32[48,16], arg4_1: f32[48],
arg5_1: f32[16,16], arg6_1: f32[16], arg7_1: f32[16], arg8_1: f32[16], arg9_1: f32[64,16],
arg10_1: f32[64], arg11_1: f32[16,64], arg12_1: f32[16]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1, arg2_1]
    outputs:  [buf2]
    internal: [buf0, buf1]

    foreach x in [0,16), reduce r in [0,16):

        op0:  buf0: f32[2,8,1] @ [x:16]  <-  arg0_1: f32[2,8,16] @ [x:16,r:16]      # sum(arg0_1[r + 16*x] over r)
            t0 = arg0_1[r + 16*x]
            r0 = sum(t0 over r)
            buf0[x] = r0

        op1:  buf1: f32[2,8,1] @ [x:16]  <-  arg0_1: f32[2,8,16] @ [x:16,r:16], buf0: f32[2,8,1] @ [x:16]      # sum(((arg0_1[r + 16*x]) - (buf0[x] / 16.0)) * ((arg0_1[r + 16*x]) - (buf0[x] / 16.0)) over r)
            t0 = arg0_1[r + 16*x]
            t1 = buf0[x]
            t2 = t1 / 16.0
            t3 = t0 - t2
            t4 = t3 * t3
            r0 = sum(t4 over r)
            buf1[x] = r0

        op2:  buf2: f32[2,8,16] @ [x:16,r:16]  <-  arg0_1: f32[2,8,16] @ [x:16,r:16], buf0: f32[2,8,1] @ [x:16], buf1: f32[2,8,1] @ [x:16], arg1_1: f32[16], arg2_1: f32[16]      # ((((arg0_1[r + 16*x]) - (buf0[x] / 16.0)) * rsqrt((buf1[x] / 16.0) + 1e-05)) * arg1_1[r]) + arg2_1[r]
            ix0 = r + 16*x
            t0 = arg0_1[ix0]
            t1 = buf0[x]
            t2 = t1 / 16.0
            t3 = t0 - t2
            t4 = buf1[x]
            t5 = t4 / 16.0
            t6 = t5 + 1e-05
            t7 = rsqrt(t6)
            t8 = t3 * t7
            t9 = arg1_1[r]
            t10 = t8 * t9
            t11 = arg2_1[r]
            t12 = t10 + t11
            buf2[ix0] = t12

kernel k1
    # implementation: extern call
    inputs:   [arg3_1, buf2]
    outputs:  [buf3]

    op3:  buf3: f32[16,48]
        extern_kernels.mm(reinterpret_tensor(buf2, [16,16], [16,1], 0),
            reinterpret_tensor(arg3_1, [16,48], [1,16], 0))

kernel k2
    inputs:   [arg4_1, buf3]
    outputs:  [buf4]

    foreach x in [0,256):

        op4:  buf4: f32[2,2,8,8] @ [x:256]  <-  arg4_1: f32[48] @ [x:256], buf3: f32[16,48] @ [x:256]      # (arg4_1[(ModularIndexing(x, 1, 8)) + 8*(ModularIndexing(x, 64, 2))]) + (buf3[384*((x//128)) + (ModularIndexing(x, 1, 8)) + 48*(ModularIndexing(x, 8, 8)) + 8*(ModularIndexing(x, 64, 2))])
            t0 = arg4_1[(ModularIndexing(x, 1, 8)) + 8*(ModularIndexing(x, 64, 2))]
            t1 = buf3[384*((x//128)) + (ModularIndexing(x, 1, 8)) + 48*(ModularIndexing(x, 8, 8)) + 8*(ModularIndexing(x, 64, 2))]
            t2 = t0 + t1
            buf4[128*((x//128)) + (ModularIndexing(x, 1, 8)) + 8*(ModularIndexing(x, 8, 8)) + 64*(ModularIndexing(x, 64, 2))] = t2

kernel k3
    inputs:   [arg4_1, buf3]
    outputs:  [buf5]

    foreach y in [0,32), foreach x in [0,8):

        op5:  buf5: f32[2,2,8,8] @ [y:32,x:8]  <-  arg4_1: f32[48] @ [y:32], buf3: f32[16,48] @ [y:32,x:8]      # (arg4_1[(ModularIndexing(y, 1, 16)) + 16]) + (buf3[48*x + 384*((y//16)) + (ModularIndexing(y, 1, 16)) + 16])
            t0 = arg4_1[(ModularIndexing(y, 1, 16)) + 16]
            t1 = buf3[48*x + 384*((y//16)) + (ModularIndexing(y, 1, 16)) + 16]
            t2 = t0 + t1
            buf5[x + 128*((y//16)) + 8*(ModularIndexing(y, 1, 16))] = t2

kernel k4
    # implementation: extern call
    inputs:   [buf4, buf5]
    outputs:  [buf6]

    op6:  buf6: f32[4,8,8]
        extern_kernels.bmm(reinterpret_tensor(buf4, [4,8,8], [64,8,1], 0),
            reinterpret_tensor(buf5, [4,8,8], [64,8,1], 0))

kernel k5
    inputs:   [buf6]
    outputs:  [buf11]
    internal: [buf7, buf8, buf9, buf10]

    foreach x in [0,32), reduce r in [0,8):

        op7:  buf7: b8[2,2,8,1] @ [x:32]  <-  buf6: f32[4,8,8] @ [x:32,r:8]      # any(not ((((buf6[r + 8*x]) * 0.35355339059327373) == ((buf6[r + 8*x]) * 0.35355339059327373)) and (abs((buf6[r + 8*x]) * 0.35355339059327373) != inf)) over r)
            ix0 = r + 8*x
            t0 = buf6[ix0]
            t1 = t0 * 0.35355339059327373
            t2 = buf6[ix0]
            t3 = t2 * 0.35355339059327373
            t4 = t1 == t3
            t5 = buf6[ix0]
            t6 = t5 * 0.35355339059327373
            t7 = abs(t6)
            t8 = t7 != inf
            t9 = t4 and t8
            t10 = not t9
            r0 = any(t10 over r)
            buf7[x] = r0

        op8:  buf8: f32[2,2,8,1] @ [x:32]  <-  buf6: f32[4,8,8] @ [x:32,r:8]      # max((buf6[r + 8*x]) * 1.0 over r)
            t0 = buf6[r + 8*x]
            t1 = t0 * 1.0
            r0 = max(t1 over r)
            buf8[x] = r0

        op9:  buf9: f32[2,2,8,1] @ [x:32]  <-  buf6: f32[4,8,8] @ [x:32,r:8]      # max((buf6[r + 8*x]) * 0.35355339059327373 over r)
            t0 = buf6[r + 8*x]
            t1 = t0 * 0.35355339059327373
            r0 = max(t1 over r)
            buf9[x] = r0

        op10:  buf10: f32[2,2,8,1] @ [x:32]  <-  buf7: b8[2,2,8,1] @ [x:32], buf6: f32[4,8,8] @ [x:32,r:8], buf8: f32[2,2,8,1] @ [x:32], buf9: f32[2,2,8,1] @ [x:32]      # sum(exp(where(not buf7[x], (((buf6[r + 8*x]) * 1.0) - buf8[x]) * 0.35355339059327373, ((buf6[r + 8*x]) * 0.35355339059327373) - buf9[x])) over r)
            t0 = buf7[x]
            t1 = not t0
            ix0 = r + 8*x
            t2 = buf6[ix0]
            t3 = t2 * 1.0
            t4 = buf8[x]
            t5 = t3 - t4
            t6 = t5 * 0.35355339059327373
            t7 = buf6[ix0]
            t8 = t7 * 0.35355339059327373
            t9 = buf9[x]
            t10 = t8 - t9
            t11 = where(t1, t6, t10)
            t12 = exp(t11)
            r0 = sum(t12 over r)
            buf10[x] = r0

        op11:  buf11: f32[2,2,8,8] @ [x:32,r:8]  <-  buf7: b8[2,2,8,1] @ [x:32], buf6: f32[4,8,8] @ [x:32,r:8], buf8: f32[2,2,8,1] @ [x:32], buf9: f32[2,2,8,1] @ [x:32], buf10: f32[2,2,8,1] @ [x:32]      # exp(where(not buf7[x], (((buf6[r + 8*x]) * 1.0) - buf8[x]) * 0.35355339059327373, ((buf6[r + 8*x]) * 0.35355339059327373) - buf9[x])) / buf10[x]
            t0 = buf7[x]
            t1 = not t0
            ix0 = r + 8*x
            t2 = buf6[ix0]
            t3 = t2 * 1.0
            t4 = buf8[x]
            t5 = t3 - t4
            t6 = t5 * 0.35355339059327373
            t7 = buf6[ix0]
            t8 = t7 * 0.35355339059327373
            t9 = buf9[x]
            t10 = t8 - t9
            t11 = where(t1, t6, t10)
            t12 = exp(t11)
            t13 = buf10[x]
            t14 = t12 / t13
            buf11[ix0] = t14

kernel k6
    inputs:   [arg4_1, buf3]
    outputs:  [buf12]

    foreach x in [0,256):

        op12:  buf12: f32[2,2,8,8] @ [x:256]  <-  arg4_1: f32[48] @ [x:256], buf3: f32[16,48] @ [x:256]      # (arg4_1[(ModularIndexing(x, 1, 8)) + 8*(ModularIndexing(x, 64, 2)) + 32]) + (buf3[384*((x//128)) + (ModularIndexing(x, 1, 8)) + 48*(ModularIndexing(x, 8, 8)) + 8*(ModularIndexing(x, 64, 2)) + 32])
            t0 = arg4_1[(ModularIndexing(x, 1, 8)) + 8*(ModularIndexing(x, 64, 2)) + 32]
            t1 = buf3[384*((x//128)) + (ModularIndexing(x, 1, 8)) + 48*(ModularIndexing(x, 8, 8)) + 8*(ModularIndexing(x, 64, 2)) + 32]
            t2 = t0 + t1
            buf12[128*((x//128)) + (ModularIndexing(x, 1, 8)) + 8*(ModularIndexing(x, 8, 8)) + 64*(ModularIndexing(x, 64, 2))] = t2

kernel k7
    # implementation: extern call
    inputs:   [buf11, buf12]
    outputs:  [buf13]

    op13:  buf13: f32[4,8,8]
        extern_kernels.bmm(reinterpret_tensor(buf11, [4,8,8], [64,8,1], 0),
            reinterpret_tensor(buf12, [4,8,8], [64,8,1], 0))

kernel k8
    inputs:   [buf13]
    outputs:  [buf14]

    foreach x in [0,256):

        op14:  buf14: f32[2,8,2,8] @ [x:256]  <-  buf13: f32[4,8,8] @ [x:256]      # copy
            t0 = buf13[128*((x//128)) + (ModularIndexing(x, 1, 8)) + 64*(ModularIndexing(x, 8, 2)) + 8*(ModularIndexing(x, 16, 8))]
            buf14[128*((x//128)) + (ModularIndexing(x, 1, 8)) + 8*(ModularIndexing(x, 8, 2)) + 16*(ModularIndexing(x, 16, 8))] = t0

kernel k9
    # implementation: extern call
    inputs:   [arg5_1, buf14]
    outputs:  [buf15]

    op15:  buf15: f32[16,16]
        extern_kernels.mm(reinterpret_tensor(buf14, [16,16], [16,1], 0),
            reinterpret_tensor(arg5_1, [16,16], [1,16], 0))

kernel k10
    inputs:   [arg0_1, arg6_1, arg7_1, arg8_1, buf15]
    outputs:  [buf18]
    internal: [buf16, buf17]

    foreach x in [0,16), reduce r in [0,16):

        op16:  buf16: f32[2,8,1] @ [x:16]  <-  arg0_1: f32[2,8,16] @ [x:16,r:16], arg6_1: f32[16], buf15: f32[16,16]      # sum((arg0_1[r + 16*x]) + (arg6_1[r] + (buf15[r + 16*x])) over r)
            ix0 = r + 16*x
            t0 = arg0_1[ix0]
            t1 = arg6_1[r]
            t2 = buf15[ix0]
            t3 = t1 + t2
            t4 = t0 + t3
            r0 = sum(t4 over r)
            buf16[x] = r0

        op17:  buf17: f32[2,8,1] @ [x:16]  <-  arg0_1: f32[2,8,16] @ [x:16,r:16], arg6_1: f32[16], buf15: f32[16,16], buf16: f32[2,8,1] @ [x:16]      # sum((((arg0_1[r + 16*x]) + (arg6_1[r] + (buf15[r + 16*x]))) - (buf16[x] / 16.0)) * (((arg0_1[r + 16*x]) + (arg6_1[r] + (buf15[r + 16*x]))) - (buf16[x] / 16.0)) over r)
            ix0 = r + 16*x
            t0 = arg0_1[ix0]
            t1 = arg6_1[r]
            t2 = buf15[ix0]
            t3 = t1 + t2
            t4 = t0 + t3
            t5 = buf16[x]
            t6 = t5 / 16.0
            t7 = t4 - t6
            t8 = t7 * t7
            r0 = sum(t8 over r)
            buf17[x] = r0

        op18:  buf18: f32[2,8,16] @ [x:16,r:16]  <-  arg0_1: f32[2,8,16] @ [x:16,r:16], arg6_1: f32[16], buf15: f32[16,16], buf16: f32[2,8,1] @ [x:16], buf17: f32[2,8,1] @ [x:16], arg7_1: f32[16], arg8_1: f32[16]      # (((((arg0_1[r + 16*x]) + (arg6_1[r] + (buf15[r + 16*x]))) - (buf16[x] / 16.0)) * rsqrt((buf17[x] / 16.0) + 1e-05)) * arg7_1[r]) + arg8_1[r]
            ix0 = r + 16*x
            t0 = arg0_1[ix0]
            t1 = arg6_1[r]
            t2 = buf15[ix0]
            t3 = t1 + t2
            t4 = t0 + t3
            t5 = buf16[x]
            t6 = t5 / 16.0
            t7 = t4 - t6
            t8 = buf17[x]
            t9 = t8 / 16.0
            t10 = t9 + 1e-05
            t11 = rsqrt(t10)
            t12 = t7 * t11
            t13 = arg7_1[r]
            t14 = t12 * t13
            t15 = arg8_1[r]
            t16 = t14 + t15
            buf18[ix0] = t16

kernel k11
    # implementation: extern call
    inputs:   [arg9_1, buf18]
    outputs:  [buf19]

    op19:  buf19: f32[16,64]
        extern_kernels.mm(reinterpret_tensor(buf18, [16,16], [16,1], 0),
            reinterpret_tensor(arg9_1, [16,64], [1,16], 0))

kernel k12
    inputs:   [arg10_1, buf19]
    outputs:  [buf20]

    foreach x in [0,1024):

        op20:  buf20: f32[2,8,64] @ [x:1024]  <-  arg10_1: f32[64] @ [x:1024], buf19: f32[16,64] @ [x:1024]      # ((arg10_1[ModularIndexing(x, 1, 64)] + (buf19[64*((x//64)) + (ModularIndexing(x, 1, 64))])) * 0.5) * (erf((arg10_1[ModularIndexing(x, 1, 64)] + (buf19[64*((x//64)) + (ModularIndexing(x, 1, 64))])) * 0.7071067811865476) + 1.0)
            ix0 = ModularIndexing(x, 1, 64)
            t0 = arg10_1[ix0]
            ix1 = 64*((x//64)) + (ModularIndexing(x, 1, 64))
            t1 = buf19[ix1]
            t2 = t0 + t1
            t3 = t2 * 0.5
            t4 = arg10_1[ix0]
            t5 = buf19[ix1]
            t6 = t4 + t5
            t7 = t6 * 0.7071067811865476
            t8 = erf(t7)
            t9 = t8 + 1.0
            t10 = t3 * t9
            buf20[ix1] = t10

kernel k13
    # implementation: extern call
    inputs:   [arg11_1, buf20]
    outputs:  [buf21]

    op21:  buf21: f32[16,16]
        extern_kernels.mm(reinterpret_tensor(buf20, [16,64], [64,1], 0),
            reinterpret_tensor(arg11_1, [64,16], [1,64], 0))

kernel k14
    inputs:   [arg0_1, arg12_1, arg6_1, buf15, buf21]
    outputs:  [buf22]

    foreach x in [0,256):

        op22:  buf22: f32[2,8,16] @ [x:256]  <-  arg0_1: f32[2,8,16] @ [x:256], arg6_1: f32[16] @ [x:256], buf15: f32[16,16] @ [x:256], arg12_1: f32[16] @ [x:256], buf21: f32[16,16] @ [x:256]      # ((arg0_1[16*((x//16)) + (ModularIndexing(x, 1, 16))]) + (arg6_1[ModularIndexing(x, 1, 16)] + (buf15[16*((x//16)) + (ModularIndexing(x, 1, 16))]))) + (arg12_1[ModularIndexing(x, 1, 16)] + (buf21[16*((x//16)) + (ModularIndexing(x, 1, 16))]))
            ix0 = 16*((x//16)) + (ModularIndexing(x, 1, 16))
            t0 = arg0_1[ix0]
            ix1 = ModularIndexing(x, 1, 16)
            t1 = arg6_1[ix1]
            t2 = buf15[ix0]
            t3 = t1 + t2
            t4 = t0 + t3
            t5 = arg12_1[ix1]
            t6 = buf21[ix0]
            t7 = t5 + t6
            t8 = t4 + t7
            buf22[ix0] = t8

# outputs
return (buf22)
```

## Program 30: softmax over a viewed bf16 tensor — bf16 cast, expand/view, and permuted alias outputs

### post_grad_fx

```python
class <lambda>(torch.nn.Module):
    def forward(self, arg0_1: "bf16[16, 128, 128]", arg1_1: "f32[1, 1, 128, 128]"):
        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:81 in forward, code: view = arg0_1.view(1, 16, 128, 128)
        view: "bf16[1, 16, 128, 128]" = torch.ops.aten.reshape.default(arg0_1, [1, 16, 128, 128]);  arg0_1 = None

        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:82 in forward, code: div = view / 16.0
        div: "bf16[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(view, 16.0);  view = None

        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:83 in forward, code: add = div + arg1_1
        add: "f32[1, 16, 128, 128]" = torch.ops.aten.add.Tensor(div, arg1_1);  div = arg1_1 = None

        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:84 in forward, code: amax = torch.amax(add, [-1], True)
        amax: "f32[1, 16, 128, 1]" = torch.ops.aten.amax.default(add, [-1], True)

        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:85 in forward, code: sub = add - amax
        sub: "f32[1, 16, 128, 128]" = torch.ops.aten.sub.Tensor(add, amax);  add = amax = None

        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:86 in forward, code: exp = torch.exp(sub)
        exp: "f32[1, 16, 128, 128]" = torch.ops.aten.exp.default(sub);  sub = None

        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:87 in forward, code: sum_1 = torch.sum(exp, [-1], True)
        sum_1: "f32[1, 16, 128, 1]" = torch.ops.aten.sum.dim_IntList(exp, [-1], True)

        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:88 in forward, code: div_1 = exp / sum_1
        div_1: "f32[1, 16, 128, 128]" = torch.ops.aten.div.Tensor(exp, sum_1);  exp = sum_1 = None

        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:89 in forward, code: convert_element_type = div_1.to(torch.bfloat16)
        convert_element_type: "bf16[1, 16, 128, 128]" = torch.ops.prims.convert_element_type.default(div_1, torch.bfloat16)

        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:90 in forward, code: expand = convert_element_type.expand(1, 16, 128, 128)
        expand: "bf16[1, 16, 128, 128]" = torch.ops.aten.expand.default(convert_element_type, [1, 16, 128, 128]);  convert_element_type = None

        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:91 in forward, code: view_1 = expand.view(16, 128, 128)
        view_1: "bf16[16, 128, 128]" = torch.ops.aten.reshape.default(expand, [16, 128, 128]);  expand = None

        # File: /home/lsakka/pytorch10/pytorch/ir_printer_examples.py:92 in forward, code: permute = view_1.permute(0, 2, 1)
        permute: "bf16[16, 128, 128]" = torch.ops.aten.permute.default(view_1, [0, 2, 1])
        return (div_1, view_1, permute)
```

### post_lowering

```text
ComputedBuffer(name='buf0', layout=FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      _, i1, i2, _ = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 128 * i2 + 16384 * i1)
      tmp1 = ops.constant(0.0625, torch.bfloat16)
      tmp2 = tmp0 * tmp1
      tmp3 = ops.to_dtype(tmp2, torch.float32, src_dtype=torch.bfloat16, use_compute_types=True)
      tmp4 = ops.load(arg1_1, r0_0 + 128 * i2)
      tmp5 = tmp3 + tmp4
      return tmp5
  ,
  ranges=[1, 16, 128, 1],
  reduction_ranges=[128],
  reduction_type=max,
  origin_node=amax,
  origins=OrderedSet([amax, add, div, view]),
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 84, in forward,
      amax = torch.amax(add, [-1], True),
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 83, in forward,
      add = div + arg1_1,
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 82, in forward,
      div = view / 16.0,
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 81, in forward,
      view = arg0_1.view(1, 16, 128, 128),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf1', layout=FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048]), data=Reduction(
  'cuda',
  torch.float32,
  def inner_fn(index, rindex):
      _, i1, i2, _ = index
      r0_0 = rindex
      tmp0 = ops.load(arg0_1, r0_0 + 128 * i2 + 16384 * i1)
      tmp1 = ops.constant(0.0625, torch.bfloat16)
      tmp2 = tmp0 * tmp1
      tmp3 = ops.to_dtype(tmp2, torch.float32, src_dtype=torch.bfloat16, use_compute_types=True)
      tmp4 = ops.load(arg1_1, r0_0 + 128 * i2)
      tmp5 = tmp3 + tmp4
      tmp6 = ops.load(buf0, i2 + 128 * i1)
      tmp7 = tmp5 - tmp6
      tmp8 = ops.exp(tmp7)
      return tmp8
  ,
  ranges=[1, 16, 128, 1],
  reduction_ranges=[128],
  reduction_type=sum,
  origin_node=sum_1,
  origins=OrderedSet([sum_1, exp, sub, add, div, view]),
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 87, in forward,
      sum_1 = torch.sum(exp, [-1], True),
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 86, in forward,
      exp = torch.exp(sub),
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 85, in forward,
      sub = add - amax,
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 83, in forward,
      add = div + arg1_1,
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 82, in forward,
      div = view / 16.0,
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 81, in forward,
      view = arg0_1.view(1, 16, 128, 128),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf2', layout=FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1]), data=Pointwise(
  'cuda',
  torch.float32,
  def inner_fn(index):
      _, i1, i2, i3 = index
      tmp0 = ops.load(arg0_1, i3 + 128 * i2 + 16384 * i1)
      tmp1 = ops.constant(0.0625, torch.bfloat16)
      tmp2 = tmp0 * tmp1
      tmp3 = ops.to_dtype(tmp2, torch.float32, src_dtype=torch.bfloat16, use_compute_types=True)
      tmp4 = ops.load(arg1_1, i3 + 128 * i2)
      tmp5 = tmp3 + tmp4
      tmp6 = ops.load(buf0, i2 + 128 * i1)
      tmp7 = tmp5 - tmp6
      tmp8 = ops.exp(tmp7)
      tmp9 = ops.load(buf1, i2 + 128 * i1)
      tmp10 = tmp8 / tmp9
      return tmp10
  ,
  ranges=[1, 16, 128, 128],
  origin_node=div_1,
  origins=OrderedSet([div_1, exp, sub, add, div, view]),
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 88, in forward,
      div_1 = exp / sum_1,
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 86, in forward,
      exp = torch.exp(sub),
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 85, in forward,
      sub = add - amax,
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 83, in forward,
      add = div + arg1_1,
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 82, in forward,
      div = view / 16.0,
  ,
  },
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 81, in forward,
      view = arg0_1.view(1, 16, 128, 128),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)


ComputedBuffer(name='buf3', layout=FixedLayout('cuda:0', torch.bfloat16, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1]), data=Pointwise(
  'cuda',
  torch.bfloat16,
  def inner_fn(index):
      _, i1, i2, i3 = index
      tmp0 = ops.load(buf2, i3 + 128 * i2 + 16384 * i1)
      tmp1 = ops.to_dtype(tmp0, torch.bfloat16, src_dtype=torch.float32, use_compute_types=True)
      return tmp1
  ,
  ranges=[1, 16, 128, 128],
  origin_node=expand,
  origins=OrderedSet([convert_element_type]),
  stack_traces = {,
    File "/home/lsakka/pytorch10/pytorch/ir_printer_examples.py", line 89, in forward,
      convert_element_type = div_1.to(torch.bfloat16),
  ,
  }
), _split_size=None, _original_inner_fn=None, _original_ranges=None, _original_reduction_ranges=None)
```

### formatted_post_lowering

```text
# device: cuda:0

# Inductor IR, post-lowering (bodies from get_default_sizes_body).
# Backend-agnostic: ranges and index expressions only.
# Loop order is the as-lowered order -- usable as-is, but no stride-based
# reordering and no dim merging has run.
#
#   foreach p0 in [0,N)   independent points; any order, any parallelism.
#                         Listed outermost to innermost; the last one varies fastest.
#   reduce  p1 in [0,N)   points combined; order unspecified
#   buf[expr]             expr is a FLAT element offset, not a subscript
#   rN = OP(v over p1)    rN is v combined across p1, one per foreach point
#   indentation           the body of the loop above

# inputs
arg0_1: bf16[16,128,128], arg1_1: f32[1,1,128,128]

# compute

# op0  max(to_dtype((arg0_1[16384*p0 + 128*p1 + p2]) * 0.0625, torch.float32) + (arg1_1[128*p1 + p2]) over p2)
buf0: f32[1,16,128,1]  <-  arg0_1: bf16[16,128,128], arg1_1: f32[1,1,128,128]
    foreach p0 in [0,16), p1 in [0,128), reduce p2 in [0,128):
        t0 = arg0_1[16384*p0 + 128*p1 + p2]
        t1 = t0 * 0.0625
        t2 = to_dtype(t1, f32)
        t3 = arg1_1[128*p1 + p2]
        t4 = t2 + t3
        r0 = max(t4 over p2)
        buf0[128*p0 + p1] = r0

# op1  sum(exp((to_dtype((arg0_1[16384*p0 + 128*p1 + p2]) * 0.0625, torch.float32) + (arg1_1[128*p1 + p2])) - (buf0[128*p0 + p1])) over p2)
buf1: f32[1,16,128,1]  <-  arg0_1: bf16[16,128,128], arg1_1: f32[1,1,128,128], buf0: f32[1,16,128,1]
    foreach p0 in [0,16), p1 in [0,128), reduce p2 in [0,128):
        t0 = arg0_1[16384*p0 + 128*p1 + p2]
        t1 = t0 * 0.0625
        t2 = to_dtype(t1, f32)
        t3 = arg1_1[128*p1 + p2]
        t4 = t2 + t3
        ix0 = 128*p0 + p1
        t5 = buf0[ix0]
        t6 = t4 - t5
        t7 = exp(t6)
        r0 = sum(t7 over p2)
        buf1[ix0] = r0

# op2  exp((to_dtype((arg0_1[16384*p0 + 128*p1 + p2]) * 0.0625, torch.float32) + (arg1_1[128*p1 + p2])) - (buf0[128*p0 + p1])) / (buf1[128*p0 + p1])
buf2: f32[1,16,128,128]  <-  arg0_1: bf16[16,128,128], arg1_1: f32[1,1,128,128], buf0: f32[1,16,128,1], buf1: f32[1,16,128,1]
    foreach p0 in [0,16), p1 in [0,128), p2 in [0,128):
        ix0 = 16384*p0 + 128*p1 + p2
        t0 = arg0_1[ix0]
        t1 = t0 * 0.0625
        t2 = to_dtype(t1, f32)
        t3 = arg1_1[128*p1 + p2]
        t4 = t2 + t3
        ix1 = 128*p0 + p1
        t5 = buf0[ix1]
        t6 = t4 - t5
        t7 = exp(t6)
        t8 = buf1[ix1]
        t9 = t7 / t8
        buf2[ix0] = t9

# op3  to_dtype(buf2[16384*p0 + 128*p1 + p2], torch.bfloat16)
buf3: bf16[1,16,128,128]  <-  buf2: f32[1,16,128,128]
    foreach p0 in [0,16), p1 in [0,128), p2 in [0,128):
        ix0 = 16384*p0 + 128*p1 + p2
        t0 = buf2[ix0]
        t1 = to_dtype(t0, bf16)
        buf3[ix0] = t1

# outputs
return (buf2, reinterpret_tensor(buf3, [16,128,128], [16384,128,1], 0), reinterpret_tensor(buf3, [16,128,128], [16384,1,128], 0))
```

### post_scheduler

```text
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', 128*d0 + d1, {d0: 16, d1: 128})]
op0.unmet_dependencies = []
op0.met_dependencies = 
    [   MemoryDep('arg0_1', 16384*d0 + 128*d1 + d2, {d0: 16, d1: 128, d2: 128}),
        MemoryDep('arg1_1', 128*d1 + d2, {d0: 16, d1: 128, d2: 128})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
    buf0.users = [
        NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False),
    ]
]
op0.group.device = cuda:0
op0.group.iteration = (2048, 128)
op0.sizes = ([16, 128], [128])
arg0_1_layout = FixedLayout('cuda:0', torch.bfloat16, size=[16, 128, 128], stride=[16384, 128, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 1, 128, 128], stride=[16384, 16384, 128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
class op0_loop_body:
    var_ranges = {p0: 16, p1: 128, p2: 128}
    index0 = 16384*p0 + 128*p1 + p2
    index1 = 128*p1 + p2
    index2 = 128*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(0.0625, torch.bfloat16)
        mul = ops.mul(load, constant)
        to_dtype = ops.to_dtype(mul, torch.float32, src_dtype = torch.bfloat16, use_compute_types = True)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(to_dtype, load_1)
        reduction = ops.reduction(torch.float32, torch.float32, 'max', add)
        get_index_2 = self.get_index('index2')
        store_reduction = ops.store_reduction('buf0', get_index_2, reduction)
        return None


op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', 128*d0 + d1, {d0: 16, d1: 128})]
op1.unmet_dependencies = [MemoryDep('buf0', 128*d0 + d1, {d0: 16, d1: 128})]
op1.met_dependencies = 
    [   MemoryDep('arg0_1', 16384*d0 + 128*d1 + d2, {d0: 16, d1: 128, d2: 128}),
        MemoryDep('arg1_1', 128*d1 + d2, {d0: 16, d1: 128, d2: 128})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (2048, 128)
op1.sizes = ([16, 128], [128])
arg0_1_layout = FixedLayout('cuda:0', torch.bfloat16, size=[16, 128, 128], stride=[16384, 128, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 1, 128, 128], stride=[16384, 16384, 128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
class op1_loop_body:
    var_ranges = {p0: 16, p1: 128, p2: 128}
    index0 = 16384*p0 + 128*p1 + p2
    index1 = 128*p1 + p2
    index2 = 128*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(0.0625, torch.bfloat16)
        mul = ops.mul(load, constant)
        to_dtype = ops.to_dtype(mul, torch.float32, src_dtype = torch.bfloat16, use_compute_types = True)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(to_dtype, load_1)
        get_index_2 = self.get_index('index2')
        load_2 = ops.load('buf0', get_index_2)
        sub = ops.sub(add, load_2)
        exp = ops.exp(sub)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', exp)
        get_index_3 = self.get_index('index2')
        store_reduction = ops.store_reduction('buf1', get_index_3, reduction)
        return None


op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', 16384*d0 + 128*d1 + d2, {d0: 16, d1: 128, d2: 128})]
op2.unmet_dependencies = 
    [   MemoryDep('buf0', 128*d0 + d1, {d0: 16, d1: 128}),
        MemoryDep('buf1', 128*d0 + d1, {d0: 16, d1: 128})]
op2.met_dependencies = 
    [   MemoryDep('arg0_1', 16384*d0 + 128*d1 + d2, {d0: 16, d1: 128, d2: 128}),
        MemoryDep('arg1_1', 128*d1 + d2, {d0: 16, d1: 128, d2: 128})]
op2.min_input_distance = 1
op2.max_input_distance = 2
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
    buf2.users = [
        NodeUser(node=SchedulerNode(name='op3'), can_inplace=True, is_weak=False),
        NodeUser(node=OUTPUT, can_inplace=False, is_weak=False),
    ]
]
op2.group.device = cuda:0
op2.group.iteration = (262144, 1)
op2.sizes = ([16, 128, 128], [])
arg0_1_layout = FixedLayout('cuda:0', torch.bfloat16, size=[16, 128, 128], stride=[16384, 128, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 1, 128, 128], stride=[16384, 16384, 128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
class op2_loop_body:
    var_ranges = {p0: 16, p1: 128, p2: 128}
    index0 = 16384*p0 + 128*p1 + p2
    index1 = 128*p1 + p2
    index2 = 128*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(0.0625, torch.bfloat16)
        mul = ops.mul(load, constant)
        to_dtype = ops.to_dtype(mul, torch.float32, src_dtype = torch.bfloat16, use_compute_types = True)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(to_dtype, load_1)
        get_index_2 = self.get_index('index2')
        load_2 = ops.load('buf0', get_index_2)
        sub = ops.sub(add, load_2)
        exp = ops.exp(sub)
        get_index_3 = self.get_index('index2')
        load_3 = ops.load('buf1', get_index_3)
        truediv = ops.truediv(exp, load_3)
        get_index_4 = self.get_index('index0')
        store = ops.store('buf2', get_index_4, truediv, None)
        return store


op3: SchedulerNode(ComputedBuffer)
op3.writes = [MemoryDep('buf3', 16384*d0 + 128*d1 + d2, {d0: 16, d1: 128, d2: 128})]
op3.unmet_dependencies = [MemoryDep('buf2', 16384*d0 + 128*d1 + d2, {d0: 16, d1: 128, d2: 128})]
op3.met_dependencies = []
op3.min_input_distance = 2
op3.max_input_distance = 3
op3.outputs = [
    buf3: ComputedBuffer
    buf3.layout = FixedLayout('cuda:0', torch.bfloat16, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
    buf3.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op3.group.device = cuda:0
op3.group.iteration = (262144, 1)
op3.sizes = ([16, 128, 128], [])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
buf3_layout = FixedLayout('cuda:0', torch.bfloat16, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
class op3_loop_body:
    var_ranges = {p0: 16, p1: 128, p2: 128}
    index0 = 16384*p0 + 128*p1 + p2
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf2', get_index)
        to_dtype = ops.to_dtype(load, torch.bfloat16, src_dtype = torch.float32, use_compute_types = True)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf3', get_index_1, to_dtype, None)
        return store
```

### formatted_post_scheduler

```text
# device: cuda:0

# Inductor IR, post scheduler-node construction (simplify_and_reorder applied).
# Backend-agnostic: ranges and index expressions only.
# Loop order optimized by pick_loop_order (small strides innermost); fusion can
# still change it, just less often. Dim merging may still be pending -- see
# config.loop_ordering_after_fusion.
#
#   foreach p0 in [0,N)   independent points; any order, any parallelism.
#                         Listed outermost to innermost; the last one varies fastest.
#   reduce  p1 in [0,N)   points combined; order unspecified
#   buf[expr]             expr is a FLAT element offset, not a subscript
#   rN = OP(v over p1)    rN is v combined across p1, one per foreach point
#   indentation           the body of the loop above

# inputs
arg0_1: bf16[16,128,128], arg1_1: f32[1,1,128,128]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf0]

    op0:  buf0: f32[1,16,128,1]  <-  arg0_1: bf16[16,128,128], arg1_1: f32[1,1,128,128]      # max(to_dtype((arg0_1[16384*p0 + 128*p1 + p2]) * 0.0625, torch.float32) + (arg1_1[128*p1 + p2]) over p2)
        foreach p0 in [0,16), p1 in [0,128), reduce p2 in [0,128):
            t0 = arg0_1[16384*p0 + 128*p1 + p2]
            t1 = t0 * 0.0625
            t2 = to_dtype(t1, f32)
            t3 = arg1_1[128*p1 + p2]
            t4 = t2 + t3
            r0 = max(t4 over p2)
            buf0[128*p0 + p1] = r0

kernel k1
    inputs:   [arg0_1, arg1_1, buf0]
    outputs:  [buf1]

    op1:  buf1: f32[1,16,128,1]  <-  arg0_1: bf16[16,128,128], arg1_1: f32[1,1,128,128], buf0: f32[1,16,128,1]      # sum(exp((to_dtype((arg0_1[16384*p0 + 128*p1 + p2]) * 0.0625, torch.float32) + (arg1_1[128*p1 + p2])) - (buf0[128*p0 + p1])) over p2)
        foreach p0 in [0,16), p1 in [0,128), reduce p2 in [0,128):
            t0 = arg0_1[16384*p0 + 128*p1 + p2]
            t1 = t0 * 0.0625
            t2 = to_dtype(t1, f32)
            t3 = arg1_1[128*p1 + p2]
            t4 = t2 + t3
            ix0 = 128*p0 + p1
            t5 = buf0[ix0]
            t6 = t4 - t5
            t7 = exp(t6)
            r0 = sum(t7 over p2)
            buf1[ix0] = r0

kernel k2
    inputs:   [arg0_1, arg1_1, buf0, buf1]
    outputs:  [buf2]

    op2:  buf2: f32[1,16,128,128]  <-  arg0_1: bf16[16,128,128], arg1_1: f32[1,1,128,128], buf0: f32[1,16,128,1], buf1: f32[1,16,128,1]      # exp((to_dtype((arg0_1[16384*p0 + 128*p1 + p2]) * 0.0625, torch.float32) + (arg1_1[128*p1 + p2])) - (buf0[128*p0 + p1])) / (buf1[128*p0 + p1])
        foreach p0 in [0,16), p1 in [0,128), p2 in [0,128):
            ix0 = 16384*p0 + 128*p1 + p2
            t0 = arg0_1[ix0]
            t1 = t0 * 0.0625
            t2 = to_dtype(t1, f32)
            t3 = arg1_1[128*p1 + p2]
            t4 = t2 + t3
            ix1 = 128*p0 + p1
            t5 = buf0[ix1]
            t6 = t4 - t5
            t7 = exp(t6)
            t8 = buf1[ix1]
            t9 = t7 / t8
            buf2[ix0] = t9

kernel k3
    inputs:   [buf2]
    internal: [buf3]

    op3:  buf3: bf16[1,16,128,128]  <-  buf2: f32[1,16,128,128]      # to_dtype(buf2[16384*p0 + 128*p1 + p2], torch.bfloat16)
        foreach p0 in [0,16), p1 in [0,128), p2 in [0,128):
            ix0 = 16384*p0 + 128*p1 + p2
            t0 = buf2[ix0]
            t1 = to_dtype(t0, bf16)
            buf3[ix0] = t1

# outputs
return (buf2, reinterpret_tensor(buf3, [16,128,128], [16384,128,1], 0), reinterpret_tensor(buf3, [16,128,128], [16384,1,128], 0))
```

### post_fusion

```text
op0_op1_op2_op3: FusedSchedulerNode(SchedulerNode,SchedulerNode,SchedulerNode,SchedulerNode)
op0_op1_op2_op3.writes = 
    [   MemoryDep('buf0', 128*d0 + d1, {d0: 16, d1: 128}),
        MemoryDep('buf1', 128*d0 + d1, {d0: 16, d1: 128}),
        MemoryDep('buf2', 16384*d0 + 128*d1 + d2, {d0: 16, d1: 128, d2: 128}),
        MemoryDep('buf3', 16384*d0 + 128*d1 + d2, {d0: 16, d1: 128, d2: 128})]
op0_op1_op2_op3.unmet_dependencies = []
op0_op1_op2_op3.met_dependencies = 
    [   MemoryDep('arg0_1', 16384*d0 + 128*d1 + d2, {d0: 16, d1: 128, d2: 128}),
        MemoryDep('arg1_1', 128*d1 + d2, {d0: 16, d1: 128, d2: 128})]
op0_op1_op2_op3.min_input_distance = 0
op0_op1_op2_op3.max_input_distance = 3
op0_op1_op2_op3.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
    buf0.users = [
        NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False),
    ]
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
    buf2.users = [
        NodeUser(node=SchedulerNode(name='op3'), can_inplace=True, is_weak=False),
        NodeUser(node=OUTPUT, can_inplace=False, is_weak=False),
    ]
    buf3: ComputedBuffer
    buf3.layout = FixedLayout('cuda:0', torch.bfloat16, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
    buf3.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op0_op1_op2_op3.snodes[0] =
op0: SchedulerNode(ComputedBuffer)
op0.writes = [MemoryDep('buf0', c0, {c0: 2048})]
op0.unmet_dependencies = []
op0.met_dependencies = 
    [   MemoryDep('arg0_1', c0, {c0: 262144}),
        MemoryDep('arg1_1', c1, {c0: 16, c1: 16384})]
op0.min_input_distance = 0
op0.max_input_distance = 0
op0.outputs = [
    buf0: ComputedBuffer
    buf0.layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
    buf0.users = [
        NodeUser(node=SchedulerNode(name='op1'), can_inplace=True, is_weak=False),
        NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False),
    ]
]
op0.group.device = cuda:0
op0.group.iteration = (2048, 128)
op0.sizes = ((16, 128), (128,))
arg0_1_layout = FixedLayout('cuda:0', torch.bfloat16, size=[16, 128, 128], stride=[16384, 128, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 1, 128, 128], stride=[16384, 16384, 128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
class op0_loop_body:
    var_ranges = {p0: 16, p1: 128, p2: 128}
    index0 = 16384*p0 + 128*p1 + p2
    index1 = 128*p1 + p2
    index2 = 128*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(0.0625, torch.bfloat16)
        mul = ops.mul(load, constant)
        to_dtype = ops.to_dtype(mul, torch.float32, src_dtype = torch.bfloat16, use_compute_types = True)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(to_dtype, load_1)
        reduction = ops.reduction(torch.float32, torch.float32, 'max', add)
        get_index_2 = self.get_index('index2')
        store_reduction = ops.store_reduction('buf0', get_index_2, reduction)
        return None
op0_op1_op2_op3.snodes[1] =
op1: SchedulerNode(ComputedBuffer)
op1.writes = [MemoryDep('buf1', c0, {c0: 2048})]
op1.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 2048})]
op1.met_dependencies = 
    [   MemoryDep('arg0_1', c0, {c0: 262144}),
        MemoryDep('arg1_1', c1, {c0: 16, c1: 16384})]
op1.min_input_distance = 1
op1.max_input_distance = 1
op1.outputs = [
    buf1: ComputedBuffer
    buf1.layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
    buf1.users = [NodeUser(node=SchedulerNode(name='op2'), can_inplace=False, is_weak=False)]
]
op1.group.device = cuda:0
op1.group.iteration = (2048, 128)
op1.sizes = ((16, 128), (128,))
arg0_1_layout = FixedLayout('cuda:0', torch.bfloat16, size=[16, 128, 128], stride=[16384, 128, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 1, 128, 128], stride=[16384, 16384, 128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
class op1_loop_body:
    var_ranges = {p0: 16, p1: 128, p2: 128}
    index0 = 16384*p0 + 128*p1 + p2
    index1 = 128*p1 + p2
    index2 = 128*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(0.0625, torch.bfloat16)
        mul = ops.mul(load, constant)
        to_dtype = ops.to_dtype(mul, torch.float32, src_dtype = torch.bfloat16, use_compute_types = True)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(to_dtype, load_1)
        get_index_2 = self.get_index('index2')
        load_2 = ops.load('buf0', get_index_2)
        sub = ops.sub(add, load_2)
        exp = ops.exp(sub)
        reduction = ops.reduction(torch.float32, torch.float32, 'sum', exp)
        get_index_3 = self.get_index('index2')
        store_reduction = ops.store_reduction('buf1', get_index_3, reduction)
        return None
op0_op1_op2_op3.snodes[2] =
op2: SchedulerNode(ComputedBuffer)
op2.writes = [MemoryDep('buf2', c0, {c0: 262144})]
op2.unmet_dependencies = [MemoryDep('buf0', c0, {c0: 2048}), MemoryDep('buf1', c0, {c0: 2048})]
op2.met_dependencies = 
    [   MemoryDep('arg0_1', c0, {c0: 262144}),
        MemoryDep('arg1_1', c1, {c0: 16, c1: 16384})]
op2.min_input_distance = 1
op2.max_input_distance = 2
op2.outputs = [
    buf2: ComputedBuffer
    buf2.layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
    buf2.users = [
        NodeUser(node=SchedulerNode(name='op3'), can_inplace=True, is_weak=False),
        NodeUser(node=OUTPUT, can_inplace=False, is_weak=False),
    ]
]
op2.group.device = cuda:0
op2.group.iteration = (262144, 1)
op2.sizes = ((16, 128, 128), ())
arg0_1_layout = FixedLayout('cuda:0', torch.bfloat16, size=[16, 128, 128], stride=[16384, 128, 1])
arg1_1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 1, 128, 128], stride=[16384, 16384, 128, 1])
buf0_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
buf1_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 1], stride=[2048, 128, 1, 2048])
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
class op2_loop_body:
    var_ranges = {p0: 16, p1: 128, p2: 128}
    index0 = 16384*p0 + 128*p1 + p2
    index1 = 128*p1 + p2
    index2 = 128*p0 + p1
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('arg0_1', get_index)
        constant = ops.constant(0.0625, torch.bfloat16)
        mul = ops.mul(load, constant)
        to_dtype = ops.to_dtype(mul, torch.float32, src_dtype = torch.bfloat16, use_compute_types = True)
        get_index_1 = self.get_index('index1')
        load_1 = ops.load('arg1_1', get_index_1)
        add = ops.add(to_dtype, load_1)
        get_index_2 = self.get_index('index2')
        load_2 = ops.load('buf0', get_index_2)
        sub = ops.sub(add, load_2)
        exp = ops.exp(sub)
        get_index_3 = self.get_index('index2')
        load_3 = ops.load('buf1', get_index_3)
        truediv = ops.truediv(exp, load_3)
        get_index_4 = self.get_index('index0')
        store = ops.store('buf2', get_index_4, truediv, None)
        return store
op0_op1_op2_op3.snodes[3] =
op3: SchedulerNode(ComputedBuffer)
op3.writes = [MemoryDep('buf3', c0, {c0: 262144})]
op3.unmet_dependencies = [MemoryDep('buf2', c0, {c0: 262144})]
op3.met_dependencies = []
op3.min_input_distance = 2
op3.max_input_distance = 3
op3.outputs = [
    buf3: ComputedBuffer
    buf3.layout = FixedLayout('cuda:0', torch.bfloat16, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
    buf3.users = [NodeUser(node=OUTPUT, can_inplace=False, is_weak=False)]
]
op3.group.device = cuda:0
op3.group.iteration = (262144, 1)
op3.sizes = ((262144,), ())
buf2_layout = FixedLayout('cuda:0', torch.float32, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
buf3_layout = FixedLayout('cuda:0', torch.bfloat16, size=[1, 16, 128, 128], stride=[262144, 16384, 128, 1])
class op3_loop_body:
    var_ranges = {p0: 262144}
    index0 = p0
    def body(self, ops):
        get_index = self.get_index('index0')
        load = ops.load('buf2', get_index)
        to_dtype = ops.to_dtype(load, torch.bfloat16, src_dtype = torch.float32, use_compute_types = True)
        get_index_1 = self.get_index('index0')
        store = ops.store('buf3', get_index_1, to_dtype, None)
        return store
```

### formatted_post_fusion

```text
# device: cuda:0

# Inductor IR, post-fusion (fuse_nodes and merge_loops applied).
# Kernel blocks are launch boundaries: inputs and outputs cross the boundary;
# internal values stay within it. Ordinary SIMD kernels show the selected iteration
# domain once; unsupported kernel kinds retain node-local operation domains.
#
#   foreach x in [0,N)   independent points; any order, any parallelism.
#                         Listed outermost to innermost; the last one varies fastest.
#   reduce  r in [0,N)   points combined; order unspecified
#   buf[expr]             expr is a FLAT element offset, not a subscript
#   T[shape] @ [x:N,r:M] non-obvious access domain; omitted for a direct match
#   rN = OP(v over r)    rN is v combined across r, one per foreach point
#   indentation           the body of the loop above

# inputs
arg0_1: bf16[16,128,128], arg1_1: f32[1,1,128,128]

# compute

kernel k0
    inputs:   [arg0_1, arg1_1]
    outputs:  [buf2]
    internal: [buf0, buf1, buf3]

    foreach x in [0,2048), reduce r in [0,128):

        op0:  buf0: f32[1,16,128,1] @ [x:2048]  <-  arg0_1: bf16[16,128,128] @ [x:2048,r:128], arg1_1: f32[1,1,128,128] @ [x:2048,r:128]      # max(to_dtype((arg0_1[r + 16384*((x//128)) + 128*(ModularIndexing(x, 1, 128))]) * 0.0625, torch.float32) + (arg1_1[r + 128*(ModularIndexing(x, 1, 128))]) over r)
            t0 = arg0_1[r + 16384*((x//128)) + 128*(ModularIndexing(x, 1, 128))]
            t1 = t0 * 0.0625
            t2 = to_dtype(t1, f32)
            t3 = arg1_1[r + 128*(ModularIndexing(x, 1, 128))]
            t4 = t2 + t3
            r0 = max(t4 over r)
            buf0[128*((x//128)) + (ModularIndexing(x, 1, 128))] = r0

        op1:  buf1: f32[1,16,128,1] @ [x:2048]  <-  arg0_1: bf16[16,128,128] @ [x:2048,r:128], arg1_1: f32[1,1,128,128] @ [x:2048,r:128], buf0: f32[1,16,128,1] @ [x:2048]      # sum(exp((to_dtype((arg0_1[r + 16384*((x//128)) + 128*(ModularIndexing(x, 1, 128))]) * 0.0625, torch.float32) + (arg1_1[r + 128*(ModularIndexing(x, 1, 128))])) - (buf0[128*((x//128)) + (ModularIndexing(x, 1, 128))])) over r)
            t0 = arg0_1[r + 16384*((x//128)) + 128*(ModularIndexing(x, 1, 128))]
            t1 = t0 * 0.0625
            t2 = to_dtype(t1, f32)
            t3 = arg1_1[r + 128*(ModularIndexing(x, 1, 128))]
            t4 = t2 + t3
            ix0 = 128*((x//128)) + (ModularIndexing(x, 1, 128))
            t5 = buf0[ix0]
            t6 = t4 - t5
            t7 = exp(t6)
            r0 = sum(t7 over r)
            buf1[ix0] = r0

        op2:  buf2: f32[1,16,128,128] @ [x:2048,r:128]  <-  arg0_1: bf16[16,128,128] @ [x:2048,r:128], arg1_1: f32[1,1,128,128] @ [x:2048,r:128], buf0: f32[1,16,128,1] @ [x:2048], buf1: f32[1,16,128,1] @ [x:2048]      # exp((to_dtype((arg0_1[r + 16384*((x//128)) + 128*(ModularIndexing(x, 1, 128))]) * 0.0625, torch.float32) + (arg1_1[r + 128*(ModularIndexing(x, 1, 128))])) - (buf0[128*((x//128)) + (ModularIndexing(x, 1, 128))])) / (buf1[128*((x//128)) + (ModularIndexing(x, 1, 128))])
            ix0 = r + 16384*((x//128)) + 128*(ModularIndexing(x, 1, 128))
            t0 = arg0_1[ix0]
            t1 = t0 * 0.0625
            t2 = to_dtype(t1, f32)
            t3 = arg1_1[r + 128*(ModularIndexing(x, 1, 128))]
            t4 = t2 + t3
            ix1 = 128*((x//128)) + (ModularIndexing(x, 1, 128))
            t5 = buf0[ix1]
            t6 = t4 - t5
            t7 = exp(t6)
            t8 = buf1[ix1]
            t9 = t7 / t8
            buf2[ix0] = t9

        op3:  buf3: bf16[1,16,128,128] @ [x:2048,r:128]  <-  buf2: f32[1,16,128,128] @ [x:2048,r:128]      # to_dtype(buf2[r + 128*x], torch.bfloat16)
            ix0 = r + 128*x
            t0 = buf2[ix0]
            t1 = to_dtype(t0, bf16)
            buf3[ix0] = t1

# outputs
return (buf2, reinterpret_tensor(buf3, [16,128,128], [16384,128,1], 0), reinterpret_tensor(buf3, [16,128,128], [16384,1,128], 0))
```
