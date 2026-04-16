(torch.compiler_inductor_codegen)=

# Code Generation

**(i.e. generating strings of Triton code)**

After fusion occurs in the [scheduler](torch.compiler_inductor_scheduler.md),
TorchInductor translates the backend-agnostic IR into fused kernels. There are
two main codegen backends: Triton (for GPU), and C++ (for CPU). In this
document, we will mostly focus on the Triton backend.

**Source**: [torch/_inductor/codegen/](https://github.com/pytorch/pytorch/tree/main/torch/_inductor/codegen)

## Overview

The scheduler works by creating a new kernel object, then reinterpreting the
backend-agnostic IR into the target destination. Throughout the Inductor
codebase, we often do abstract analysis by custom interpretation of the IR
nodes, whether that's for range analysis or dtype propagation. Codegen is
similar, although now we are reinterpreting the operators and translating them
to Triton codegen.

To avoid having to thread globals throughout the codebase, we use a pattern for
accessing thread local global values under `V`. In codegen, we set both the
[global](https://github.com/pytorch/pytorch/blob/532389fe9ed9fbc787a67cd3d4ffe22c7cc1c8ab/torch/_inductor/codegen/simd.py#L1943)
`V.kernel` object and set `V.ops` so that op interpretation is routed through
our kernel. The kernel will execute the FX graph nodes of the
[LoopBody](https://github.com/pytorch/pytorch/blob/532389fe9ed9fbc787a67cd3d4ffe22c7cc1c8ab/torch/_inductor/loop_body.py#L91)
that are persisted on the
[SchedulerNode](https://github.com/pytorch/pytorch/blob/532389fe9ed9fbc787a67cd3d4ffe22c7cc1c8ab/torch/_inductor/scheduler.py#L1409-L1416).

As a reminder, here is a representative FX graph:

```python
def forward(self, ops):
    index0 = self.index0
    load = ops.load('x_1', index0, False);  index0 = None
    index0_1 = self.index0
    load_1 = ops.load('y_1', index0_1, False);  index0_1 = None
    add = ops.add(load, load_1);  load = load_1 = None
    index0_2 = self.index0
    store = ops.store('buf0', index0_2, add, None);  ops = index0_2 = add = None
    return store
```

We call this function with `V.get_ops_handler()`, and that passes in the `ops`
object, which has all of the requisite codegen functions defined on it. For
example,
[this is how ops.load is defined for Triton](https://github.com/pytorch/pytorch/blob/532389fe9ed9fbc787a67cd3d4ffe22c7cc1c8ab/torch/_inductor/codegen/triton.py#L3125):

```python
def load(self, name: str, index: sympy.Expr, upcast: bool = False):
    var = self.args.input(name)
    index, mask = self.indexing(index)
    line = f"tl.load({var} + {index}, {mask})"
    if upcast:
        line += ".to(tl.float32)"
```

The end result is a kernel that looks like this:

```python
@pointwise_heuristics(size_hints=[32])
@triton.jit
def kernel0(in_ptr0, in_ptr1, out_ptr0, ks0, xnumel, XBLOCK : tl.constexpr):
    xoffset = tl.program_id(0) * XBLOCK
    xindex = xoffset + tl.reshape(tl.arange(0, XBLOCK), [XBLOCK])
    xmask = xindex < xnumel
    x0 = xindex
    tmp0 = tl.load(in_ptr0 + x0, xmask)
    tmp1 = tl.load(in_ptr1 + x0, xmask)
    tmp2 = tmp0 + tmp1
    tl.store(out_ptr0 + x0, tmp2, xmask)
```

## Reductions

Mapping pointwise operators to their Triton equivalents is mostly
straightforward (with indexing and masking requiring more care). Reductions are
more complicated. Before we attempt to codegen reductions, we first decide what
fused nodes will occur in which reduction loops in
[generate_node_schedule](https://github.com/pytorch/pytorch/blob/e882c761dd2bd2f32d1ac5b6f846c9951564e9e7/torch/_inductor/codegen/simd.py#L1287-L1381).

As an example:

```python
torch._logging.set_logs(output_code=True)

@torch.compile()
def foo(x):
    return x + relu(), x.sum()

foo(torch.rand([256, 256], device="cuda"))
```

For this function, we schedule `x.relu()` in the reduction loop so that we only
need to do one pass over the data. For something like `x.sum().relu()`, the
relu would occur after the reduction loop.

TorchInductor generates two forms of reductions by default currently:
[looped reductions](https://gist.github.com/shunting314/056d43d35907e87efb883970b35c17d4)
which iterate over the reduction dimension, and
[persistent reductions](https://gist.github.com/shunting314/39e7c00ff8bb2055942ed5a3255d61ca)
which load the entire reduction dimension in memory. For reductions which reduce
over large amounts of data into a single element, TorchInductor may
[split the kernel into two](https://github.com/pytorch/pytorch/blob/036eb1f65dc6ed5e1e4b88a94e20afe6e3f356fe/torch/_inductor/ir.py#L1280-L1293)
for additional parallelism. This transformation occurs prior to codegen. An
alternate, single-kernel approach is being developed but not yet enabled, called
`cooperative_reductions`.

### Reduction-Pointwise Fusion

The scheduler can fuse multiple reductions and pointwise operations. It does
that by sticking all the scheduler nodes into a fused scheduler node. At code
generation time, the fused scheduler node is transformed into a single kernel
by iterating through the individual schedule nodes and running their respective
bodies under the Triton codegen interpreter. This operation is tricky for
reduction-pointwise fusion since we need to flush the preceding reduction. It's
performed by `codegen_body`, which codegens the body in the sense that it
consumes the various partial string results into `self.body`. For the looped
reduction, the various interpreter functions take care to not accumulate text
into `self.body` directly (they check `self.inside_reduction` to figure this
out).

## Templates

TorchInductor allows you to register Triton kernels as templated functions which
will get instantiated with input shapes, strides, and other metadata. The most
commonly dispatched template is matrix multiplication, but there are also
templates for
[persistent_tma_mm](https://github.com/pytorch/pytorch/blob/036eb1f65dc6ed5e1e4b88a94e20afe6e3f356fe/torch/_inductor/kernel/mm.py#L236),
[grouped_mm](https://github.com/pytorch/pytorch/blob/036eb1f65dc6ed5e1e4b88a94e20afe6e3f356fe/torch/_inductor/kernel/mm_grouped.py#L124),
convolution, batched jagged mms, and a few others. TorchInductor will
epilogue-fuse pointwise kernels into the templates.

Templates are usually registered with an `ExternKernelChoice` (such as CuBlas),
which gives an alternative kernel implementation to Triton. At lowering time, we
compile Triton templates for various configuration instantiations, benchmark
them against ExternKernels, and select a Triton template only if it is the
fastest option. This decision is made taking into account the possible fusions
that could occur with the TritonTemplate, but not with the Extern choice. Triton
templates are only autotuned with `max-autotune` or
`max-autotune-no-cudagraphs`. To minimize compilation time, these templates get
compiled asynchronously and then benchmarked later.

:::{note}
Not covered: CUTLASS templates, CPP Codegen, Wrapper code.
:::

:::{seealso}
For a guide to profiling generated Inductor kernels, including environment
variables for unique kernel naming and individual kernel benchmarking, see
[GPU Profiling](torch.compiler_inductor_profiling.md).
:::
