# Localization

Locality domains are a feature introduced in CUDA 13.4. They allow developers
to program each locality domain of modern GPUs, enabling better locality of
reference. This follows the general trend of moving compute to data instead of
moving data to compute.

At a high level, this involves 1) creating an SM-localized `GreenContext`
object and then 2) passing kernels localized memory to operate on within that
context. In its simplest form, this works as follows:

```python
import torch

dev = torch.cuda.current_device()
dom_id = 0
(ctx,) = torch.cuda.GreenContext.split(locality_domain_ids=(dom_id,), device_id=dev)
allocator = torch.cuda.LocalizedAllocator(dom_id, device=dev)
with torch.cuda.device(dev):
    pool = torch.cuda.MemPool(allocator.allocator())

with torch.cuda.use_mem_pool(pool, device=dev):
    foo = torch.randn(64, 8, device=f"cuda:{dev}")
(bar,) = torch.cuda.execute_on_streams(
    (ctx.Stream(),), torch.nn.functional.relu, (foo,)
)
```

This example allocates a localized tensor `foo` entirely on locality domain 0
and then passes that localized tensor to `relu`. `relu` is itself limited to
execute only on locality domain 0, and thus all of its memory reads are local.

Mixing up the data and compute localization is possible:

```python
import torch

dev = torch.cuda.current_device()
allocator2 = torch.cuda.LocalizedAllocator(1, device=dev)
with torch.cuda.device(dev):
    pool2 = torch.cuda.MemPool(allocator2.allocator())

with torch.cuda.use_mem_pool(pool2, device=dev):
    foo = torch.randn(64, 8, device=f"cuda:{dev}")
(ctx,) = torch.cuda.GreenContext.split(locality_domain_ids=(0,), device_id=dev)
(bar,) = torch.cuda.execute_on_streams(
    (ctx.Stream(),), torch.nn.functional.relu, (foo,)
)
```

This _succeeds_, but the kernel runs entirely on SMs associated with locality
domain 0, and thus must fetch data from locality domain 1 for every read of
`foo`. Performance will be considerably worse.

Localization is analogous to NUMA (non-uniform memory access).

## Automated optimizations

`torch.cuda.localize` enables automatic localization of entire networks:

```python
import torch

mod = torch.nn.Linear(256, 256, device="cuda").eval()
x = torch.randn(8, 256, device="cuda")

with torch.inference_mode():
    mod = torch.cuda.localize(mod)
    compiled_mod = torch.compile(mod, mode="reduce-overhead")
    out = compiled_mod(x)
```

Localization and subsequent forward calls must run inside `torch.inference_mode()`.
Keep `TORCH_CUBLAS_WORKSPACE_CACHE` unset or `0` before starting the process.
Compiling the localized module with `mode="reduce-overhead"`, as shown above, is strongly recommended.

The internal `_register_localizable_op` helper can register custom operations
prior to invoking `torch.cuda.localize`; it is not a public API.
Registering a `module_type` alongside the op lets
`torch.cuda.localize` rewrite that module's parameters when it walks a model,
the same way it handles `torch.nn.Linear`.

### How it works

`torch.cuda.localize` walks the module tree for module types registered
in `_LOCALIZABLE_OPS` (`torch.nn.Linear` by default). For each matching module, it
swaps out the parameters named by the op's localized argument positions -- `weight` and
`bias` for a linear -- with a `torch.Tensor` subclass. That subclass transforms every
localizable op into a fork/join pattern: it runs the op twice (once per green
stream), and it pre-splits the weights into localized allocations. It then
concatenates the two outputs to form a single (unlocalized) tensor, so that
downstream ops are unaffected.

### Issues and limitations

#### CUDA Graphs are necessary to reduce overhead

For every localizable op, we do considerable additional work:

* An event is enqueued in the caller stream;
* Two green streams each wait on that caller event;
* Two kernel launches instead of one;
* Each green stream records an event on itself;
* The caller stream waits on both green events;
* A concatenate operation produces a non-localized tensor; and
* The green streams wait for the caller to finish consuming their outputs before
  reusing the branch buffers.

This adds latency per localizable op and can dominate performance for small ops.
There is also additional overhead during `torch.cuda.localize`, as it reallocates
weights using a slower mechanism than the normal allocator.

CUDA graphs reduce CPU launch overhead by replaying the captured work.
Use `torch.compile(..., mode='reduce-overhead')` to enable CUDA graphs.

#### No cost model

`torch.cuda.localize` does not include a cost model at present. It
localizes every registered op unconditionally, and not all ops benefit from
localization. As the overheads described above imply, localization has a
non-zero cost that can worsen performance on some workloads.

As a result, localization is likely to speed up some parts of a model and slow
down others.

#### Numerics change

For linear operations, localization splits output channels. Each branch computes
complete output elements, and concatenation joins them without additional
arithmetic. Different GEMM shapes and execution resources can change kernel
selection and floating-point rounding, so results may differ numerically.
Training with localization enabled is currently unsupported.
