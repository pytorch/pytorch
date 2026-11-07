# Localization

Locality domains are a feature introduced in CUDA 13.4. They allow developers
to program each locality domain of modern GPUs, enabling better locality of
reference. This follows the general trend of moving compute to data instead of
moving data to compute.

The birds-eye view is 1) we create an SM-localized GreenContext object and then
2) we pass kernels localized memory to operate on within that context. In its
barest form, this works as follows:

```python
  dev = torch.cuda.current_device()
  dom_id = 0
  (ctx,) = torch.cuda.GreenContext.split(locality_domain_ids=(dom_id,), device_id=dev)
  allocator = torch.cuda.LocalizedAllocator(dom_id, device=dev)
  with torch.cuda.device(dev):
    pool = torch.cuda.MemPool(allocator.allocator())

  with torch.cuda.use_mem_pool(pool, device=dev):
    foo = torch.randn((64,8), device='cuda')
  (bar,) = torch.cuda.execute_on_streams(
    (ctx.Stream(),), torch.nn.functional.relu, (foo,)
  )
```

This example allocates a localized tensor `foo` entirely on locality domain 0
and then passes that localized tensor to `relu`. `relu` is itself limited to
execute only on locality domain 0, and thus all of its memory reads are local.

Mixing up the data and compute localization is possible:

```python
  allocator2 = torch.cuda.LocalizedAllocator(1, device=dev)
  with torch.cuda.device(dev):
    pool2 = torch.cuda.MemPool(allocator2.allocator())

  with torch.cuda.use_mem_pool(pool2, device=dev):
    foo = torch.randn((64, 8), device='cuda')
  (ctx,) = torch.cuda.GreenContext.split(locality_domain_ids=(0,), device_id=dev)
  (bar,) = torch.cuda.execute_on_streams(
    (ctx.Stream(),), torch.nn.functional.relu, (foo,)
  )
```

This _succeeds_, but the kernel runs entirely on SMs associated with locality
domain 0, and thus must fetch data from locality domain 1 for every read of
`foo`. Performance will be considerably worse.

Localization is a form of NUMA.

## Automated optimizations

We introduce the new torch.cuda.localize API to enable automatic localization
of entire networks. Usage is simple:

```python
  mod: torch.nn.Module = ...
  inference: bool = torch.is_inference_mode_enabled()
  mod = torch.cuda.localize(mod, mark_static_halves=inference)
  mod(...)
```

We strongly encourage users to `torch.compile(..., mode='reduce-overhead')` the
resulting module.

Projects that have custom operations that they wish localized should use
`register_localizable_op` to register said ops prior to invoking
`torch.cuda.localize`. Registering a `module_type` alongside the op lets
`torch.cuda.localize` rewrite that module's parameters when it walks a model,
the same way it handles `torch.nn.Linear`.

### How it works

The torch.cuda.localize API call searches for supported operations
(`_LOCALIZABLE_OPS` in the code; `torch.nn.functional.linear` and its exported
`aten.linear` shape are registered by default). When it finds one, it swaps out
the parameters feeding that op's localized argument positions -- `weight` and
`bias` for a linear -- with a torch.Tensor subclass. That subclass will transform every
localizable op into a fork/join pattern: it runs the op twice (once per green
stream), and it pre-splits the weights into localized allocations. It will then
concatenate the two outputs back together to form a single (unlocalized)
tensor, so that downstream ops are none the wiser about what happened.

### Issues and limitations

#### CUDA Graphs are necessary to reduce overhead

For every localizable op, we do considerable additional work:

* An event is enqueued in the main stream;
* Two green streams each wait on that 'main' event;
* Two kernel launches instead of one;
* Each green stream records an event on their localized green stream;
* The main stream waits on both green events; and
* A concatenate operation produces a non-localized tensor.

This can add 10--50 microseconds of additional latency per localizable op. If
the op was already running in ~10us, these overheads will dominate performance.
There is also additional overhead during torch.cuda.localize as it reallocates
weights using a slower mechanism than the normal allocator.

The solution to this problem is CUDA graphs. Raw CUDA graphs are best, but
`torch.compile(..., mode='reduce-overhead')` is the more ergonomic solution.
There is active but as-yet out-of-tree work to improve the overheads associated
with `mode='reduce-overhead'`.

#### No cost model

`torch.cuda.localize` does not include any cost model at present. This means it
will blindly localize every op that you tell it to, and not all ops will
benefit from localization. Worse, as the overheads mentioned above imply, there
is non-zero cost to localizing that will worsen performance on some workloads.

Thus: it is likely localization will benefit part of your model, and slowdown
other parts.

#### Numerics change

Localization will change the numerics of any reduction operations (matmuls
involve a reduction). In many cases this is equivalent to switching to a
"split-k" GEMM implementation and so has the same limitations: because we
change where partial sums are accumulated, this can affect numerics in subtle
ways.

In practice we have found simple networks do not change behavior with
localization enabled or not, and theoretically localization should lead to
*more* accurate numerics. This is perhaps little consolation to a model author
whose model accuracy declines in practice. One theoretical method to ameliorate
these declines is---just like with quantization---to train the final epoch or
last `N` batches *with* localization.

#### Integration gotchas

* For torchao, you must quantize and *then* localize; localization followed by
quantization will cause the quantization to (silently) erase the localization.

* torchao requires some additional ops to be implemented on e.g. Float8Tensor.
This work is ongoing.

* vLLM and SGLang contain custom ops for localization, and so applying
torch.cuda.localize on top of those ops will not be valuable. For the vLLM /
SGLang developers out there: we have prototyped using torch.cuda.localize
*instead* of those custom ops and it works great. You might consider relying on
torch for this kind of localization.

* Certain cuBlasLt GEMMs can cause `CUBLAS_STATUS_NOT_SUPPORTED` errors when
launching GEMMs on localized green streams. One possible cause is unsupported
KMD / UMD driver combinations. Report all such issues to NVIDIA.
