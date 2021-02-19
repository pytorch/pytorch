> :warning: **This is an experimental feature**

# Static Runtime

The premise of this approach is that a small subset of neural networks are well represented by a
completely flattened dataflow graph.
TorchScript supports a far more feature programming paradigm,
so many models will not work out of the box.

## Assumptions

This is a list of current assumptions for use with
this feature.

- Inference only execution

After `torch.jit.freeze` and inlining/constant propagation is run on the model:

- No control flow
- No submodule invocations
- No references to `self`
- Inlined weights (i.e. no calls to `GetAttr`)

## Threading model
Static runtime supports two execution modes.

Mode 1: single-threaded with no parallelism except for intra-op parallelism.
For this mode, you can do either:
```
  // m is the TorchScript module
  auto runtime = StaticRuntime(m, opts);
  auto output = runtime.run(args, kwargs);
```
or
```
  auto mod = PrepareForStaticRuntime(m);
  auto runtime = StaticRuntime(mod, opts);
  auto output = runtime.run(args, kwargs);
```
Mode 2: similar to data parallelism, run the same model for different inputs
on different threads at the same time. In this case, run
`PrepareForStaticRuntime` to prepare the graph for Static Runtime. You
should have one InferenceModule instance per model, and one Static Runtime instance
per running thread. To avoiding creating StaticRuntime on the fly, use a
synchronized stack (i.e. `boost::lockfree::stack`) to cache all the Static
Runtime instances in your code.
```
  // initialization
  auto mod = PrepareForStaticRuntime(m);
  // 128 is good for most cases. Pick a number that works for you
  boost::lockfree::stack<std::shared_ptr<StaticRuntime>,
    boost::lockfree::fixed_sized<true>> pool(128);

  // inference
  std::shared_ptr<StaticRuntime> runtime = nullptr;
  pool.pop(runtime);
  if (!runtime) {
    runtime = std::make_shared<StaticRuntime>(mod, opts);
  }
  auto output = runtime->run(args, kwargs);
  pool.push(runtime);
```

## Planned features

- Memory planning
- Operator dispatch inlining
- Operator subsitution
- Weight layout transformations (pre-packing)
- Lowering to `torch.jit.tensorexpr`
