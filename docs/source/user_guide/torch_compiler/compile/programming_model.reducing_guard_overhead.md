# Reducing Guard Overhead

Every time you call a `torch.compile`d function, Dynamo runs a set of *guards*
before dispatching to a compiled artifact. Guards check that the assumptions
made at compile time (tensor shapes and dtypes, `nn.Module` attributes, global
state, etc.) still hold, so that it is sound to reuse the compiled code (for the
conceptual background, see [Guards](programming_model.dynamo_core_concepts) in
Dynamo Core Concepts). This check happens on **every call**, so for functions
that are cheap relative to their guard set, guard evaluation can become a
measurable fraction of runtime.

Closely related is the *pre-graph bytecode* that Dynamo runs before entering the
compiled graph to marshal inputs (for example, looking up parameters and buffers
to pass into the graph). Like guards, this runs on every call.

This page collects the knobs for reducing that per-call overhead. This is
distinct from reducing *compilation* time; if your problem is slow or repeated
compilation, see [Dealing with Recompilations](programming_model.recompilation).

```{warning}
Most of the options below are **unsafe** by design: they trade soundness for
speed by dropping or skipping guards. They assume your model code does not
mutate the guarded state (e.g. `nn.Module` attributes or globals) between calls
after warmup. If that assumption is violated, `torch.compile` may silently run a
stale compiled artifact and produce **incorrect results**. Only enable them once
you understand the assumption each one makes.
```

To see *which* guards are generated, use tlparse or `TORCH_LOGS=guards`; see
[tlparse / TORCH_TRACE](programming_model.observability). To measure how much
guard evaluation *costs*, profile the compiled function and look for the
`TorchDynamo Cache Lookup` event, which times guard evaluation on each call:

```python
from torch.profiler import profile, ProfilerActivity

with profile(activities=[ProfilerActivity.CPU]) as prof:
    opt_mod(x)
prof.export_chrome_trace("trace.json")  # inspect in chrome://tracing
```

Compare that time against the compiled function's total run time to decide
whether the overhead is worth reducing.

## 1. Reduce pre-graph bytecode time with `install_free_tensors`

```python
import torch._dynamo

with torch._dynamo.config.patch(install_free_tensors=True):
    # torch.compile your model / call the compiled function here
    ...
```

This installs free tensors (such as parameters and buffers) as graph attributes
rather than graph inputs, which reduces the pre-graph bytecode work needed to
marshal them into the graph. As a side effect it also changes how those
parameters and buffers are guarded (they no longer need a per-input tensor
match), so you may see a small guard-overhead improvement as well. That guard
improvement is often small on its own, because most of the per-call cost is in
*accessing* the object being checked rather than in the guard check itself.

`install_free_tensors` is `False` by default (it originated as an export aid for
producing a consistent number of graph inputs), so you must opt in via the
config patch above.

## 2. Skip guards on `nn.Module`s with `guard_filter_fn`

`torch.compile` accepts a `guard_filter_fn` option that decides, guard by guard,
which guards to keep. `torch.compiler` ships several ready-made filters. The most
impactful for guard overhead is skipping guards on `nn.Module`s:

```python
import torch

opt_mod = torch.compile(
    mod,
    options={"guard_filter_fn": torch.compiler.skip_guard_on_all_nn_modules_unsafe},
)
```

This typically gives a large reduction in guard overhead, since models tend to
have many module attributes that are guarded but never mutated. Note the
interaction with pre-graph bytecode: skipping these guards means the cache of
parameter/buffer pointers goes cold, so some of the overhead you removed from
guards **shifts into pre-graph bytecode**. To see the full benefit, combine this
with `install_free_tensors=True` above — apply **both (1) and (2)** together.

Other filters available in `torch.compiler` (all `_unsafe`, same caveats):

- `skip_guard_on_inbuilt_nn_modules_unsafe` — skip guards only on built-in
  modules like `torch.nn.Linear`.
- `skip_guard_on_globals_unsafe` — skip guards on globals.
- `keep_tensor_guards_unsafe` — keep only tensor guards (optionally including
  parameters).
- `keep_portable_guards_unsafe` — keep only guards that are portable across
  Python and non-Python environments (global-state, shape-env, and non-global
  tensor guards).
- `skip_all_guards_unsafe` — drop **all** guards; removes every safety guarantee,
  use with extreme caution.

## 3. Try `use_recursive_dict_tags_for_guards` first

```python
import torch._dynamo

torch._dynamo.config.use_recursive_dict_tags_for_guards = True
```

This speeds up guard execution for nested `nn.Module`s by recursively checking
dict tags to avoid running the full guard set. It relies on a fairly complicated
mechanism using low-level CPython features. A few concerns have been raised in
OSS issues, so it is off by default and may be revisited. It is worth trying
**before** options (1) and (2): if it works for your model, you will not need to
skip `nn.Module` guards (2), though you would still benefit from
`install_free_tensors` (1).

## 4. Skip guard evaluation after warmup with `skip_guard_eval_unsafe`

```python
import torch

# 1. Warm up: run the compiled model with a sufficient variety of inputs until
#    no further recompilation occurs.
# 2. Then switch stance to run only the minimal set of differentiating guards.
with torch.compiler.set_stance(skip_guard_eval_unsafe=True):
    # steady-state inference / training iterations
    ...
```

Once you have warmed up the compiled model to the point where no further
recompilations happen, `skip_guard_eval_unsafe` runs only the minimal set of
guards needed to differentiate between the compiled artifacts you already have,
skipping the rest. Unlike the options above, this cannot be set at
`torch.compile` time — you must enable it in the serving/training loop *after*
warmup. If the no-more-recompilation assumption fails (a genuinely new input
arrives), there is a risk of silently producing incorrect results, hence the
`unsafe` in the name.

## Putting it together

- Start by measuring where the per-call time goes (guards vs. pre-graph bytecode)
  with tlparse / `TORCH_LOGS=guards`.
- Optionally try `use_recursive_dict_tags_for_guards=True` first.
- Otherwise, apply `install_free_tensors=True` **and** a `guard_filter_fn` such
  as `skip_guard_on_all_nn_modules_unsafe` together — one reduces guard overhead,
  the other keeps that saving from reappearing as pre-graph bytecode overhead.
- For steady-state serving after warmup, consider
  `set_stance(skip_guard_eval_unsafe=True)`.

## Further reading

For a deeper mental model of guards — compile units (a graph plus its guard set),
why the guard set is large, and worked profiler benchmarks for each of the
techniques above — see the developer blog post
[Inside torch.compile Guards](https://docs.pytorch.org/devlogs/dynamo/2025-06-04-inside-torch-compile-guards/).
