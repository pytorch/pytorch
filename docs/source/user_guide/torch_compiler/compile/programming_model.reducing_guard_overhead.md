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

### Label guards by region with `guard_ctx`

Use {func}`torch.compiler.guard_ctx` and
{func}`torch.compiler.register_guard_filter` to apply a policy to guards from a
particular region. Labels are compile-time metadata: entering a labeled region
does not add operations to the FX graph. A label alone does not remove guards.

Register filters before compiling, using unique names for your application or
library. The following dictionary policy is user-defined; PyTorch does not
provide a built-in `skip_dict_guards` helper.

```python
import torch


@torch.compiler.register_guard_filter("example.no_dict_guard")
def no_dict_guard(entries):
    return [entry.guard_type != "DICT_KEYS_MATCH" for entry in entries]


@torch.compiler.register_guard_filter("example.no_guards")
def no_guards(entries):
    return [False for entry in entries]


def fn2(args):
    return args["x"] + args["offset"]


def fn3(args):
    return args["x"].sin()


@torch.compiler.guard_ctx("example.no_dict_guard")
def fn(args):
    y = fn2(args)
    with torch.compiler.guard_ctx("example.no_guards"):
        z = fn3(args)
    return y + z


compiled_fn = torch.compile(fn, fullgraph=True)
args = {"x": torch.ones(2), "offset": 2}
torch.testing.assert_close(compiled_fn(args), args["x"] + 2 + args["x"].sin())
```

The decorator labels guards created while tracing `fn` and its call to `fn2`.
The explicit inner context labels new guards created while tracing `fn3`.
Nested contexts use the innermost label and restore the previous label on exit,
including across graph breaks and exceptions. Nested traces start without the
outer trace's label unless they explicitly enter their own context. Use
non-empty, exact built-in `str` labels for uninterrupted `fullgraph=True`
tracking. String subclasses work eagerly but require fallback when traced.

You can also enter a context before a compiled call starts tracing:

```python
compiled_fn2 = torch.compile(fn2, fullgraph=True)
with torch.compiler.guard_ctx("example.no_dict_guard"):
    torch.testing.assert_close(compiled_fn2(args), args["x"] + 2)
```

Labels attach at actual **Guard creation**. A Guard created in region A and
installed later in region B keeps A's label; installation uses the current label
only if the Guard has no creation label. Allocating a lazy variable tracker does
not itself create its guards: a Guard first created when that variable is
realized in B receives B's label. If semantically duplicate guards are installed
in several regions, their attached labels are unioned.

At guard finalization, each registered filter whose label appears on at least one
guard receives one batch containing only entries carrying that label. Each entry
exposes `name`, `guard_type`, `is_global`,
the original Guard in `orig_guard`, and all attached labels as a `frozenset` in
`labels`. This happens **before C++ guard construction**: regional entries have
`has_value=False`, and `derived_guard_types` and `code_parts` are empty tuples.
Filter using the available metadata, without relying on source values or built
guard code. Guard types and inspection data have no backward compatibility
guarantee.

Return a sequence with exactly one built-in `bool` per entry: `True` keeps the
guard and `False` rejects it. Integers and tensor booleans are invalid. Every
registered filter matching a guard's labels runs, even if another already
accepted it. Decisions are ORed: the guard is dropped only when **all matching
filters reject it**. Guards without any registered matching label are kept;
an unregistered label does not override a matching filter's rejection. The
compile-wide `options={"guard_filter_fn": ...}` policy then runs on the
survivors, with its existing built-entry metadata, and can reject them too.

Registrations are process-global and survive `torch.compiler.reset()` and
`torch._dynamo.reset()`. Registering the same name twice raises `ValueError`;
there is no public unregister operation. Finalization takes a snapshot of the
registry before invoking callbacks. Registrations made during a callback affect
later finalizations. Registration does not change cached artifacts retroactively,
and callbacks do not run on cache hits. Enter the desired context for the call
that actually traces; adding a context around an existing cache hit does not
relabel the saved guards.

```{warning}
Dropping a guard transfers responsibility for its assumption to you. If that
assumption changes, a cache hit can silently reuse stale code and return wrong
results. In the example, dropping `DICT_KEYS_MATCH` does not disable every
dictionary check: surviving child guards may still require dictionary key-order
managers, which can reject a changed key order. Unlabeled ambient guards and
backend AOTAutograd alias/storage relations are outside the regional policy.
Even a filter returning all `False` does not promise to remove every check or
constraint from a compiled call.
```

Ordinary AOT compilation and the legacy `caching_precompile` mode support
regional filtering. Saved artifacts retain the resulting survivor policy;
creation labels and attached label metadata are transient and are not saved.
Loading rebuilds those survivors without invoking registered regional callbacks.
The prototype `torch.compiler.precompile.capture` with `DynamoTracer` rejects
regional decisions that actually drop a guard, before constructing guards,
because its safety reporting cannot account for these drops. Keep-all regional
filters work with that prototype path.

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
