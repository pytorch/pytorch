(torch.compiler_inductor_caching)=

# Compiled Graph Caching

## Overview

The goal of the TorchInductor cache is straightforward: to avoid compile
overhead by reusing artifacts previously compiled for the same graph, input
shapes, and settings. The base implementation stores cache entries and compiled
artifacts on the filesystem. In fact, TorchInductor already emits generated
code to the filesystem, so each cache *entry* is essentially just a record with
some metadata about the compiled graph and a pointer to the location of the
generated code. (There's also some complexity specific to supporting symbolic
shapes, described below). See
[CompiledFxGraph](https://github.com/pytorch/pytorch/blob/e0fff31ae31bf3fc7eec39391f90f4893a27ee27/torch/_inductor/output_code.py#L397);
this class doubles as the cache entry which we serialize to a file on disk, as
well as the callable representing the compiled graph.

A cache lookup involves computing a key from the graph and inputs, checking for
a record on the filesystem corresponding to that key, locating the compiled
artifact that entry references, and loading it, thereby skipping all
TorchInductor compilation.

For context, the TorchInductor cache is actually one piece of a cache
hierarchy. From bottom to top, we have:

- **Triton cache**. TorchInductor generates Triton kernels and leverages the
  Triton compiler to generate cubin. The Triton compiler implements its own
  on-disk cache to avoid recompilation when it sees the same kernel.
- **TorchInductor cache**, a.k.a, the FXGraphCache (described here). This cache
  avoids regenerating Triton when unnecessary, but still calls the Triton
  compiler. It therefore relies on the Triton cache to avoid the Triton
  compilation.
- **AOTAutograd cache**. This component caches at the AOT Dispatch (or AOT
  Autograd) level to cache the forward and backward together. There are some
  situations where we cannot cache the joint graph. Therefore, if the
  AOTAutograd cache cannot be used, we bypass it and "fall through" to the
  TorchInductor cache.

## Core Concepts

Below we provide some finer details on the main parts of the TorchInductor
cache.

### Key Calculation

Computing a stable cache key requires careful consideration. To avoid serving
an illegal entry, the cache key should consider *anything* that could affect
TorchInductor's code generation. See the implementation of
[FxGraphHashDetails](https://www.internalfb.com/code/search?q=repo%3Afbcode%20class%20FxGraphHashDetails)
for an up-to-date list of everything the key calculation considers. Here are
some of the important considerations:

- The GraphModule. For caching, we use the representation returned from
  `GraphModule.__reduce__()`, which is currently a text string of the generated
  code (reduce is used for pickling; more on that below).
- All the `example_inputs` and the `kwargs`. But for tensors, we consider only
  the tensor metadata (shape, stride, device, etc).
- All TorchInductor config settings in `_inductor/config.py`, except for a few
  blocklisted settings that can not materially affect code generation, but would
  prevent legitimate caching, e.g., because they include a username. Note that
  custom pass settings are allowed to be callables. To use these settings and
  still benefit from caching, we require that users implement a
  [specific interface](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/custom_graph_pass.py)
  to indicate how to include the callable in the key calculation.
- Any global settings that could affect codegen, e.g.,
  `torch.are_deterministic_algorithms_enabled()`.
- A hash of all the TorchInductor source code.
- Triton version, cuda version, etc.
- ... and many more ...

The key calculation creates an `FxGraphHashDetails` object with all relevant
fields populated, pickles the object to create a byte string, and then computes
a sha256 hash for the key. See
[FxGraphCachePickler](https://github.com/pytorch/pytorch/blob/e0fff31ae31bf3fc7eec39391f90f4893a27ee27/torch/_inductor/codecache.py#L498).
The rationale for this approach is:

1. Pickling should be sufficient to capture all salient characteristics of any
   object, as long as we also hash the source code, since unpickling would
   recreate the original object. The assumption is that any custom
   `__reduce__` methods are sound.
2. Any new object types that make their way into the cache key in the future,
   e.g., via the graph args, are automatically included in the key calculation
   without any updates to the key calculation logic. If those objects don't
   pickle, then we'll avoid caching that particular graph altogether.

The main complication of the pickle approach is handling a few cases where real
pickling is too restrictive. The most important example is the handling of the
tensor inputs, where we only want to consider the tensor metadata and not the
actual values. To achieve this, `FxGraphCachePickler` is a custom Pickler class
that overrides the dispatch table for a few types (see the various
`_reduce_*` methods in that class). When pickling tensor objects, for example,
we instead create a new object containing only the metadata.

### Dynamic Shapes

Dynamic shapes present an additional complexity for caching because
TorchInductor can specialize on inputs, which can cause the creation of new
guards when symbolic inputs are involved. That means:

1. Before serving a cache entry, we need to ensure that any relevant guard
   assumptions are valid.
2. After serving a cache entry, we need to make sure the proper guards are
   added to the environment.

We support that added complexity with the following:

- Among the metadata we store with each cache entry, we also compute and
  include a string representation of a guards expression that's appropriate for
  validating any symbols for Tensor arguments that have symbolic bounds. On
  cache lookup, we evaluate those guards in the current context to validate that
  a cached entry can be served.
- A given graph could theoretically have multiple compiled versions,
  corresponding to different sets of guards. Therefore, we store cache entries
  with an additional directory layer, in the form:
  `<temp dir>/<fx graph hash>/<serialized metadata>`.
- On cache lookup, we compute the key from the graph details, iterate over all
  leaf files in the corresponding subdirectory, deserialize the entry, and
  evaluate its guards expression. If the evaluation succeeds, we have a cache
  hit and serve that entry. Note that we make no attempt to somehow serve the
  "optimal" entry if multiple entries would be valid.
- Finally, on a cache hit, we need to make sure any guards that would have been
  created during compilation are added to the current context.

### Frozen Parameters

For inference use cases, TorchInductor can "freeze" the model parameters (i.e.,
treat them as constants) in order to enable further optimization like constant
folding. Frozen parameters create a challenge for caching, however, because:

1. The parameters can be very large and we don't want to store the full values
   in the cache entries.
2. Even if we did store the full values in the cache, we'd presumably need to
   include those values in the key calculation, but hashing can be
   prohibitively expensive. Also, including the constant values would prevent
   serving a cache entry for an identical graph (and input shapes), but with
   different parameter values.

TorchInductor stores constant values as attributes on the Python module
containing the compiled graph (frozen parameters or otherwise). To support the
treatment of constants as attributes, we store *small* constants in the cache
entry and recreate the attributes when loading from the cache. For *large*
constants (like those we see from frozen parameters), we instead save the
mapping from the attribute names in the `GraphLowering` to the original name of
the attribute in the `GraphModule`. When we create the Python module from the
cache entry, we then look up the constants from the current `GraphModule` and
apply those constants as attributes.

## Some Usage Details

All cache artifacts (Triton, AOTAutograd, TorchInductor) are written to the
location pointed to by the environment variable `TORCHINDUCTOR_CACHE_DIR`. By
default, the cache directory is at: `/tmp/torchinductor_$USER`.

Many unit tests and benchmarking scripts use
[fresh_cache](https://github.com/pytorch/pytorch/blob/e0fff31ae31bf3fc7eec39391f90f4893a27ee27/torch/_inductor/utils.py#L1307)
(aliased to `fresh_inductor_cache`) to achieve cache isolation. This context
manager manipulates `TORCHINDUCTOR_CACHE_DIR` to ensure an empty cache
directory. Note that all TorchInductor unit tests under `test/inductor/` should
derive from a `TestCase` base class that already leverages `fresh_cache` (except
for the cache-specific tests). For debugging, you can disable fresh caches with
`TORCH_COMPILE_DEBUG=1` so that generated artifacts are available in
`/tmp/torchinductor_$USER`.

Caching is enabled by default, but can be disabled via a few settings (See also
[torch_compile_caching_configuration_tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_configuration_tutorial.html)):

- `TORCHINDUCTOR_FX_GRAPH_CACHE=0`: Disable the *local* TorchInductor cache.
- `TORCHINDUCTOR_FX_GRAPH_REMOTE_CACHE=0`: Disable the *remote* TorchInductor
  cache.
- `TORCHINDUCTOR_AUTOGRAD_CACHE=0`: Disable the *local* AOTAutograd cache.
- `TORCHINDUCTOR_AUTOGRAD_REMOTE_CACHE=0`: Disable the *remote* AOTAutograd
  cache.
- `TORCHINDUCTOR_FORCE_DISABLE_CACHES=1`: Disable *all* caches.

Finally, debugging the cause of cache misses is tricky. Typically, we want to
investigate why the key calculation differs between two graphs we expect to be
identical. The first step is to enable debug logging in the caching module:
`TORCH_LOGS=+torch._inductor.codecache`. When debug logging is enabled, we
print the value and sha256 hash of every component in the key calculation
(graph, example_inputs, config settings, and so on). When debugging with tlparse
dumps, e.g., for internal MAST jobs, that output is available for each compile
ID in artifacts named like: `fx_graph_cache_miss_NN.json`,
`aotautograd_cache_hit_NN.json`, etc. Comparing the log output allows you to at
least see which individual components in the key calculation vary between the
two graphs.

## Links

See some current public documentation here:
[torch_compile_caching_tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html)
