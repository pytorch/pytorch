(torch.compiler_inductor_caching)=

# Compiled Graph Caching

TorchInductor caches compiled graphs to avoid redundant compilation on
subsequent runs. This caching system operates at multiple levels, from
individual Triton kernel configurations to entire compiled FX graphs.

**Source**: [torch/_inductor/codecache.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/codecache.py),
[torch/_inductor/output_code.py](https://github.com/pytorch/pytorch/blob/main/torch/_inductor/output_code.py)

## Cache Hierarchy

TorchInductor uses a multi-level caching strategy:

```
┌─────────────────────────────────────┐
│  AOT Autograd Cache                 │  Caches traced forward/backward graphs
├─────────────────────────────────────┤
│  FXGraphCache                       │  Caches compiled Inductor output
├─────────────────────────────────────┤
│  Triton Autotuning Cache            │  Caches kernel config benchmarks
├─────────────────────────────────────┤
│  Triton Compilation Cache           │  Caches compiled Triton binaries
└─────────────────────────────────────┘
```

- **AOT Autograd Cache** — caches the traced and decomposed forward/backward
  FX graphs produced by AOT Autograd, avoiding re-tracing.
- **FXGraphCache** — the primary Inductor cache. Stores the full compiled
  output (generated kernels, wrapper code, metadata) so the entire Inductor
  compilation pipeline can be skipped on a cache hit.
- **Triton Autotuning Cache** — stores the results of kernel configuration
  benchmarks so autotuning doesn't need to re-benchmark on subsequent runs.
  See [Autotuning](torch.compiler_inductor_autotuning.md) for details.
- **Triton Compilation Cache** — Triton's own cache for compiled GPU binaries,
  stored in `~/.triton/cache/` by default.

## FXGraphCache

`FxGraphCache` is the central caching mechanism in TorchInductor. It stores
compiled graphs on disk and, optionally, in a remote cache for sharing across
machines.

### How It Works

1. **Key computation** — when a graph is ready for compilation, TorchInductor
   gathers all relevant details (graph structure, input metadata, system
   settings) into an `FxGraphHashDetails` object, pickles it using a
   deterministic custom pickler (`FxGraphCachePickler`), and computes a hash
   to produce the cache key.

2. **Lookup** — the cache directory for that key is checked. If entries exist,
   their guard expressions are evaluated against the current symbolic context.
   A successful guard evaluation means a cache hit.

3. **On hit** — the cached `CompiledFxGraph` is deserialized and returned.
   Kernel binaries are loaded from their own disk cache locations. Any guards
   that would have been created during compilation are replayed into the
   current context.

4. **On miss** — the graph is compiled normally. The resulting
   `CompiledFxGraph` (including references to generated kernel locations) is
   serialized and stored in the cache directory.

### What Goes Into the Cache Key

The `FxGraphHashDetails` class captures everything that could affect the
compiled output:

- **Graph structure** — the FX graph module itself (nodes, operations, shapes)
- **Example inputs** — tensor shapes, strides, dtypes, and device information
- **Compile kwargs** — any keyword arguments passed to the compile function
- **Alignment checks** — which inputs require alignment verification
- **System settings** — deterministic algorithm settings, CUDA matmul precision
  settings, and device properties
- **User-defined Triton kernels** — source code and configurations of any
  custom Triton kernels referenced in the graph
- **Cache key tag** — an optional user-provided tag for cache partitioning

### Storage Layout

Cache entries are stored on disk in the following structure:

```
<cache_dir>/fxgraph/<key_prefix>/<full_key>/<serialized_entry>
```

A single graph hash can have **multiple entries**, each corresponding to
different guard expressions. This supports dynamic shapes where the same
graph structure may compile differently depending on symbolic constraints.

### Local and Remote Caching

FXGraphCache supports both local disk caching and remote caching:

- **Local cache** — enabled by default (`fx_graph_cache = True`). Stores
  entries in the local filesystem.
- **Remote cache** — optional. Allows sharing cached compilations across
  different machines or CI runs. Controlled by `fx_graph_remote_cache`.

When both are enabled, the local cache is checked first. On a local miss,
the remote cache is consulted. On a complete miss, the compiled result is
written to both caches.

## CompiledFxGraph

The `CompiledFxGraph` class serves a dual role: it is both the **cache entry**
(what gets serialized and stored) and the **callable** (what gets executed at
runtime). It contains:

- References to the generated kernel code and compiled binaries
- The wrapper function source code
- Guard expressions for dynamic shape validation
- Metadata about the compilation (inputs to check, output strides, etc.)

When loaded from cache, `CompiledFxGraph` reconstructs the callable by loading
the kernel binaries from their respective cache locations.

## Dynamic Shapes and Guards

Dynamic shapes introduce complexity for caching because the same graph
structure may need different compiled versions depending on symbolic
constraints.

TorchInductor handles this by:

1. **Storing guard expressions** — each cache entry includes a guards
   expression that encodes the symbolic constraints under which the compiled
   code is valid (e.g., `s0 >= 1`, `s0 <= 2048`).

2. **Multi-entry directories** — a single cache key (based on graph structure)
   can map to multiple entries, each with different guard expressions.

3. **Guard evaluation on lookup** — when checking the cache, each entry's
   guards are evaluated against the current symbolic context. The first entry
   whose guards are satisfied is returned.

This means that a graph compiled with `batch_size=32` and another with
`batch_size=64` can coexist in the cache under the same graph hash, each
with its own guards.

## Frozen Parameters

When compiling models for inference, parameters are often **frozen** (embedded
as constants in the graph rather than passed as inputs). This creates
challenges for caching:

- Frozen parameter values become part of the graph, so any change to model
  weights invalidates the cache.
- Large models with frozen parameters produce large cache entries.
- Different checkpoint versions of the same model architecture will not share
  cache entries.

For inference workloads where model weights change between runs, consider
whether the caching benefits outweigh the overhead of cache invalidation.

## Configuration Reference

| Config | Env Variable | Default | Description |
|--------|-------------|---------|-------------|
| `fx_graph_cache` | `TORCHINDUCTOR_FX_GRAPH_CACHE` | `True` | Enable local FX graph caching. |
| `fx_graph_remote_cache` | — | `None` | Enable remote FX graph caching. `None` means off for OSS. |
| `bundle_triton_into_fx_graph_cache` | — | `None` | Bundle Triton compilation artifacts into the FX graph cache entries. |
| `autotune_local_cache` | — | `True` | Enable local caching of autotuning results. |
| `autotune_remote_cache` | — | `None` | Enable remote autotuning cache. `None` means off for OSS. |
| `force_disable_caches` | — | `False` | Disable all caching. Useful for debugging or benchmarking compilation time. |

:::{seealso}
For a tutorial on using Inductor caching in practice, see the
[TorchInductor GPU Compilation Caching Tutorial](https://docs.pytorch.org/tutorials/recipes/torch_compile_caching_tutorial.html).
:::
