# Runtime and Performance Issues

## Table of Contents

1. [ProfilingGraphExecutor](#profilinggraphexecutor)
2. [NNC / TensorExpr Bugs](#nnc--tensorexpr-bugs)
3. [Static Runtime Gaps](#static-runtime-gaps)
4. [Serialization Issues](#serialization-issues)

---

## ProfilingGraphExecutor

The ProfilingGraphExecutor (PGE) implements TorchScript's JIT features:
1. First execution: instruments graph with profiling nodes (records dtype, shape, etc.)
2. Second execution: collects profiled info, removes profiling nodes, applies optimizations, adds guards

### Memory overhead (2+ instances)

PGE profiles inputs and compiles optimized kernels, consuming significant memory. For CPU-only preproc workloads, this provides no benefit.

**Workaround:** `torch.jit.getProfilingMode() = False`

### Thread serialization via compile_mutex (2+ instances)

`GraphExecutor::run()` acquires a `compile_mutex` on every forward call, fully serializing execution across threads. Sharing a single JIT module across threads causes 2-3x latency regression.

**Workaround:** Use `module.clone()` for per-thread copies, or use `PytorchPredictorContainer`'s method-level caching.

### First-invocation warmup (3+ instances, 1 SEV)

The optimized execution plan is generated lazily on first invocation, holding a lock. Other threads block during initialization. In SEV S593199, slow model preloading during warmup caused crash-looping in production.

**Workaround:** Run a single warmup request per model before routing real traffic.

### Profiling executor long optimization (1 SEV)

DCE (dead code elimination) in PGE optimization can take several minutes, triggering distributed monitoring that thinks the process is stuck (S495086).

**How to detect PGE issues:** Look for "profiling" in context (JIT profiling, not strobelight/torch.profiler), PGE-specific graph passes in the stacktrace, or issues that only appear on the 2nd execution.

**Disabling PGE:** `torch._C._get_graph_executor_optimize(False)` — may cause performance regressions.

---

## NNC / TensorExpr Bugs

NNC/TensorExpr performs operator fusions using info from PGE to generate fused kernels (analogous to Inductor in PT2).

### Wrong device allocation (2+ instances)

Model on `cuda:1` runs correctly on first invocation, but NNC allocates memory on `cuda:0` on second invocation.

**Workaround:** `torch._C._get_graph_executor_optimize(False)` (disables fusions, may regress performance).

### Locale-dependent code generation (1 instance)

NVRTC changes the program locale, and NNC's CUDA code generation inserts thousands-separators (e.g., `1,024` instead of `1024`). Fixed via PR: imbue ostream to fix locale.

### Silent crashes on second forward pass (1+ instances)

Operator fusion (NNC/TExpr/NVFuser) can produce silent crashes (no error, no stack trace) on the second forward pass on CUDA.

**Workaround:** Disable all fusers:
```python
torch._C._jit_override_can_fuse_on_cpu(False)
torch._C._jit_override_can_fuse_on_gpu(False)
```

### Common NNC issues

- Requires LLVM upgrades whenever LLVM is updated internally
- Can error/crash unexpectedly
- Can hit exponential/very slow compilation behavior

---

## Static Runtime Gaps

Static Runtime is **super deprecated** (no experts left, no oncall support) but still used widely.

### Key limitation

Static Runtime requires additional graph passes — notably `torch.jit.freeze` — before the graph is passed to it. Freezing has additional limitations not present for normal models (e.g., `ModuleList`/`ModuleDict` often don't work).

### prim::Enter/prim::Exit not supported

`record_function` context managers generate these ops, which Static Runtime cannot handle. Blocks SR enablement and causes 50% performance regression without SR.

**Workaround:** Remove `record_function` from inference code paths.

**Disabling Static Runtime:** Can cause performance regressions. Either disable it or change the model to avoid the limitation.

---

## Serialization Issues

### Invalid/wrong file format (2+ instances)

Users attempt to load non-TorchScript files with `torch.jit.load`, getting "PytorchStreamReader failed reading zip archive."

**Resolution:** Verify the file is actually a TorchScript-saved model (saved via `torch.jit.save`).

### Custom ops must be pre-registered (2+ instances)

`torch.jit.load` fails with "Unknown builtin op" if the custom op library isn't loaded first. Especially problematic with Triton kernels in AOTI-published models.

**Resolution:** Ensure `torch.library.load_library()` or equivalent is called before `torch.jit.load`.

### Forward compatibility (1 SEV)

Model snapshots can become incompatible across PyTorch versions (S544252).

### Compilation units and load isolation

If you `torch.jit.load` two separate files, they load into different compilation units. Two files both defining `MyCustomClass` will be treated as completely independent classes.

### Other serialization issues

- `_LazyImportWrap` objects cannot be resolved as type annotations during scripting
- `toPyObject` `InvalidTag` errors: suspected serialization regression producing corrupted type information
