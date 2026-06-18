---
name: numerics-debugging
description: Debug numeric / bitwise divergence between two PyTorch runs that should agree (eager vs torch.compile, eager vs aot_fx_trace, distributed vs single-process, TF32 vs fp32, before vs after a refactor) by capturing plain torch.utils._debug_mode.DebugMode string dumps with tensor hashes and comparing them to localize the first diverging op. Use when the user reports loss drift, mismatched outputs, "numerics don't match", "bitwise divergence", "find which op diverges", or invokes /numerics-debugging.
---

# Numerics Debugging (DebugMode-based)

When two runs that *should* produce the same numbers don't, the goal is to
find the **first op where they diverge**, then decide whether that op is the
cause (its inputs matched but its output didn't) or just a carrier (its inputs
already differed). This skill captures `torch.utils._debug_mode.DebugMode`
string dumps from one step of each run, with tensor hashes enabled, then
compares the two dumps.

There is no bundled tooling. You author a small throwaway capture script in
`agent_space/` from the recipe in
[references/capture-recipe.md](references/capture-recipe.md), run it once per
side, and compare the plain `debug_string()` text dumps.

## DebugMode Settings

Use DebugMode's own string dump as the debugging artifact. Turn on:

- `record_realtensor=True`
- `record_nn_module=True`
- `record_ids=True`
- `record_stack_trace=True` when source context is useful
- `record_profiler_context=True`
- `run_compile_with_interpreter=True` for AOT/eager compiled regions when
  module metadata matters
- `DebugMode.log_tensor_hashes(hash_fn="norm", hash_inputs=True)`

Use `record_nn_module=True` for layer names. Do not monkey-patch modules or add
framework-specific layer markers; if module names are weak, rely on stack traces,
profiler scopes, and the surrounding op sequence in the plain dump.

## Workflow

1. **Make the two runs comparable.** They MUST share dtype, seed, and inputs,
   and run deterministically. A precision change (bf16 vs fp32) makes every op
   diverge and the diff is useless. Set the same `torch.manual_seed(...)`
   before model init and before the step, use the same batch, and enable
   `torch.use_deterministic_algorithms(True)` where possible.
   For training, first do a one-step scalar check: compare the loss and a
   deterministic grad norm. If the loss matches bitwise but the grad norm
   diverges, focus the dump comparison on backward.
2. **Probe** a tiny `DebugMode.debug_string()` dump (recipe, step 1) to confirm
   hashes, tensor IDs, stack traces, and module names appear in this
   environment.
3. **Capture one step per run** with the capture function (recipe, step 2),
   writing each run's plain dump to its own file (`run_A_debug.txt`,
   `run_B_debug.txt`). The step can be inference-only or forward+backward;
   capture on a warmed-up step if step 0 has legitimate one-time init/compile
   differences.
4. **Diff the two dumps** ([references/comparing-runs.md](references/comparing-runs.md)):
   inspect the plain text and report the first op whose output hash differs.

## What gets captured per op

For each recorded op in the plain dump:

- **Module/profiler context** — from `record_nn_module=True` and
  `record_profiler_context=True`.
- **Tensor IDs** — from `record_ids=True`.
- **Source summaries** — from `record_stack_trace=True` and
  `debug_string(show_stack_trace=True)`.
- **Output hash** — DebugMode's `norm` hash of the result.
- **Input hashes** — `norm` hash of each input *before* the op ran. This is
  what catches in-place mutations (e.g. `_fused_adam_` returns `None`, so it
  has no output hash, but its input hashes still move).

## Interpreting the diff

- **All hashes identical** -> the runs are bitwise equal; the bug is elsewhere
  (data loading, loss, optimizer state, RNG consumption order).
- **First divergence at op K, and K's input hashes already differ** -> the
  real divergence is upstream; follow the inputs back to their producing op
  and keep walking until you find an op whose **inputs match but output
  differs**. That op is the root cause.
- **Everything diverges from the very first op** -> almost always a dtype or
  seed/determinism mismatch, not a real bug. Fix setup (step 1) and recapture.
- **A tiny last-digit difference that doesn't propagate** -> often
  reduction-order noise, not a bug. Use `hash_fn="hash_tensor"` if strict
  bitwise equality is required.

This was validated on a real divergence: fp32 vs TF32 matmul on the same model
matched on every forward op but first diverged on a **backward** `mm`, whose
inputs matched -> correctly fingering the TF32 matmul kernel itself as the
source. Divergence often first crosses the hash's precision floor in backward
(larger reductions) even when forward already differs in lower digits.

## Cost

Capture adds overhead on the single captured step only. Because the context
wraps just one step, steady-state training is untouched.

## References

- [references/capture-recipe.md](references/capture-recipe.md) — the probe,
  relevant DebugMode flags, example dump snippets, single-step training checks,
  and plain `debug_string()` capture.
- [references/comparing-runs.md](references/comparing-runs.md) — how to compare
  the plain dumps and handle common key/order drift.
