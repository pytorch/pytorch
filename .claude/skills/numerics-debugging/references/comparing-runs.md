# Comparing DebugMode Dumps

Use the plain `dm.debug_string()` output as the source of truth. Do not require
a custom parser or a custom row format. A text diff is usually enough to find
the first meaningful mismatch.

## What to compare

Start with the first operator line where the output `hash` differs. Then inspect
that same line's `input_hash` values:

- Matching `input_hash`, different output `hash`: this op is the likely source.
- Different `input_hash`: this op is only carrying an upstream divergence.
- Missing output hash on an in-place or `None`-returning op: compare
  `input_hash`; moved input hashes can be the mutation.

Use tensor IDs from `record_ids=True` to follow values between nearby lines.
Use module markers from `record_nn_module=True`, profiler scopes, and stack
trace summaries to keep the search localized to the same layer/module.

## Pairing lines

Prefer direct visual pairing by module scope, operator name, tensor shape, and
nearby tensor IDs. If the dumps are too different for direct pairing:

1. First recapture with `record_stack_trace=True` and
   `debug_string(show_stack_trace=True)`.
2. For compiled/AOT regions, try `run_compile_with_interpreter=True`.
3. Only after that, pair positionally by dispatch order and treat the result as
   weaker evidence.

## Training Runs

Before diffing long training dumps, run a deterministic single-step check for
loss and grad norm:

- Loss differs: the first mismatch is probably in forward or in setup before
  forward.
- Loss matches bitwise but grad norm differs: forward likely matched; inspect
  the first backward hash mismatch.
- Both match: inspect optimizer state, RNG/data loading after the checked step,
  or later iterations.

## Common Mismatches

- Recompute can shorten or shift module scopes.
- Extra bookkeeping, communication, or in-place ops can shift later op order.
- Compiled/traced runs may lower one eager op into several lower-level ops.
- Repeated subgraphs can make fuzzy pairing drift by one repeated block.
- Backward ops may have weaker module attribution than forward ops.

## Before Calling It A Bug

Confirm both runs used the same dtype, seed, inputs, RNG behavior, and
determinism settings. If setup differs, every hash can diverge and the dump will
not identify a meaningful kernel or operator bug.
