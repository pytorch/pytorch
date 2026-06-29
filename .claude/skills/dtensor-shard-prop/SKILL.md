---
name: dtensor-shard-prop
description: Implement or debug PyTorch DTensor sharding propagation rules. Use for DTensor shard prop issues, missing operator rules, wrong placements, register_single_dim_strategy, RuntimeSchemaInfo cache bugs, ExplicitRedistributionContext tests, or torch.distributed.tensor._ops.strategy_validation results.
---

# DTensor Shard Prop

Use `register_single_dim_strategy` for sharding propagation rules, regardless of how an older rule was written. Other sharding rule styles are deprecated for new work.

## Where To Work

- Put almost all changes in `torch/distributed/tensor/_ops/`.
- Read other files under `torch/distributed/tensor/` only to understand DTensor machinery.
- Find existing rules and tests first:
  ```bash
  rg -n "aten\\.<op>|register_.*strategy|register_prop_rule" torch/distributed/tensor/_ops
  rg -n "<op>" test/distributed/tensor
  ```
- Use older `register_op_strategy` or `register_prop_rule` code as reference only when migrating or understanding existing behavior.
- `register_single_dim_strategy` adds the full-`Replicate` case automatically; do not add it yourself.
- In rare cases, update `OpDispatcher` custom op handlers when single-dim strategies cannot express the behavior.
- Treat `RuntimeSchemaInfo` bugs as shard prop cache bugs: over-caching can happen when args that affect sharding are missing from the cache key.
- Register the exact ATen overload. Normalize negative dims before comparing with `Shard(dim)`.
- Add `RuntimeSchemaInfo` for args that affect sharding validity, such as `dim`, `keepdim`, masks, shape args, or output arity.
- Specify one output placement per Tensor output. Use `None` only for non-Tensor outputs or absent optional Tensor outputs.

## Debugging

- Inspect the `OpSchema` to see DTensor input specs and static args.
- Compare selected placements against expected placements.
- Check `needs_redistribute` and `redistribute_schema` for unexpected communication.
- If behavior changes across calls, suspect `RuntimeSchemaInfo` or the shard prop cache key.

## Tests

For unit tests:

- Compare `full_tensor()` numerics against a single-node reference.
- Use `ExplicitRedistributionContext` to ensure no implicit redistributions happen.
- Assert expected output placements.
- Keep tests minimal, but cover the added rules, especially corner cases.
- Common test homes: `test_dtensor_ops.py` for runtime behavior, `test_op_strategy.py` for strategy logic, and `test_common_rules.py` for lower-level propagation rules.

Run the strategy validator and report it. Budget the whole validation at roughly
5-10 minutes; never launch an unbounded full run. The default (no flags) does
expensive per-sample missing-detection (a graph search over placement
combinations for every sample) and can take an hour or more. Do not run it.

Two-step workflow:

1. Correctness gate (the hard requirement). `--incorrect-only` skips
   missing-detection and is cheap, so run it over all samples (no `--max-samples`):

   ```bash
   python -m torch.distributed.tensor._ops.strategy_validation --op <op> --incorrect-only
   ```

   This must report `0` incorrect and `> 0` correct. If it is still slow or you
   are iterating, add `--max-samples <N>`.

2. Missing-case survey (for the report). Run the full mode only with a small
   `--max-samples` (start around `20`, raise only if needed). Missing patterns
   repeat across samples, so a small cap surfaces them quickly; always state the
   cap in the report. Never run this step uncapped.

   ```bash
   python -m torch.distributed.tensor._ops.strategy_validation --op <op> --max-samples 20
   ```

   Missing cases print under "Possibly missing" as `inputs -> output` lines using
   short placement codes (`R`, `S(dim)`, `P(reduce_op)`); add `--show-repro` for
   concrete sample reproducers. Do not grep these out by the words
   Shard/Partial/Replicate.

Run the validator in the background and poll, since even capped runs can take a
few minutes; do not block a single foreground call on it.

If the OpInfo exists, `0` incorrect cases and `> 0` correct cases are required. Nonzero missing cases can be acceptable, but only after investigating and justifying them. Missing cases may come from false-positive-prone OpInfo samples, such as all-zero tensors causing `Partial` ambiguity.

If incorrect or false-positive cases appear, investigate and fix or add rules. If they remain, understand the samples and document why.

If no OpInfo exists, consider adding one with adversarial cases that stress the sharding rules.

## PR Or Commit Report

Include:

- Description of the op and its sharding-related mathematical properties, such as passthrough dimensions, contracting dimensions, and behavior on other dimensions.
- Unconditional rules: sharding rules that hold regardless of inputs.
- Conditional rules: rules that depend on inputs, such as a reduction `dim`.
- Incorrect or added rules: rules removed or added by the change.
- Justification for each rule.
- Strategy validator summary: op, number of correct cases, incorrect cases, and missing cases, plus justification for any missing rules.
