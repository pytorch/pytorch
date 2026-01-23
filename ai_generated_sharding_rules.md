I want you to write a DTensor sharding rule PR for the given operator. The steps are, given an operator:

1) Exhaustively enumerate all possible arg/kwarg combinations for the operator that affects operator semantics (e.g. different modes, kwarg values, having/not having optional arguments, etc.)
  2) For each arg/kwarg combination, adversarially generate several sample inputs for the operator, that stress tests proposed sharding rules. Ensure for shardable dimensions, tensor shapes are at least 8.
  3) Additionally for each arg/kwarg combination, exhaustively enumerate all possible input/output placement combinations on the input/output tensors (i.e. sharding rules) across Replicate/_ShardPlaceholder, as well as Partial(reduce_op=...) options. YOU MUST USE the functions enumerate_placements() and/or enumerate_all_placement_combos() in `torch/testing/_internal/distributed/_tensor/common_dtensor.py`.
    4) For each proposed sharding rule:
      5) For each generated sample input, take the (operator, sample input, sharding rule) instance, and write a Python script that explicitly calls the single-dim validation infra in validate_sharding_rule_sample() (torch/testing/_internal/distributed/_tensor/common_dtensor.py). YOU MUST USE THIS FOR VALIDATION, DO NOT WRITE YOUR OWN VERSION. Execute this on 4 GPUs. If it fails, eliminate the sharding rule.
  6) This gives us a list of sharding rules for this operator that unconditionally work, on the set of adversarially-generated sample inputs. Now analyze the operator semantics, and propose additional conditional sharding prop rules, that are dependent on input conditions. Common ones are dimension-dependent, for example you cannot maintain sharding on a indexing dimension. For this case, consider additional placements in DTensor (e.g. NormPartial, MaskPartial) as well.
    7) For each conditional sharding rule, validate the sharding rule against the subset of generated sample inputs that satisfy the condition. If the subset is non-empty, and there are no failures, include the conditional sharding rule.
8) For the operator, considering all arg/kwarg combinations, and all unconditional & conditional rules discovered, write a PR that registers a single-dim strategy (register_single_dim_strategy) for the operator. Write tests that verify all proposed sample inputs, and all proposed sharding rules. Keep this minimal, no fluff, not too many comments, avoid writing defensive code.
9) Add these changes into a new branch for the operator.


Details:

- The test setup should be on 4 GPUs, running real data.
- For 1), generate the local tensors (one tensor per-rank). This allows us to consider adversarial edge-cases where data on each rank is not equal. Additionally, generate the exact tensor values, not just the shapes/strides, since the values will matter for Partial placements.
- For 2) & 3), rely on the code in `torch/testing/_internal/distributed/_tensor/common_dtensor.py`: enumerate_placements() and enumerate_all_placement_combos().
- Regarding handling of ShardPlaceholder in 3):
  - Some strategies work for Shard/StridedShard with any split_factor, but some may only work for Shard. To distinguish this, for each ShardPlaceholder combination, replace the placeholder with Shard, and StridedShard(split_factor = 2 and 4 and 8). If all of those are valid, keep ShardPlaceholder in the final strategy. If not, use Shard.
