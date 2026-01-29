Sharding Rule Tester (ShRT)
---------------------------

We want to automatically validate every sharding rule registered with DTensor.

Sharding rules describe valid combinations of input placements and output placements for an operator.
A combination is valid if and only if the local input tensors for those placements can be fed directly to the operator
on every rank, and the local output tensors can be directly wrapped in a DTensor with the specified output placements.
The output DTensors should have the same values as the operator would have produced on full input tensors.

Goals
------

A few nice properties for this test system:
1. It should automatically detect which operators to test (by looking at DTensor's registration system)
2. It should leverage torch's op DB to find valid input args/kwargs for each operator to test
3. It should use LocalTensor to produce correct results without requiring slow nccl initialization
4. It should focus on testing the sharding rules over a 1-D mesh, since rules can compose over N-D meshes
5. It should enumerate all possible 1-D mesh placements for the tensor inputs to the operator and determine the ground
   truth of which placement combinations are correct or incorrect
6. It should be possible to run as a test in CI that fails if a sharding strategy either produces an invalid rule, or forgets to produce a valid one
7. Optionally, it should support an on-disk (source controlled) database file that lets it skip testing combinations that
   have already been validated and have not changed. It could determine which combinations have changed by factoring
   the test into 2 steps: first, get the sharding strategies for the operator for specified inputs.  Then, if any of these strategies mismatch the cached already tested strategies, run the test on the operator with real tensor values,
   otherwise, continue to the next combination.
8. If using a database file, there should be a local util (script) in torch tools that lets a developer update the cache
   to account for a fixed sharding rule
9. It should have an XFAIL system that tracks already failing combinations, and allows CI to pass until the XFails are addressed and updated with fixes.

Details
-------
There are 3 kinds of sharding rules in DTensor Today
1. Sharding rules (register_sharding_rule) - old, not used much, should not bother testing
2. Sharding Strategies (register_sharding_strategy) - newer, used by most ops, should probably test
3. Single-dim sharding strategies - newest, not used widely yet, but important to test

Consider the tradeoffs between testing (2) and (3).  It might be possible to test (2) on a single mesh dim and reuse the same test infra as developed for (3).  Note that it is possible type 2 rules can produce invalid multi-dim rules, and these would be invisible to our test.  It might be worth the tradeoff to leave this gap in test coverage and avoid more expensive combinatorics in test enumeration, especially since we are trying to migrate type (2) rules to type (3) anyway.
