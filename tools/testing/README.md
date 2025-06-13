# Testing in CI

This README only discusses what happens in CI through [`run_test.py`][run_test].

## Sharding

PyTorch has a large test suite that takes a long time to execute. To make
testing faster, the testing infrastructure shards, or divides tests across
multiple machines, to reduce total execution time.  The sharding system balances
tests across multiple shards based on their estimated execution time and tries
to distribute tests evenly to minimize the overall test runtime.

Sharding is called by `run_test.py` and can be found in
[test_selections.py][test_selections].

1. Inputs:
   - Shard information such as shard id and total number of shards is set in the
   [workflow files][pull], which are then passed to run_test.py
   - Test files are ordered according to target determination
   - Test timing information:
     - The system uses historical test execution times to make informed sharding decisions
     - Test times are calculated and uploaded by [test-infra][update_test_times],
     pulled during CI by the build job, and saved as a build artifact to ensure
     consistent data across shards
     - If timing data is unavailable for a test, it uses a round-robin approach

2. Shard Calculation:
   - Test files are divided into "serial" and "parallel" groups:
     - Serial tests: Run one at a time (e.g., distributed tests or tests with special requirements)
     - Parallel tests: Can run concurrently with other tests
   - First, serial tests are distributed across shards
   - Then, parallel tests are distributed to balance load across all shards.  See below about parallelism

3. Long-running Test Splitting:
   - Test files exceeding a time threshold (default: 10 minutes) ares split into
   smaller segments.  These are also commonly referred to as shards but will be
   called segments in this README.
   - Test files run with [pytest-shard] flags, which automatically distributes
   tests in the test file among segments
   - Test segments can be seen in the logs by searching for `Serial`, and
   generally look like `<test file> <segment id>/<num segments>`

## Parallelism

In addition to sharding, `run_test.py` also runs multiple test segments on
different processes at the same time on a single machine.

Apply the `@serialTest()` decorator to a unittest to ensure it runs serially if that
is needed for correctness.

Environment variables:
- `PYTORCH_TEST_RUN_EVERYTHING_IN_SERIAL`: If set to "1", runs all tests serially
- `NUM_PROCS`: Controls the number of parallel processes (varies by platform)
- `PYTORCH_TEST_CUDA_MEM_LEAK_CHECK`: Special mode for memory leak checking

## Other

**Reruns**: `run_test.py` includes retry mechanisms to handle test flakiness and
provide better test reliability.  Each unittest gets three subprocess runs with
three runs per subprocess, a total of nine retries.  The first subprocess run is
usually bundled with other unittests, so if a test has exactly 3 failures before
succeeding, this generally means it succeeded in a new process and may have
failed previously due to issues with global state.

**Timeouts**: After sharding and separation into test segments, each test
segment is expected to take no more than 10 minutes.  If a segment takes longer
than 30 minutes, it will be killed via keyboard interrupt and retried like a
normal failure.

**Logs**: To improve log readibility when running in parallel, test logs are not
piped directly to stdout, and only logs corresponding to failing test segments
are printed after it finishes, so timestamps in GHA logs may not be accurate.
All logs can be found in the job artifacts.


## Debugging CI

Most information about how tests were sharded or ran can be found in raw logs.
Some common questions are:

* **Determine which shard/job a test ran on**: Go to the commit page on HUD and use the raw log search in the workflow box
* **Find sharding information**: Search for `Serial` in the raw logs
* **Determine if a test was skipped**: Determine which shard the test ran on, download the log or test report artifacts for that shard/job, and look for that test in the artifacts. Artifacts can be found on HUD, either through the "Artifacts" below the job or ctrl-f job id (the second/last set of numbers in the GHA url) on the HUD commit page.


Page maintainers: @pytorch-dev-infra
<br>
Last verified: 2025-06-13
<!-- Still need to add sections on disabled tests, slow tests, mem leak check, rerun disabled tests, cpp tests, TD -->

[pull]: https://github.com/pytorch/pytorch/blob/ce44877961af7c8ec618f525853ce7edf3efa4eb/.github/workflows/pull.yml#L60
[update_test_times]: https://github.com/pytorch/test-infra/blob/b32b7d6d44e1d90544214e3d960f32172c5c8b87/.github/workflows/update-test-times.yml
[test_selections]: https://github.com/pytorch/pytorch/blob/ce44877961af7c8ec618f525853ce7edf3efa4eb/tools/testing/test_selections.py
[pytest-shard]: https://pypi.org/project/pytest-shard/
[run_test]: https://github.com/pytorch/pytorch/blob/77f884c2ec62df9df930ae86e9b8437364900346/test/run_test.py
