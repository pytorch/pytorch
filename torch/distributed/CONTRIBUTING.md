# Contributing to PyTorch Distributed

Please go through PyTorch's top level [Contributing Guide](../../CONTRIBUTING.md) before proceeding with this guide.

[PyTorch Distributed Overview](https://pytorch.org/tutorials//beginner/dist_overview.html) is a great starting point with a lot of tutorials, documentation and design docs covering PyTorch Distributed. We would highly recommend going through some of that material before you start working on PyTorch Distributed.

In this document, we mostly focus on some of the code structure for PyTorch distributed and implementation details.

### Onboarding Tasks

A list of onboarding tasks can be found [here](https://github.com/pytorch/pytorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22module%3A+distributed%22+label%3A%22topic%3A+bootcamp%22) and [here](https://github.com/pytorch/pytorch/issues?q=is%3Aopen+is%3Aissue+label%3A%22module%3A+distributed%22+label%3Apt_distributed_rampup).


## Testing Your Changes

All the unit tests can be found under the [test/distributed](../../test/distributed) directory and RPC tests in particular are under [test/distributed/rpc](../../test/distributed/rpc). A few examples on how to run unit tests:

TODO: things we should document

- base-classes for tests
- c++ vs python tests
- gpu tests


```
# Run the c10d unit tests.
python test/distributed/test_c10d_common.py
python test/distributed/test_c10d_gloo.py
python test/distributed/test_c10d_nccl.py

# Run the Store tests.
python test/distributed/test_store.py

# Run Process Group Wrapper tests.
python test/distributed/test_pg_wrapper.py

# Run distributed tests, including tests for Distributed Data Parallel.
python test/run_test.py --verbose -i distributed/test_distributed_spawn

# Run a single test in the test_distributed_spawn test suite.
touch /tmp/barrier && TEMP_DIR="/tmp" BACKEND="nccl" WORLD_SIZE="2" python test/distributed/test_distributed_spawn.py -v TestDistBackendWithSpawn.test_ddp_profiling_torch_profiler

# Run a specific test method. Uses pytest (pip install pytest).
# ProcessGroup gloo/nccl test
pytest -vs test/distributed/test_c10d_common.py -k test_multi_limit_single_dtype
```
