# Contributing to PyTorch Distributed

Please go through PyTorch's top level [Contributing Guide](../../CONTRIBUTING.md) before proceeding with this guide.

[PyTorch Distributed Overview](https://pytorch.org/tutorials//beginner/dist_overview.html) is a great starting point with a lot of tutorials, documentation and design docs covering PyTorch Distributed. We highly recommend going through some of that material before you start working on PyTorch Distributed.

In this document, we mostly focus on some of the code structure for PyTorch distributed and implementation details.

### Onboarding Tasks

A list of onboarding tasks can be found [here](https://github.com/pytorch/pytorch/issues?q=is%3Aopen%20is%3Aissue%20label%3A%22pt_distributed_rampup%22).


## Code Pointers

The relevant code for different modules is either inside the c++ C10D library or the torch python library.

#### Collectives and Communication Library (C10D)

This is the place to look if you are trying to find low-level communication APIs, process group creation, etc.

- API layer: [torch/distributed/distributed_c10d.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/distributed_c10d.py)
- Python Bindings: [torch/csrc/distributed/c10d/init.cpp](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/init.cpp)
- Implementations: [torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/ProcessGroupNCCL.cpp)

#### DTensor

- API layer: ([torch/distributed/_tensor/api.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_tensor/api.py))
- Implementation: see other files in the same folder

#### Distributed Data Parallel (DDP)

- API layer: [torch/nn/parallel/distributed.py](https://github.com/pytorch/pytorch/blob/main/torch/nn/parallel/distributed.py)
- Reducer (backend that schedules allreduces): [torch/csrc/distributed/c10d/reducer.cpp](https://github.com/pytorch/pytorch/blob/main/torch/csrc/distributed/c10d/reducer.cpp)
- Mixed Precision Hooks: [torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/algorithms/ddp_comm_hooks/default_hooks.py)
#### Fully Sharded Data Parallel (FSDP)

- FSDP: [torch/distributed/fsdp/api.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/fsdp/api.py)
- FSDP2: [torch/distributed/_composable/fsdp/fully_shard.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/_composable/fsdp/fully_shard.py)
- Implementations are contained in other files in the same folder as the API for each variant

#### Tensor Parallel (TP)

- API layer: [torch/distributed/tensor/parallel/api.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/tensor/parallel/api.py)
- Implementation: see other files in the same folder

#### Pipeline Parallel (PP)

- Pipeline Schedules: [torch/distributed/pipelining/schedules.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/schedules.py)
- Pipeline Stage: [torch/distributed/pipelining/stage.py](https://github.com/pytorch/pytorch/blob/main/torch/distributed/pipelining/stage.py)


## Adding Tests

You should write tests for your changes just like in other parts of PyTorch, but you may need to use some test infrastructure to run either multi-process tests on multiple GPUs, or use a FakeProcessGroup to mock out communications.

Most testing can be done from python, and you can find existing python tests [here](https://github.com/pytorch/pytorch/tree/main/test/distributed).

For an example of using the MultiProcessTestCase to run a test on multiple GPUs, see tests in [test_c10d_nccl.py](https://github.com/pytorch/pytorch/blob/main/test/distributed/test_c10d_nccl.py)

## Testing Your Changes

All the unit tests can be found under the [test/distributed](../../test/distributed) directory and RPC tests in particular are under [test/distributed/rpc](../../test/distributed/rpc). A few examples on how to run unit tests:


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
