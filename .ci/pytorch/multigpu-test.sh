#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Testing pytorch"
time python test/run_test.py --include test_cuda_multigpu test_cuda_primary_ctx --verbose

# Disabling tests to see if they solve timeout issues; see https://github.com/pytorch/pytorch/issues/70015
# python tools/download_mnist.py --quiet -d test/cpp/api/mnist
# OMP_NUM_THREADS=2 TORCH_CPP_TEST_MNIST_PATH="test/cpp/api/mnist" build/bin/test_api
time python test/run_test.py --verbose -i distributed/test_c10d_common
time python test/run_test.py --verbose -i distributed/test_c10d_gloo
time python test/run_test.py --verbose -i distributed/test_c10d_nccl
time python test/run_test.py --verbose -i distributed/test_c10d_spawn_gloo
time python test/run_test.py --verbose -i distributed/test_c10d_spawn_nccl
time python test/run_test.py --verbose -i distributed/test_store
time python test/run_test.py --verbose -i distributed/test_pg_wrapper
time python test/run_test.py --verbose -i distributed/rpc/cuda/test_tensorpipe_agent
# FSDP tests
for f in test/distributed/fsdp/*.py ; do time python test/run_test.py --verbose -i "${f#*/}" ; done
# ShardedTensor tests
time python test/run_test.py --verbose -i distributed/checkpoint/test_checkpoint
time python test/run_test.py --verbose -i distributed/checkpoint/test_file_system_checkpoint
time python test/run_test.py --verbose -i distributed/_shard/sharding_spec/test_sharding_spec
time python test/run_test.py --verbose -i distributed/_shard/sharding_plan/test_sharding_plan
time python test/run_test.py --verbose -i distributed/_shard/sharded_tensor/test_sharded_tensor
time python test/run_test.py --verbose -i distributed/_shard/sharded_tensor/test_sharded_tensor_reshard
time python test/run_test.py --verbose -i distributed/_shard/sharded_tensor/ops/test_embedding
time python test/run_test.py --verbose -i distributed/_shard/sharded_tensor/ops/test_embedding_bag
time python test/run_test.py --verbose -i distributed/_shard/sharded_tensor/ops/test_binary_cmp
time python test/run_test.py --verbose -i distributed/_shard/sharded_tensor/ops/test_init
time python test/run_test.py --verbose -i distributed/_shard/sharded_optim/test_sharded_optim

# DTensor tests
time python test/run_test.py --verbose -i distributed/_tensor/test_device_mesh.py
time python test/run_test.py --verbose -i distributed/_tensor/test_random_ops.py

# DTensor/TP tests
time python test/run_test.py --verbose -i distributed/tensor/parallel/test_ddp_2d_parallel
time python test/run_test.py --verbose -i distributed/tensor/parallel/test_fsdp_2d_parallel
time python test/run_test.py --verbose -i distributed/tensor/parallel/test_tp_examples

# Other tests
time python test/run_test.py --verbose -i test_cuda_primary_ctx
time python test/run_test.py --verbose -i test_optim -- -k optimizers_with_varying_tensors
time python test/run_test.py --verbose -i test_foreach -- -k test_tensors_grouping
assert_git_not_dirty
