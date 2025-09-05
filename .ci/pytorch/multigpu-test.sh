#!/bin/bash

# Required environment variable: $BUILD_ENVIRONMENT
# (This is set by default in the Docker images we build, so you don't
# need to set it yourself.

# shellcheck source=./common.sh
source "$(dirname "${BASH_SOURCE[0]}")/common.sh"

echo "Testing pytorch"
# When adding more tests, please use HUD to see which shard is shorter
if [[ "${SHARD_NUMBER:-1}" == "1" ]]; then
    # FSDP tests
    for f in test/distributed/fsdp/*.py ; do time python test/run_test.py --verbose -i "${f#*/}" ; done
fi

if [[ "${SHARD_NUMBER:-2}" == "2" ]]; then
    time python test/run_test.py --include test_cuda_multigpu test_cuda_primary_ctx --verbose

    # Disabling tests to see if they solve timeout issues; see https://github.com/pytorch/pytorch/issues/70015
    # python tools/download_mnist.py --quiet -d test/cpp/api/mnist
    # OMP_NUM_THREADS=2 TORCH_CPP_TEST_MNIST_PATH="test/cpp/api/mnist" build/bin/test_api
    time python test/run_test.py --verbose -i distributed/test_c10d_common
    time python test/run_test.py --verbose -i distributed/test_c10d_gloo
    time python test/run_test.py --verbose -i distributed/test_c10d_nccl
    time python test/run_test.py --verbose -i distributed/test_c10d_spawn_gloo
    time python test/run_test.py --verbose -i distributed/test_c10d_spawn_nccl
    time python test/run_test.py --verbose -i distributed/test_compute_comm_reordering
    time python test/run_test.py --verbose -i distributed/test_store
    time python test/run_test.py --verbose -i distributed/test_symmetric_memory
    time python test/run_test.py --verbose -i distributed/test_pg_wrapper
    time python test/run_test.py --verbose -i distributed/rpc/cuda/test_tensorpipe_agent

    # ShardedTensor tests
    time python test/run_test.py --verbose -i distributed/checkpoint/test_checkpoint
    time python test/run_test.py --verbose -i distributed/checkpoint/test_file_system_checkpoint
    time python test/run_test.py --verbose -i distributed/_shard/sharding_spec/test_sharding_spec
    time python test/run_test.py --verbose -i distributed/_shard/sharding_plan/test_sharding_plan
    time python test/run_test.py --verbose -i distributed/_shard/sharded_tensor/test_sharded_tensor
    time python test/run_test.py --verbose -i distributed/_shard/sharded_tensor/test_sharded_tensor_reshard

    # functional collective tests
    time python test/run_test.py --verbose -i distributed/test_functional_api

    # DTensor tests
    time python test/run_test.py --verbose -i distributed/tensor/test_random_ops
    time python test/run_test.py --verbose -i distributed/tensor/test_dtensor_compile
    time python test/run_test.py --verbose -i distributed/tensor/test_utils.py

    # DeviceMesh test
    time python test/run_test.py --verbose -i distributed/test_device_mesh

    # DTensor/TP tests
    time python test/run_test.py --verbose -i distributed/tensor/parallel/test_tp_examples
    time python test/run_test.py --verbose -i distributed/tensor/parallel/test_tp_random_state

    # FSDP2 tests
    time python test/run_test.py --verbose -i distributed/_composable/fsdp/test_fully_shard_training -- -k test_2d_mlp_with_nd_mesh

    # ND composability tests
    time python test/run_test.py --verbose -i distributed/_composable/test_composability/test_2d_composability
    time python test/run_test.py --verbose -i distributed/_composable/test_composability/test_pp_composability

    # Other tests
    time python test/run_test.py --verbose -i test_cuda_primary_ctx
    time python test/run_test.py --verbose -i test_optim -- -k test_forloop_goes_right_direction_multigpu
    time python test/run_test.py --verbose -i test_optim -- -k test_mixed_device_dtype
    time python test/run_test.py --verbose -i test_foreach -- -k test_tensors_grouping
fi
assert_git_not_dirty
