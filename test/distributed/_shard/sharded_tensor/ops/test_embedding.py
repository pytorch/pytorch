# Owner(s): ["oncall: distributed"]

import sys

import torch
import torch.distributed as dist
from torch.distributed._shard import (
    shard_parameter,
)
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
    run_tests,
)
from torch.testing._internal.distributed._shard.sharded_tensor import (
    TEST_GPU_NUM,
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._shard.sharded_tensor._test_ops_common import (
    clone_module_parameter,
    generate_chunk_sharding_specs_for_test,
    generate_local_weight_sharding_params_for_test,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedEmbedding(ShardedTensorTestBase):
    def _run_sharded_embedding(
        self,
        spec,
        input_size,
        num_embeddings,
        embedding_dim,
        max_norm=None,
        norm_type=2.0,
        padding_idx=None,
    ):
        # Use same seed.
        torch.manual_seed(0)
        local_embedding = torch.nn.Embedding(
            num_embeddings,
            embedding_dim,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_idx,
        ).cuda(self.rank)

        sharded_embedding = torch.nn.Embedding(
            num_embeddings,
            embedding_dim,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_idx,
        )

        # Copy the weights from local embedding
        sharded_embedding.weight = clone_module_parameter(
            local_embedding, "weight"
        )

        # Shard the parameter.
        shard_parameter(sharded_embedding, "weight", spec)

        # Run sharded computation
        torch.manual_seed(self.rank)  # inputs different on each rank
        inp = torch.randint(0, num_embeddings, tuple(input_size)).cuda(self.rank)
        sharded_output = sharded_embedding(inp)

        # If max_norm is set, we need to ensure that the renorm has been applied across
        # inputs from all ranks.
        if max_norm is not None:
            gathered_inputs = [torch.zeros_like(inp) for _ in range(TEST_GPU_NUM)]
            dist.all_gather(gathered_inputs, inp)
            unique_inp = torch.unique(torch.cat(gathered_inputs))
            local_embedding(unique_inp)

        # Run local computation
        local_output = local_embedding(inp)

        # Compare local weight and shared one to ensure the renorm
        # as expected.
        if max_norm is not None:
            sharded_dim = spec.dim
            sharded_weight = sharded_embedding.weight.local_shards()[0].tensor
            (start_pos, chunk_size) = generate_local_weight_sharding_params_for_test(
                local_embedding.weight, sharded_dim, TEST_GPU_NUM, spec, self.rank
            )
            local_weight_narrowed = local_embedding.weight.narrow(
                sharded_dim, start_pos, chunk_size
            )
            self.assertEqual(local_weight_narrowed, sharded_weight)

        # Verify
        self.assertEqual(local_output, sharded_output)

        # Validate for torch.nn.functional.embedding version.
        local_output = torch.nn.functional.embedding(
            inp,
            local_embedding.weight,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_idx,
        )
        sharded_output = torch.nn.functional.embedding(
            inp,
            sharded_embedding.weight,
            max_norm=max_norm,
            norm_type=norm_type,
            padding_idx=padding_idx,
        )

        self.assertEqual(local_output, sharded_output)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_embedding_colwise(self):
        for spec in generate_chunk_sharding_specs_for_test(1):
            self._run_sharded_embedding(spec, [5, 4], 17, 12)
            self._run_sharded_embedding(spec, [6, 7, 6], 21, 11)
            self._run_sharded_embedding(spec, [8, 6, 5, 4], 23, 13)
            self._run_sharded_embedding(spec, [8, 6, 5, 4, 7], 23, 16)
            self._run_sharded_embedding(spec, [4], 15, 14)
            self._run_sharded_embedding(spec, [34], 15, 14, padding_idx=10)
            self._run_sharded_embedding(spec, [8, 6, 5, 4], 23, 13, padding_idx=12)
            self._run_sharded_embedding(
                spec, [4, 5, 6], 23, 13, max_norm=2.5,
            )
            self._run_sharded_embedding(
                spec, [12, 7, 16], 23, 13, max_norm=2.5,
            )
            self._run_sharded_embedding(
                spec, [8, 16, 20], 12, 12, max_norm=1.25, norm_type=1.0,
            )
            self._run_sharded_embedding(spec, [30], 15, 14, max_norm=2.0)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(TEST_GPU_NUM)
    @requires_nccl()
    def test_sharded_embedding_rowwise(self):
        for spec in generate_chunk_sharding_specs_for_test(0):
            # Test even split.
            self._run_sharded_embedding(spec, [5, 12], 16, 22)
            self._run_sharded_embedding(spec, [5, 4], 32, 12)
            self._run_sharded_embedding(spec, [6, 7, 6], 64, 11)
            self._run_sharded_embedding(
                spec, [5, 12], 16, 22, max_norm=2.5,
            )
            self._run_sharded_embedding(spec, [6, 7, 6], 64, 11, padding_idx=30)
            self._run_sharded_embedding(
                spec, [6, 5, 3], 26, 11, max_norm=2.0,
            )

            # Test uneven split.
            self._run_sharded_embedding(spec, [8, 6, 5, 4], 19, 11)
            self._run_sharded_embedding(spec, [6, 7, 6], 21, 11)
            self._run_sharded_embedding(spec, [4], 21, 11)
            self._run_sharded_embedding(spec, [8, 6, 5, 4], 21, 11, padding_idx=10)
            self._run_sharded_embedding(
                spec, [6, 5, 8], 28, 5, max_norm=2.0,
            )
            self._run_sharded_embedding(spec, [4], 14, 11, max_norm=2.5)


if __name__ == "__main__":
    run_tests()
