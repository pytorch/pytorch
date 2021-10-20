import sys

import torch
from torch.distributed._sharded_tensor import (
    shard_parameter,
)
from torch.testing._internal.common_distributed import (
    requires_nccl,
    skip_if_lt_x_gpu,
)
from torch.testing._internal.common_utils import (
    TEST_WITH_DEV_DBG_ASAN,
)
from torch.testing._internal.distributed._sharded_tensor import (
    ShardedTensorTestBase,
    with_comms,
)
from torch.testing._internal.distributed._sharded_tensor._test_ops_common import (
    generate_chunk_sharding_specs_for_test,
)

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedEmbedding(ShardedTensorTestBase):
    def _run_sharded_embedding(self, spec, input_size, num_embeddings, embedding_dim):
        # Use same seed.
        torch.manual_seed(0)
        local_embedding = torch.nn.Embedding(num_embeddings, embedding_dim).cuda(
            self.rank
        )

        sharded_embedding = torch.nn.Embedding(num_embeddings, embedding_dim)

        # Copy the weights from local embedding
        sharded_embedding.weight = local_embedding.weight

        # Shard the parameter.
        shard_parameter(sharded_embedding, "weight", spec)

        # Run sharded computation
        torch.manual_seed(self.rank)  # inputs different on each rank
        inp = torch.randint(num_embeddings, tuple(input_size)).cuda(self.rank)
        sharded_output = sharded_embedding(inp)

        # Run local computation
        local_output = local_embedding(inp)

        # Verify
        self.assertEqual(local_output, sharded_output)

        # Validate for torch.nn.functional.embedding version.
        local_output = torch.nn.functional.embedding(inp, local_embedding.weight)
        sharded_output = torch.nn.functional.embedding(inp, sharded_embedding.weight)

        self.assertEqual(local_output, sharded_output)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_embedding_colwise(self):
        for spec in generate_chunk_sharding_specs_for_test(1):
            self._run_sharded_embedding(spec, [5, 4], 17, 12)
            self._run_sharded_embedding(spec, [6, 7, 6], 21, 11)
            self._run_sharded_embedding(spec, [8, 6, 5, 4], 23, 13)
            self._run_sharded_embedding(spec, [8, 6, 5, 4, 7], 23, 16)
            self._run_sharded_embedding(spec, [4], 15, 14)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_embedding_rowwise(self):
        for spec in generate_chunk_sharding_specs_for_test(0):
            # Test even split.
            self._run_sharded_embedding(spec, [5, 12], 16, 22)
            self._run_sharded_embedding(spec, [5, 4], 32, 12)
            self._run_sharded_embedding(spec, [6, 7, 6], 64, 11)

            # Test uneven split.
            self._run_sharded_embedding(spec, [8, 6, 5, 4], 19, 11)
            self._run_sharded_embedding(spec, [6, 7, 6], 21, 11)
            self._run_sharded_embedding(spec, [4], 21, 11)
