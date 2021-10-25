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
    run_tests,
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


class TestShardedEmbeddingBag(ShardedTensorTestBase):
    def _run_sharded_embedding_bag(
        self, spec, input_size, num_embeddings, embedding_dim, mode, offset_size=None
    ):
        # Use same seed.
        torch.manual_seed(0)
        local_embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings, embedding_dim, mode=mode
        ).cuda(self.rank)

        sharded_embedding_bag = torch.nn.EmbeddingBag(
            num_embeddings, embedding_dim, mode=mode
        )

        # Copy the weights from local embedding bag.
        sharded_embedding_bag.weight = local_embedding_bag.weight

        # Shard the parameter.
        shard_parameter(sharded_embedding_bag, "weight", spec)

        # Run sharded computation
        torch.manual_seed(self.rank)  # inputs different on each rank
        inp = torch.randint(0, num_embeddings, tuple(input_size)).cuda(self.rank)
        per_sample_weights = None
        if mode == "sum":
            per_sample_weights = torch.rand(*input_size).cuda(self.rank)

        offsets = None
        if len(input_size) == 1:
            # We need to generate certain length offset for each rank.
            # The current implementation and dist API does not support
            # the case when the offset has different lengths.
            # input_size[0] >> offset_size, so the while loop will not
            # for too long.
            while offsets is None or (offsets.size(0) != offset_size):
                offsets = torch.randint(input_size[0], (offset_size,))
                offsets[0] = 0
                offsets = (
                    torch.unique(offsets, sorted=True).contiguous().cuda(self.rank)
                )

        sharded_output = sharded_embedding_bag(
            inp, offsets=offsets, per_sample_weights=per_sample_weights
        )

        # Run local computation
        local_output = local_embedding_bag(
            inp, offsets=offsets, per_sample_weights=per_sample_weights
        )

        # Verify
        self.assertEqual(local_output, sharded_output)

        # Validate for torch.nn.functional.embedding_bag version.
        local_output = torch.nn.functional.embedding_bag(
            inp,
            local_embedding_bag.weight,
            offsets=offsets,
            mode=mode,
            per_sample_weights=per_sample_weights,
        )
        sharded_output = torch.nn.functional.embedding_bag(
            inp,
            sharded_embedding_bag.weight,
            offsets=offsets,
            mode=mode,
            per_sample_weights=per_sample_weights,
        )

        self.assertEqual(local_output, sharded_output)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_embedding_bag_colwise(self):
        return
        for spec in generate_chunk_sharding_specs_for_test(1):
            self._test_sharded_embedding_bag_with_test_cases(spec)

    @with_comms(init_rpc=False)
    @skip_if_lt_x_gpu(4)
    @requires_nccl()
    def test_sharded_embedding_bag_rowwise(self):
        for spec in generate_chunk_sharding_specs_for_test(0):
            self._test_sharded_embedding_bag_with_test_cases(spec)

    def _test_sharded_embedding_bag_with_test_cases(self, spec):
        self._run_sharded_embedding_bag(spec, [5, 5], 17, 14, "sum")
        self._run_sharded_embedding_bag(spec, [5, 4], 17, 12, "mean")
        self._run_sharded_embedding_bag(spec, [6, 7], 21, 11, "max")
        self._run_sharded_embedding_bag(spec, [8, 6], 24, 13, "sum")
        self._run_sharded_embedding_bag(spec, [4, 3], 16, 14, "max")
        self._run_sharded_embedding_bag(spec, [8], 23, 13, "sum", offset_size=3)
        self._run_sharded_embedding_bag(spec, [5], 17, 12, "mean", offset_size=2)
        self._run_sharded_embedding_bag(spec, [12], 16, 12, "max", offset_size=4)


if __name__ == "__main__":
    run_tests()
