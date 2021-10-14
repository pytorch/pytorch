import sys

import torch
from torch.distributed._sharded_tensor import (
    shard_parameter,
)
from torch.distributed._sharding_spec import (
    ChunkShardingSpec,
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

if TEST_WITH_DEV_DBG_ASAN:
    print(
        "Skip dev-asan as torch + multiprocessing spawn have known issues",
        file=sys.stderr,
    )
    sys.exit(0)


class TestShardedTensorOpsEmbeddingBag(ShardedTensorTestBase):
    def _run_sharded_embedding_bag(self, spec, input_size, weight_size, mode):
        # Use same seed.
        torch.manual_seed(0)
        local_embedding_bag = torch.nn.EmbeddingBag(*weight_size, mode=mode).cuda(
            self.rank
        )

        sharded_embedding_bag = torch.nn.EmbeddingBag(*weight_size, mode=mode)

        # Copy the weights from local embedding bag.
        sharded_embedding_bag.weight = local_embedding_bag.weight

        # Shard the parameter.
        shard_parameter(sharded_embedding_bag, "weight", spec)

        # Run sharded computation
        torch.manual_seed(self.rank)  # inputs different on each rank
        inp = torch.randint(weight_size[0], tuple(input_size)).cuda(self.rank)
        per_sample_weights = None
        if mode == "sum":
            per_sample_weights = torch.rand(*input_size).cuda(self.rank)

        offsets = None
        if len(input_size) == 1:
            offsets = torch.randint(input_size[0], (2,)).cuda(self.rank)
            offsets[0] = 0

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
    def test_sharded_embedding_colwise(self):
        spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
                "rank:3/cuda:3",
            ],
        )

        self._run_sharded_embedding_bag(spec, [5, 4], [17, 12], "mean")
        self._run_sharded_embedding_bag(spec, [6, 7], [21, 11], "max")
        self._run_sharded_embedding_bag(spec, [8, 6], [23, 13], "sum")
        self._run_sharded_embedding_bag(spec, [4, 3], [15, 14], "max")
        self._run_sharded_embedding_bag(spec, [8], [23, 13], "sum")
        self._run_sharded_embedding_bag(spec, [5], [17, 12], "mean")

        # Test different ordering.
        spec = ChunkShardingSpec(
            dim=1,
            placements=[
                "rank:3/cuda:3",
                "rank:0/cuda:0",
                "rank:1/cuda:1",
                "rank:2/cuda:2",
            ],
        )

        self._run_sharded_embedding_bag(spec, [5, 5], [17, 12], "sum")
        self._run_sharded_embedding_bag(spec, [6, 7], [21, 11], "mean")
        self._run_sharded_embedding_bag(spec, [8, 6], [23, 13], "mean")
        self._run_sharded_embedding_bag(spec, [4, 3], [15, 14], "max")
        self._run_sharded_embedding_bag(spec, [4], [15, 14], "max")
        self._run_sharded_embedding_bag(spec, [5], [17, 12], "sum")
